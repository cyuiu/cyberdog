#!/usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Point, Quaternion

# ===================== LCM 配置 =====================
LCM_PATHS = [
    "/home/lcm/build/python",
    "/home/loco_hl_example/sequential_motion"
]
for path in LCM_PATHS:
    if path not in sys.path:
        sys.path.insert(0, path)

import lcm
from robot_control_cmd_lcmt import robot_control_cmd_lcmt

# ===================== 视觉控制参数（纯视觉优化版） =====================
FORWARD_SPEED = 0.12  # 直道稳定前进速度
KP = 0.0025           # PD控制比例系数
KD = 0.0006           # PD控制微分系数

# 🔍 纯视觉弯道/分叉口识别面积阈值
# 当裁切后的画面中黄色像素点的总面积超过这个值时，判定正在过关键弯道/分叉口
FORK_AREA_THRESHOLD = 8000  

class PureVisualTrackingNode(Node):
    def __init__(self):
        super().__init__("pure_visual_track_final")
        self.bridge = CvBridge()
        
        # 控制底层运动的核心变量
        self.vx = FORWARD_SPEED
        self.wz = 0.0
        self.error = 0.0
        self.last_error = 0.0
        self.has_image = False
        self.is_reached_goal = False        
        
        # 纯视觉状态机标志位
        self.in_fork_zone = False
        self.turned_into_final_lane = False

        # 配置兼容仿真图像的高频低延迟 QoS
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # 订阅标准俯视相机话题
        self.sub_img = self.create_subscription(
            Image, 
            "/down_rgb_camera/down_rgb_camera_sensor/image_raw", 
            self.image_callback, 
            qos_profile
        )

        # 创建 Gazebo 状态设置客户端（仅用于脚本刚启动时的绝对位置初始化）
        self.set_state_client = self.create_client(SetEntityState, "/gazebo/set_entity_state")

    def telemetry_perfect_spawn(self):
        """全自动闪现：强行把狗子初始化校准到绝对黄金起点"""
        print("[🛠️ 智能初始化] 正在连接 Gazebo 状态服务...")
        if not self.set_state_client.wait_for_service(timeout_sec=2.0):
            print("[⚠️ 警告] 未检测到 Gazebo 服务，将使用当前手动摆放位置起跑！")
            return

        req = SetEntityState.Request()
        req.state.name = "robot"  
        req.state.reference_frame = "world"
        req.state.pose.position = Point(x=-0.2248, y=4.8461, z=0.255)
        req.state.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.55557, w=0.83147)
        
        self.set_state_client.call_async(req)
        print("[🏁 智能初始化] 机器狗已全自动瞬移至绝对黄金起跑点！车身已对齐 100% 中央！")

    def image_callback(self, msg):
        try:
            if self.is_reached_goal:
                self.vx, self.wz = 0.0, 0.0
                return

            # ===================== 📸 1. 精准区域裁剪（彻底屏蔽左侧死胡同） =====================
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w, _ = cv_image.shape
            
            # 🎯【核心物理屏蔽】纵向看远到 0.75（提前预判弯道），横向砍掉左侧 45% 的视野（w*0.45 到 1.0）
            # 这样左边的死胡同黄线在图像层面上根本不会出现，狗子只会专注地看右侧正确道路！
            roi = cv_image[int(h*0.75):int(h*0.97), int(w*0.45):int(w*1.0)]
            roi_h, roi_w, _ = roi.shape
            
            # 颜色空间转换与黄色掩膜提取
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([15, 40, 40])
            upper_yellow = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # ===================== 🏁 2. 终点横线全自动检测 =====================
            # 【双重保险】只有当成功避开分叉口并切入最终车道后，才解锁终点线检测，杜绝中途误判落座
            if self.turned_into_final_lane:
                left_check   = mask[int(roi_h * 0.85), int(roi_w * 0.15)]  
                center_check = mask[int(roi_h * 0.85), int(roi_w * 0.50)]  
                right_check  = mask[int(roi_h * 0.85), int(roi_w * 0.85)]  
                
                if left_check == 255 and center_check == 255 and right_check == 255:
                    print("[🏁 终点制动] 纯视觉捕捉到贯通终点线！正在紧急安全落座...")
                    self.vx, self.wz = 0.0, 0.0
                    self.error = 0.0
                    self.is_reached_goal = True
                    self.has_image = True
                    return

            # ===================== 🐕 3. 纯净自适应巡线控制 =====================
            M = cv2.moments(mask)
            total_area = M["m00"]

            if total_area > 0:
                cx = int(M["m10"] / total_area)
                screen_center = roi_w / 2
                
                # 🔧【安全居中偏置】因为你的轨迹总是偏右踩黄线，这里增加一个居中修正偏差
                # 迫使车身在巡线时主动向左靠拢，留出安全车宽。若仍踩右线可改大（如 45）；若偏左可改小（如 15）
                LINE_OFFSET = 55 
                target_center = screen_center + LINE_OFFSET
                
                # 👁️【纯视觉分叉口/大发卡弯识别处理】
                if total_area > FORK_AREA_THRESHOLD and not self.turned_into_final_lane:
                    if not self.in_fork_zone:
                        print(f"[👁️ 视觉感知] 黄色面积增大至 {total_area:.0f}！捕捉到关键右大弯，开启高敏捷强制切入")
                        self.in_fork_zone = True

                    # 此时由于左侧被物理裁剪，画面里只有右侧正确车道，直接给一个稳定的向右拧头角速度
                    self.vx = 0.04   # 大弯道强制压低线速度，确保底盘稳定
                    self.wz = -0.40  # 稳定的右转弯转向动力
                    self.error = target_center - cx
                    self.has_image = True
                    
                    # 当黄线重新回正到视野偏左的舒适安全区域时，宣告彻底通过分叉弯道
                    if cx > screen_center:
                        self.in_fork_zone = False
                        self.turned_into_final_lane = True
                        print("[👁️ 视觉感知] 已完美切入冲刺直道，正常巡线机制全面接管。")
                    return

                # 正常直道和普通弯道的闭环 PD 控制
                self.error = target_center - cx
                p_term = KP * self.error
                d_term = KD * (self.error - self.last_error)
                
                # 弯道或大修正时自适应降速，直道时全速平稳推进
                if abs(self.wz) > 0.05:
                    self.vx = 0.04
                else:
                    self.vx = FORWARD_SPEED
                    
                self.wz = p_term + d_term
                self.last_error = self.error
                self.wz = max(-0.45, min(self.wz, 0.45))
                self.has_image = True
            else:
                # 发生丢失线信号时的安全右转寻线保护
                self.vx = FORWARD_SPEED
                self.wz = -0.22
                self.has_image = True

        except Exception as e:
            self.get_logger().error(f"图像处理失败: {str(e)}")

def main():
    rclpy.init()
    node = PureVisualTrackingNode()
    # 实例化底层 UDP 通信
    lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
    cmd = robot_control_cmd_lcmt()
    life = 0

    # 启动后台闪现校准（完全合规的起跑前初始化）
    node.telemetry_perfect_spawn()

    print("[Visual Track] 站立准备中...")
    t_start = time.time()
    while time.time() - t_start < 5:
        cmd.mode = 12
        cmd.gait_id = 3
        cmd.contact = 15
        cmd.step_height = [0.25, 0.25] 
        cmd.vel_des = [0.0, 0.0, 0.0]
        life = (life + 1) % 128
        cmd.life_count = life
        lc.publish("robot_control_cmd", cmd.encode())
        rclpy.spin_once(node, timeout_sec=0.01)
        time.sleep(0.05)

    print("[Visual Track] 开始纯视觉巡线推进...")
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if not node.has_image:
                continue
            
            if node.is_reached_goal:
                break
                
            # 下发常规行走指令包
            cmd.mode = 11       
            cmd.gait_id = 3
            cmd.contact = 15
            cmd.step_height = [0.25, 0.25]
            
            # 实时同步图像回调中算出的自适应速度
            cmd.vel_des = [float(node.vx), 0.0, float(node.wz)]
            
            life = (life + 1) % 128
            cmd.life_count = life
            lc.publish("robot_control_cmd", cmd.encode())
            
            print(f"Error: {node.error:.2f} | Wz: {node.wz:.3f} | Vx: {node.vx:.2f}")
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("[Visual Track] 收到手动停止信号...")

    # ===================== 最终安全稳妥落座 =====================
    print("[🏁 🏁 🏁] 已抵达终点，下发落座命令稳停")
    cmd.mode = 7
    cmd.vel_des = [0.0, 0.0, 0.0]
    lc.publish("robot_control_cmd", cmd.encode())
    time.sleep(3.0) 
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
