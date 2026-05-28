#!/usr/bin/env python3
import sys
import time
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
# 导入 ROS2 QoS 与 Service 模块
from rclpy.qos import QoSProfile, ReliabilityPolicy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Point, Quaternion

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


# ===================== 视觉控制参数（纯净高敏捷版） =====================
FORWARD_SPEED = 0.12  # 直道稳定速度
KP = 0.0025           # 解放比例系数！起跑线锁死后，可以用高敏捷度确保极其居中
KD = 0.0006           # 微分项

class PerfectStartTrackingNode(Node):
    def __init__(self):
        super().__init__("perfect_start_track3")
        self.bridge = CvBridge()

        # LCM 初始化
        self.lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd = robot_control_cmd_lcmt()

        # 控制变量
        self.error = 0.0
        self.last_error = 0.0
        self.wz = 0.0
        self.has_image = False

        # 状态机变量
        self.forced_right_turn_done = False
        self.is_turning_right = False
        self.turn_start_time = 0.0
        self.is_reached_goal = False
        self.first_image_time = 0.0

        # 位置跟踪变量
        self.latest_pose = None
        self.received_pose = False

        # 配置兼容仿真图像的 QoS
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # 订阅专属物理俯视相机话题
        self.sub_img = self.create_subscription(
            Image,
            "/down_rgb_camera/down_rgb_camera_sensor/image_raw",
            self.image_callback,
            qos_profile
        )

        # 订阅模型状态用于位置跟踪
        self.create_subscription(
            ModelStates,
            "/gazebo/model_states",
            self.on_model_states,
            QoSProfile(depth=10)
        )

        # 创建 ROS 2 客户端，用于全自动设置机器人绝对位置（彻底抹平手动摆放误差）
        self.set_state_client = self.create_client(SetEntityState, "/gazebo/set_entity_state")

    def on_model_states(self, msg):
        try:
            index = list(msg.name).index("robot")
        except ValueError:
            self.received_pose = False
            return

        pose = msg.pose[index]
        q = pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.latest_pose = (pose.position.x, pose.position.y, pose.position.z, yaw)
        self.received_pose = True

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def walk_to_next_level_start(self):
        """走到第四关的起始位置 (3.0777, 7.0474)，朝向 1.5646 rad"""
        target_x = 3.0777
        target_y = 7.0474
        target_yaw = 1.5646
        pos_tolerance = 0.15
        yaw_tolerance = 0.1
        walk_speed = 0.12
        turn_speed = 0.25

        print(f"[🚶 走向第四关] 开始走到第四关起始位置: ({target_x}, {target_y})")

        # 先站立起来
        print("[🔄 站立] 正在站立...")
        for _ in range(60):  # 站立3秒
            rclpy.spin_once(self, timeout_sec=0.01)
            self.cmd.mode = 12
            self.cmd.gait_id = 3
            self.cmd.contact = 15
            self.cmd.vel_des = [0.0, 0.0, 0.0]
            self.cmd.life_count = (self.cmd.life_count + 1) % 128
            self.lc.publish("robot_control_cmd", self.cmd.encode())
            time.sleep(0.05)
        print("[✅ 站立] 站立完成")

        max_walk_time = 60  # 最大行走时间60秒
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.latest_pose is None:
                time.sleep(0.05)
                continue

            # 检查是否超时
            elapsed = time.time() - start_time
            if elapsed > max_walk_time:
                print(f"[⏰ 超时] 已行走{elapsed:.1f}秒，停止行走继续运行")
                self.cmd.mode = 7
                self.cmd.vel_des = [0.0, 0.0, 0.0]
                self.lc.publish("robot_control_cmd", self.cmd.encode())
                return

            x, y, z, yaw = self.latest_pose
            dx = target_x - x
            dy = target_y - y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < pos_tolerance:
                print(f"[✅ 到达] 已到达第四关起始位置附近，距离: {dist:.3f}m")
                break

            target_angle = math.atan2(dy, dx)
            yaw_error = self.normalize_angle(target_angle - yaw)

            if abs(yaw_error) > yaw_tolerance:
                vyaw = turn_speed if yaw_error > 0 else -turn_speed
                self.cmd.mode = 11
                self.cmd.gait_id = 3
                self.cmd.vel_des = [0.0, 0.0, vyaw]
                self.cmd.life_count = (self.cmd.life_count + 1) % 128
                self.lc.publish("robot_control_cmd", self.cmd.encode())
            else:
                self.cmd.mode = 11
                self.cmd.gait_id = 3
                self.cmd.vel_des = [walk_speed, 0.0, 0.0]
                self.cmd.life_count = (self.cmd.life_count + 1) % 128
                self.lc.publish("robot_control_cmd", self.cmd.encode())

            print(f"[🚶 走向第四关] dist={dist:.3f}m, yaw_err={yaw_error:.3f}rad")
            time.sleep(0.05)

        print("[🔄 调整朝向] 正在调整朝向对准第四关...")
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.latest_pose is None:
                time.sleep(0.05)
                continue
            _, _, _, yaw = self.latest_pose
            yaw_error = self.normalize_angle(target_yaw - yaw)
            if abs(yaw_error) < yaw_tolerance:
                break
            vyaw = turn_speed if yaw_error > 0 else -turn_speed
            self.cmd.mode = 11
            self.cmd.gait_id = 3
            self.cmd.vel_des = [0.0, 0.0, vyaw]
            self.cmd.life_count = (self.cmd.life_count + 1) % 128
            self.lc.publish("robot_control_cmd", self.cmd.encode())
            time.sleep(0.05)

        self.cmd.mode = 7
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.lc.publish("robot_control_cmd", self.cmd.encode())
        print("[✅ 完成] 已到达第四关起始位置，准备开始第四关")

    def walk_to_start_position(self):
        """自行走到本关起始位置，高精度版本"""
        target_x = -0.2149
        target_y = 4.8571
        target_yaw = 1.1699  # 67.03度
        pos_tolerance = 0.10  # 位置精度10cm
        yaw_tolerance = 0.05  # 角度精度约3度
        walk_speed = 0.10     # 行走速度
        turn_speed = 0.18     # 转向速度
        max_adjust_time = 30  # 最大调整时间30秒

        print(f"[🚶 走向起始位] 开始走到第三关起始位置: ({target_x}, {target_y}), yaw={target_yaw:.4f}")

        # 先站立起来
        print("[🔄 站立] 正在站立...")
        for _ in range(60):  # 站立3秒
            rclpy.spin_once(self, timeout_sec=0.01)
            self.cmd.mode = 12
            self.cmd.gait_id = 3
            self.cmd.contact = 15
            self.cmd.vel_des = [0.0, 0.0, 0.0]
            self.cmd.life_count = (self.cmd.life_count + 1) % 128
            self.lc.publish("robot_control_cmd", self.cmd.encode())
            time.sleep(0.05)
        print("[✅ 站立] 站立完成")

        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.latest_pose is None:
                time.sleep(0.05)
                continue

            # 检查是否超时
            elapsed = time.time() - start_time
            if elapsed > max_adjust_time:
                print(f"[⏰ 超时] 已调整{elapsed:.1f}秒，停止调整继续运行")
                self.cmd.mode = 7
                self.cmd.vel_des = [0.0, 0.0, 0.0]
                self.lc.publish("robot_control_cmd", self.cmd.encode())
                return

            x, y, z, yaw = self.latest_pose
            dx = target_x - x
            dy = target_y - y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < pos_tolerance:
                # 已到达起始位置附近，精细调整朝向
                yaw_error = self.normalize_angle(target_yaw - yaw)
                if abs(yaw_error) < yaw_tolerance:
                    # 到达目标，保持稳定2秒
                    self.cmd.mode = 7
                    self.cmd.vel_des = [0.0, 0.0, 0.0]
                    self.lc.publish("robot_control_cmd", self.cmd.encode())
                    print(f"[✅ 到达] 已精确到达起始位置: ({x:.4f}, {y:.4f}), yaw={yaw:.4f}")
                    print("[⏳ 稳定] 等待2秒确保稳定...")
                    time.sleep(2.0)
                    return
                else:
                    # 精细调整朝向，使用更小的速度
                    vyaw = 0.08 if yaw_error > 0 else -0.08
                    self.cmd.mode = 11
                    self.cmd.gait_id = 3
                    self.cmd.vel_des = [0.0, 0.0, vyaw]
                    self.cmd.life_count = (self.cmd.life_count + 1) % 128
                    self.lc.publish("robot_control_cmd", self.cmd.encode())
                    print(f"[🔄 精调朝向] yaw_err={yaw_error:.4f}rad ({math.degrees(yaw_error):.2f}°)")
                    time.sleep(0.05)
                    continue

            # 走向起始位置
            target_angle = math.atan2(dy, dx)
            yaw_error = self.normalize_angle(target_angle - yaw)

            if abs(yaw_error) > yaw_tolerance:
                vyaw = turn_speed if yaw_error > 0 else -turn_speed
                self.cmd.mode = 11
                self.cmd.gait_id = 3
                self.cmd.vel_des = [0.0, 0.0, vyaw]
            else:
                self.cmd.mode = 11
                self.cmd.gait_id = 3
                self.cmd.vel_des = [walk_speed, 0.0, 0.0]

            self.cmd.life_count = (self.cmd.life_count + 1) % 128
            self.lc.publish("robot_control_cmd", self.cmd.encode())

            print(f"[🚶 走向起始位] dist={dist:.4f}m ({dist*100:.1f}cm), yaw_err={yaw_error:.4f}rad ({math.degrees(yaw_error):.2f}°)")
            time.sleep(0.05)

    def image_callback(self, msg):
        try:
            if self.is_reached_goal:
                return

            if self.first_image_time == 0.0:
                self.first_image_time = time.time()
            
            elapsed_time = time.time() - self.first_image_time

            # ===================== ⏳ 1. 智能分叉口拐弯状态机 =====================
            # 既然起点现在是 100% 完美的，13.2秒的拦截时间点将会变得极其精准，绝对不会有半秒误差！
            if elapsed_time > 13.2 and not self.forced_right_turn_done and not self.is_turning_right:
                print(f"[智能控制] 准时到达危险分叉口！触发右转拧头机制！")
                self.is_turning_right = True
                self.turn_start_time = time.time()

            if self.is_turning_right:
                if time.time() - self.turn_start_time < 2.5: 
                    self.wz = -0.50  # 完美的盲转力度
                    self.has_image = True
                    return
                else:
                    self.is_turning_right = False
                    self.forced_right_turn_done = True
                    print("[智能控制] 转向成功，切入最终冲刺道！开始监测终点黄线...")

            # ===================== 📸 2. 正常读取相机画面巡线 =====================
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w, _ = cv_image.shape
            
            # 物理级大低头 + 剪裁
            roi = cv_image[int(h*0.82):int(h*0.97), int(w*0.35):int(w*1.0)]
            roi_h, roi_w, _ = roi.shape
            
            # 颜色分割
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([15, 40, 40])
            upper_yellow = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # ===================== 🏁 3. 终点横线全自动碰撞检测 =====================
            if self.forced_right_turn_done:
                left_check   = mask[int(roi_h * 0.85), int(roi_w * 0.15)]  
                center_check = mask[int(roi_h * 0.85), int(roi_w * 0.50)]  
                right_check  = mask[int(roi_h * 0.85), int(roi_w * 0.85)]  
                
                if left_check == 255 and center_check == 255 and right_check == 255:
                    print("[🏁 终点制动] 探测到脚尖前全贯通横向终点线！正在紧急安全落座...")
                    self.wz = 0.0
                    self.error = 0.0
                    self.is_reached_goal = True
                    self.has_image = True
                    return

            # ===================== 🐕 4. 纯净自适应巡线（告别死偏置） =====================
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                screen_center = roi_w / 2
                
                # 【回归纯净】不加任何奇奇怪怪的偏置！
                # 因为起点完美对齐了，狗子会天生无比丝滑地走在正中央，两边踩线的隐患直接从物理上被消灭！
                self.error = screen_center - cx
                
                # PD 控制器
                p_term = KP * self.error
                d_term = KD * (self.error - self.last_error)
                self.wz = p_term + d_term
                self.last_error = self.error
                
                self.wz = max(-0.45, min(self.wz, 0.45))
                self.has_image = True
            else:
                self.wz = -0.22
                self.has_image = True

        except Exception as e:
            self.get_logger().error(f"图像处理失败: {str(e)}")


def main():
    rclpy.init()
    node = PerfectStartTrackingNode()
    lc = node.lc
    cmd = node.cmd
    life = 0

    # ===================== 🚀 走到起始位置 =====================
    node.walk_to_start_position()

    print("[Visual Track] 站立准备中...")
    t_start = time.time()
    while time.time() - t_start < 5:
        cmd.mode = 12
        cmd.gait_id = 3
        cmd.contact = 15
        cmd.vel_des = [0.0, 0.0, 0.0]
        life = (life + 1) % 128
        cmd.life_count = life
        lc.publish("robot_control_cmd", cmd.encode())
        rclpy.spin_once(node, timeout_sec=0.01)
        time.sleep(0.05)

    print("[Visual Track] 开始视觉巡线...")
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if not node.has_image:
                # 等待图像期间继续维持站立，切到 mode=11 前不要停
                cmd.mode = 12
                cmd.vel_des = [0.0, 0.0, 0.0]
                life = (life + 1) % 128
                cmd.life_count = life
                lc.publish("robot_control_cmd", cmd.encode())
                if int(time.time()) % 3 == 0:
                    print("[等待中] 还没收到相机图像，请检查话题 /down_rgb_camera/down_rgb_camera_sensor/image_raw 是否有数据...")
                time.sleep(0.05)
                continue
            
            if node.is_reached_goal:
                break

            # 检查 y 坐标，到达 y>=7 时停止
            if node.latest_pose is not None:
                _, y, _, _ = node.latest_pose
                if y >= 7.0:
                    print(f"[🏁 y坐标到达] y={y:.3f} >= 7.0，停止巡线")
                    node.is_reached_goal = True
                    break
                
            cmd.mode = 11       
            cmd.gait_id = 3
            cmd.contact = 15
            cmd.step_height = [0.25, 0.25]
            
            if node.is_turning_right or abs(node.wz) > 0.05:
                current_vx = 0.04  
            else:
                current_vx = FORWARD_SPEED  
            
            cmd.vel_des = [current_vx, 0.0, node.wz]
            
            life = (life + 1) % 128
            cmd.life_count = life
            lc.publish("robot_control_cmd", cmd.encode())
            
            print(f"Error: {node.error:.2f} | Out Wz: {node.wz:.3f} | Vx: {current_vx:.2f}")
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("[Visual Track] 收到手动停止信号...")

    # ===================== 最终稳妥落座 =====================
    print("[🏁 🏁 🏁] 已安全抵达最终终点线前，收尾落座稳停")
    cmd.mode = 7
    cmd.vel_des = [0.0, 0.0, 0.0]
    lc.publish("robot_control_cmd", cmd.encode())
    time.sleep(3.0)

    # ===================== 走到第四关起始位置 =====================
    node.walk_to_next_level_start()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
