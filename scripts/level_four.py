#!/usr/bin/env python3
import math
import sys
import time

# --- 路径配置 ---
LCM_PYTHON_PATH = "/home/lcm/build/python"
CONTROL_PATH = "/home/loco_hl_example/sequential_motion"
if LCM_PYTHON_PATH not in sys.path: sys.path.insert(0, LCM_PYTHON_PATH)
if CONTROL_PATH not in sys.path: sys.path.insert(0, CONTROL_PATH)

import lcm
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from rclpy.qos import qos_profile_sensor_data

try:
    from robot_control_cmd_lcmt import robot_control_cmd_lcmt
except ImportError:
    import robot_control_cmd_lcmt

class SequentialTaskNode(Node):
    def __init__(self):
        super().__init__("sequential_task_node")
        
        # --- 目标坐标点 (从你的数据中精准提取) ---
        self.p1 = (3.08, 6.96)   # 初始起立点
        self.p2 = (2.13, 7.13)   # 横移目标点
        self.p3 = (2.09, 9.68)   # 冲刺结束点，开始准备蹲下
        self.p4 = (2.02, 10.98)  # 最终蹲着走到的目标点
        
        self.state = "STAND_UP"
        self.latest_pose = None
        self.state_counter = 0  # 用于非阻塞状态内的计时
        
        # --- LCM 初始化 ---
        self.lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd = robot_control_cmd_lcmt()
        
        self.create_subscription(ModelStates, "/gazebo/model_states", self.on_pose, qos_profile_sensor_data)
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("任务启动：站立 -> 横移 -> 冲刺 -> 压低身体 -> 蹲走")

    def on_pose(self, msg):
        try:
            # 这里的名字需要匹配 Gazebo 中的机器人 Model 名
            idx = msg.name.index("robot") 
            curr = msg.pose[idx]
            q = curr.orientation
            # 四元数转 Yaw
            yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))
            self.latest_pose = (curr.position.x, curr.position.y, yaw)
        except (ValueError, IndexError):
            pass

    def publish(self, vx, vy, vyaw, mode=11, gait=3, body_height=None, step_height=None):
        """
        vx: 前进速度
        vy: 侧移速度 (正向左)
        body_height: 身体绝对高度(米)，None 表示不修改
        step_height: 摆腿离地高度 [前腿, 后腿]，None 表示不修改
        """
        self.cmd.mode = mode
        self.cmd.gait_id = gait
        self.cmd.contact = 15 if mode == 11 else 0
        self.cmd.vel_des = [float(vx), float(vy), float(vyaw)]
        self.cmd.duration = 0 if mode == 11 else self.cmd.duration
        if body_height is not None:
            self.cmd.pos_des = [0.0, 0.0, float(body_height)]
        if step_height is not None:
            self.cmd.step_height = [float(step_height[0]), float(step_height[1])]
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def set_body_height(self, height, duration=500):
        """通过 mode=21 位置插值设置身体绝对高度(米)，站姿约 0.15-0.22m。"""
        self.cmd.mode = 21
        self.cmd.gait_id = 0
        self.cmd.contact = 0       # 清掉前阶段 mode=11 残留的 contact=15
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.rpy_des = [0.0, 0.0, 0.0]
        self.cmd.pos_des = [0.0, 0.0, float(height)]
        self.cmd.duration = duration
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def control_loop(self):
        if self.latest_pose is None:
            self.get_logger().warn("等待位姿数据...", throttle_duration_sec=2.0)
            return
            
        x, y, yaw = self.latest_pose

        # 阶段 1: 初始起立
        if self.state == "STAND_UP":
            self.get_logger().info("阶段 1: 正在起立...")
            for _ in range(30):
                self.publish(0, 0, 0, mode=12) # 恢复站立模式
                time.sleep(0.05)
            self.state = "MOVE_TO_P2"

        # 阶段 2: 横移至 P2 (x=3.08 -> x=2.13)
        elif self.state == "MOVE_TO_P2":
            dx = self.p2[0] - x
            if abs(dx) > 0.1:
                # 给一个正的 Vy 让狗往左横着走
                self.publish(0.0, 0.2, 0.0) 
                self.get_logger().info(f"阶段 2: 横移中... x={x:.2f}", throttle_duration_sec=1.0)
            else:
                self.state = "MOVE_TO_P3"

        # 阶段 3: 快速直行至 P3 (y=7.13 -> y=9.68)
        elif self.state == "MOVE_TO_P3":
            dy = self.p3[1] - y
            if dy > 0.1:
                # 快速前进
                self.publish(0.4, 0.0, 0.0)
                self.get_logger().info(f"阶段 3: 冲刺中... y={y:.2f}", throttle_duration_sec=1.0)
            else:
                self.state = "SET_CROUCH"

        # 阶段 4: 用 mode=21 压低身体到蹲姿高度（持续发多条确保生效）
        elif self.state == "SET_CROUCH":
            if self.state_counter == 0:
                self.get_logger().info("阶段 4: 压低身体到蹲姿...")
            self.set_body_height(0.08, duration=500)  # 绝对高度 0.10m
            self.state_counter += 1
            if self.state_counter >= 12:  # 12 周期 = 1.2 秒
                self.state = "SQUAT_WALK"
                self.state_counter = 0

        # 阶段 5: 蹲着走到终点 P4 (y=9.68 -> y=10.98)
        elif self.state == "SQUAT_WALK":
            dy = self.p4[1] - y
            if dy > 0.05:
                # 先 mode=21 强制压低身体，再 mode=11 走路
                self.set_body_height(0.10, duration=0)
                self.publish(0.06, 0.0, 0.0, mode=11, gait=27,
                             step_height=[0.05, 0.05])
                self.get_logger().info(f"阶段 5: 蹲着走... y={y:.2f}", throttle_duration_sec=1.0)
            else:
                self.state = "FINAL_STOP"

        # 阶段 6: 到达目的地彻底趴下
        elif self.state == "FINAL_STOP":
            self.get_logger().info("任务完成，彻底趴下。")
            self.publish(0, 0, 0, mode=7) # 进入阻尼模式或待机模式
            self.state = "FINISHED"

def main():
    rclpy.init()
    node = SequentialTaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()