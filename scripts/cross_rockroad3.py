#!/usr/bin/env python3
import math
import sys
import time

LCM_PYTHON_PATH = "/home/lcm/build/python"
CONTROL_PATH = "/home/loco_hl_example/sequential_motion"

if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)
if CONTROL_PATH not in sys.path:
    sys.path.insert(0, CONTROL_PATH)

import lcm
import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from robot_control_cmd_lcmt import robot_control_cmd_lcmt


class CrossRockroad3(Node):
    def __init__(self):
        super().__init__("cross_rockroad3")

        self.declare_parameter("model_name", "robot")
        self.declare_parameter("target_yaw", 0.0)
        self.declare_parameter("yaw_deadband", 0.06)
        self.declare_parameter("turn_speed", 0.09)
        self.declare_parameter("forward_speed", 0.35)
        self.declare_parameter("step_height", 0.25)
        self.declare_parameter("target_x", 3.00)
        self.declare_parameter("max_abs_y", 0.35)
        self.declare_parameter("max_time", 70.0)
        self.declare_parameter("control_period", 0.1)

        self.declare_parameter("entry_target_x", 3.06)
        self.declare_parameter("entry_target_y", 0.88)
        self.declare_parameter("entry_target_yaw", 1.57)
        self.declare_parameter("entry_lateral_speed", 0.10)
        self.declare_parameter("entry_forward_speed", 0.06)
        self.declare_parameter("entry_x_deadband", 0.06)
        self.declare_parameter("entry_y_deadband", 0.05)

        self.model_name = self.get_parameter("model_name").value
        self.target_yaw = float(self.get_parameter("target_yaw").value)
        self.yaw_deadband = float(self.get_parameter("yaw_deadband").value)
        self.turn_speed = float(self.get_parameter("turn_speed").value)
        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.step_height = float(self.get_parameter("step_height").value)
        self.target_x = float(self.get_parameter("target_x").value)
        self.max_abs_y = float(self.get_parameter("max_abs_y").value)
        self.max_time = float(self.get_parameter("max_time").value)
        self.control_period = float(self.get_parameter("control_period").value)

        self.entry_target_x = float(self.get_parameter("entry_target_x").value)
        self.entry_target_y = float(self.get_parameter("entry_target_y").value)
        self.entry_target_yaw = float(self.get_parameter("entry_target_yaw").value)
        self.entry_lateral_speed = float(self.get_parameter("entry_lateral_speed").value)
        self.entry_forward_speed = float(self.get_parameter("entry_forward_speed").value)
        self.entry_x_deadband = float(self.get_parameter("entry_x_deadband").value)
        self.entry_y_deadband = float(self.get_parameter("entry_y_deadband").value)

        self.lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd = self.make_cmd()

        self.latest_pose = None
        self.received_pose = False
        self.state = "ALIGN_YAW"
        self.start_time = None
        self.final_stopped = False
        self.last_zone = ""

        self.stones = [
            ("stone_1", 0.606, 0.887),
            ("stone_2", 1.106, 1.387),
            ("stone_3", 1.606, 1.887),
            ("stone_4", 2.106, 2.387),
        ]

        self.create_subscription(
            ModelStates,
            "/gazebo/model_states",
            self.on_model_states,
            qos_profile_sensor_data,
        )
        self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            "cross_rockroad3 | speed=%.2f | step_height=%.2f | target_x=%.2f | entry=(%.2f, %.2f, %.2f)"
            % (
                self.forward_speed,
                self.step_height,
                self.target_x,
                self.entry_target_x,
                self.entry_target_y,
                self.entry_target_yaw,
            )
        )

        self.recovery_stand()

    def make_cmd(self):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = 3
        msg.contact = 15
        msg.duration = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.rpy_des = [0.0, 0.0, 0.0]
        msg.pos_des = [0.0, 0.0, 0.0]
        msg.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.ctrl_point = [0.0, 0.0, 0.0]
        msg.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.step_height = [self.step_height, self.step_height]
        return msg

    def publish_cmd(self, vx, vy, yaw):
        self.cmd.mode = 11
        self.cmd.gait_id = 3
        self.cmd.contact = 15
        self.cmd.duration = 0
        self.cmd.vel_des = [float(vx), float(vy), float(yaw)]
        self.cmd.step_height = [self.step_height, self.step_height]
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def damper_stop(self):
        self.cmd.mode = 7
        self.cmd.gait_id = 0
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.duration = 0
        self.cmd.life_count = (self.cmd.life_count + 1) % 128
        self.lc.publish("robot_control_cmd", self.cmd.encode())

    def walk_to_next_level_start(self):
        """走到第二关的起始位置 (3.0985, 0.6795)，朝向 1.417 rad"""
        target_x = 3.0985
        target_y = 0.6795
        target_yaw = 1.417
        pos_tolerance = 0.15
        yaw_tolerance = 0.1
        walk_speed = 0.12
        turn_speed = 0.25

        self.get_logger().info(f"开始走到第二关起始位置: ({target_x}, {target_y})")

        # 先站立起来
        self.get_logger().info("正在站立...")
        for _ in range(60):  # 站立3秒
            rclpy.spin_once(self, timeout_sec=0.01)
            self.cmd.mode = 12
            self.cmd.gait_id = 0
            self.cmd.contact = 0
            self.cmd.vel_des = [0.0, 0.0, 0.0]
            self.cmd.life_count = (self.cmd.life_count + 1) % 128
            self.lc.publish("robot_control_cmd", self.cmd.encode())
            time.sleep(0.05)
        self.get_logger().info("站立完成")

        max_walk_time = 60
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.latest_pose is None:
                time.sleep(0.05)
                continue

            elapsed = time.time() - start_time
            if elapsed > max_walk_time:
                self.get_logger().info(f"已行走{elapsed:.1f}秒超时，停止行走")
                self.publish_cmd(0.0, 0.0, 0.0)
                return

            x, y, z, yaw = self.latest_pose
            dx = target_x - x
            dy = target_y - y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < pos_tolerance:
                self.get_logger().info(f"已到达第二关起始位置附近，距离: {dist:.3f}m")
                break

            target_angle = math.atan2(dy, dx)
            yaw_error = self.normalize_angle(target_angle - yaw)

            if abs(yaw_error) > yaw_tolerance:
                vyaw = turn_speed if yaw_error > 0 else -turn_speed
                self.publish_cmd(0.0, 0.0, vyaw)
            else:
                self.publish_cmd(walk_speed, 0.0, 0.0)

            self.get_logger().info(
                f"走向第二关: dist={dist:.3f}m, yaw_err={yaw_error:.3f}rad",
                throttle_duration_sec=1.0,
            )
            time.sleep(0.05)

        self.get_logger().info("调整朝向对准第二关...")
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
            self.publish_cmd(0.0, 0.0, vyaw)
            time.sleep(0.05)

        self.publish_cmd(0.0, 0.0, 0.0)
        self.get_logger().info("已到达第二关起始位置，准备开始第二关")

    def recovery_stand(self):
        self.get_logger().info("Recovery stand for 5 seconds...")
        self.cmd.mode = 12
        self.cmd.gait_id = 0
        self.cmd.contact = 0
        self.cmd.vel_des = [0.0, 0.0, 0.0]
        self.cmd.duration = 5000

        end_time = time.time() + 5.0
        while time.time() < end_time:
            self.cmd.life_count = (self.cmd.life_count + 1) % 128
            self.lc.publish("robot_control_cmd", self.cmd.encode())
            time.sleep(0.05)

        self.cmd = self.make_cmd()
        self.get_logger().info("Recovery stand done.")

    def on_model_states(self, msg):
        try:
            index = list(msg.name).index(self.model_name)
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

    def get_zone(self, x):
        for name, x0, x1 in self.stones:
            if x0 <= x <= x1:
                return name

        for i in range(len(self.stones) - 1):
            left = self.stones[i]
            right = self.stones[i + 1]
            if left[2] < x < right[1]:
                return "gap_%d_%d" % (i + 1, i + 2)

        if x < self.stones[0][1]:
            return "approach"
        if x > self.stones[-1][2]:
            return "after_rockroad"
        return "unknown"

    def control_loop(self):
        if self.final_stopped:
            self.publish_cmd(0.0, 0.0, 0.0)
            return

        if not self.received_pose or self.latest_pose is None:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.get_logger().info("state=WAIT_POSE | action=stop")
            return

        x, y, z, yaw = self.latest_pose
        yaw_error = self.normalize_angle(yaw - self.target_yaw)

        if self.state == "ALIGN_YAW" and abs(yaw_error) > self.yaw_deadband:
            yaw_cmd = self.turn_speed if yaw_error < 0.0 else -self.turn_speed
            action = "turn_left" if yaw_error < 0.0 else "turn_right"
            self.publish_cmd(0.0, 0.0, yaw_cmd)
            self.get_logger().info(
                "state=ALIGN_YAW | action=%s | yaw=%.3f | pose=(%.3f, %.3f, %.3f)"
                % (action, yaw, x, y, z)
            )
            return

        if self.state == "ALIGN_YAW":
            self.state = "CROSS"
            self.start_time = time.time()
            self.get_logger().info(
                "state=CROSS | start pose=(%.3f, %.3f, %.3f) | yaw=%.3f"
                % (x, y, z, yaw)
            )

        elapsed = time.time() - self.start_time if self.start_time else 0.0
        zone = self.get_zone(x)

        if zone != self.last_zone:
            self.get_logger().info("zone changed: %s | x=%.3f" % (zone, x))
            self.last_zone = zone

        if self.state == "CROSS" and x >= self.target_x:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.state = "SHIFT_TO_ENTRY"
            self.get_logger().info(
                "state=SHIFT_TO_ENTRY | reason=target_x | pose=(%.3f, %.3f, %.3f) | yaw=%.3f"
                % (x, y, z, yaw)
            )
            return

        if self.state == "SHIFT_TO_ENTRY":
            y_error = self.entry_target_y - y

            if abs(y_error) > self.entry_y_deadband:
                vy = self.entry_lateral_speed if y_error > 0.0 else -self.entry_lateral_speed
                self.publish_cmd(0.0, vy, 0.0)
                self.get_logger().info(
                    "state=SHIFT_TO_ENTRY | y=%.3f/%.3f | vy=%.2f | pose=(%.3f, %.3f, %.3f)"
                    % (y, self.entry_target_y, vy, x, y, z)
                )
                return

            self.publish_cmd(0.0, 0.0, 0.0)
            self.state = "ADJUST_ENTRY_X"
            self.get_logger().info("state=ADJUST_ENTRY_X | y=%.3f" % y)
            return

        if self.state == "ADJUST_ENTRY_X":
            x_error = self.entry_target_x - x

            if abs(x_error) > self.entry_x_deadband:
                vx = self.entry_forward_speed if x_error > 0.0 else -0.5 * self.entry_forward_speed
                self.publish_cmd(vx, 0.0, 0.0)
                self.get_logger().info(
                    "state=ADJUST_ENTRY_X | x=%.3f/%.3f | vx=%.2f | y=%.3f"
                    % (x, self.entry_target_x, vx, y)
                )
                return

            self.publish_cmd(0.0, 0.0, 0.0)
            self.state = "ALIGN_ENTRY_YAW"
            self.get_logger().info("state=ALIGN_ENTRY_YAW | x=%.3f" % x)
            return

        if self.state == "ALIGN_ENTRY_YAW":
            yaw_error = self.normalize_angle(yaw - self.entry_target_yaw)

            if abs(yaw_error) > self.yaw_deadband:
                yaw_cmd = self.turn_speed if yaw_error < 0.0 else -self.turn_speed
                action = "turn_left" if yaw_error < 0.0 else "turn_right"
                self.publish_cmd(0.0, 0.0, yaw_cmd)
                self.get_logger().info(
                    "state=ALIGN_ENTRY_YAW | action=%s | yaw=%.3f/%.3f | err=%.3f | yaw_cmd=%.2f"
                    % (action, yaw, self.entry_target_yaw, yaw_error, yaw_cmd)
                )
                return

            self.publish_cmd(0.0, 0.0, 0.0)
            self.final_stopped = True
            self.get_logger().info(
                "state=FINAL_STOP | reason=entry_ready | pose=(%.3f, %.3f, %.3f) | yaw=%.3f"
                % (x, y, z, yaw)
            )
            return

        if self.state == "CROSS" and abs(y) > self.max_abs_y:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.final_stopped = True
            self.get_logger().info(
                "state=FINAL_STOP | reason=y_limit | y=%.3f | pose=(%.3f, %.3f, %.3f)"
                % (y, x, y, z)
            )
            return

        if elapsed >= self.max_time:
            self.publish_cmd(0.0, 0.0, 0.0)
            self.final_stopped = True
            self.get_logger().info(
                "state=FINAL_STOP | reason=max_time | time=%.2f | pose=(%.3f, %.3f, %.3f)"
                % (elapsed, x, y, z)
            )
            return

        yaw_correction = 0.0
        if abs(yaw_error) > 0.03:
            yaw_correction = -0.6 * yaw_error
            yaw_correction = max(-0.12, min(0.12, yaw_correction))

        # Gaps get a little more commitment so the robot does not hesitate between slabs.
        vx = self.forward_speed
        if zone.startswith("gap"):
            vx = max(self.forward_speed, 0.12)

        self.publish_cmd(vx, 0.0, yaw_correction)
        self.get_logger().info(
            "state=CROSS | zone=%s | vx=%.2f | x=%.3f/%.3f | y=%.3f | z=%.3f | yaw=%.3f | yaw_cmd=%.3f | time=%.1f/%.1f"
            % (
                zone,
                vx,
                x,
                self.target_x,
                y,
                z,
                yaw,
                yaw_correction,
                elapsed,
                self.max_time,
            )
        )


def main():
    rclpy.init()
    node = CrossRockroad3()

    try:
        # 使用循环代替 rclpy.spin，以便检测任务完成
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.final_stopped:
                node.get_logger().info("任务完成，准备走到第二关起始位置")
                break
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, damper stop")
        node.damper_stop()

    node.publish_cmd(0.0, 0.0, 0.0)
    # 走到第二关起始位置
    node.walk_to_next_level_start()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()