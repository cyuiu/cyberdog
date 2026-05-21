#!/usr/bin/env python3
"""
Sequential task: stand → lateral move → sprint → crouch walk → stop.

Crouch height is switched dynamically via ROS2 "yaml_parameter" topic,
same approach as set.py. Normal locomotion scripts are unaffected.

States:
  STAND_UP   – recovery stand (mode=12)
  MOVE_TO_P2 – lateral move to p2
  MOVE_TO_P3 – sprint forward to p3
  SET_CROUCH – send crouch params via ROS2, then zero-vel locomotion
               to let the height filter lower the body
  SQUAT_WALK – mode=11 + gait_id=3 slow forward at crouch height
  FINAL_STOP – restore normal height, damper stop
"""
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
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from rclpy.qos import qos_profile_sensor_data
from cyberdog_msg.msg import YamlParam

try:
    from robot_control_cmd_lcmt import robot_control_cmd_lcmt
except ImportError:
    import robot_control_cmd_lcmt


# ── Height parameters (same as set.py) ────────────────────────────
CROUCH_HEIGHT = [0.0, 0.0, 0.15]
NORMAL_HEIGHT = [0.0, 0.0, 0.25]
NORMAL_HEIGHT_MOTION = [0.0, 0.0, 0.225]
NORMAL_HEIGHT_STAIR = [0.0, 0.0, 0.225]

CROUCH_PARAMS = [
    ("des_roll_pitch_height",        YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("des_roll_pitch_height_motion", YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("des_roll_pitch_height_stair",  YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("step_height_max",              YamlParam.DOUBLE,       0.04),
]

RESTORE_PARAMS = [
    ("des_roll_pitch_height",        YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT),
    ("des_roll_pitch_height_motion", YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT_MOTION),
    ("des_roll_pitch_height_stair",  YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT_STAIR),
    ("step_height_max",              YamlParam.DOUBLE,       0.06),
]

# Time (in control cycles at 10 Hz) for zero-vel locomotion during crouch settle
CROUCH_SETTLE_CYCLES = 30  # ~3 seconds


def make_yaml_param(name, kind, value, is_user=True):
    """Build a YamlParam ROS2 message."""
    msg = YamlParam()
    msg.name = name
    msg.kind = kind
    msg.is_user = 1 if is_user else 0
    if kind == YamlParam.VEC_X_DOUBLE:
        arr = [0.0] * 12
        for i, v in enumerate(value):
            arr[i] = float(v)
        msg.vecxd_value = arr
    elif kind == YamlParam.DOUBLE:
        msg.double_value = float(value)
    return msg


class SequentialTaskNode(Node):
    def __init__(self):
        super().__init__("sequential_task_node")

        # --- 目标坐标点 ---
        self.p1 = (3.08, 6.96)   # 初始起立点
        self.p2 = (2.13, 7.13)   # 横移目标点
        self.p3 = (2.09, 9.68)   # 冲刺结束点，开始准备蹲下
        self.p4 = (2.02, 10.98)  # 最终蹲着走到的目标点

        self.state = "STAND_UP"
        self.latest_pose = None
        self.state_counter = 0

        # --- LCM 初始化 ---
        self.lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd = robot_control_cmd_lcmt()

        # --- ROS2 持久 publisher（不在回调里临时创建）---
        self.yaml_pub = self.create_publisher(YamlParam, "yaml_parameter", 10)

        self.create_subscription(
            ModelStates, "/gazebo/model_states",
            self.on_pose, qos_profile_sensor_data,
        )
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            "Task: stand → lateral → sprint → crouch (ROS2 params) → walk → stop"
        )

    def on_pose(self, msg):
        try:
            idx = msg.name.index("robot")
            curr = msg.pose[idx]
            q = curr.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            self.latest_pose = (curr.position.x, curr.position.y, yaw)
        except (ValueError, IndexError):
            pass

    # ── ROS2 param helpers ────────────────────────────────────────
    def send_yaml_params(self, param_list):
        """Publish yaml parameters via the persistent ROS2 publisher."""
        for name, kind, value in param_list:
            ros_msg = make_yaml_param(name, kind, value)
            self.yaml_pub.publish(ros_msg)
            time.sleep(0.05)

    # ── LCM helpers ───────────────────────────────────────────────
    def publish(self, vx, vy, vyaw, mode=11, gait=3,
                body_height=None, step_height=None):
        """Send one robot_control_cmd via LCM."""
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

    def recovery_stand_blocking(self):
        """Blocking recovery stand for ~1.5s."""
        for _ in range(30):
            self.publish(0, 0, 0, mode=12)
            time.sleep(0.05)

    # ── Main control loop (10 Hz) ─────────────────────────────────
    def control_loop(self):
        if self.latest_pose is None:
            self.get_logger().warn("Waiting for pose...", throttle_duration_sec=2.0)
            return

        x, y, yaw = self.latest_pose

        # ── Stage 1: Stand up ─────────────────────────────────
        if self.state == "STAND_UP":
            self.get_logger().info("Stage 1: Recovery stand...")
            self.recovery_stand_blocking()
            self.state = "MOVE_TO_P2"

        # ── Stage 2: Lateral move to P2 ────────────────────────
        elif self.state == "MOVE_TO_P2":
            dx = self.p2[0] - x
            if abs(dx) > 0.1:
                self.publish(0.0, 0.2, 0.0)
                self.get_logger().info(
                    f"Stage 2: lateral move x={x:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "MOVE_TO_P3"

        # ── Stage 3: Sprint to P3 ──────────────────────────────
        elif self.state == "MOVE_TO_P3":
            dy = self.p3[1] - y
            if dy > 0.1:
                self.publish(0.4, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 3: sprint y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "SET_CROUCH"
                self.state_counter = 0

        # ── Stage 4: Crouch via ROS2 params + zero-vel settle ──
        elif self.state == "SET_CROUCH":
            if self.state_counter == 0:
                self.get_logger().info("Stage 4: Sending crouch params via ROS2...")
                self.send_yaml_params(CROUCH_PARAMS)
                self.get_logger().info("Stage 4: Zero-vel locomotion — lowering body...")

            self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
            self.state_counter += 1

            if self.state_counter >= CROUCH_SETTLE_CYCLES:
                self.state = "SQUAT_WALK"
                self.state_counter = 0
                self.get_logger().info("Stage 4 done, entering crouch walk.")

        # ── Stage 5: Crouch walk to P4 ─────────────────────────
        elif self.state == "SQUAT_WALK":
            dy = self.p4[1] - y
            if dy > 0.05:
                self.publish(0.06, 0.0, 0.0, mode=11, gait=3,
                             step_height=[0.05, 0.05])
                self.get_logger().info(
                    f"Stage 5: crouch walk y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "FINAL_STOP"
                self.state_counter = 0

        # ── Stage 6: Restore height + damper stop ──────────────
        elif self.state == "FINAL_STOP":
            if self.state_counter == 0:
                self.get_logger().info("Stage 6: Restoring normal height...")
                self.send_yaml_params(RESTORE_PARAMS)
            self.publish(0, 0, 0, mode=7)
            self.state_counter += 1

            if self.state_counter >= 20:  # ~2 seconds
                self.get_logger().info("Task complete.")
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
