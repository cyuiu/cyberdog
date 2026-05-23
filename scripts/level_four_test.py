#!/usr/bin/env python3
"""
Test script for level_four P5→P8 segment.
Starts at P5 (standing), runs the same stages as level_four from P5 onward.

States:
  INIT_STAND       – restore height + recovery stand
  MOVE_TO_P6       – walk forward to p6 (standing)
  LATERAL_TO_P7    – lateral move to p7 (standing)
  ALIGN_YAW_AT_P7  – adjust yaw at p7
  MOVE_TO_P8       – walk forward to p8
  ALIGN_YAW_AT_P8  – rotate to face initial target yaw at p8
  MOVE_TO_P9       – walk forward to p9
  REVERSE_TO_P8    – reverse walk back to p8
  ALIGN_YAW_P8_REV – align yaw to -2.1007 rad at p8
  MOVE_TO_P10      – walk forward to p10
  ALIGN_YAW_AT_P10 – rotate to initial target yaw at p10
  MOVE_TO_P11      – walk forward to p11
  FINAL_STOP       – damper stop
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


# ── Height parameters (same as level_four) ──────────────────────────
NORMAL_HEIGHT = [0.0, 0.0, 0.25]
NORMAL_HEIGHT_MOTION = [0.0, 0.0, 0.225]
NORMAL_HEIGHT_STAIR = [0.0, 0.0, 0.225]

RESTORE_PARAMS = [
    ("des_roll_pitch_height",        YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT),
    ("des_roll_pitch_height_motion", YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT_MOTION),
    ("des_roll_pitch_height_stair",  YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT_STAIR),
    ("step_height_max",              YamlParam.DOUBLE,       0.06),
]

# Direction alignment
YAW_ALIGN_TARGET = 1.5646     # rad (~90°, face +y)
YAW_ALIGN_TOLERANCE = 0.02
YAW_ALIGN_KP = 0.8
YAW_ALIGN_MAX_VYAW = 0.3
YAW_ALIGN_MIN_VYAW = 0.08

# Position alignment
POS_ALIGN_TOLERANCE = 0.03
POS_ALIGN_KP = 0.5
POS_ALIGN_MAX_VEL = 0.15


def make_yaml_param(name, kind, value, is_user=True):
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


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class TestTaskNode(Node):
    def __init__(self):
        super().__init__("test_task_node")

        # --- 目标坐标点（与 level_four 完全一致）---
        self.p6 = (0.9917, 8.1413)
        self.p7 = (1.8430, 8.2192)
        self.p8 = (0.9750, 9.7024)
        self.p9 = (0.9688, 10.8809)
        self.p10 = (-0.0873, 7.8036)
        self.p11 = (-0.0927, 8.8680)

        self.state = "INIT_STAND"
        self.latest_pose = None
        self.state_counter = 0

        # --- LCM 初始化 ---
        self.lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd = robot_control_cmd_lcmt()

        # --- ROS2 持久 publisher ---
        self.yaml_pub = self.create_publisher(YamlParam, "yaml_parameter", 10)

        self.create_subscription(
            ModelStates, "/gazebo/model_states",
            self.on_pose, qos_profile_sensor_data,
        )
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Test: P6 → P7 → align → P8 → stop")

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

    # ── ROS2 param helpers ────────────────────────────────────
    def send_yaml_params(self, param_list):
        for name, kind, value in param_list:
            ros_msg = make_yaml_param(name, kind, value)
            self.yaml_pub.publish(ros_msg)
            time.sleep(0.05)

    # ── LCM helpers ───────────────────────────────────────────
    def publish(self, vx, vy, vyaw, mode=11, gait=3,
                body_height=None, step_height=None):
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

    # ── Main control loop (10 Hz) ─────────────────────────────
    def control_loop(self):
        if self.latest_pose is None:
            self.get_logger().warn("Waiting for pose...", throttle_duration_sec=2.0)
            return

        x, y, yaw = self.latest_pose

        # ── Stage 0: Stand up ──────────────────────────────────
        if self.state == "INIT_STAND":
            self.get_logger().info("Stage 0: Restoring normal height...")
            self.send_yaml_params(RESTORE_PARAMS)
            time.sleep(0.5)
            self.get_logger().info("Stage 0: Recovery stand...")
            self.recovery_stand_blocking()
            self.state = "MOVE_TO_P6"

        # ── Stage 1: Walk forward to P6 (standing) ─────────────
        if self.state == "MOVE_TO_P6":
            dy = self.p6[1] - y
            if dy > 0.1:
                self.publish(0.15, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 1: walking to P6 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "LATERAL_TO_P7"
                self.state_counter = 0

        # ── Stage 2: Lateral move to P7 (standing) ─────────────
        elif self.state == "LATERAL_TO_P7":
            dx = self.p7[0] - x
            if abs(dx) > 0.1:
                self.publish(0.0, -0.2, 0.0)
                self.get_logger().info(
                    f"Stage 2: lateral move to P7 x={x:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "ALIGN_YAW_AT_P7"
                self.state_counter = 0

        # ── Stage 3: Adjust yaw at P7 ──────────────────────────
        elif self.state == "ALIGN_YAW_AT_P7":
            yaw_target = 2.0183  # P7 目标朝向（与 level_four 一致）
            yaw_err = normalize_angle(yaw_target - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 3: aligned, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "MOVE_TO_P8"
                self.state_counter = 0
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw)
                self.get_logger().info(
                    f"Stage 3: aligning yaw — yaw={yaw:.3f}  "
                    f"target={yaw_target:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 4: Walk forward to P8 ────────────────────────
        elif self.state == "MOVE_TO_P8":
            dy = self.p8[1] - y
            if dy > 0.1:
                self.publish(0.15, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 4: walking to P8 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "ALIGN_YAW_AT_P8"
                self.state_counter = 0

        # ── Stage 5: Adjust yaw at P8 ───────────────────────────
        elif self.state == "ALIGN_YAW_AT_P8":
            yaw_err = normalize_angle(YAW_ALIGN_TARGET - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 5: aligned, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "MOVE_TO_P9"
                self.state_counter = 0
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw)
                self.get_logger().info(
                    f"Stage 5: aligning yaw — yaw={yaw:.3f}  "
                    f"target={YAW_ALIGN_TARGET:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 6: Walk forward to P9 ─────────────────────────
        elif self.state == "MOVE_TO_P9":
            dy = self.p9[1] - y
            if dy > 0.1:
                self.publish(0.15, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 6: walking to P9 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "REVERSE_TO_P8"
                self.state_counter = 0

        # ── Stage 7: Reverse walk back to P8 ────────────────────
        elif self.state == "REVERSE_TO_P8":
            dx_world = self.p8[0] - x
            dy_world = self.p8[1] - y
            dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 7: reversed to P8, dist={dist:.3f} m"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "ALIGN_YAW_P8_REV"
                self.state_counter = 0
            else:
                body_vx = POS_ALIGN_KP * (dx_world * math.cos(yaw) + dy_world * math.sin(yaw))
                body_vy = POS_ALIGN_KP * (-dx_world * math.sin(yaw) + dy_world * math.cos(yaw))
                body_vx = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vx))
                body_vy = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vy))
                self.publish(body_vx, body_vy, 0.0)
                self.get_logger().info(
                    f"Stage 7: reversing — x={x:.3f} y={y:.3f}  "
                    f"dist={dist:.3f}  vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 8: Align yaw at P8 (reverse direction) ────────
        elif self.state == "ALIGN_YAW_P8_REV":
            yaw_target = -2.1007
            yaw_err = normalize_angle(yaw_target - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 8: aligned, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "MOVE_TO_P10"
                self.state_counter = 0
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw)
                self.get_logger().info(
                    f"Stage 8: aligning yaw — yaw={yaw:.3f}  "
                    f"target={yaw_target:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 9: Walk forward to P10 ────────────────────────
        elif self.state == "MOVE_TO_P10":
            dx_world = self.p10[0] - x
            dy_world = self.p10[1] - y
            dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 9: reached P10, dist={dist:.3f} m"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "ALIGN_YAW_AT_P10"
                self.state_counter = 0
            else:
                body_vx = POS_ALIGN_KP * (dx_world * math.cos(yaw) + dy_world * math.sin(yaw))
                body_vy = POS_ALIGN_KP * (-dx_world * math.sin(yaw) + dy_world * math.cos(yaw))
                body_vx = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vx))
                body_vy = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vy))
                self.publish(body_vx, body_vy, 0.0)
                self.get_logger().info(
                    f"Stage 9: walking to P10 — x={x:.3f} y={y:.3f}  "
                    f"dist={dist:.3f}  vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 10: Align yaw to initial direction at P10 ─────
        elif self.state == "ALIGN_YAW_AT_P10":
            yaw_err = normalize_angle(YAW_ALIGN_TARGET - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 10: aligned, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "MOVE_TO_P11"
                self.state_counter = 0
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw)
                self.get_logger().info(
                    f"Stage 10: aligning yaw — yaw={yaw:.3f}  "
                    f"target={YAW_ALIGN_TARGET:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 11: Walk forward to P11 ───────────────────────
        elif self.state == "MOVE_TO_P11":
            dy = self.p11[1] - y
            if dy > 0.1:
                self.publish(0.15, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 11: walking to P11 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "FINAL_STOP"
                self.state_counter = 0

        # ── Stage 12: Damper stop ──────────────────────────────
        elif self.state == "FINAL_STOP":
            self.publish(0, 0, 0, mode=7)
            self.state_counter += 1

            if self.state_counter >= 20:  # ~2 seconds
                self.get_logger().info("Test complete.")
                self.state = "FINISHED"


def main():
    rclpy.init()
    node = TestTaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C caught, restoring normal height...")
        node.send_yaml_params(RESTORE_PARAMS)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
