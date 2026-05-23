#!/usr/bin/env python3
"""
Test script for level_four P10→P11 segment.
Starts at P10 (standing), runs the same stages as level_four from P10 onward.

States:
  INIT_STAND       – restore height + recovery stand
  ALIGN_YAW_AT_P10 – rotate to initial target yaw at p10
  SET_CROUCH_2     – send crouch params, zero-vel settle at p10
  ALIGN_DIRECTION_3 – re-align before crouch walk at p10
  SQUAT_WALK_2     – crouch walk to p11
  STAND_UP_AT_P11   – restore normal height, settle at p11
  MOVE_TO_P12       – walk forward to p12 (standing)
  TURN_AROUND_2     – in-place 180° turn at p12
  MOVE_TO_P11_BACK  – walk back to p11 (standing)
  SET_CROUCH_3      – send crouch params, zero-vel settle at p11
  SQUAT_WALK_3      – crouch walk back to p10
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

CROUCH_SETTLE_CYCLES = 30  # ~3 seconds

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


class TestTaskNode2(Node):
    def __init__(self):
        super().__init__("test_task_node_2")

        # --- 目标坐标点（与 level_four 完全一致）---
        self.p10 = (-0.0873, 7.8036)
        self.p11 = (-0.1482, 10.2624)
        self.p12 = (-0.1482, 10.9000)

        self.state = "INIT_STAND"
        self.latest_pose = None
        self.state_counter = 0
        self.turn_target_yaw = 0.0

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
        self.get_logger().info("Test 2: P10 → crouch walk → P11 → stop")

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
            self.state = "ALIGN_YAW_AT_P10"

        # ── Stage 1: Align yaw to initial direction at P10 ─────
        elif self.state == "ALIGN_YAW_AT_P10":
            yaw_err = normalize_angle(YAW_ALIGN_TARGET - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 1: aligned, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "SET_CROUCH_2"
                self.state_counter = 0
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw)
                self.get_logger().info(
                    f"Stage 1: aligning yaw — yaw={yaw:.3f}  "
                    f"target={YAW_ALIGN_TARGET:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 2: Crouch via ROS2 params at P10 ─────────────
        elif self.state == "SET_CROUCH_2":
            if self.state_counter == 0:
                self.get_logger().info("Stage 2: Sending crouch params via ROS2...")
                self.send_yaml_params(CROUCH_PARAMS)
                self.get_logger().info("Stage 2: Zero-vel locomotion — lowering body...")

            self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
            self.state_counter += 1

            if self.state_counter >= CROUCH_SETTLE_CYCLES:
                self.state = "ALIGN_DIRECTION_3"
                self.state_counter = 0

        # ── Stage 3: Re-align before crouch walk ───────────────
        elif self.state == "ALIGN_DIRECTION_3":
            yaw_err = normalize_angle(YAW_ALIGN_TARGET - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 3: aligned, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
                self.state = "SQUAT_WALK_2"
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw, mode=11, gait=3)
                self.get_logger().info(
                    f"Stage 3: aligning — yaw={yaw:.3f}  "
                    f"target={YAW_ALIGN_TARGET:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 4: Crouch walk to P11 ────────────────────────
        elif self.state == "SQUAT_WALK_2":
            dy = self.p11[1] - y
            if dy > 0.05:
                self.publish(0.06, 0.0, 0.0, mode=11, gait=3,
                             step_height=[0.05, 0.05])
                self.get_logger().info(
                    f"Stage 4: crouch walk to P11 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "STAND_UP_AT_P11"
                self.state_counter = 0

        # ── Stage 5: Stand up at P11 ────────────────────────────
        elif self.state == "STAND_UP_AT_P11":
            if self.state_counter == 0:
                self.get_logger().info("Stage 5: Restoring normal height, standing up...")
                self.send_yaml_params(RESTORE_PARAMS)

            self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
            self.state_counter += 1

            if self.state_counter >= CROUCH_SETTLE_CYCLES:
                self.state = "MOVE_TO_P12"
                self.state_counter = 0

        # ── Stage 6: Walk forward to P12 (standing) ─────────────
        elif self.state == "MOVE_TO_P12":
            dy = self.p12[1] - y
            if dy > 0.1:
                self.publish(0.15, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 6: walking to P12 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "TURN_AROUND_2"
                self.state_counter = 0

        # ── Stage 7: In-place 180° turn at P12 ──────────────────
        elif self.state == "TURN_AROUND_2":
            if self.state_counter == 0:
                self.turn_target_yaw = normalize_angle(yaw + math.pi)
                self.get_logger().info(
                    f"Stage 7: Turning 180° — yaw={yaw:.3f} → target={self.turn_target_yaw:.3f}"
                )

            yaw_err = normalize_angle(self.turn_target_yaw - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 7: turn complete, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "MOVE_TO_P11_BACK"
                self.state_counter = 0
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw)
            self.state_counter += 1

        # ── Stage 8: Walk back to P11 (standing) ────────────────
        elif self.state == "MOVE_TO_P11_BACK":
            dx_world = self.p11[0] - x
            dy_world = self.p11[1] - y
            dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 8: reached P11, dist={dist:.3f} m"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "SET_CROUCH_3"
                self.state_counter = 0
            else:
                body_vx = POS_ALIGN_KP * (dx_world * math.cos(yaw) + dy_world * math.sin(yaw))
                body_vy = POS_ALIGN_KP * (-dx_world * math.sin(yaw) + dy_world * math.cos(yaw))
                body_vx = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vx))
                body_vy = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vy))
                self.publish(body_vx, body_vy, 0.0)
                self.get_logger().info(
                    f"Stage 8: moving to P11 — x={x:.3f} y={y:.3f}  "
                    f"dist={dist:.3f}  vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 9: Crouch via ROS2 params at P11 ──────────────
        elif self.state == "SET_CROUCH_3":
            if self.state_counter == 0:
                self.get_logger().info("Stage 9: Sending crouch params via ROS2...")
                self.send_yaml_params(CROUCH_PARAMS)
                self.get_logger().info("Stage 9: Zero-vel locomotion — lowering body...")

            self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
            self.state_counter += 1

            if self.state_counter >= CROUCH_SETTLE_CYCLES:
                self.state = "SQUAT_WALK_3"
                self.state_counter = 0

        # ── Stage 10: Crouch walk to P10 ────────────────────────
        elif self.state == "SQUAT_WALK_3":
            dx_world = self.p10[0] - x
            dy_world = self.p10[1] - y
            dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 11: crouch walked to P10, dist={dist:.3f} m"
                )
                self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
                self.state = "FINAL_STOP"
                self.state_counter = 0
            else:
                body_vx = POS_ALIGN_KP * (dx_world * math.cos(yaw) + dy_world * math.sin(yaw))
                body_vy = POS_ALIGN_KP * (-dx_world * math.sin(yaw) + dy_world * math.cos(yaw))
                body_vx = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vx))
                body_vy = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vy))
                self.publish(body_vx, body_vy, 0.0, mode=11, gait=3)
                self.get_logger().info(
                    f"Stage 11: crouch walking — dist={dist:.3f}  "
                    f"vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 11: Damper stop ────────────────────────────────
        elif self.state == "FINAL_STOP":
            self.publish(0, 0, 0, mode=7)
            self.state_counter += 1

            if self.state_counter >= 20:  # ~2 seconds
                self.get_logger().info("Test 2 complete.")
                self.state = "FINISHED"


def main():
    rclpy.init()
    node = TestTaskNode2()
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
