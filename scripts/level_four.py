#!/usr/bin/env python3
"""
Sequential task: stand → lateral move → sprint → crouch walk → reverse → turn → return.

Crouch height is switched dynamically via ROS2 "yaml_parameter" topic,
same approach as set.py. Normal locomotion scripts are unaffected.

States:
  STAND_UP         – recovery stand (mode=12)
  ALIGN_DIRECTION  – rotate in place to face target direction
  ALIGN_POSITION   – move to exact p1 start position
  MOVE_TO_P2       – lateral move to p2
  MOVE_TO_P3       – sprint forward to p3
  ALIGN_POSITION_2 – fine-tune position to exact p3 before crouch
  SET_CROUCH       – send crouch params via ROS2, then zero-vel locomotion
                     to let the height filter lower the body
  ALIGN_DIRECTION_2 – re-align before crouch walk (correct drift)
  SQUAT_WALK       – mode=11 + gait_id=3 slow forward at crouch height
  RETURN_TO_P3     – reverse walk back to p3 (crouch)
  STAND_UP_AT_P3   – restore normal height, settle at p3
  TURN_AROUND      – in-place 180° turn at p3 (standing)
  MOVE_TO_P2_BACK  – walk back to p2 (standing)
  ALIGN_YAW_AT_P2  – rotate to face initial target yaw at p2
  MOVE_TO_P5       – lateral move to p5 (standing)
  MOVE_TO_P6       – walk forward to p6 (standing)
  LATERAL_TO_P7    – lateral move to p7 (standing)
  ALIGN_YAW_AT_P7  – adjust yaw at p7
  MOVE_TO_P8       – walk forward to p8
  ALIGN_YAW_AT_P8  – rotate to face initial target yaw at p8
  MOVE_TO_P9       – walk forward to p9
  REVERSE_TO_P8     – reverse walk back to p8
  ALIGN_YAW_P8_REV  – align yaw to -2.1007 rad at p8
  MOVE_TO_P10       – walk forward to p10
  ALIGN_YAW_AT_P10  – rotate to initial target yaw at p10
  SET_CROUCH_2      – send crouch params, zero-vel settle at p10
  ALIGN_DIRECTION_3  – re-align before crouch walk at p10
  SQUAT_WALK_2      – crouch walk to p11
  STAND_UP_AT_P11    – restore normal height, settle at p11
  MOVE_TO_P12        – walk forward to p12 (standing)
  TURN_AROUND_2      – in-place 180° turn at p12
  MOVE_TO_P11_BACK   – walk back to p11 (standing)
  SET_CROUCH_3       – send crouch params, zero-vel settle at p11
  SQUAT_WALK_3       – crouch walk back to p10
  FINAL_STOP        – damper stop
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

# Direction alignment
YAW_ALIGN_TARGET = 1.5646     # rad (~90°, face +y), fixed desired yaw
YAW_ALIGN_TOLERANCE = 0.02    # rad (~1.1°), stop rotating when error below this
YAW_ALIGN_KP = 0.8            # proportional gain (lower = smoother approach)
YAW_ALIGN_MAX_VYAW = 0.3      # max angular speed (rad/s), slow for precision
YAW_ALIGN_MIN_VYAW = 0.08     # min angular speed to overcome static friction

# Position alignment
POS_ALIGN_TOLERANCE = 0.03    # m, stop moving when distance error below this
POS_ALIGN_KP = 0.5            # proportional gain for position correction
POS_ALIGN_MAX_VEL = 0.15      # max correction speed (m/s)


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


def normalize_angle(angle):
    """将角度归一化到 [-π, π] 范围。"""
    return math.atan2(math.sin(angle), math.cos(angle))


class SequentialTaskNode(Node):
    def __init__(self):
        super().__init__("sequential_task_node")

        # --- 目标坐标点 ---
        self.p1 = (3.0777, 7.0474)  # 初始起立点
        self.p2 = (2.0026, 7.0393)  # 横移目标点
        self.p3 = (2.0729, 9.6959)  # 冲刺结束点，开始准备蹲下
        self.p4 = (2.1071, 11.0007)  # 最终蹲着走到的目标点
        self.p5 = (0.9734, 7.1048)  # 最后横移目标点
        self.p6 = (0.9917, 8.1413)  # 前进目标点
        self.p7 = (1.8430, 8.2192)  # 横移终点，调整朝向
        self.p8 = (0.9750, 9.7024)  # 前进终点
        self.p9 = (0.9688, 10.8809)  # 前进终点
        self.p10 = (-0.0873, 7.8036)  # 最终前进点
        self.p11 = (-0.1482, 10.2624)  # 蹲走最终点
        self.p12 = (-0.1482, 10.9000)  # 站姿前进终点

        self.state = "STAND_UP"
        self.latest_pose = None
        self.state_counter = 0
        self.turn_target_yaw = 0.0

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
            "Task: stand → lateral → sprint → crouch walk → reverse → 180° turn → return to P2 → stop"
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
            # 无论上次如何退出，启动就恢复正常高度，防止蹲姿参数残留
            self.get_logger().info("Stage 1: Restoring normal height...")
            self.send_yaml_params(RESTORE_PARAMS)
            time.sleep(0.5)
            self.get_logger().info("Stage 1: Recovery stand...")
            self.recovery_stand_blocking()
            self.state = "ALIGN_DIRECTION"

        # ── Stage 2: Align direction ───────────────────────────
        elif self.state == "ALIGN_DIRECTION":
            yaw_err = normalize_angle(YAW_ALIGN_TARGET - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 2: aligned, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "ALIGN_POSITION"
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                # keep sign and enforce minimum magnitude
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw)
                self.get_logger().info(
                    f"Stage 2: aligning — yaw={yaw:.3f}  "
                    f"target={YAW_ALIGN_TARGET:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 3: Align position to P1 ──────────────────────
        elif self.state == "ALIGN_POSITION":
            dx = self.p1[0] - x
            dy = self.p1[1] - y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 3: position aligned, dist={dist:.3f} m"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "MOVE_TO_P2"
            else:
                # world-frame error → body-frame velocity
                body_vx = POS_ALIGN_KP * (dx * math.cos(yaw) + dy * math.sin(yaw))
                body_vy = POS_ALIGN_KP * (-dx * math.sin(yaw) + dy * math.cos(yaw))
                # limit speed
                body_vx = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vx))
                body_vy = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vy))
                self.publish(body_vx, body_vy, 0.0)
                self.get_logger().info(
                    f"Stage 3: position aligning — x={x:.3f} y={y:.3f}  "
                    f"dist={dist:.3f}  vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 4: Lateral move to P2 ────────────────────────
        elif self.state == "MOVE_TO_P2":
            dx = self.p2[0] - x
            if abs(dx) > 0.1:
                self.publish(0.0, 0.2, 0.0)
                self.get_logger().info(
                    f"Stage 4: lateral move x={x:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "MOVE_TO_P3"

        # ── Stage 5: Sprint to P3 ──────────────────────────────
        elif self.state == "MOVE_TO_P3":
            dy = self.p3[1] - y
            if dy > 0.1:
                self.publish(0.4, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 5: sprint y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "ALIGN_POSITION_2"
                self.state_counter = 0

        # ── Stage 6: Align position to P3 ──────────────────────
        elif self.state == "ALIGN_POSITION_2":
            dx = self.p3[0] - x
            dy = self.p3[1] - y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 6: position aligned, dist={dist:.3f} m"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "SET_CROUCH"
                self.state_counter = 0
            else:
                body_vx = POS_ALIGN_KP * (dx * math.cos(yaw) + dy * math.sin(yaw))
                body_vy = POS_ALIGN_KP * (-dx * math.sin(yaw) + dy * math.cos(yaw))
                body_vx = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vx))
                body_vy = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vy))
                self.publish(body_vx, body_vy, 0.0)
                self.get_logger().info(
                    f"Stage 6: position aligning — x={x:.3f} y={y:.3f}  "
                    f"dist={dist:.3f}  vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 7: Crouch via ROS2 params + zero-vel settle ──
        elif self.state == "SET_CROUCH":
            if self.state_counter == 0:
                self.get_logger().info("Stage 7: Sending crouch params via ROS2...")
                self.send_yaml_params(CROUCH_PARAMS)
                self.get_logger().info("Stage 7: Zero-vel locomotion — lowering body...")

            self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
            self.state_counter += 1

            if self.state_counter >= CROUCH_SETTLE_CYCLES:
                self.state = "ALIGN_DIRECTION_2"
                self.state_counter = 0
                self.get_logger().info("Stage 7 done, re-aligning before crouch walk.")

        # ── Stage 8: Re-align before crouch walk ───────────────
        elif self.state == "ALIGN_DIRECTION_2":
            yaw_err = normalize_angle(YAW_ALIGN_TARGET - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 8: aligned, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
                self.state = "SQUAT_WALK"
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw, mode=11, gait=3)
                self.get_logger().info(
                    f"Stage 8: aligning — yaw={yaw:.3f}  "
                    f"target={YAW_ALIGN_TARGET:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 9: Crouch walk to P4 ─────────────────────────
        elif self.state == "SQUAT_WALK":
            dy = self.p4[1] - y
            if dy > 0.05:
                self.publish(0.06, 0.0, 0.0, mode=11, gait=3,
                             step_height=[0.05, 0.05])
                self.get_logger().info(
                    f"Stage 9: crouch walk y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "RETURN_TO_P3"
                self.state_counter = 0

        # ── Stage 10: Reverse walk back to P3 ─────────────────
        elif self.state == "RETURN_TO_P3":
            # world-frame error from current pos → P3
            dx_world = self.p3[0] - x
            dy_world = self.p3[1] - y
            dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 10: returned to P3, dist={dist:.3f} m"
                )
                self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
                self.state = "STAND_UP_AT_P3"
                self.state_counter = 0
                self.get_logger().info("Stage 10 done, standing up at P3.")
            else:
                # rotate world error → body frame (same as ALIGN_POSITION)
                body_vx = POS_ALIGN_KP * (dx_world * math.cos(yaw) + dy_world * math.sin(yaw))
                body_vy = POS_ALIGN_KP * (-dx_world * math.sin(yaw) + dy_world * math.cos(yaw))
                body_vx = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vx))
                body_vy = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vy))
                self.publish(body_vx, body_vy, 0.0, mode=11, gait=3)
                self.get_logger().info(
                    f"Stage 10: reversing — x={x:.3f} y={y:.3f}  "
                    f"dist={dist:.3f}  vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 11: Stand up at P3 ──────────────────────────
        elif self.state == "STAND_UP_AT_P3":
            if self.state_counter == 0:
                self.get_logger().info("Stage 11: Restoring normal height, standing up...")
                self.send_yaml_params(RESTORE_PARAMS)

            self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
            self.state_counter += 1

            if self.state_counter >= CROUCH_SETTLE_CYCLES:
                self.state = "TURN_AROUND"
                self.state_counter = 0
                self.get_logger().info("Stage 11 done, starting 180° turn (standing).")

        # ── Stage 12: In-place 180° turn at P3 (standing) ─────
        elif self.state == "TURN_AROUND":
            if self.state_counter == 0:
                self.turn_target_yaw = normalize_angle(yaw + math.pi)
                self.get_logger().info(
                    f"Stage 12: Turning 180° — yaw={yaw:.3f} → target={self.turn_target_yaw:.3f}"
                )

            yaw_err = normalize_angle(self.turn_target_yaw - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 12: turn complete, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
                self.state = "MOVE_TO_P2_BACK"
                self.state_counter = 0
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw, mode=11, gait=3)
                self.get_logger().info(
                    f"Stage 12: turning — yaw={yaw:.3f}  "
                    f"target={self.turn_target_yaw:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )
            self.state_counter += 1

        # ── Stage 13: Walk back to P2 (standing) ───────────────
        elif self.state == "MOVE_TO_P2_BACK":
            dx_world = self.p2[0] - x
            dy_world = self.p2[1] - y
            dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 13: reached P2, dist={dist:.3f} m"
                )
                self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
                self.state = "ALIGN_YAW_AT_P2"
                self.state_counter = 0
                self.get_logger().info("Stage 13 done, aligning yaw to initial direction.")
            else:
                body_vx = POS_ALIGN_KP * (dx_world * math.cos(yaw) + dy_world * math.sin(yaw))
                body_vy = POS_ALIGN_KP * (-dx_world * math.sin(yaw) + dy_world * math.cos(yaw))
                body_vx = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vx))
                body_vy = max(-POS_ALIGN_MAX_VEL, min(POS_ALIGN_MAX_VEL, body_vy))
                self.publish(body_vx, body_vy, 0.0, mode=11, gait=3)
                self.get_logger().info(
                    f"Stage 13: moving to P2 — x={x:.3f} y={y:.3f}  "
                    f"dist={dist:.3f}  vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 14: Align yaw at P2 ──────────────────────────
        elif self.state == "ALIGN_YAW_AT_P2":
            yaw_err = normalize_angle(YAW_ALIGN_TARGET - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 14: aligned, yaw_err={yaw_err:.3f} rad"
                )
                self.publish(0.0, 0.0, 0.0)
                self.state = "MOVE_TO_P5"
                self.state_counter = 0
            else:
                vyaw = YAW_ALIGN_KP * yaw_err
                if abs(vyaw) < YAW_ALIGN_MIN_VYAW:
                    vyaw = YAW_ALIGN_MIN_VYAW if yaw_err > 0 else -YAW_ALIGN_MIN_VYAW
                vyaw = max(-YAW_ALIGN_MAX_VYAW, min(YAW_ALIGN_MAX_VYAW, vyaw))
                self.publish(0.0, 0.0, vyaw)
                self.get_logger().info(
                    f"Stage 14: aligning yaw — yaw={yaw:.3f}  "
                    f"target={YAW_ALIGN_TARGET:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 15: Lateral move to P5 (standing) ────────────
        elif self.state == "MOVE_TO_P5":
            dx = self.p5[0] - x
            if abs(dx) > 0.1:
                self.publish(0.0, 0.2, 0.0)
                self.get_logger().info(
                    f"Stage 15: lateral move to P5 x={x:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "MOVE_TO_P6"
                self.state_counter = 0

        # ── Stage 16: Walk forward to P6 (standing) ─────────────
        elif self.state == "MOVE_TO_P6":
            dy = self.p6[1] - y
            if dy > 0.1:
                self.publish(0.15, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 16: walking to P6 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "LATERAL_TO_P7"
                self.state_counter = 0

        # ── Stage 17: Lateral move to P7 (standing) ─────────────
        elif self.state == "LATERAL_TO_P7":
            dx = self.p7[0] - x
            if abs(dx) > 0.1:
                self.publish(0.0, -0.2, 0.0)
                self.get_logger().info(
                    f"Stage 17: lateral move to P7 x={x:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "ALIGN_YAW_AT_P7"
                self.state_counter = 0

        # ── Stage 18: Adjust yaw at P7 ──────────────────────────
        elif self.state == "ALIGN_YAW_AT_P7":
            yaw_target = 2.0183  # P7 目标朝向
            yaw_err = normalize_angle(yaw_target - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 18: aligned, yaw_err={yaw_err:.3f} rad"
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
                    f"Stage 18: aligning yaw — yaw={yaw:.3f}  "
                    f"target={yaw_target:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 19: Walk forward to P8 ────────────────────────
        elif self.state == "MOVE_TO_P8":
            dy = self.p8[1] - y
            if dy > 0.1:
                self.publish(0.15, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 19: walking to P8 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "ALIGN_YAW_AT_P8"
                self.state_counter = 0

        # ── Stage 20: Adjust yaw at P8 ──────────────────────────
        elif self.state == "ALIGN_YAW_AT_P8":
            yaw_err = normalize_angle(YAW_ALIGN_TARGET - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 20: aligned, yaw_err={yaw_err:.3f} rad"
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
                    f"Stage 20: aligning yaw — yaw={yaw:.3f}  "
                    f"target={YAW_ALIGN_TARGET:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 21: Walk forward to P9 ────────────────────────
        elif self.state == "MOVE_TO_P9":
            dy = self.p9[1] - y
            if dy > 0.1:
                self.publish(0.15, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 21: walking to P9 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "REVERSE_TO_P8"
                self.state_counter = 0

        # ── Stage 22: Reverse walk back to P8 ───────────────────
        elif self.state == "REVERSE_TO_P8":
            dx_world = self.p8[0] - x
            dy_world = self.p8[1] - y
            dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 22: reversed to P8, dist={dist:.3f} m"
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
                    f"Stage 22: reversing — x={x:.3f} y={y:.3f}  "
                    f"dist={dist:.3f}  vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 23: Align yaw at P8 (reverse direction) ───────
        elif self.state == "ALIGN_YAW_P8_REV":
            yaw_target = -2.1007  # 倒退到P8后的目标朝向
            yaw_err = normalize_angle(yaw_target - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 23: aligned, yaw_err={yaw_err:.3f} rad"
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
                    f"Stage 23: aligning yaw — yaw={yaw:.3f}  "
                    f"target={yaw_target:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 24: Walk forward to P10 ───────────────────────
        elif self.state == "MOVE_TO_P10":
            dx_world = self.p10[0] - x
            dy_world = self.p10[1] - y
            dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 24: reached P10, dist={dist:.3f} m"
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
                    f"Stage 24: walking to P10 — x={x:.3f} y={y:.3f}  "
                    f"dist={dist:.3f}  vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 25: Align yaw to initial direction at P10 ─────
        elif self.state == "ALIGN_YAW_AT_P10":
            yaw_err = normalize_angle(YAW_ALIGN_TARGET - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 25: aligned, yaw_err={yaw_err:.3f} rad"
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
                    f"Stage 25: aligning yaw — yaw={yaw:.3f}  "
                    f"target={YAW_ALIGN_TARGET:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 26: Crouch via ROS2 params at P10 ────────────
        elif self.state == "SET_CROUCH_2":
            if self.state_counter == 0:
                self.get_logger().info("Stage 26: Sending crouch params via ROS2...")
                self.send_yaml_params(CROUCH_PARAMS)
                self.get_logger().info("Stage 26: Zero-vel locomotion — lowering body...")

            self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
            self.state_counter += 1

            if self.state_counter >= CROUCH_SETTLE_CYCLES:
                self.state = "ALIGN_DIRECTION_3"
                self.state_counter = 0

        # ── Stage 27: Re-align before crouch walk ──────────────
        elif self.state == "ALIGN_DIRECTION_3":
            yaw_err = normalize_angle(YAW_ALIGN_TARGET - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 27: aligned, yaw_err={yaw_err:.3f} rad"
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
                    f"Stage 27: aligning — yaw={yaw:.3f}  "
                    f"target={YAW_ALIGN_TARGET:.3f}  err={yaw_err:.3f}  vyaw={vyaw:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 28: Crouch walk to P11 ───────────────────────
        elif self.state == "SQUAT_WALK_2":
            dy = self.p11[1] - y
            if dy > 0.05:
                self.publish(0.06, 0.0, 0.0, mode=11, gait=3,
                             step_height=[0.05, 0.05])
                self.get_logger().info(
                    f"Stage 28: crouch walk to P11 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "STAND_UP_AT_P11"
                self.state_counter = 0

        # ── Stage 29: Stand up at P11 ───────────────────────────
        elif self.state == "STAND_UP_AT_P11":
            if self.state_counter == 0:
                self.get_logger().info("Stage 29: Restoring normal height, standing up...")
                self.send_yaml_params(RESTORE_PARAMS)

            self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
            self.state_counter += 1

            if self.state_counter >= CROUCH_SETTLE_CYCLES:
                self.state = "MOVE_TO_P12"
                self.state_counter = 0

        # ── Stage 30: Walk forward to P12 (standing) ────────────
        elif self.state == "MOVE_TO_P12":
            dy = self.p12[1] - y
            if dy > 0.1:
                self.publish(0.15, 0.0, 0.0)
                self.get_logger().info(
                    f"Stage 30: walking to P12 y={y:.2f}",
                    throttle_duration_sec=1.0,
                )
            else:
                self.state = "TURN_AROUND_2"
                self.state_counter = 0

        # ── Stage 31: In-place 180° turn at P12 ─────────────────
        elif self.state == "TURN_AROUND_2":
            if self.state_counter == 0:
                self.turn_target_yaw = normalize_angle(yaw + math.pi)
                self.get_logger().info(
                    f"Stage 31: Turning 180° — yaw={yaw:.3f} → target={self.turn_target_yaw:.3f}"
                )

            yaw_err = normalize_angle(self.turn_target_yaw - yaw)

            if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 31: turn complete, yaw_err={yaw_err:.3f} rad"
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

        # ── Stage 32: Walk back to P11 (standing) ───────────────
        elif self.state == "MOVE_TO_P11_BACK":
            dx_world = self.p11[0] - x
            dy_world = self.p11[1] - y
            dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 32: reached P11, dist={dist:.3f} m"
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
                    f"Stage 32: moving to P11 — x={x:.3f} y={y:.3f}  "
                    f"dist={dist:.3f}  vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 33: Crouch via ROS2 params at P11 ─────────────
        elif self.state == "SET_CROUCH_3":
            if self.state_counter == 0:
                self.get_logger().info("Stage 33: Sending crouch params via ROS2...")
                self.send_yaml_params(CROUCH_PARAMS)
                self.get_logger().info("Stage 33: Zero-vel locomotion — lowering body...")

            self.publish(0.0, 0.0, 0.0, mode=11, gait=3)
            self.state_counter += 1

            if self.state_counter >= CROUCH_SETTLE_CYCLES:
                self.state = "SQUAT_WALK_3"
                self.state_counter = 0

        # ── Stage 34: Crouch walk to P10 ───────────────────────
        elif self.state == "SQUAT_WALK_3":
            dy_world = self.p10[1] - y
            dx_world = self.p10[0] - x
            dist = math.sqrt(dx_world * dx_world + dy_world * dy_world)

            if dist < POS_ALIGN_TOLERANCE:
                self.get_logger().info(
                    f"Stage 35: crouch walked to P10, dist={dist:.3f} m"
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
                    f"Stage 35: crouch walking — dist={dist:.3f}  "
                    f"vx={body_vx:.3f} vy={body_vy:.3f}",
                    throttle_duration_sec=1.0,
                )

        # ── Stage 35: Damper stop ───────────────────────────────
        elif self.state == "FINAL_STOP":
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
        node.get_logger().info("Ctrl+C caught, restoring normal height...")
        node.send_yaml_params(RESTORE_PARAMS)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
