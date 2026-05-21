#!/usr/bin/env python3
"""
Crouch walk using normal locomotion (mode=11, gait_id=3).

Height is switched dynamically via ROS2 "yaml_parameter" topic at runtime:
  - Normal height comes from the default YAML loaded by legged_simparam.cpp
  - Before crouch walking, this script publishes lowered height params via ROS2
  - After walking, normal height params are restored

Other scripts (manual_drive.py, test_builtin.py, etc.) are unaffected.

Flow:
  1. Recovery stand (mode=12) at normal height
  2. Publish crouch height params via ROS2 yaml_parameter topic
  3. Zero-velocity locomotion (mode=11, gait_id=3, vel=0) — body lowers
  4. Slow forward crouch walk
  5. Restore normal height params
  6. Damper stop (mode=7)
"""
import sys
import time

LCM_PYTHON_PATH = "/home/lcm/build/python"
if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)

import lcm
import rclpy
from rclpy.node import Node
from cyberdog_msg.msg import YamlParam
from robot_control_cmd_lcmt import robot_control_cmd_lcmt

LCM_URL = "udpm://239.255.76.67:7671?ttl=255"

# ── Height parameters ──────────────────────────────────────────────
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


def make_cmd():
    """Create a fresh locomotion command (mode=11, gait_id=3)."""
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
    msg.step_height = [0.05, 0.05]
    return msg


def publish_lcm(lc, msg):
    msg.life_count = (msg.life_count + 1) % 128
    lc.publish("robot_control_cmd", msg.encode())


def recovery_stand(lc, msg):
    """Recovery stand, then reset command to locomotion defaults."""
    print("[1] Recovery stand (mode=12) for 5s...")
    msg.mode = 12
    msg.gait_id = 0
    msg.vel_des = [0.0, 0.0, 0.0]
    msg.duration = 5000

    end_time = time.time() + 5.0
    while time.time() < end_time:
        publish_lcm(lc, msg)
        time.sleep(0.05)

    new_msg = make_cmd()
    for slot in msg.__slots__:
        setattr(msg, slot, getattr(new_msg, slot))
    print("    Stand done, command reset to locomotion defaults.")


def heartbeat(lc, msg, duration_sec):
    """Send the same LCM command repeatedly for duration_sec seconds."""
    for _ in range(int(duration_sec / 0.05)):
        publish_lcm(lc, msg)
        time.sleep(0.05)


def send_params(node, param_list):
    """Publish a list of (name, kind, value) tuples via ROS2 yaml_parameter topic."""
    pub = node.create_publisher(YamlParam, "yaml_parameter", 10)
    for name, kind, value in param_list:
        ros_msg = make_yaml_param(name, kind, value)
        pub.publish(ros_msg)
        time.sleep(0.05)  # let the simulator process each param
    node.destroy_publisher(pub)


def main():
    # ── ROS2 init ──
    rclpy.init()
    ros_node = Node("crouch_walk_params")

    # ── LCM init ──
    lc = lcm.LCM(LCM_URL)
    msg = make_cmd()

    print("=" * 55)
    print("Crouch walk via mode=11 + gait_id=3 (normal locomotion)")
    print("Height set dynamically via ROS2 yaml_parameter topic")
    print("  crouch: des_roll_pitch_height = 0.15")
    print("  normal: des_roll_pitch_height = 0.25 (restored after)")
    print("=" * 55)

    try:
        # ---- 1. Recovery stand at normal height ----
        recovery_stand(lc, msg)

        # ---- 2. Send crouch height params via ROS2 ----
        print("[2] Sending crouch height params via ROS2...")
        send_params(ros_node, CROUCH_PARAMS)
        print("    Crouch params sent.")

        # ---- 3. Zero-velocity locomotion: let height filter settle ----
        print("[3] Zero-vel locomotion (mode=11, gait_id=3) for 3s — lowering body...")
        msg.vel_des = [0.0, 0.0, 0.0]
        heartbeat(lc, msg, 3.0)

        # ---- 4. Slow forward crouch walk ----
        print("[4] Crouch walk forward (vx=0.08) for 8s...")
        msg.vel_des = [0.08, 0.0, 0.0]
        heartbeat(lc, msg, 8.0)

        # ---- 5. Restore normal height params ----
        print("[5] Restoring normal height params...")
        send_params(ros_node, RESTORE_PARAMS)
        print("    Normal params restored.")

        # ---- 6. Damper stop ----
        print("[6] Damper stop (mode=7)...")
        msg.mode = 7
        msg.gait_id = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.duration = 0
        heartbeat(lc, msg, 1.0)
        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted — restoring normal height + damper stop...")
        send_params(ros_node, RESTORE_PARAMS)
        msg.mode = 7
        msg.gait_id = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.duration = 0
        publish_lcm(lc, msg)
        time.sleep(0.5)

    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
