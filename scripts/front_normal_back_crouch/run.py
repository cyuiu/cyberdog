#!/usr/bin/env python3
"""
前腿站立后腿蹲走 - 用于下台阶
机制：通过 ROS2 yaml_parameter 全身蹲下(0.15m)，再用 pitch 把前腿抬高
前腿 = 站立高度，后腿 = 蹲姿高度
"""
import sys
import time

LCM_PYTHON_PATH = "/home/lcm/build/python"
SCRIPTS_PATH = "/home/loco_hl_example/scripts"
if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

import lcm
import rclpy
from rclpy.node import Node
from cyberdog_msg.msg import YamlParam
from robot_control_cmd_lcmt import robot_control_cmd_lcmt

LCM_URL = "udpm://239.255.76.67:7671?ttl=255"

# ── 高度参数 ──────────────────────────────────────────────
# 全身蹲下到 0.15m，然后用 pitch 抬前腿
CROUCH_HEIGHT = [0.0, 0.0, 0.15]          # 蹲姿：全身 0.15m
NORMAL_HEIGHT = [0.0, 0.0, 0.25]          # 正常：全身 0.25m
NORMAL_HEIGHT_MOTION = [0.0, 0.0, 0.225]
NORMAL_HEIGHT_STAIR = [0.0, 0.0, 0.225]

# 蹲姿参数：降低身体 + 限制步高
CROUCH_PARAMS = [
    ("des_roll_pitch_height",        YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("des_roll_pitch_height_motion", YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("des_roll_pitch_height_stair",  YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("step_height_max",              YamlParam.DOUBLE,       0.04),
]

# 恢复正常参数
RESTORE_PARAMS = [
    ("des_roll_pitch_height",        YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT),
    ("des_roll_pitch_height_motion", YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT_MOTION),
    ("des_roll_pitch_height_stair",  YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT_STAIR),
    ("step_height_max",              YamlParam.DOUBLE,       0.06),
]


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


def make_cmd():
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
    print("    Stand done.")


def heartbeat(lc, msg, duration_sec):
    for _ in range(int(duration_sec / 0.05)):
        publish_lcm(lc, msg)
        time.sleep(0.05)


def send_params(node, param_list):
    pub = node.create_publisher(YamlParam, "yaml_parameter", 10)
    for name, kind, value in param_list:
        ros_msg = make_yaml_param(name, kind, value)
        pub.publish(ros_msg)
        time.sleep(0.05)
    node.destroy_publisher(pub)


def main():
    rclpy.init()
    ros_node = Node("front_stand_back_crouch")

    lc = lcm.LCM(LCM_URL)
    msg = make_cmd()

    print("=" * 55)
    print("前腿站立后腿蹲走 (下台阶用)")
    print("  1. 全身蹲下到 0.15m (via ROS2)")
    print("  2. pitch 抬高前腿 (后腿保持蹲姿)")
    print("  3. 前进")
    print("=" * 55)

    try:
        # ---- 1. 正常站立 ----
        recovery_stand(lc, msg)

        # ---- 2. 发送蹲姿参数：全身降到 0.15m ----
        print("[2] Sending crouch params (body height → 0.15m)...")
        send_params(ros_node, CROUCH_PARAMS)

        # ---- 3. 等待身体降下 ----
        print("[3] Waiting for body to lower (3s)...")
        msg.vel_des = [0.0, 0.0, 0.0]
        heartbeat(lc, msg, 3.0)

        # ---- 4. 加 pitch 抬高前腿，后腿保持蹲姿 ----
        # pitch 正值 → 前高后低
        # 蹲姿 0.15m + pitch 0.25 → 前腿约 0.25m，后腿约 0.15m
        print("[4] Applying pitch to raise front legs (pitch=0.25)...")
        msg.rpy_des = [0.0, 0.25, 0.0]
        msg.vel_des = [0.0, 0.0, 0.0]
        heartbeat(lc, msg, 2.0)

        # ---- 5. 前进 ----
        print("[5] Walking forward (vx=0.08) for 10s...")
        msg.vel_des = [0.08, 0.0, 0.0]
        heartbeat(lc, msg, 10.0)

        # ---- 6. 恢复正常 ----
        print("[6] Restoring normal height...")
        send_params(ros_node, RESTORE_PARAMS)
        msg = make_cmd()
        heartbeat(lc, msg, 2.0)

        # ---- 7. 停止 ----
        print("[7] Damper stop...")
        msg.mode = 7
        msg.gait_id = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        heartbeat(lc, msg, 1.0)
        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted — restoring...")
        send_params(ros_node, RESTORE_PARAMS)
        msg.mode = 7
        msg.gait_id = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        publish_lcm(lc, msg)
        time.sleep(0.5)

    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
