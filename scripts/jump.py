#!/usr/bin/env python3
"""
横向跳跃测试脚本
通过速度脉冲模拟横向跳跃效果
跳跃距离约 0.44m
"""
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
from cyberdog_msg.msg import YamlParam
from robot_control_cmd_lcmt import robot_control_cmd_lcmt

LCM_URL = "udpm://239.255.76.67:7671?ttl=255"

# ── 高度参数 ──────────────────────────────────────────────────────
CROUCH_HEIGHT = [0.0, 0.0, 0.15]  # 蹲下蓄力高度
NORMAL_HEIGHT = [0.0, 0.0, 0.25]  # 正常高度

CROUCH_PARAMS = [
    ("des_roll_pitch_height",        YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("des_roll_pitch_height_motion", YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("des_roll_pitch_height_stair",  YamlParam.VEC_X_DOUBLE, CROUCH_HEIGHT),
    ("step_height_max",              YamlParam.DOUBLE,       0.04),
]

RESTORE_PARAMS = [
    ("des_roll_pitch_height",        YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT),
    ("des_roll_pitch_height_motion", YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT),
    ("des_roll_pitch_height_stair",  YamlParam.VEC_X_DOUBLE, NORMAL_HEIGHT),
    ("step_height_max",              YamlParam.DOUBLE,       0.06),
]

# ── 跳跃参数（可调整）──────────────────────────────────────────────
JUMP_DISTANCE = 0.44      # 跳跃距离 (m)
JUMP_VY = 0.8             # 横向跳跃速度 (m/s)，增大可跳更远
JUMP_TIME = JUMP_DISTANCE / JUMP_VY  # 跳跃持续时间 (s)

# ── 蓄力蹲下参数 ─────────────────────────────────────────────────
CROUCH_SETTLE_TIME = 1.0  # 蹲下蓄力时间 (s)


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
        time.sleep(0.05)
    node.destroy_publisher(pub)


def main():
    # ── ROS2 init ──
    rclpy.init()
    ros_node = Node("jump_test_node")

    # ── LCM init ──
    lc = lcm.LCM(LCM_URL)
    msg = make_cmd()

    print("=" * 55)
    print("横向跳跃测试脚本")
    print(f"  跳跃距离: {JUMP_DISTANCE} m")
    print(f"  跳跃速度: {JUMP_VY} m/s")
    print(f"  跳跃时间: {JUMP_TIME:.3f} s")
    print("=" * 55)

    try:
        # ---- 1. 站立准备 ----
        print("[1] 站立准备 (mode=12) 3秒...")
        msg.mode = 12
        msg.gait_id = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.duration = 3000
        heartbeat(lc, msg, 3.0)
        msg = make_cmd()
        print("    站立完成")

        # ---- 2. 蹲下蓄力 ----
        print("[2] 蹲下蓄力...")
        send_params(ros_node, CROUCH_PARAMS)
        print("    蹲下参数已发送")

        # 零速度蹲下，让身体降低
        msg.vel_des = [0.0, 0.0, 0.0]
        heartbeat(lc, msg, CROUCH_SETTLE_TIME)
        print(f"    蹲下蓄力 {CROUCH_SETTLE_TIME} 秒完成")

        # ---- 3. 横向跳跃 ----
        print(f"[3] 横向跳跃！vy={JUMP_VY} m/s, 持续 {JUMP_TIME:.3f} 秒...")
        msg.vel_des = [0.0, JUMP_VY, 0.0]  # 横向跳跃
        heartbeat(lc, msg, JUMP_TIME)
        print("    跳跃脉冲完成")

        # ---- 4. 立即停止 ----
        print("[4] 停止...")
        msg.vel_des = [0.0, 0.0, 0.0]
        heartbeat(lc, msg, 0.5)

        # ---- 5. 恢复正常高度 ----
        print("[5] 恢复正常高度...")
        send_params(ros_node, RESTORE_PARAMS)
        msg.vel_des = [0.0, 0.0, 0.0]
        heartbeat(lc, msg, 1.0)
        print("    高度恢复完成")

        # ---- 6. 最终停止 ----
        print("[6] 最终停止 (mode=12)...")
        msg.mode = 12
        msg.gait_id = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.duration = 2000
        heartbeat(lc, msg, 2.0)
        print("完成！")

    except KeyboardInterrupt:
        print("\n中断 - 恢复正常高度并停止...")
        send_params(ros_node, RESTORE_PARAMS)
        msg.mode = 12
        msg.vel_des = [0.0, 0.0, 0.0]
        publish_lcm(lc, msg)
        time.sleep(1.0)

    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
