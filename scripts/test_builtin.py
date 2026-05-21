#!/usr/bin/env python3
"""Minimal test: mode=11 + built-in trot (gait_id=3). No custom gaits, no special params."""
import sys
import time

LCM_PYTHON_PATH = "/home/lcm/build/python"
if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)

import lcm
from robot_control_cmd_lcmt import robot_control_cmd_lcmt

LCM_URL = "udpm://239.255.76.67:7671?ttl=255"
LOOP_RATE = 0.005


def publish(lc, msg):
    msg.life_count = (msg.life_count + 1) % 128
    lc.publish("robot_control_cmd", msg.encode())


def heartbeat(lc, msg, duration_sec):
    for _ in range(int(duration_sec / LOOP_RATE)):
        publish(lc, msg)
        time.sleep(LOOP_RATE)


def main():
    lc = lcm.LCM(LCM_URL)
    msg = robot_control_cmd_lcmt()

    print("Minimal test: built-in trot (mode=11, gait_id=3)")
    print("If this doesn't walk → locomotion itself is broken.")
    print("If this walks → problem is with gait_id=110 setup.\n")

    try:
        # Recovery stand
        print("[1] Recovery stand 3s...")
        msg.mode = 12
        msg.gait_id = 0
        msg.duration = 3000
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.rpy_des = [0.0, 0.0, 0.0]
        msg.pos_des = [0.0, 0.0, 0.0]
        msg.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.ctrl_point = [0.0, 0.0, 0.0]
        msg.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.step_height = [0.0, 0.0]
        msg.contact = 0x0F
        msg.value = 0
        heartbeat(lc, msg, 3.0)

        # Built-in trot walk
        print("[2] Walk with built-in trot (gait_id=3) 5s...")
        msg.mode = 11
        msg.gait_id = 3
        msg.duration = 0
        msg.vel_des = [0.1, 0.0, 0.0]
        msg.rpy_des = [0.0, 0.0, 0.0]
        msg.pos_des = [0.0, 0.0, 0.0]
        msg.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.ctrl_point = [0.0, 0.0, 0.0]
        msg.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.step_height = [0.0, 0.0]
        msg.contact = 0
        msg.value = 0
        heartbeat(lc, msg, 5.0)

        # Stop
        print("[3] Damper stop...")
        msg.mode = 7
        heartbeat(lc, msg, 1.0)
        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted")
        msg.mode = 7
        heartbeat(lc, msg, 1.0)


if __name__ == "__main__":
    main()
