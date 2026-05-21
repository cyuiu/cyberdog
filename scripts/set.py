#!/usr/bin/env python3
"""
Crouch walk using normal locomotion (mode=11, gait_id=3) with lowered height params.

The height reduction comes from the crouch YAML config loaded by the simulator
(cyberdog2-ctrl-user-parameters-crouch.yaml), not from customized_gait or mode=62.

Flow:
  1. Recovery stand (mode=12) — normal stand height
  2. Zero-velocity locomotion (mode=11, gait_id=3, vel=0) — body gradually lowers
     as the height filter converges to des_roll_pitch_height=0.15
  3. Slow forward crouch walk (mode=11, gait_id=3, vel>0)
  4. Damper stop (mode=7)
"""
import sys
import time

LCM_PYTHON_PATH = "/home/lcm/build/python"
if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)

import lcm
from robot_control_cmd_lcmt import robot_control_cmd_lcmt

LCM_URL = "udpm://239.255.76.67:7671?ttl=255"


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


def publish(lc, msg):
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
        publish(lc, msg)
        time.sleep(0.05)

    new_msg = make_cmd()
    # Copy fields back (lc handle stays the same, msg object replaced)
    for slot in msg.__slots__:
        setattr(msg, slot, getattr(new_msg, slot))
    print("    Stand done, command reset to locomotion defaults.")


def heartbeat(lc, msg, duration_sec):
    """Send the same command repeatedly for duration_sec seconds."""
    for _ in range(int(duration_sec / 0.05)):
        publish(lc, msg)
        time.sleep(0.05)


def main():
    lc = lcm.LCM(LCM_URL)
    msg = make_cmd()

    print("=" * 55)
    print("Crouch walk via mode=11 + gait_id=3 (normal locomotion)")
    print("Height from crouch YAML: des_roll_pitch_height = 0.15")
    print("=" * 55)

    try:
        # ---- 1. Recovery stand ----
        recovery_stand(lc, msg)
        # msg is now a fresh locomotion command (mode=11, gait_id=3, vel=0)

        # ---- 2. Zero-velocity locomotion: let height filter settle ----
        print("[2] Zero-vel locomotion (mode=11, gait_id=3) for 3s — lowering body...")
        msg.vel_des = [0.0, 0.0, 0.0]
        heartbeat(lc, msg, 3.0)

        # ---- 3. Slow forward crouch walk ----
        print("[3] Crouch walk forward (vx=0.08) for 8s...")
        msg.vel_des = [0.08, 0.0, 0.0]
        heartbeat(lc, msg, 8.0)

        # ---- 4. Damper stop ----
        print("[4] Damper stop (mode=7)...")
        msg.mode = 7
        msg.gait_id = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.duration = 0
        heartbeat(lc, msg, 1.0)
        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted — damper stop...")
        msg.mode = 7
        msg.gait_id = 0
        msg.vel_des = [0.0, 0.0, 0.0]
        msg.duration = 0
        publish(lc, msg)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
