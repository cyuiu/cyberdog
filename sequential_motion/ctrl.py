#!/usr/bin/env python3
import os
import select
import sys
import termios
import time
import tty

LCM_PYTHON_PATH = "/home/lcm/build/python"
CANDIDATE_CONTROL_PATHS = [
    os.getcwd(),
    "/home/loco_example/sequential_motion",
    "/home/loco_example/loco_hl_example/sequential_motion",
]

if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)

for path in CANDIDATE_CONTROL_PATHS:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import lcm
from robot_control_cmd_lcmt import robot_control_cmd_lcmt


def make_cmd(step_height):
    msg = robot_control_cmd_lcmt()
    msg.mode = 11
    msg.gait_id = 3
    msg.contact = 15
    msg.life_count = 0
    msg.duration = 0
    msg.vel_des = [0.0, 0.0, 0.0]
    msg.rpy_des = [0.0, 0.0, 0.0]
    msg.pos_des = [0.0, 0.0, 0.0]
    msg.acc_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    msg.ctrl_point = [0.0, 0.0, 0.0]
    msg.foot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    msg.step_height = [step_height, step_height]
    msg.value = 0
    return msg


def publish(lc, msg):
    msg.life_count = (msg.life_count + 1) % 128
    lc.publish("robot_control_cmd", msg.encode())


def set_locomotion(msg, step_height):
    msg.mode = 11
    msg.gait_id = 3
    msg.contact = 15
    msg.duration = 0
    msg.step_height = [step_height, step_height]


def damper_stop(lc, msg):
    msg.mode = 7
    msg.gait_id = 0
    msg.contact = 0
    msg.vel_des = [0.0, 0.0, 0.0]
    msg.duration = 0
    publish(lc, msg)


def recovery_stand(lc, msg):
    print("\nRecovery stand for 5 seconds...")
    msg.mode = 12
    msg.gait_id = 0
    msg.contact = 0
    msg.vel_des = [0.0, 0.0, 0.0]
    msg.duration = 5000

    end_time = time.time() + 5.0
    while time.time() < end_time:
        publish(lc, msg)
        time.sleep(0.05)

    print("Recovery stand done.")


def read_key(timeout):
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if readable:
        return sys.stdin.read(1)
    return ""


def print_help():
    print("Manual2 drive:")
    print("  r: recovery stand")
    print("  w/s: forward/backward")
    print("  a/d: turn left/right")
    print("  q/e: move left/right")
    print("  space: stop velocity")
    print("  H or h: increase step height")
    print("  L or l: decrease step height")
    print("  z: damper stop")
    print("  Ctrl+C: damper stop and quit")


def main():
    lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")

    vx_step = 0.03
    vy_step = 0.03
    yaw_step = 0.05

    max_vx = 0.40
    max_vy = 0.25
    max_yaw = 0.45

    step_height = 0.08
    step_height_step = 0.02
    min_step_height = 0.02
    max_step_height = 0.35

    msg = make_cmd(step_height)

    old_settings = termios.tcgetattr(sys.stdin)
    print_help()

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            key = read_key(0.05)

            if key == "r":
                recovery_stand(lc, msg)
                msg = make_cmd(step_height)

            elif key == "w":
                msg.vel_des[0] = min(max_vx, msg.vel_des[0] + vx_step)

            elif key == "s":
                msg.vel_des[0] = max(-max_vx, msg.vel_des[0] - vx_step)

            elif key == "a":
                msg.vel_des[2] = min(max_yaw, msg.vel_des[2] + yaw_step)

            elif key == "d":
                msg.vel_des[2] = max(-max_yaw, msg.vel_des[2] - yaw_step)

            elif key == "q":
                msg.vel_des[1] = min(max_vy, msg.vel_des[1] + vy_step)

            elif key == "e":
                msg.vel_des[1] = max(-max_vy, msg.vel_des[1] - vy_step)

            elif key == " ":
                msg.vel_des = [0.0, 0.0, 0.0]

            elif key in ("H", "h"):
                step_height = min(max_step_height, step_height + step_height_step)
                print("\nstep_height increased to %.2f" % step_height)

            elif key in ("L", "l"):
                step_height = max(min_step_height, step_height - step_height_step)
                print("\nstep_height decreased to %.2f" % step_height)

            elif key == "z":
                print("\nDamper stop.")
                damper_stop(lc, msg)
                msg = make_cmd(step_height)

            set_locomotion(msg, step_height)
            publish(lc, msg)

            sys.stdout.write(
                "\rvx=%.2f vy=%.2f yaw=%.2f step_h=%.2f      "
                % (msg.vel_des[0], msg.vel_des[1], msg.vel_des[2], step_height)
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nCtrl+C, damper stop and quit.")
        damper_stop(lc, msg)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()