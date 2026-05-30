#!/usr/bin/env python3
import select
import sys
import termios
import time
import tty

LCM_PYTHON_PATH = "/home/lcm/build/python"
if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)

import lcm
from robot_control_cmd_lcmt import robot_control_cmd_lcmt


LCM_URL = "udpm://239.255.76.67:7671?ttl=255"


def make_cmd():
    msg = robot_control_cmd_lcmt()
    msg.mode = 11
    msg.gait_id = 3
    msg.contact = 15
    msg.duration = 0
    msg.vel_des = [0.0, 0.0, 0.0]
    msg.rpy_des = [0.0, 0.3, 0.0]  # 初始正pitch，前高后低
    msg.pos_des = [0.0, 0.0, 0.25]
    msg.step_height = [0.05, 0.05]
    return msg


def publish(lc, msg):
    msg.life_count = (msg.life_count + 1) % 128
    lc.publish("robot_control_cmd", msg.encode())



def set_damper(lc, msg):
    msg.mode = 7
    msg.gait_id = 0
    msg.vel_des = [0.0, 0.0, 0.0]
    msg.duration = 0
    publish(lc, msg)


def set_recovery_stand(lc, msg):
    msg.mode = 12
    msg.gait_id = 0
    msg.vel_des = [0.0, 0.0, 0.0]
    msg.duration = 5000

    end_time = time.time() + 5.0
    while time.time() < end_time:
        publish(lc, msg)
        time.sleep(0.05)


def read_key(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.read(1)
    return None


def main():
    lc = lcm.LCM(LCM_URL)
    msg = make_cmd()

    vx_step = 0.03
    vy_step = 0.03
    yaw_step = 0.15
    max_vx = 0.15
    max_vy = 0.10
    max_yaw = 0.50
    pitch_step = 0.05
    max_pitch = 0.60
    min_pitch = -0.50
    body_step = 0.02
    min_body = 0.15
    max_body = 0.40

    old_settings = termios.tcgetattr(sys.stdin)

    print("Manual drive:")
    print("  r: recovery stand")
    print("  w/s: forward/back")
    print("  a/d: turn left/right")
    print("  q/e: move left/right")
    print("  t/g: pitch +/- (前高后低/前低后高)")
    print("  y/h: body height +/-")
    print("  space: stop velocity")
    print("  z: damper stop")
    print("  Ctrl+C: damper stop and quit")

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            key = read_key(0.1)

            if key == "r":
                print("\nRecovery stand...")
                set_recovery_stand(lc, msg)
                msg = make_cmd()
            elif key == "w":
                msg.vel_des[0] = min(max_vx, msg.vel_des[0] + vx_step)
            elif key == "s":
                msg.vel_des[0] = max(-max_vx, msg.vel_des[0] - vx_step)
            elif key == "q":
                msg.vel_des[1] = min(max_vy, msg.vel_des[1] + vy_step)
            elif key == "e":
                msg.vel_des[1] = max(-max_vy, msg.vel_des[1] - vy_step)
            elif key == "a":
                msg.vel_des[2] = min(max_yaw, msg.vel_des[2] + yaw_step)
            elif key == "d":
                msg.vel_des[2] = max(-max_yaw, msg.vel_des[2] - yaw_step)
            elif key == "t":
                msg.rpy_des[1] = min(max_pitch, msg.rpy_des[1] + pitch_step)
            elif key == "g":
                msg.rpy_des[1] = max(min_pitch, msg.rpy_des[1] - pitch_step)
            elif key == "y":
                msg.pos_des[2] = min(max_body, msg.pos_des[2] + body_step)
            elif key == "h":
                msg.pos_des[2] = max(min_body, msg.pos_des[2] - body_step)
            elif key == " ":
                msg.vel_des = [0.0, 0.0, 0.0]
            elif key == "z":
                set_damper(lc, msg)
                msg = make_cmd()

            publish(lc, msg)
            print(
                "\rvx={:.2f} vy={:.2f} yaw={:.2f} pitch={:.2f} body_h={:.2f}      ".format(
                    msg.vel_des[0], msg.vel_des[1], msg.vel_des[2],
                    msg.rpy_des[1], msg.pos_des[2]
                ),
                end="",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\nStopping...")
        set_damper(lc, msg)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()