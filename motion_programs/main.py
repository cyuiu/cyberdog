"""
Rich motion program runner for CyberDog loco high-level interface.

Usage:
    cd loco_hl_example/motion_programs
    python3 main.py
    python3 main.py 1
    python3 main.py built_in_moonwalk
"""
import os
import sys
import time

import lcm
import toml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEQUENTIAL_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "sequential_motion"))
if SEQUENTIAL_DIR not in sys.path:
    sys.path.append(SEQUENTIAL_DIR)

from robot_control_cmd_lcmt import robot_control_cmd_lcmt


CATALOG_PATH = os.path.join(CURRENT_DIR, "catalog.toml")
PROGRAM_DIR = os.path.join(CURRENT_DIR, "programs")


def load_catalog():
    catalog = toml.load(CATALOG_PATH)
    return catalog.get("program", [])


def print_catalog(programs):
    print("Available motion programs:")
    for index, item in enumerate(programs, start=1):
        print(f"{index:>2}. {item['key']} | {item['title']} | {item['description']}")


def find_program(programs, selector):
    if selector is None:
        return None

    if selector.isdigit():
        index = int(selector) - 1
        if 0 <= index < len(programs):
            return programs[index]
        raise ValueError("Program index out of range.")

    selector_lower = selector.lower()
    for item in programs:
        candidates = [
            item["key"],
            item["title"],
            item["file"],
        ]
        if any(selector_lower in candidate.lower() for candidate in candidates):
            return item

    raise ValueError(f"Program not found: {selector}")


def choose_program(programs):
    print_catalog(programs)
    print("\nInput a program number:")
    selector = input().strip()
    if not selector:
        raise ValueError("Empty selection.")
    return find_program(programs, selector)


def fill_message(msg, step):
    msg.mode = step["mode"]
    msg.gait_id = step["gait_id"]
    msg.contact = step["contact"]
    msg.value = step["value"]
    msg.duration = step["duration"]
    msg.life_count += 1

    for index in range(3):
        msg.vel_des[index] = step["vel_des"][index]
        msg.rpy_des[index] = step["rpy_des"][index]
        msg.pos_des[index] = step["pos_des"][index]
        msg.acc_des[index] = step["acc_des"][index]
        msg.acc_des[index + 3] = step["acc_des"][index + 3]
        msg.foot_pose[index] = step["foot_pose"][index]
        msg.ctrl_point[index] = step["ctrl_point"][index]

    for index in range(2):
        msg.step_height[index] = step["step_height"][index]


def run_program(program):
    program_path = os.path.join(PROGRAM_DIR, program["file"])
    steps = toml.load(program_path).get("step", [])

    if not steps:
        raise ValueError(f"No steps found in {program_path}")

    print(f"Run program: {program['title']}")
    print(f"File: {program_path}")
    print(f"Description: {program['description']}")

    lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
    msg = robot_control_cmd_lcmt()

    try:
        for index, step in enumerate(steps, start=1):
            fill_message(msg, step)
            lc.publish("robot_control_cmd", msg.encode())
            print(
                f"step {index:>2}: mode={msg.mode}, gait_id={msg.gait_id}, duration={msg.duration}"
            )
            time.sleep(0.1)

        for _ in range(75):
            lc.publish("robot_control_cmd", msg.encode())
            time.sleep(0.2)
    except KeyboardInterrupt:
        msg.mode = 7
        msg.gait_id = 0
        msg.duration = 0
        msg.life_count += 1
        lc.publish("robot_control_cmd", msg.encode())
        print("\nInterrupted, send PureDamper.")


def main():
    programs = load_catalog()
    if not programs:
        raise ValueError("Catalog is empty.")

    selector = sys.argv[1].strip() if len(sys.argv) > 1 else None
    try:
        program = find_program(programs, selector) if selector else choose_program(programs)
    except ValueError as exc:
        print(exc)
        print()
        print_catalog(programs)
        sys.exit(1)

    run_program(program)
    sys.exit(0)


if __name__ == "__main__":
    main()
