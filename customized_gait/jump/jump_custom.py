#!/usr/bin/env python3
"""
横向跳跃自定义步态执行脚本
使用 USERGAIT (mode=11, gait_id=110) 实现跳跃
"""
import lcm
import sys
import time
import toml
import copy
import math

LCM_PYTHON_PATH = "/home/lcm/build/python"
CONTROL_PATH = "/home/loco_hl_example/sequential_motion"
CUSTOM_GAIT_PATH = "/home/loco_hl_example/customized_gait"
if LCM_PYTHON_PATH not in sys.path:
    sys.path.insert(0, LCM_PYTHON_PATH)
if CONTROL_PATH not in sys.path:
    sys.path.insert(0, CONTROL_PATH)
if CUSTOM_GAIT_PATH not in sys.path:
    sys.path.insert(0, CUSTOM_GAIT_PATH)

from robot_control_cmd_lcmt import robot_control_cmd_lcmt
from file_send_lcmt import file_send_lcmt

LCM_URL = "udpm://239.255.76.67:7671?ttl=255"
JUMP_DIR = "/home/loco_hl_example/customized_gait/jump"

# 基础命令模板
robot_cmd = {
    'mode': 0, 'gait_id': 0, 'contact': 0, 'life_count': 0,
    'vel_des': [0.0, 0.0, 0.0],
    'rpy_des': [0.0, 0.0, 0.0],
    'pos_des': [0.0, 0.0, 0.0],
    'acc_des': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'ctrl_point': [0.0, 0.0, 0.0],
    'foot_pose': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'step_height': [0.0, 0.0],
    'value': 0, 'duration': 0
}


def main():
    lcm_cmd = lcm.LCM(LCM_URL)
    lcm_usergait = lcm.LCM(LCM_URL)
    usergait_msg = file_send_lcmt()
    cmd_msg = robot_control_cmd_lcmt()

    try:
        print("=" * 55)
        print("横向跳跃自定义步态执行")
        print("=" * 55)

        # ---- 1. 站立准备 ----
        print("[1] 站立准备 (mode=12) 8秒...")
        cmd_msg.mode = 12
        cmd_msg.gait_id = 0
        cmd_msg.duration = 8000
        # 持续发送站立命令
        for i in range(160):  # 8秒，每50ms发送一次
            cmd_msg.life_count += 1
            lcm_cmd.publish("robot_control_cmd", cmd_msg.encode())
            time.sleep(0.05)
        print("    站立完成，等待稳定...")
        time.sleep(2.0)  # 额外等待2秒确保完全稳定
        print("    稳定完成")

        # ---- 2. 加载步态定义文件 ----
        print("[2] 加载跳跃步态定义...")
        file_obj_gait_def = open(f"{JUMP_DIR}/Gait_Def_jump.toml", 'r')
        gait_def_data = file_obj_gait_def.read()
        file_obj_gait_def.close()
        print(f"    文件大小: {len(gait_def_data)} bytes")
        usergait_msg.data = gait_def_data
        lcm_usergait.publish("user_gait_file", usergait_msg.encode())
        time.sleep(0.5)
        print("    步态定义已发送")

        # ---- 3. 加载步态参数文件 ----
        print("[3] 加载跳跃步态参数...")
        # 先处理参数文件，转换为完整格式
        steps = toml.load(f"{JUMP_DIR}/Gait_Params_jump.toml")
        full_steps = {'step': [robot_cmd]}
        k = 0
        for i in steps['step']:
            cmd = copy.deepcopy(robot_cmd)
            cmd['duration'] = i['duration']
            if i['type'] == 'usergait':
                cmd['mode'] = 11  # LOCOMOTION
                cmd['gait_id'] = 110  # USERGAIT
                cmd['vel_des'] = i['body_vel_des']
                cmd['rpy_des'] = i['body_pos_des'][0:3]
                cmd['pos_des'] = i['body_pos_des'][3:6]
                cmd['foot_pose'][0:2] = i['landing_pos_des'][0:2]
                cmd['foot_pose'][2:4] = i['landing_pos_des'][3:5]
                cmd['foot_pose'][4:6] = i['landing_pos_des'][6:8]
                cmd['ctrl_point'][0:2] = i['landing_pos_des'][9:11]
                cmd['step_height'][0] = math.ceil(i['step_height'][0] * 1e3) + math.ceil(i['step_height'][1] * 1e3) * 1e3
                cmd['step_height'][1] = math.ceil(i['step_height'][2] * 1e3) + math.ceil(i['step_height'][3] * 1e3) * 1e3
                cmd['acc_des'] = i['weight']
                cmd['value'] = i['use_mpc_traj']
                cmd['contact'] = math.floor(i['landing_gain'] * 1e1)
                cmd['ctrl_point'][2] = i['mu']
            if k == 0:
                full_steps['step'] = [cmd]
            else:
                full_steps['step'].append(cmd)
            k = k + 1

        # 保存完整参数文件
        f = open(f"{JUMP_DIR}/Gait_Params_jump_full.toml", 'w')
        f.write("# Gait Params for Jump (Full)\n")
        f.writelines(toml.dumps(full_steps))
        f.close()

        # 继续发送站立命令保持机器人站立
        print("    等待步态加载...")
        cmd_msg.mode = 12
        cmd_msg.gait_id = 0
        cmd_msg.duration = 0
        for i in range(20):  # 等待1秒，持续发送站立命令
            cmd_msg.life_count += 1
            lcm_cmd.publish("robot_control_cmd", cmd_msg.encode())
            time.sleep(0.05)

        # 发送参数文件
        file_obj_gait_params = open(f"{JUMP_DIR}/Gait_Params_jump_full.toml", 'r')
        gait_params_data = file_obj_gait_params.read()
        file_obj_gait_params.close()
        print(f"    参数文件大小: {len(gait_params_data)} bytes")
        usergait_msg.data = gait_params_data
        lcm_usergait.publish("user_gait_file", usergait_msg.encode())
        time.sleep(0.1)
        print("    步态参数已发送")

        # ---- 4. 执行跳跃步态（完全按 moonwalk 方式）----
        print("[4] 执行跳跃步态...")
        user_gait_list = open(f"{JUMP_DIR}/Usergait_List_jump.toml", 'r')
        steps = toml.load(user_gait_list)
        for step in steps['step']:
            cmd_msg.mode = step['mode']
            cmd_msg.value = step['value']
            cmd_msg.contact = step['contact']
            cmd_msg.gait_id = step['gait_id']
            cmd_msg.duration = step['duration']
            for i in range(3):
                cmd_msg.vel_des[i] = step['vel_des'][i]
                cmd_msg.rpy_des[i] = step['rpy_des'][i]
                cmd_msg.pos_des[i] = step['pos_des'][i]
                cmd_msg.acc_des[i] = step['acc_des'][i]
                cmd_msg.acc_des[i + 3] = step['acc_des'][i + 3]
                cmd_msg.foot_pose[i] = step['foot_pose'][i]
                cmd_msg.ctrl_point[i] = step['ctrl_point'][i]
            for i in range(2):
                cmd_msg.step_height[i] = step['step_height'][i]
            cmd_msg.life_count += 1
            lcm_cmd.publish("robot_control_cmd", cmd_msg.encode())
            print(f"    发送: mode={step['mode']}, gait_id={step['gait_id']}, duration={step['duration']}ms")
            time.sleep(0.1)
        user_gait_list.close()
        print("    跳跃步态命令已发送")

        # ---- 5. 心跳保持（等待步态执行完成）----
        print("[5] 心跳保持...")
        for i in range(75):  # 15秒，与 moonwalk 一致
            lcm_cmd.publish("robot_control_cmd", cmd_msg.encode())
            time.sleep(0.2)

        # ---- 6. 最终停止 ----
        print("[6] 最终停止 (mode=12)...")
        cmd_msg.mode = 12
        cmd_msg.gait_id = 0
        cmd_msg.life_count += 1
        lcm_cmd.publish("robot_control_cmd", cmd_msg.encode())
        time.sleep(2.0)
        print("完成！")

    except KeyboardInterrupt:
        print("\n中断 - 停止机器人...")
        cmd_msg.mode = 12
        cmd_msg.gait_id = 0
        cmd_msg.life_count += 1
        lcm_cmd.publish("robot_control_cmd", cmd_msg.encode())
        time.sleep(1.0)

    except Exception as e:
        print(f"错误: {e}")
        cmd_msg.mode = 12
        cmd_msg.gait_id = 0
        cmd_msg.life_count += 1
        lcm_cmd.publish("robot_control_cmd", cmd_msg.encode())

    sys.exit()


if __name__ == '__main__':
    main()
