# CyberDog 仿真蹲姿行走实现方案

## 目标

在 CyberDog 仿真环境中实现低姿态（蹲姿）行走，机身高度从默认的 ~0.23m 降到 ~0.15m，同时不影响正常的直立行走功能。

## 整体架构

仿真系统由三个进程组成：

```
Gazebo (物理引擎)
    └── legged_simparam.cpp    ← 加载 YAML 配置，通过共享内存发给控制器
                                ← 提供 ROS2 "yaml_parameter" topic 用于运行时改参数

控制器 (locomotion)
    └── convex_mpc_loco_gaits.cpp  ← MPC 步态控制，pos_cmd_min_ 限制最低高度

脚本 (Python)
    └── set.py                 ← 通过 LCM 发送 robot_control_cmd（mode=11 行走）
                               ← 通过 ROS2 发送 YamlParam（动态改高度参数）
```

**关键点：** 仿真模式下，locomotion 参数由 `legged_simparam.cpp` 从 YAML 加载，通过共享内存发给控制器。控制器内部不做 YAML 文件读取。

## 涉及的关键参数

| 参数名 | 类型 | 默认值 | 蹲姿值 | 作用 |
|--------|------|--------|--------|------|
| `des_roll_pitch_height` | Vec3 | [0, 0, 0.25] | [0, 0, 0.15] | 正常行走时目标机身高度 (z) |
| `des_roll_pitch_height_motion` | Vec3 | [0, 0, 0.225] | [0, 0, 0.15] | motion 模式目标高度 |
| `des_roll_pitch_height_stair` | Vec3 | [0, 0, 0.225] | [0, 0, 0.15] | 楼梯模式目标高度 |
| `step_height_max` | double | 0.06 | 0.04 | 最大步高（蹲姿保守值） |
| `pos_cmd_min_` | C++ 硬编码 | 0.23 → 0.15 | 0.15 | 位置指令 z 轴下限，低于此值会被钳制 |

## 修改的文件

### 1. `legged_simparam.cpp`（核心入口）

**路径：** `/home/cyberdog_sim/src/cyberdog_simulator/cyberdog_gazebo/src/legged_simparam.cpp`

**改动：** 保持加载原始 YAML（不改文件名），让默认行为是正常高度。

```cpp
// 第 46 行，加载默认参数（正常高度 0.25）
user_parameters_.DefineAndInitializeFromYamlFile(
    GetLocoConfigDirectoryPath() + "cyberdog2-ctrl-user-parameters.yaml"
);
```

**原理：** 此文件在 Gazebo 启动时运行。它通过共享内存将 YAML 参数发送给控制器。这个文件也订阅了 ROS2 topic `"yaml_parameter"`，允许运行时热更新单个参数（见 `HandleYamlParam` 回调函数，第 180 行）。保持加载原始 YAML 确保默认行为是正常高度，蹲姿由脚本在运行时通过 ROS2 动态切换。

### 2. `convex_mpc_loco_gaits.cpp`（高度下限）

**路径：** `/home/cyberdog_sim/src/cyberdog_locomotion/control/src/convex_mpc/convex_mpc_loco_gaits.cpp`

**改动：** 降低位置指令 z 轴的最小值限制，从 0.23 改为 0.15。两处都需要改（`#ifdef USE_ABSOLUTE_ODOM_FOR_ALL` 分支和 `#else` 分支）。

```cpp
// 第 202 行 (#ifdef 分支)
pos_cmd_rel_min_ << 0.0, 0.0, 0.15;

// 第 209 行 (#else 分支)
pos_cmd_min_ << 0.0, 0.0, 0.15;
```

**原理：** 控制器在计算位置指令时，会调用 `WrapRange(pos_des_, pos_cmd_min_, pos_cmd_max_)` 做限幅。如果只改 YAML 不改这个 C++ 常量，当 `des_roll_pitch_height` 设到 0.15 时，位置指令会被 `pos_cmd_min_` 钳制住而无法真正下降。0.15 是目标蹲姿高度，如果遇到不稳可以回退到 0.17。

### 3. 新建 `cyberdog2-ctrl-user-parameters-crouch.yaml`（蹲姿参数模板）

**路径：** `/home/cyberdog_sim/src/cyberdog_locomotion/common/config/cyberdog2-ctrl-user-parameters-crouch.yaml`

**说明：** 这是从原始 YAML 复制的蹲姿参数文件，修改了 4 个参数。**当前方案不通过文件名切换使用它**，而是通过 ROS2 topic 在运行时动态下发这些参数值。保留此文件作为蹲姿参数的参考模板，也方便后续如果在不需要 ROS2 的场景下（例如纯实机测试），直接通过修改 `legged_simparam.cpp` 中的文件名来切换。

```yaml
# 第 102-104 行
des_roll_pitch_height        : [0.0, 0, 0.15]
des_roll_pitch_height_motion : [0.0, 0, 0.15]
des_roll_pitch_height_stair  : [0.0, 0, 0.15]

# 第 213 行
step_height_max     : 0.04
```

### 4. `set.py`（蹲走控制脚本）

**路径：** `/home/loco_hl_example/scripts/set.py`

**改动：** 完全重写。从原来的 "mode=62 + customized_gait + LCM 文件上传" 方案，改为 "mode=11 + gait_id=3 + ROS2 动态参数" 方案。

**执行流程：**

```
┌─────────────────────────────────────────────────────────┐
│ 1. Recovery stand (mode=12, LCM)                        │
│    狗以正常高度 0.25 站起来                                │
├─────────────────────────────────────────────────────────┤
│ 2. ROS2 发送 crouch 参数 (yaml_parameter topic)          │
│    把 des_roll_pitch_height 从 0.25 改为 0.15            │
│    legged_simparam.cpp 的 HandleYamlParam 回调处理        │
│    通过共享内存同步发给控制器                               │
├─────────────────────────────────────────────────────────┤
│ 3. 零速 locomotion (mode=11, gait_id=3, vel=0, 3 秒)     │
│    控制器高度滤波器逐渐将机身降到 0.15                       │
├─────────────────────────────────────────────────────────┤
│ 4. 蹲走前进 (mode=11, gait_id=3, vx=0.08, 8 秒)          │
│    在低姿态下缓慢行走                                      │
├─────────────────────────────────────────────────────────┤
│ 5. ROS2 恢复默认参数                                      │
│    des_roll_pitch_height 恢复为 0.25                      │
├─────────────────────────────────────────────────────────┤
│ 6. Damper stop (mode=7, LCM)                             │
└─────────────────────────────────────────────────────────┘
```

**为什么不直接用 LCM 改参数：** `legged_simparam.cpp` 的参数更新接口是 ROS2 topic `"yaml_parameter"`，消息类型为 `cyberdog_msg::msg::YamlParam`。LCM 通道 `"robot_control_cmd"` 只能发控制指令（mode、vel_des 等），不能改底层控制器参数。因此脚本需要同时使用 LCM（控制指令）和 ROS2（参数更新）。

### 为什么不用 customized_gait (mode=62)

customized_gait 需要：
1. 定义 Gait_Def 和 Gait_Params TOML 文件（步态时序、落脚点、身体姿态）
2. 通过 LCM `"user_gait_file"` 通道上传
3. 所有底层参数在 TOML 中硬编码，调试困难

而普通 locomotion (mode=11, gait_id=3) 直接使用内置 trot 步态，只需改 YAML 高度参数。更简单、更可靠、更容易调试。

## 通信架构总结

```
set.py
  ├── LCM → "robot_control_cmd" 通道
  │         mode, gait_id, vel_des, duration 等控制指令
  │         控制器直接读取并执行
  │
  └── ROS2 → "yaml_parameter" topic
              YamlParam(name, kind, value, is_user)
              legged_simparam.cpp 订阅 → 共享内存 → 控制器更新参数
```

## 构建命令

```bash
cd /home/cyberdog_sim
source /opt/ros/galactic/setup.bash
source install/setup.bash
colcon build --merge-install --packages-select cyberdog_locomotion cyberdog_gazebo
source install/setup.bash
```

## 运行命令

```bash
# 每个新终端都需要 source 环境
source /opt/ros/galactic/setup.bash
source /home/cyberdog_sim/install/setup.bash

# 运行蹲走脚本
python3 /home/loco_hl_example/scripts/set.py
```

## 参数调节指南

如果蹲姿不稳或摔倒，按优先级尝试：

1. **提高目标高度** — 在 `set.py` 中把 `CROUCH_HEIGHT` 从 `[0.0, 0.0, 0.15]` 改为 `[0.0, 0.0, 0.17]`
2. **同步修改 C++ 下限** — 在 `convex_mpc_loco_gaits.cpp` 中把 `pos_cmd_min_` 和 `pos_cmd_rel_min_` 的 z 值从 0.15 改为 0.17
3. **降低前进速度** — 在 `set.py` 第 4 步中把 `vx=0.08` 改小（如 0.05）
4. **延长降高时间** — 把第 3 步零速 locomotion 的时间从 3 秒增加到 5 秒

## 备用方案：硬编码切换（不推荐）

如果 ROS2 参数下发不可用，可以回退到硬编码方式：修改 `legged_simparam.cpp` 第 46 行，把 YAML 文件名改为 `cyberdog2-ctrl-user-parameters-crouch.yaml`。代价是**全局默认高度变为蹲姿**，正常行走也受影响。不推荐在需要频繁切换高/低姿态的场景下使用。
