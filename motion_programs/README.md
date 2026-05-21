# 运动程序扩展示例

该目录基于 `运动控制模块.pdf` 附录“表1. 模式动作定义列表”补充了一组新的高层动作程序，目标是让仓库除了原有的 `basic_motion`、`sequential_motion`、`customized_gait` 之外，再多出一批可以直接运行和复用的现成动作。

## 目录说明

- `main.py`：动作程序选择器 / 执行器
- `catalog.toml`：动作程序清单
- `programs/*.toml`：具体动作序列

## 当前内置程序

- `01_left_hand_wave.toml`：`mode=62, gait_id=1`，握左手
- `02_sit_down.toml`：`mode=62, gait_id=3`，坐下
- `03_hip_swing.toml`：`mode=62, gait_id=4`，扭屁股
- `04_head_twist.toml`：`mode=62, gait_id=5`，扭头
- `05_stretch_body.toml`：`mode=62, gait_id=6`，伸懒腰
- `06_ballet_dance.toml`：`mode=62, gait_id=11`，芭蕾舞
- `07_built_in_moonwalk.toml`：`mode=62, gait_id=12`，内置太空步
- `08_push_up.toml`：`mode=62, gait_id=34`，俯卧撑
- `09_trot_slow_circle.toml`：`mode=11, gait_id=27`，慢跑转圈
- `10_bound_turn.toml`：`mode=11, gait_id=7`，四足跳跑转向

## 运行方式

```bash
cd loco_hl_example/motion_programs
python3 main.py
```

也可以直接传入序号或关键字：

```bash
python3 main.py 7
python3 main.py built_in_moonwalk
```

## 设计约定

- 所有程序默认以 `Recovery stand (mode=12)` 起始；
- 动作完成后统一切到 `PureDamper (mode=7)` 收尾，便于连续调试；
- `mode=62` 的预制动作尽量使用不同 `gait_id`，避免和原仓库示例重复；
- locomotion 类动作保留了较保守的速度与步高参数，更适合先在仿真里验证。
