# 运动控制高层接口示例

详细使用方法参照[铁蛋运动控制二次开发接口文档](https://miroboticslab.github.io/blogs/#/cn/cyberdog_loco_cn)。

## 仓库内示例

- `basic_motion`：基础动作接口
- `sequential_motion`：读取 `.toml` 顺序执行动作
- `customized_gait`：用户自定义步态
- `motion_programs`：新增的动作程序库，内置 10 个可直接运行的动作

## 新增 `motion_programs`

该目录依据 `运动控制模块.pdf` 附录“表1. 模式动作定义列表”补充，当前已新增以下程序：

- `01_left_hand_wave.toml`：`mode=62, gait_id=1`
- `02_sit_down.toml`：`mode=62, gait_id=3`
- `03_hip_swing.toml`：`mode=62, gait_id=4`
- `04_head_twist.toml`：`mode=62, gait_id=5`
- `05_stretch_body.toml`：`mode=62, gait_id=6`
- `06_ballet_dance.toml`：`mode=62, gait_id=11`
- `07_built_in_moonwalk.toml`：`mode=62, gait_id=12`
- `08_push_up.toml`：`mode=62, gait_id=34`
- `09_trot_slow_circle.toml`：`mode=11, gait_id=27`
- `10_bound_turn.toml`：`mode=11, gait_id=7`

运行方式：

```bash
cd motion_programs
python3 main.py
```

也可以直接传入编号或关键字：

```bash
python3 main.py 8
python3 main.py push_up
```
