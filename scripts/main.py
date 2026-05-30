#!/usr/bin/env python3
"""
CyberDog 闯关主程序
按顺序调用每一关的运行文件
"""

import sys
import subprocess
import os

# 关卡配置：(关卡名称, 脚本路径)
LEVELS = [
    ("第一关：过石板路", "/home/loco_hl_example/scripts/cross_rockroad3.py"),
    ("第二关：击球", "/home/loco_hl_example/scripts/hitball.py"),
    ("第三关：视觉巡线", "/home/loco_hl_example/scripts/race3.py"),
    ("第四关：level_four", "/home/loco_hl_example/scripts/level_four.py"),
    ("第五关：斜坡行走", "/home/loco_hl_example/scripts/r52.py"),
    ("第六关：足球", "/home/loco_hl_example/scripts/football.py"),
]

def run_level(name, script_path):
    """运行单个关卡"""
    print(f"\n{'='*60}")
    print(f"🚀 开始运行：{name}")
    print(f"{'='*60}\n")

    if not os.path.exists(script_path):
        print(f"❌ 错误：找不到脚本 {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path)
        )
        if result.returncode == 0:
            print(f"\n✅ {name} 运行完成")
            return True
        else:
            print(f"\n⚠️ {name} 运行异常，返回码：{result.returncode}")
            return False
    except KeyboardInterrupt:
        print(f"\n🛑 {name} 被手动中断")
        return False
    except Exception as e:
        print(f"\n❌ {name} 运行出错：{e}")
        return False

def main():
    print("🎮 CyberDog 闯关系统启动")
    print(f"共 {len(LEVELS)} 关待挑战\n")

    for i, (name, script) in enumerate(LEVELS, 1):
        print(f"\n>>> 第 {i}/{len(LEVELS)} 关")
        success = run_level(name, script)

        if not success:
            print(f"\n⚠️ {name} 未成功完成，继续下一关...")

    print("\n🏁 闯关结束！")

if __name__ == "__main__":
    main()
