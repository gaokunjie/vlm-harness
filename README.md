# dobot-rua

Dobot Magician + 吸盘 + 三路相机的 Robot-Use Agent 项目：
自然语言指令 → 指定颜色物块 → 放到指定颜色线交点。

- 设计文档：`../docs/superpowers/specs/2026-11-07-rua-line-intersection-design.md`
- 实施计划：`../docs/superpowers/plans/2026-11-07-rua-line-intersection-implementation.md`
- 进度：`docs/PROGRESS.md`

## 快速开始

```bash
bash scripts/setup_vendor.sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e vendor/uni-robot-agent -e ".[dev]"
pytest
```

需要用户/硬件参与的步骤见 `docs/HANDOFF.md`；当前状态见 `docs/PROGRESS.md`。

## 标定（8×7 / 25 mm ChArUco）

```bash
python scripts/calibrate_intrinsics.py --camera front --samples 20
python scripts/calibrate_hand_eye.py --mode wrist --camera wrist --samples 15
python scripts/calibrate_hand_eye.py --mode fixed --camera front --samples 15
python scripts/measure_targets.py
```

详见 `docs/HANDOFF.md` §3。

## 目录

- `src/dobot_rua/`：几何、运动学、工作空间、感知、硬件、integration、评估
- `vendor/uni-robot-agent/`：pinned 上游 `f3e0ec0`
- `configs/`：机器人限位、FK、相机、目标、任务、模型
- `tests/`：离线测试
