# dobot-rua

Dobot Magician + 吸盘 + 三路相机的 Robot-Use Agent 项目：
自然语言指令 → 指定颜色物块 → 放到指定颜色线交点。

- 设计文档：`../docs/superpowers/specs/2026-11-07-rua-line-intersection-design.md`
- 实施计划：`../docs/superpowers/plans/2026-11-07-rua-line-intersection-implementation.md`
- 进度：`docs/PROGRESS.md`

## 快速开始

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e vendor/uni-robot-agent -e . --extra dev
pytest
```

## 目录

- `src/dobot_rua/`：几何、运动学、工作空间、感知、硬件、integration、评估
- `vendor/uni-robot-agent/`：pinned 上游 `f3e0ec0`
- `configs/`：机器人限位、FK、相机、目标、任务、模型
- `tests/`：离线测试
