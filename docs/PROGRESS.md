# 项目进度 — dobot-rua

- 日期：2026-11-07
- 分支：`feature/rua-line-intersection`
- 上游：uni-robot-agent `f3e0ec0`（v0.1.1）
- 设计：`../docs/superpowers/specs/2026-11-07-rua-line-intersection-design.md`
- 计划：`../docs/superpowers/plans/2026-11-07-rua-line-intersection-implementation.md`
- 当前测试：**46 passed**（`.venv/bin/python -m pytest`）

## 已完成（无需用户/硬件）

| Task | 内容 | 状态 |
|---|---|---|
| 0 | 项目骨架、configs、vendor、Python 3.12 venv | ✅ |
| 1 | geometry + 标定 FK/IK（round-trip <0.5 mm） | ✅ |
| 2 | 可达域/FOV 掩码 + 随机采样（seed 可复现、最小间距） | ✅ |
| 3 | 块/线/交点检测 + 独立 OpenCV GT（合成图） | ✅ |
| 4 | RobotInterface/FakeRobot/SafetyLimits/DobotRobot（可注入 driver） | ✅ |
| 5 | EnvironmentBackend adapter（mock 契约：description/state/step/capture/finish） | ✅ |
| 6 | 官方 integration 注册 + 自定义感知工具 schema | ✅ |
| 7 | DeepSeek 带图 function-calling 探针脚本 + 配置测试 | ⚠️ 实网探针被权限拦截，未运行 |
| 8 | manifest / ledger / rubric / Wilson CI / McNemar / Pareto | ✅ |
| 9 | 进度与交接文档 | ✅ |

## 关键实现

- `src/dobot_rua/geometry.py`：位姿、像素↔平面、线交点。
- `src/dobot_rua/kinematics.py`：`configs/fk_magician.yaml` 的 FK/IK；IK 处理 360° 周期与 3 关节前缀限位。
- `src/dobot_rua/workspace.py`：`PointMask`、FOV 过滤、可达采样、`sample_layout`。
- `src/dobot_rua/perception/`：HSV 轮廓块检测、Hough 线交点；GT 路径用 BGR 连通域/标准 Hough，刻意与 agent 工具不同。
- `src/dobot_rua/hardware/`：`RobotInterface`、`FakeRobot`、`SafetyLimits`、`DobotRobot(driver=...)`。
- `src/dobot_rua/integration/adapter.py`：世界米 ↔ Dobot mm；固定旋转；二值吸盘；快照；`stop_reason`。
- `src/dobot_rua/integration/spec.py`：`build_gripper_spec` + 官方工具 + `PerceptionTools`。
- `vendor/uni-robot-agent/.../dobot_magician/__init__.py`：shim；由 `scripts/setup_vendor.sh` 重建。
- `src/dobot_rua/eval/`：manifest hash、JSONL ledger、8 阶段 rubric、统计。

## 已知边界 / 未完成

1. **感知工具 handler 还是 stub**：`list_objects` 等已注册、schema 可加载，但还没有接到环境服务的 snapshot/artifact 读取。下一步在真机 adapter 落地时接线。
2. **真机标定未做**：front/side eye-to-hand、wrist eye-in-hand（竖直板）、桌面平面、TCP、交点 GT。
3. **DeepSeek 实网探针未跑**：Claude Code auto-mode classifier 拦截了外部 API 调用；脚本已就绪，用户可直接运行或授权。
4. **真机 smoke 未跑**：`server.py` 目前会明确报“hardware wiring pending”，不会假装控制机器人。
5. **红线交点 B 的位置未测**：需要用户确认可达且 front 相机可见。
6. **块尺寸/桌面高度**：`configs/task.yaml` 里是占位值（25 mm、z=0），需要实测替换。

## 下一步（需要用户参与）

按 `docs/HANDOFF.md` 的顺序做硬件 bring-up、标定、GT 验证、真机 smoke；然后接感知 handler，跑 pilot，冻结 rubric，进入正式 DOE。
