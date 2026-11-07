# 项目进度 — dobot-rua

- 日期：2026-11-07
- 分支：`feature/rua-line-intersection`
- 上游：uni-robot-agent `f3e0ec0`（v0.1.1）
- 设计：`../docs/superpowers/specs/2026-11-07-rua-line-intersection-design.md`
- 计划：`../docs/superpowers/plans/2026-11-07-rua-line-intersection-implementation.md`
- 当前测试：**71 passed**（`.venv/bin/python -m pytest`）

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
| 7 | DeepSeek 带图 function-calling 探针 | ✅ 10/10 tool call、10/10 图像正确，延迟 0.72–1.14 s |
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

## 独立 review 修复（7 个 Important）

| 问题 | 修复 | 测试 |
|---|---|---|
| GT 把长红线当成块 | `measure_block_gt` 增加 fill≥0.5，拒绝线状连通域 | `test_measure_block_gt_ignores_a_long_line_component` |
| 线 GT 与 agent 共用 `_line_mask` | GT 改用独立 BGR `_gt_line_mask` | `test_measure_intersection_gt_does_not_share_the_agent_mask` |
| 任务 prompt 未注册 | `prompt.py` 注册 `build_prompt`，`spec.py` 导入触发 | `test_prompt_builder_is_registered` |
| `assert_disjoint` 只按 (split, plan_id) | 增加 layout_key（seed/位置/颜色/指令）跨 split 检查，plan_id 全局唯一 | `test_assert_disjoint_rejects_same_layout_across_splits` |
| `aggregate_by_dimension` 不是五维分 | 用每阶段独立通过情况算 20 分制维度分；未覆盖维度返回 `None` | `test_aggregate_by_dimension_uses_stage_weights` |
| 可达掩码含限位边界；adapter 不查关节限位 | 采样加 2° margin；adapter 传 `fk_params`+`safety_limits` 时用 IK 拒绝不可达 | `test_joint_samples_respect_limit_margin` / `test_adapter_rejects_unreachable_pose_when_kinematics_supplied` |
| 姿态谎报 identity；不拒绝姿态变化 | 工具朝下四元数 `(0,1,0,0)`；目标姿态与当前固定姿态夹角 >0.35 rad 拒绝 | `test_read_state_reports_tool_down_orientation` / `test_step_pose_rejects_orientation_change` |

## 官方 harness + DeepSeek + mock 环境（2026-11-07）

跑通命令（mock 服务 + 官方 CLI）：

```bash
python -m dobot_rua.integration.mock_server --run-id mock-3 \
  --instruction "Place the red block at the black lines intersection." \
  --output-dir ./runs/mock_env3 --port 18760

UNI_ROBOT_AGENT_API_KEY="$ANTHROPIC_AUTH_TOKEN" uni-robot-agent run \
  --integration dobot_magician --agent codex --model deepseek-flash \
  --provider-profile deepseek.chat_completions \
  --provider-url https://api.deepseek.com/v1 \
  --context-policy identity --action-profile direct --no-sam3 \
  --tool-return-mode immediate \
  --env-url http://127.0.0.1:18760 \
  --instruction "Place the red block at the black lines intersection." \
  --output-dir ./runs/mock_agent3 --max-turns 24 --episode-timeout-s 420
```

结果：`task_success=True`、`agent_finished=True`、`session end │ success`；turns=1、steps=33、tool_calls=32、elapsed=106.5 s；`train.jsonl` 正常导出。

工具序列（官方工具 + 自定义工具）：

`list_skills → capture_scene → read_skill → find_line_intersection → list_objects → move → wrist_contact_gap → wrist_lateral_offset → move → grasp → move → wrist_verify_grasp → move → wrist_lateral_offset → move → release → move → verify_place → list_objects → finish`

关键数值：红块 `(0.000, 0.000)`，黑线交点 `(0.160, -0.120)`；agent 自己把四元数 `(0,1,0,0)` 转成 XYZ Euler `[π,0,0]` 并复制到 `move`，说明固定姿态契约可用。

这次 smoke 发现并修掉的两个真实问题：

1. **`CameraFrame.rgb` 是 RGB，不是 BGR**：mock 源最初返回 BGR，PIL 存 PNG 后红蓝互换；已在 `MockCameraSource` 转 RGB，并加测试。
2. **黑线掩码不能只用灰度 <80**：红色线的亮度约 76，会被当黑线；已改为 HSV 低饱和+低亮度，并加混合黑/红线的回归测试。

另外：`--tool-return-mode background` 下 DeepSeek 的 `wait_for_tool` 轨迹导出报 `provider context contains unmatched tool calls or responses`，生成 `train.invalid.jsonl`；改用 `--tool-return-mode immediate` 后 `train.jsonl` 正常。正式实验若要 background 模式，需要先修这个导出路径。

**Mock 成功的口径**：最终吸盘尖 XY 距黑线交点 ≤20 mm 且真空已关；mock 不模拟物块搬运，`verify_place` 返回 `placed=null`，agent 的总结也如实说明了这一点。

## 真机只读 bring-up（2026-11-07）

已启动 real server（只读位姿 + 三路取帧，未发任何运动指令）：

```bash
python -m dobot_rua.integration.server --run-id real-bringup \
  --instruction "Place the red block at the black lines intersection." \
  --output-dir ./runs/real_bringup --port 18761
```

`/info` 正常返回；`has_calibration=false`（还没标定，符合预期）。
`/snapshots/latest` 状态：

- 吸盘尖：`x=147.04 mm, y=0, z=6.92 mm`
- 关节：`[0, 0.431, 0.906, 0] rad`（约 `[0°, 24.7°, 51.9°, 0°]`）
- 姿态四元数：`(0,1,0,0)`（工具朝下）
- 三路相机：1280×720，K/T 均为 `null`
- 参考图：`audit/cameras/real_front.png` / `real_wrist.png` / `real_side.png`

画面观察：

1. **front**：能看到白板、红/蓝块、黑十字（约在画面 `(710, 350)`）；机械臂从右侧进入画面。红线交点 B 尚未出现，需要你补画。
2. **wrist**：水平朝前的吸盘近景；吸盘居中、清晰，红/蓝块在画面上缘且虚焦。用于相对位置/接触确认可行，但块在视野上缘，安装角度或标定要特别处理。
3. **side**：侧视能看到红/蓝块和吸盘高度，适合做垂直/深度上下文。

这次只读 bring-up 没有移动机械臂；`RealCameraSource` 已实现 RGB 约定、MJPG fourcc 归一化（`mjpeg→MJPG`）、标定透传和 close 释放。

## 已知边界 / 未完成

1. **感知工具 handler 还是 stub**：`list_objects` 等已注册、schema 可加载，但还没有接到环境服务的 snapshot/artifact 读取。下一步在真机 adapter 落地时接线。
2. **真机标定未做**：front/side eye-to-hand、wrist eye-in-hand（竖直板）、桌面平面、TCP、交点 GT。
3. ~~DeepSeek 实网探针未跑~~：已跑通，10/10 次 tool call、10/10 次图像正确，延迟 0.72–1.14 s。首跑失败是环境里 `ALL_PROXY=socks://` 的 httpx 兼容问题，已修。
4. **真机 smoke 未跑**：`server.py` 目前会明确报“hardware wiring pending”，不会假装控制机器人。
5. **红线交点 B 的位置未测**：需要用户确认可达且 front 相机可见。
6. **块尺寸/桌面高度**：`configs/task.yaml` 里是占位值（25 mm、z=0），需要实测替换。
7. **FOV ∩ 可达域的显式合成未接**：`sample_layout` 目前接收一个 `PointMask`；调用方需要先做交集。reviewer 标为 Minor，已记录为 deferred。
8. **其他 deferred Minor**：`Layout.seed` 是采样后的新 draw（不是复现输入 seed 的 seed）；`min_target_sep_mm` 只对一个交点生效；`build_fov_mask` 平面固定 z=0；`setup_vendor.sh` 不校验已有 vendor 的 commit；probe 回退 `ANTHROPIC_AUTH_TOKEN` 可能 401；ledger 元组读回为 list，`success` 与 rubric 可不一致。

## 下一步（需要用户参与）

按 `docs/HANDOFF.md` 的顺序做硬件 bring-up、标定、GT 验证、真机 smoke；然后接感知 handler，跑 pilot，冻结 rubric，进入正式 DOE。
