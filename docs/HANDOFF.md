# 交接：需要用户/硬件参与的步骤

本文只列“我不能独立完成”的部分；已完成部分见 `PROGRESS.md`。

## 0. 环境复现

```bash
cd /home/kelly/harness+qwen/dobot-rua
bash scripts/setup_vendor.sh          # 重建 pinned vendor + integration shim
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e vendor/uni-robot-agent -e ".[dev]"
pytest                                 # 期望 46 passed
```

## 1. DeepSeek 探针（已完成）

已跑通：`--n 10` → **10/10 tool call、10/10 图像正确**，延迟 0.72–1.14 s。
当前环境用 `ANTHROPIC_AUTH_TOKEN` 也能访问 `https://api.deepseek.com/v1`。

复跑命令：

```bash
export UNI_ROBOT_AGENT_API_KEY='<deepseek key>'   # 或 DEEPSEEK_API_KEY / ANTHROPIC_AUTH_TOKEN
.venv/bin/python scripts/probe_deepseek.py --n 10
```

- 脚本会处理 `ALL_PROXY=socks://` 的 httpx 兼容问题；HTTP(S)_PROXY 保留。
- 期望：`ok_calls=10`、`image_ok=10`、`errors=[]`。
- 如果 401/403，说明当前 key 不适用于 `https://api.deepseek.com/v1`，需要单独的 DeepSeek API key。

## 1b. 官方 harness mock smoke（已完成）

已跑通一次完整的官方工具链路（DeepSeek + Codex runtime + `dobot_magician` integration + mock environment）：

- `task_success=True`、`agent_finished=True`、32 tool calls、106.5 s。
- 结果目录：`runs/mock_agent3/dobot_magician_codex_1794054815_09b058ea/`。
- 复跑命令见 `docs/PROGRESS.md` 的「官方 harness + DeepSeek + mock 环境」。
- 建议加 `--tool-return-mode immediate`：background 模式下 DeepSeek 的 `wait_for_tool` 轨迹会触发训练导出校验错误。
- 这只验证工具/协议/模型链路，不验证真机运动、标定或物块搬运。

## 2. 硬件 bring-up（需要机械臂/相机）

前置检查：

```bash
ls -l /dev/dobot_follower /dev/video0 /dev/video2 /dev/video4
v4l2-ctl --list-devices
```

要确认/记录：

1. 机械臂回零、ready、j1–j4 限位、吸盘 ON/OFF 延迟。
2. 三路相机取帧；wrist 画面里吸盘与物块的关系是否稳定。
3. 锁定曝光/白平衡/对焦（沿用 VLA 项目的相机参数即可，但代码要新写）。
4. 桌面高度、块尺寸、吸盘尖端 TCP。
5. 黑线交点 A、红线交点 B 是否都在 front FOV 和机械臂可达域内。

## 3. 标定

### 3.1 front / side：eye-to-hand

- 棋盘格内参；固定外参到机器人基座。
- 用机器人触碰 5–10 个桌面点验证，记录 `calibration_bias_mm`。
- 目标：重投影 <2 px，XY 误差 <3 mm。

### 3.2 wrist：eye-in-hand（竖直标定板）

- wrist 相机竖直安装、光轴水平向前；**棋盘格不能放桌面**。
- 把棋盘格固定在竖直支架/墙面/第二根臂夹持的刚性板上。
- 机械臂走 12–20 个位姿，`cv2.calibrateHandEye`；留出 3–5 个位姿验证。
- 标定吸盘尖在 wrist 图像中的参考位置，用于接触间隙/附着判断。
- 目标：3D 误差 <2–3 mm。

### 3.3 交点 GT

- 高分辨率 front 原图 + 独立线拟合 + 人工复核一次。
- 可选机器人触碰验证。
- 写入 `configs/targets.yaml` 的 `intersection_gt_xy_m`、`measured_at`、`calibration_hash`。

### 3.4 GT 验证

- 20–30 个静态块位置：OpenCV GT vs 机器人触碰/人工标注。
- 目标：bias <2 mm、随机误差 <3 mm。
- 通过后冻结 GT 管线版本/hash。

## 4. 真机 adapter smoke

1. 用 `DobotRobot` + 真实相机源构造 `DobotAdapter`；**必须传 `fk_params=load_fk(...)` 和 `safety_limits=SafetyLimits.from_yaml(...)`**，否则 adapter 只查 x/y/z 包围盒，不查关节可达性。
2. 工具朝向按“朝下”四元数 `(0,1,0,0)` 上报；`move` 目标姿态必须复制实测姿态，偏离超过 0.35 rad 会被拒绝。
3. 先只做 `read_state` / `capture` / 单步小位移 / 吸盘 ON-OFF。
3. 确认：
   - `description` 的 `rotation="fixed"`、`gripper="binary"` 被官方 `move/grasp/release` 接受；
   - `step_pose` 的 `stop_reason`、`position_error_mm` 合理；
   - 快照里三路图像+关节角同一时刻；
   - 吸盘 `grasp`→ON、`release`→OFF。
4. 再跑 3 条端到端 smoke，确认无超限/碰撞。

## 5. 接感知 handler

`src/dobot_rua/integration/tools.py` 目前是 schema + “not wired” stub。需要在环境服务里：

- 从 `episode.latest_snapshot` / `artifact_mirror` 读取 front/side/wrist 图像；
- 调用 `perception/blocks.py`、`lines.py`、wrist 近景模块；
- 返回 `ToolOutput` + debug 图；
- 不在工具里返回 GT。

## 6. 安全

- 所有真机动作前过 `SafetyLimits.check_joints/check_step` 和 adapter 工作空间检查。
- 急停/报警立即停运动、关吸盘。
- 首次运行用低速、小步、手放在急停上。
- `close()` 必须关吸盘。

## 7. 建议顺序

1. `pytest`（离线）
2. DeepSeek 探针
3. 相机/机械臂 bring-up
4. front/side 标定
5. wrist 竖直板标定
6. 交点 GT + 验证
7. 真机 adapter smoke
8. 感知 handler 接线
9. pilot 10–15 条
10. 冻结 rubric，正式 DOE
