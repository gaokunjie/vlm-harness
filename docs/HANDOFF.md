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

## 1. DeepSeek 探针

实网探针被 Claude Code 的 auto-mode classifier 拦截，需要你运行或授权：

```bash
export UNI_ROBOT_AGENT_API_KEY='<deepseek key>'
.venv/bin/python scripts/probe_deepseek.py --n 1
```

- 脚本也会回退使用 `DEEPSEEK_API_KEY` 或 `ANTHROPIC_AUTH_TOKEN`。
- 期望：至少 1 次 `tool_calls`，`latencies_s` 有值；失败会打印 HTTP 错误。
- 如果 401/403，说明当前 key 不适用于 `https://api.deepseek.com/v1`，需要单独的 DeepSeek API key。

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

1. 用 `DobotRobot` + 真实相机源构造 `DobotAdapter`。
2. 先只做 `read_state` / `capture` / 单步小位移 / 吸盘 ON-OFF。
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
