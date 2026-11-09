# 坐标帧契约（dobot-rua）

- 日期：2026-11-09
- 状态：v1；标定、adapter、感知代码以此为准。

## 1. 帧定义

| 帧 | 定义 | 谁提供 | 谁使用 |
|---|---|---|---|
| `W` world/base | 固件基座系；x 前、y 左、z 上 | 固件 `pose()` / `move_to` | adapter、FK、外参 |
| `F` flange | 固件报告和接受的末端点；EE offset = 0 | 固件 `pose()` | 硬件层、wrist 手眼 |
| `T` suction tip | 吸盘尖 = `F + (0, 0, -L)`，L = `tip_length_mm` = 66.6 mm | 实测（安装面→吸盘唇口） | `control_frame="suction_tip"` |
| `C` optical | OpenCV：x 右、y 下、z 前 | `K` + distortion | 感知 |
| `B` board | ChArUco 板系，z = 0 | `solvePnP` | 标定 |

## 2. 硬规则

1. **单位**：harness 世界系用米；`DobotAdapter` 在硬件边界转毫米；`cameras.yaml` 里的 T 用米；`tcp_mm` 用毫米；时间用秒。
2. **刚体性**：所有 `T_world_from_optical` / `T_tool_from_optical` 必须满足 `RᵀR = I`、`det(R) = +1`、底行 `[0,0,0,1]`。`load_camera_specs` 在加载时拒绝反射。
3. **板系**：`solvePnP` 返回的板 `+z` 通常背离相机（在基座系里朝下）。转成 level-board 约定必须用刚体 `Rx(180) = diag(1, -1, -1)`。**`diag(1, 1, -1)` 是反射（det = -1），禁止使用。**
4. **吸盘尖**：吸盘竖直、装在 j4 轴心上，所以 `tip = flange + (0, 0, -L)`，`flange_cmd = tip_target + (0, 0, +L)`。L 是 `tip_length_mm`（实测 66.6 mm，末端安装面→吸盘唇口）。x/y 不变；j4 只绕吸盘自身轴转，不影响位置。不要再用 3 维工具系 TCP。
5. **wrist**：`T_tool_from_optical` 是相对**三轴法兰系 F**，不是吸盘尖 T。`capture()` 传给相机源的是 F；若改成 T，必须重做 wrist 手眼。
6. **标定验收（DOE 前置）**：板重投影 <2 px；触点/held-out 3D RMS <2 mm、max <3 mm；闭环 10–20 个独立点 bias <2 mm、随机 <3 mm；每 session touch bias <3 mm，超了暂停。
7. **失败关闭**：外参 `det ≠ +1`、残差超限、`calibration_hash` 缺失、或 `tcp_mm` 未测，都不得用于正式 trial。
8. **回零固定动作**：每次回零后 j4 必须逆时针转 88° 并保持（`post_home_j4_offset_deg`）。只用 `scripts/home_robot.py` 或 `DobotRobot.home()` 回零；标定扫位姿用关节空间命令锁住 j4=88，不要用 Cartesian `r`。
9. **前视平面模型（2026-11-09）**：`T_world_from_optical` 是两平面 PnP 的主模型；`H_table`/`H_top` 由 T 派生。对应点是吸盘中心 `p_cup_board=(25,75,0) mm`，不是板中心。块顶用 `z = table_z + block_height`，线交点用 `table_z`。`calibrated_region` 外的点必须拒绝；front 没有 VALID planar 时感知工具 fail-closed。旧的 canonical `front.T_world_from_optical` 已重命名为 `T_world_from_optical_invalid_stale_2026-11-09`，不得再用。

## 3. 2026-11-08 前视外参为什么作废

`configs/cameras.yaml` 里原来的 front `T_world_from_optical`（已重命名为 `T_world_from_optical_invalid_reflection_2026-11-08`）`det(R) = -1`：它是反射，不是刚体变换，会把每个投影点翻一面。

根因是 `level_board.py` 用 `diag(1, 1, -1)` 表示“板 +z 朝下”；正确刚体是 `Rx(180) = diag(1, -1, -1)`。同一次标定还把固件法兰当成吸盘尖（TCP 未建模），并用板平面当世界 z = 0，导致约 25 mm 的 z 偏差。该矩阵不得用于任何感知或 trial。

## 4. 当前待办（需要硬件）

1. ~~实测 `tcp_mm`~~：已实测 **L = 66.6 mm**（末端安装面→吸盘唇口）；固件 `GetEndEffectorParams=(0,0,0)`，确认控制器报的是法兰。
2. **底座被移动过**：front/side 外参、桌面 z、targets、旧触点 session 全部作废；先回零。
3. 同一 session 重拍大板 + 8–10 个触点（记录 tip = flange − L），跑 `scripts/finish_front_extrinsic.py solve`（默认 `--holdout 2`，均匀留出）；fit 和 held-out 都 ≤3 mm 才保存。
4. 闭环独立验证 10–20 点（+x/+y、近/远），bias <2 mm、随机 <3 mm。

## 5. 验证与失败关闭

- `scripts/validate_front_extrinsic.py --move` 用 `tip_length_mm` 下发吸盘尖目标（法兰 = tip + (0,0,L)）；`--tip-length-mm` 可覆盖 config。
- 感知工具在 front 外参缺失或非 proper 时返回 `success=false` + 明确错误，不再静默产生 NaN/IndexError。
- 当前 config 的 front `T_world_from_optical` 为空；在重新标定并通过 3 mm 门限前，不要跑绝对位置 trial。
