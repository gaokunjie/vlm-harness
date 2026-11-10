# 底座 j1 尺度差异（2026-11-10）

## 结论先行

这台 Dobot Magician 的**底座物理转角**不是固件报告/命令的 j1，而是：

```text
j1_physical = 1.26932 × j1_firmware
```

因此固件返回的 `pose().x/y`、`j1_deg` 都落在“固件坐标系”里，和相机看到的物理世界相差一个随 j1 增大的角度误差。y 方向在 `j1≈16.7°` 时报告 60 mm、实际约 80 mm；这是本轮抓取失败的**主因**，不是 front 手眼本身。

修复已落盘：`configs/robot_limits.yaml` 的 `base_j1_scale: 1.26932`，以及 `src/dobot_rua/hardware/dobot.py` 的坐标换算。

## 现象

- 绿块→红线交点的 agent 抓放失败；front 手眼被怀疑。
- 旧的 front 两平面 PnP（板贴吸盘、`p_cup=(25,75)`）残差 **37.5 px / holdout 41.1 px**，完全不可用。
- 用同一批 `board_pose + tip` 数据反解，发现误差随 y 平滑变化（y=−20 时偏 −12 mm，y=+60 时偏 +13 mm），且在 `z=−58` 与 `z=−35` 两个高度**几乎完全相同**。
- 误差不随深度变化，排除相机焦距/主点/畸变这类内参问题；指向机器人 x/y 平面的运动学尺度误差。

## 测量与证据

### 1. 尺子测试（物理位移）

固定 `x=200 mm`、法兰 `z=41 mm`（tip z=−30 mm），命令 y 从 0 到 60 mm：

| 状态 | 固件报告 y | 尺子实测 |
|---|---:|---:|
| 修复前 | 60 mm | **约 80 mm** |
| 修复后（j1 除以 1.26932） | 47.5 mm（固件帧） | **约 60 mm** |

### 2. 相机 + 标定板拟合（22 个位姿）

板贴在吸盘上，front 相机解 `T_cam_board`，板坐标 `p_cup=(75,25,0)`（用户坐标轴与 OpenCV 板坐标轴互换：用户 (25,75) = OpenCV (75,25)），机器人 tip 为 `flange − (0,0,71)`。

拟合模型：

```text
tip_physical = (r·cos(k·j1_fw), r·sin(k·j1_fw), tip_z)
r = hypot(x_fw, y_fw),  j1_fw = atan2(y_fw, x_fw)
```

结果：

- `k = 1.26932`
- 3D 刚体残差从 **9.0 mm** 降到 **3.4 mm**
- 残差不再随 y 平滑漂移

### 3. 排除“横向工具偏移”

只转 j4（88° → 0°，j1/j2/j3 不变），板坐标 `(75,25)` 点的投影只移动约 **1.5 mm**。说明吸盘尖基本在 j4 轴线上，不存在明显的水平工具偏移；黑块是轴对称/居中支架。因此误差来自底座 j1，而不是 `tip = flange − (0,0,L)` 漏了水平项。

## 代码改动

### 配置

`configs/robot_limits.yaml`：

```yaml
post_home_j4_offset_deg: 88.0
base_j1_scale: 1.26932
```

### 硬件层

`src/dobot_rua/hardware/dobot.py`：

- 新增 `physical_from_firmware(x_mm, y_mm, base_j1_scale) -> (x_phys, y_phys, j1_phys)`
- 新增 `firmware_from_physical(x_mm, y_mm, base_j1_scale) -> (x_fw, y_fw)`
- `DobotRobot.__init__(..., base_j1_scale=1.0)`
- `read_state()`：固件 x/y/j1 → 物理坐标
- `move_joints()`：物理 j1 → 固件 j1（除以 scale）
- `move_pose()`：物理 x/y → 固件 x/y（极角除以 scale）后再下发 `MOVL_XYZ`

### 调用方

- `src/dobot_rua/integration/server.py`：从 `configs/robot_limits.yaml` 读 `base_j1_scale` 并传给 `DobotRobot`
- `scripts/home_robot.py`、`scripts/calibrate_front_planar.py`：同样传入
- 测试：`tests/test_hardware_base_scale.py`
  - 正/反变换 round-trip
  - `move_joints` 下发缩放后的 j1
  - `read_state` 返回物理 j1

## 验证

- 尺子：修复后命令物理 y=60 mm，实测约 60 mm。
- 全量 `pytest -q` 通过。
- front 两平面重扫（校正后的物理坐标）：
  - 20 个有效位姿，marker 重投影 0.7–1.3 px
  - 3D 刚体残差 **3.34 mm**
  - 像素残差 **9.19 px RMS / 20.09 px max**
  - 状态仍为 **PROVISIONAL**（未过正式 2 px / 3 px 门限）

## 对 front 标定的影响

同一批数据、同一 `p_cup=(75,25,0)`：

| 版本 | 3D 刚体残差 | 像素残差 |
|---|---:|---:|
| 修复前（固件坐标） | 9.0 mm | 22.2 px |
| 修复后（物理坐标） | 3.3 mm | 9.2 px |

同时修正了 `table_z_m`：从 `−0.060` 改为 **`−0.093`**（8 个触点的实测板面/桌面 z）。旧的 −60 mm 是“台架高度”读数，不是机器人 z 坐标系；它会让桌面上的所有投影产生约 33 mm 的平面偏差。

## 注意事项

1. `base_j1_scale` 是**全局标量**。拟合中同时放半径尺度可把残差从 3.4 mm 降到 2.7 mm，但未采用；如后续精度不够，可再拟合 `j1_scale + radius_scale`。
2. 该值针对**当前底座/皮带/固件组合**。更换底座、皮带、固件或机器人后必须重测。
3. 重测方法：尺子法（命令物理 y=0→60 mm，读实际位移）或相机-板法（拟合 `k`），两者误差应在 2–3% 内。
4. front 标定仍是 PROVISIONAL：纸板在吸盘上轻微弯曲，9.2 px 残差主要来自标定板本身。正式 DOE 前需换硬质衬底并重新标定到 `VALID`。
