# 动作识别模块（action module）接入说明

本模块是**下游消费者**：读取感知层的 `poses_3d.json`，输出动作事件
`actions.json`。不负责检测/姿态/ReID/3D 重建。

```text
perception（感知层）                     action module（本模块）
┌────────────────────────────┐          ┌──────────────────────────────┐
│ RF-DETR 检测（人/球）        │          │ 1. 适配（schema 桥接）        │
│ RTMPose 姿态                │  poses_  │    - ball_measurements       │
│ ReID + 3D 重建              │ ──────▶  │    - hoop_3d（坐标系）        │
│ 轨迹可视化（俯视/骨架）       │ 3d.json │ 2. 球轨迹后处理               │
└────────────────────────────┘          │    弹道分段/弹跳/飞行-运球状态   │
                                        │ 3. 规则引擎                   │
                                        │    持球状态机 + 事件判定        │
                                        └──────────────┬───────────────┘
                                                       ▼
                                                actions.json
                                       （pass/shoot/layup/rebound/
                                         follow_up/block + make/miss
                                         + 逐帧 possession）
```

## 一条命令跑完

```bash
python tools/run_action_pipeline.py \
    --poses <perception output>/poses/poses_3d.json \
    --output-dir <out> \
    --extrinsics <perception>/assets/extrinsic_parameters/extrinsics_new_calibration.json \
    --intrinsics <perception>/assets/intrinsics_parameters/undistorted_intrinsics_correct.json \
    --print-stats
```

产物：`actions.json`（事件 + `possession` 逐帧持球人）。

## 支持两种输入 schema

| 输入来源 | 识别方式 | 处理 |
|---|---|---|
| **perception 模块**（`schema_version = "2.0-rfdetr-rtmpose"`） | 有 `balls_3d` / `balls_3d_predicted` / `balls_2d`，无 `ball_measurements` | 用 `balls_3d`（排除 predicted 维持帧）构建 `ball_measurements`；用标注的篮筐像素 + perception 标定**三角化篮筐**（perception 不输出篮筐） |
| **本模块自带 pipeline 输出** | 有 `ball_measurements` + 同目录 `hoop_3d.json` | 直接复用（坐标系一致） |

两种情况都不需要改感知层代码；`tools/adapt_perception.py` 是桥接器，也可单独调用。

## 依赖

动作模块（`src/ball_trajectory/` + `src/action_rules/`）**只依赖 numpy/scipy/json**，
不需要 torch / onnxruntime / mmpose——可以在一台只装了 numpy/scipy 的机器上
跑完整条动作链（只要拿到 `poses_3d.json`）。

## 球检测的两种配置（对比实验结论）

| 配置 | 球 3D 观测覆盖 | 各视角球检出 | 说明 |
|---|---|---|---|
| 2XL 单模型（mixed-fp16 ONNX） | 82.7% | 60-72% | 通用模型，球召回偏低 |
| **2XL 人 + 微调球（hybrid）** | **92.0%** | **65-81%** | 人的框/mask 来自 2XL，球的框来自微调检测器 |

hybrid 模式已在感知层实现（`rfdetr.backend: hybrid` +
`rfdetr.ball_checkpoint_path: <微调权重>`），感知层其余部分（姿态/ReID/3D）不变。

## 坐标与篮筐

- 两个模块各自的世界系由各自的标定文件定义；动作模块不做跨系变换，
  所有几何（篮筐、三分半径、距筐距离）都用输入文件所在坐标系的数值。
- perception 的标定假设标准半场（`court_world_bounds [0,15,0,14]`）；
  本模块自带 pipeline 的重标定按实测场地（`[0,13,0,12.5]`，篮筐
  (6.5, 1.7, 3.05)）。**接入 perception 时用感知层的标定**，篮筐由
  桥接器三角化得到（当前实测 ≈ (7.64, 1.81, 3.05)）。
- 换场地/换标定后，只需更新 `--extrinsics/--intrinsics`，桥接器会重新
  三角化篮筐；也可用 `--hoop-center x y z` 直接指定。

## 对比实验：球检测配置（900-1800 帧，同一规则引擎）

| 配置 | 球 3D 观测 | view1/2/3/4 检出 | 动作匹配 | 区间多检 |
|---|---|---|---|---|
| 2XL 单模型（通用） | 82.7% | 72/68/60/49% | 5/5 | 7 |
| **2XL 人 + 微调球（hybrid）** | **88.7%** | **81/79/65/49%** | 4/5 | 12 |
| 本模块自带 pipeline（微调球） | 92.0% | 81/79/65/50% | 5/5 | 7 |

结论：
- **微调球检测把 2D 检出率提升 6-11pp**（三个配置里 2D 检出率由检测器决定：
  换成微调模型后队友 pipeline 的 view1/2/3 检出率与我们的完全一致）。
- 3D 观测率的差异（88.7 vs 92.0）来自**三角化/在线滤波阶段**，不是检测。
- 动作匹配在 900-1800（5 个 GT 事件）上三者接近——样本太小，
  完整对比见全片评估。

## 评估注意：id_map 必须按 pipeline 重建

每个 pipeline 的 track ID 是内部编号（跨视角关联用，不是人工身份），
**两个 pipeline 的编号不通用**。评估前用事件共现自动重建 id_map：

```python
# 每个 GT id 取 ±60 帧内同类型事件出现最多的 track 作为映射
co[gt_id][our_actor] += 1  ->  id_map[gt_id] = argmax
```

用错 id_map 会把"检测正确但编号不同"误判成 wrong_player（实测会低估 20-40%）。
