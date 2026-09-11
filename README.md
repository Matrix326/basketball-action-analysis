# basketball-action-analysis

多视角篮球视频的**感知 + 动作识别**：四路同步视频 → 检测/姿态/ReID/3D 重建 →
轨迹与可视化 → 规则动作识别（动作事件 + 命中判定 + 逐帧持球人）。

```
四路同步视频 + 相机标定
        │
        ▼
┌───────────────────────────┐
│ perception/                │  检测（hybrid：2XL 人 + 微调球）
│  感知层（追踪 + 可视化）    │  RTMPose 姿态 → ReID → 3D 重建
│                           │  → 轨迹视频 / 俯视轨迹 / 3D 骨架动画
└──────────┬────────────────┘  篮筐定位（hoop_detection）
           │  poses_3d.json（3D 骨架 + 球 + quality）
           ▼
┌───────────────────────────┐
│ a_r/rule_based_code/       │  球轨迹后处理（弹道分段/弹跳/状态）
│  动作识别模块              │  → 规则引擎（持球状态机 + 事件判定）
└──────────┬────────────────┘
           │  actions.json（pass/shoot/layup/rebound/follow_up/block
           ▼                 + make/miss + 逐帧 possession）
        动作事件 / 高光 / GT 评估
```

---

## 1. 模块任务与接口

| 模块 | 任务 | 输入 | 输出 | 入口 |
|---|---|---|---|---|
| **`perception/`** | 检测（人/球）、RTMPose 姿态、跨视角 ReID、3D 重建、轨迹与可视化、篮筐定位 | 四路同步去畸变 MP4；内参/外参 JSON；`config/config.yaml` | `poses/poses_3d.json`（3D 骨架 + 球 + quality）、`poses/tracks/*.jsonl`、轨迹/俯视/3D 骨架视频、`hoop_3d.json` | `src/run_rfdetr_full_pipeline.py` |
| **`action_recognition/rule_based_code/`** | 规则动作识别：球轨迹后处理 + 持球状态机 + 事件判定 | `poses_3d.json`（上面那个）；可选 `hoop_3d.json` | `actions.json`（事件 + 命中 + 逐帧持球人）、`ball_trajectory.json` | `tools/run_action_pipeline.py` |
| `action_recognition/actionclip/` | ActionCLIP 学习式动作识别（对比方案） | 视频 / SpaceJam 数据 | 动作分类结果 | `actionclip_yolo_sliding_window.py` |
| `action_recognition/FROSTER/` | FROSTER 学习式动作识别（对比方案） | 视频 / SpaceJam 数据 | 动作分类结果 | 见 `FROSTER/README.md` |

**关键接口约定**

- `poses_3d.json`（schema `2.0-rfdetr-rtmpose`）：
  - `poses_3d[frame][id]`：17×3 世界坐标（COCO-17 顺序，米）
  - `balls_2d[frame][view]`：球 2D 像素框；`balls_3d[frame]`：球 3D 位置
  - `balls_3d_predicted[frame]`：true = 预测/维持帧（非观测）
  - `quality[frame][id]`：重投影误差、有效关键点数、预测标记
  - 本仓库自带 pipeline 的输出额外含 `ball_measurements`（观测 + 视角数）
- `actions.json`：`actions[]`（type / frame / end_frame / actor_id / receiver_id /
  result / three_point 等）+ `possession`（逐帧持球人）+ `stats`
- **track ID 是各 pipeline 内部编号，互不通用**——GT 评估前按 pipeline 重建
  id_map（见 `action_recognition/rule_based_code/docs/integration.md`）

---

## 2. 跑整个项目（端到端）

### 2.1 环境

```bash
# 感知层（GPU 机器；模型走 Git LFS）
cd "<REPO_ROOT>"
git lfs install --local
git lfs pull --include="perception/models/**" --exclude=""
python -m pip install -r perception/requirements.txt

# 动作模块（只依赖 numpy/scipy/yaml，可与感知层同机或另机）
python -m pip install -r action_recognition/rule_based_code/requirements.txt
```

### 2.2 感知层：视频 → 3D 骨架 + 球 + 可视化

```bash
cd "<REPO_ROOT>/perception"
export TMPDIR="$PWD/output/tmp"; export MPLCONFIGDIR="$PWD/output/cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

# 预检（路径/标定/视频帧范围）
python src/check_inputs.py --config config/config.yaml --start-frame 0 --end-frame 19739

# 完整流程（检测 + 姿态 + ReID + 3D + 轨迹/俯视/3D 骨架可视化）
CUDA_VISIBLE_DEVICES=0 python src/run_rfdetr_full_pipeline.py \
  --config config/config.yaml --start-frame 0 --end-frame 19739 \
  --output-base output/full

# 篮筐定位（YOLO 检测 + 标定三角化 → hoop_3d.json）
python src/hoop_detection/run_hoop_detection.py \
  --config config/config.yaml --start-frame 900 --end-frame 1800
```

产物：`output/full/poses/poses_3d.json` + 各视角轨迹视频 + 俯视轨迹 + 3D 骨架动画。

### 2.3 动作识别：3D 数据 → 动作事件

```bash
cd "<REPO_ROOT>/action_recognition/rule_based_code"
python tools/run_action_pipeline.py \
  --poses ../../perception/output/full/poses/poses_3d.json \
  --output-dir output/actions \
  --extrinsics ../../perception/assets/extrinsic_parameters/extrinsics_new_calibration.json \
  --intrinsics ../../perception/assets/intrinsics_parameters/undistorted_intrinsics_correct.json \
  --print-stats
```

产物：`output/actions/actions.json`（动作事件 + 命中判定 + 逐帧持球人）。

### 2.4 评估（有 GT 时）

```bash
cd "<REPO_ROOT>/action_recognition/rule_based_code"
python src/action_rules/evaluate_gt.py \
  --gt <gt>/1-3v3-action.json \
  --actions output/actions/actions.json \
  --id-map <id_map>.json \
  --start-frame 0 --end-frame 19739 --tolerance 50 \
  --output /tmp/gt_eval.json
```

---

## 3. 配置

| 文件 | 管什么 |
|---|---|
| `perception/config/config.yaml` | 检测（`rfdetr.backend: hybrid` = 2XL 人 + 微调球）、姿态、ReID、3D、球跟踪、轨迹与可视化、`hoop_detection`、**相机标定路径与球场范围** |
| `action_recognition/rule_based_code/config/config.yaml` | 球轨迹后处理（弹道分段/补全/平滑参数）、动作规则阈值（持球/出手/接球距离、三分半径、命中容差、补篮窗口…）、输入输出路径 |

换场地必改：`perception/config` 的标定路径与 `court_world_bounds`，
`action_recognition/rule_based_code/config` 的 `three_point_radius_m`；篮筐 3D 由
`hoop_detection` 工具或动作模块的桥接器重新生成。

---

## 4. 目录结构

```text
basketball-action-analysis/
├── perception/                    # 感知层：检测/姿态/ReID/3D/可视化/篮筐
│   ├── src/rfdetr_pipeline/       # 检测（hybrid 后端）+ 姿态 + ReID + 3D
│   ├── src/track/                 # 轨迹生成与自适应跳变平滑（可视化）
│   ├── src/hoop_detection/        # 篮筐检测 + 3D 三角化
│   ├── src/run_rfdetr_full_pipeline.py   # 完整流程入口
│   ├── config/ assets/ models/    # 配置 / 标定 / 权重（LFS）
│   └── docs/interfaces.md         # 输入输出接口细节
└── action_recognition/
    ├── rule_based_code/           # 动作识别模块（规则）
    │   ├── src/ball_trajectory/   # 球轨迹后处理
    │   ├── src/action_rules/      # 规则引擎 + GT 评估
    │   ├── tools/                 # 桥接器 + 一键入口
    │   ├── docs/integration.md    # 模块边界 / 接入 / 对比实验
    │   └── config/                # 动作参数
    ├── actionclip/                # ActionCLIP 动作识别（对比方案）
    └── FROSTER/                   # FROSTER 动作识别（对比方案）
```

---

## 5. 文档索引

- 感知层：`perception/README.md`（快速运行、模型与环境、后端选择）、
  `perception/docs/interfaces.md`（帧号/坐标/schema 细节）
- 动作模块：`action_recognition/rule_based_code/README.md`（球轨迹后处理与规则引擎的
  完整逻辑、全部参数、评估结果、实现要点）、
  `action_recognition/rule_based_code/docs/integration.md`（模块接入、坐标约定、对比实验）
- 对比方案：`action_recognition/actionclip/README.md`、`action_recognition/FROSTER/README.md`
