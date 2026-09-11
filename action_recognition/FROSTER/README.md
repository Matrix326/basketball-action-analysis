# FROSTER 篮球动作识别 (Spacejam)

本项目基于 [**FROSTER: Frozen CLIP is a Strong Teacher for Open-vocabulary Action Recognition**](https://arxiv.org/abs/2402.03241)（ICLR 2024，[官方仓库](https://github.com/Visual-AI/FROSTER)）微调而来，用于**篮球比赛视频中的多人动作识别**，包含三条完整流水线：

1. **骨架预处理** —— 为训练数据叠加人体骨架（可选但推荐）
2. **模型微调** —— 在 Spacejam 篮球动作数据集上微调 TemporalClipVideo
3. **多人滑动窗口推理** —— YOLO 检测 + 跨帧追踪 + 滑动窗口动作识别 + 可视化

```
视频 ──► YOLO 人物检测/追踪 ──► 人框扩展成正方形 ──► 缩放 224×224 ──► 滑动窗口采样 8 帧 ──► FROSTER 分类 ──► 可视化标注
```

---

## 项目结构

```
FROSTER/
├── tools/
│   ├── train_spacejam.py              # 训练脚本（单卡/多卡/教师蒸馏）
│   ├── precompute_skeleton.py         # 离线骨架预处理
│   └── froster_yolo_sliding_window.py # 多人滑动窗口推理 + 可视化
├── configs/Spacejam/
│   ├── TemporalCLIP_vitb16_spacejam.yaml   # 训练/推理主配置（8 帧）
│   ├── spacejam_skeleton.yaml              # 骨架数据训练配置
│   └── TemporalCLIP_vitb16_spacejam_os.yaml # 16 帧实验配置
├── slowfast/                         # PySlowFast 框架（含 FROSTER 模型）
├── data/spacejam/                    # 数据集
└── output/teacher/                   # 训练输出（checkpoints 等）
```

---

## 环境依赖

- Python 3.8，PyTorch 1.11.0，torchvision 0.12.0（推荐使用 conda 环境 `ivnet`）
- [PySlowFast](https://github.com/facebookresearch/SlowFast)（仓库内已包含）
- `ultralytics`（YOLO 检测 + 姿态估计）
- `opencv-python`、`pyav`、`tensorboard`

```bash
conda create -n ivnet python=3.8
conda activate ivnet
pip install torch==1.11.0 torchvision==0.12.0
pip install ultralytics opencv-python av tensorboard
```

---

## 数据准备

数据集为 Spacejam 格式，目录下需要以下文件：

```
data/spacejam/
├── train.csv                    # 每行: <视频路径> <标签id>
├── val.csv                      # 每行: <视频路径> <标签id>
├── index_label_mapping_enhancev2.json   # 标签id -> 动作名映射
└── class_weights.json           # 类别权重（类别不平衡时使用）
```

- 当前数据集规模：`train.csv` 约 3.9 万条，`val.csv` 约 3300 条
- 视频为单人动作片段（128×176 小分辨率，约 16 帧）

---

## 骨架预处理（可选但推荐）

训练前可先为视频叠加人体骨架（黄色关键点 + 绿色骨架线），可以提高模型对姿态信息的利用。处理结果输出到独立目录（如 `data/spacejam_skeleton`），并自动重建 `train.csv` / `val.csv` 指向新视频。

```bash
python tools/precompute_skeleton.py \
    --data-dir data/spacejam \
    --pose-model /data/ljy23/project/pose/pose/model/yolo11x-pose.pt \
    --suffix _skeleton
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--pose-model` | yolo11x-pose.pt | 姿态估计模型（yolov8n-pose 更快） |
| `--conf` | 0.40 | 姿态检测置信度阈值 |
| `--draw-mode` | `overlay` | 绘制模式：overlay / skeleton_only / skeleton_on_black |
| `--line-thick` | 1 | 骨架线条粗细 |
| `--kpt-radius` | 2 | 关键点圆点半径 |
| `--max-persons` | 1 | 每帧最多画几个人（0 = 全部） |
| `--overwrite` | 关闭 | 覆盖已存在的骨架视频 |

> 关键点颜色统一为**黄色** `(0,255,255)`，连线统一为**绿色** `(0,255,0)`。
> ⚠️ 推理时若使用骨架数据训练的模型，必须开启推理脚本的 `--skeleton-mode`，否则输入分布不匹配。

---

## 训练

### 单卡训练

```bash
python tools/train_spacejam.py --config configs/Spacejam/TemporalCLIP_vitb16_spacejam.yaml
```

### 多卡训练

```bash
torchrun --nproc_per_node=4 tools/train_spacejam.py --config configs/Spacejam/TemporalCLIP_vitb16_spacejam.yaml
```

### 教师蒸馏训练（可选）

冻结的原始 CLIP 模型作为教师，微调模型作为学生，用特征/分类蒸馏约束学生：

```bash
python tools/train_spacejam.py \
    --config configs/Spacejam/TemporalCLIP_vitb16_spacejam.yaml \
    --use-teacher --distill-weight 1.0
```

### 骨架数据训练

修改配置文件中的 `DATA.PATH_TO_DATA_DIR` 指向骨架目录（或直接使用 `configs/Spacejam/spacejam_skeleton.yaml`）：

```bash
python tools/train_spacejam.py --config configs/Spacejam/spacejam_skeleton.yaml
```

### 关键配置（TemporalCLIP_vitb16_spacejam.yaml）

| 配置项 | 值 | 说明 |
| --- | --- | --- |
| `DATA.NUM_FRAMES` | 8 | 每个片段采样帧数 |
| `DATA.SAMPLING_RATE` | 16 | 帧采样间隔 |
| `DATA.TRAIN_CROP_SIZE` | 224 | 训练裁剪尺寸（短边随机缩放 [224,256] 后随机裁剪） |
| `DATA.TEST_CROP_SIZE` | 224 | 测试裁剪尺寸 |
| `MODEL.MODEL_NAME` | TemporalClipVideo | 模型结构 |
| `MODEL.TEMPORAL_MODELING_TYPE` | expand_temporal_view | 时间建模方式 |
| `TRAIN.BATCH_SIZE` | 32 | 批大小 |
| `SOLVER.BASE_LR` | 3.33e-6 | 初始学习率（cosine 衰减） |
| `SOLVER.MAX_EPOCH` | 20 | 训练轮数 |

### 训练输出

训练产物保存在配置的 `OUTPUT_DIR`（默认 `output/teacher`）：

```
output/teacher/
├── checkpoints/checkpoint_epoch_000018.pyth   # 周期 checkpoint
├── best_model.pyth                            # 验证集最优
└── final_model.pyth                           # 最终模型
```

---

## 推理（多人滑动窗口）

输入一段比赛视频，自动检测并追踪所有球员，对每个球员用滑动窗口逐段识别动作，输出带标注的可视化视频。

```bash
python tools/froster_yolo_sliding_window.py \
    --video /path/to/game.mp4 \
    --checkpoint output/teacher/checkpoints/checkpoint_epoch_00018.pyth \
    --config configs/Spacejam/TemporalCLIP_vitb16_spacejam.yaml
```

**如果模型是用骨架数据训练的，务必加骨架模式：**

```bash
python tools/froster_yolo_sliding_window.py \
    --video /path/to/game.mp4 \
    --checkpoint output/teacher/checkpoints/checkpoint_epoch_00018.pyth \
    --config configs/Spacejam/TemporalCLIP_vitb16_spacejam.yaml \
    --skeleton-mode
```

### 推理流程

1. **人物检测**：YOLOv8 检测每帧人物（`classes=[0]`）
2. **跨帧追踪**：基于检测框中心的最近邻匹配，关联同一个人
3. **裁剪**：以检测框中心扩展（`--expand-ratio 1.7`）成正方形，缩放到 224×224
4. **滑动窗口**：窗口 32 帧、步长 14，窗口内隔 4 帧采样出 8 帧
5. **识别**：FROSTER 输出 10 类动作概率，取 top-1 标注在视频上

### 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--start-frame` / `--end-frame` | 1300 / 1800 | 处理帧范围（-1 到末尾） |
| `--yolo-model` | yolov8n.pt | 人物检测模型 |
| `--conf-thres` | 0.3 | 检测置信度阈值 |
| `--expand-ratio` | 1.7 | 检测框扩展比例（人物在框内的占比 ≈ 1/1.7） |
| `--window-len` / `--stride` | 32 / 14 | 滑动窗口长度与步长 |
| `--input-size` | 224 | 模型输入尺寸（需与训练一致） |
| `--out-filename` | 自动生成 | 输出视频路径 |
| `--fps` | 15 | 输出视频帧率 |
| `--skeleton-mode` | 关闭 | 输入模型前叠加骨架（与 precompute_skeleton.py 参数一致） |
| `--pose-model` | yolo11x-pose.pt | 姿态估计模型（骨架模式用） |
| `--skeleton-line-thick` / `--skeleton-kpt-radius` / `--skeleton-max-persons` / `--skeleton-pose-conf` | 1 / 2 / 1 / 0.40 | 骨架绘制参数（与预处理脚本一致） |
| `--save-clips` | 关闭 | 保存每个窗口的模型输入片段，便于检查 |

### 输出

- 可视化视频：默认保存到 `output/spacejam/<视频名>_f<起>_<止>_froster.mp4`，每名球员有独立颜色框 + `P0: 动作名 (概率)` 标注
- 终端输出每个球员/全局的动作窗口分布统计

---

## 训练与推理输入一致性说明

| | 训练 | 推理 |
| --- | --- | --- |
| 空间尺寸 | 224×224（随机裁剪） | 224×224（人框扩展 1.7× 后正方形裁剪） |
| 帧数 | 8 帧（NUM_FRAMES=8） | 32 帧窗口隔 4 帧采样 → 8 帧 |
| 归一化 | CLIP 统计量 (/255 + 均值方差) | 相同 |
| 数据形态 | 原视频 或 骨架视频 | 与训练数据形态保持一致（骨架模型开 `--skeleton-mode`） |

> 训练视频是单人片段（人物几乎占满画面）；推理时人物占框内约 58%（1/1.7 扩展），如效果不佳可调小 `--expand-ratio`（如 1.2~1.3）以匹配训练尺度。

---

## 论文信息

FROSTER 通过冻结 CLIP 骨干 + 微调时间建模 + 知识蒸馏，在开放词汇动作识别上达到 SOTA（base-to-novel 与 cross-dataset 两个设置）。

- 论文：[arXiv:2402.03241](https://arxiv.org/abs/2402.03241)
- 官方仓库：[Visual-AI/FROSTER](https://github.com/Visual-AI/FROSTER)

```bibtex
@inproceedings{
  huang2024froster,
  title={FROSTER: Frozen CLIP is a Strong Teacher for Open-Vocabulary Action Recognition},
  author={Xiaohu Huang and Hao Zhou and Kun Yao and Kai Han},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## License

FROSTER 基于 [`CC BY-NC-SA 4.0 license`](https://creativecommons.org/licenses/by-nc-sa/4.0/)。

## Acknowledgement

本仓库构建于 [`OpenVCLIP`](https://github.com/wengzejia1/Open-VCLIP)、[`PySlowFast`](https://github.com/facebookresearch/SlowFast) 与 [`CLIP`](https://github.com/openai/CLIP) 之上。
