# ActionCLIP 篮球动作识别（SpaceJam 微调）

基于 ActionCLIP 的篮球动作识别项目，支持多人视频中按球员做滑动窗口动作识别与可视化。

## 目录结构

```
actionclip/
├── train_actionclip_spacejam.py        # 训练脚本
├── actionclip_yolo_sliding_window.py   # 推理脚本（YOLO检测 + 滑动窗口识别）
├── train_spacejam.sh                   # 多卡训练启动脚本
├── models/
│   ├── load.py                         # 模型加载（预训练/微调权重）
│   ├── actionclip.py                   # ActionCLIP 模型定义
│   └── adapter.py                      # 时序适配器
└── data/
    └── spacejam/
        ├── label_map.txt               # 标签映射（每行一个类名，行号=标签索引）
        ├── action_descriptions.txt     # 动作文本描述（简单）
        ├── action_descriptions_enhance.txt  # 动作文本描述（详细，推荐）
        ├── train.csv                   # 训练集标注（需自备）
        └── val.csv                     # 验证集标注（需自备）
```

## 环境依赖

- Python 3.8+，PyTorch（支持 CUDA）
- [MMAction2](https://github.com/open-mmlab/mmaction2) 及 mmengine（提供视频解码管线 DecordInit/SampleFrames 等）
- [CLIP](https://github.com/openai/CLIP)（`pip install git+https://github.com/openai/CLIP.git`）
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（推理用，`pip install ultralytics`）
- decord, opencv-python, torchvision, tqdm, numpy

```bash
pip install torch torchvision decord opencv-python tqdm
pip install ultralytics
pip install git+https://github.com/openai/CLIP.git
# 按 MMAction2 官方文档安装 mmaction2 与 mmengine
```

## 数据准备

### 数据集说明

本项目使用 **SpaceJam 篮球动作数据集**，包含 10 类篮球动作。视频为按人物裁剪的短片段（每人一段），并通过翻转/旋转/平移等数据增强扩充样本量。

数据集文件夹组织如下：

```
spacejam/
├── examples/                  # 原始视频片段（按球员裁剪的短 mp4）
│   ├── 0000000.mp4
│   ├── 0000001.mp4
│   └── ...
└── augmented-examples/        # 数据增强后的视频（与原始片段共享标签）
    ├── 0000000_flipped_rotate_30.mp4        # 翻转+旋转30°
    ├── 0000000_flipped_rotate_330.mp4       # 翻转+旋转330°
    ├── 0000000_flipped_translate_-32_0.mp4  # 翻转+平移
    ├── 0000000_rotate_30.mp4                # 旋转
    ├── 0000000_translate_32_0.mp4           # 平移
    └── ...
```

- `examples/`：原始片段，文件名 `0000000.mp4`、`0000001.mp4` ... 递增
- `augmented-examples/`：对原始片段做 `flipped`（翻转）、`rotate_{angle}`（旋转）、`translate_{dx}_{dy}`（平移）等增强，文件名带增强方式后缀，与原始片段共享同一标签
- 10 个类别（对应 `label_map.txt`）：`block`、`pass`、`run`、`dribble`、`shoot`、`ball in hand`、`defense`、`pick`、`no_action`、`walk`

> 视频路径在 `train.csv`/`val.csv` 中以绝对路径记录。若数据集迁移到其他机器，需批量替换 csv 中的路径前缀。

### 1. 标注文件 `train.csv` / `val.csv`

每行一条样本，格式为 `视频路径,标签索引`（逗号分隔）：

```
/path/to/video_001.mp4,3
/path/to/video_002.mp4,0
/path/to/video_003.mp4,4
```

- 标签索引从 0 开始，对应 `label_map.txt` 的行号
- 视频路径建议用绝对路径

### 2. 标签映射 `label_map.txt`

每行一个类名，**行号即为标签索引**，顺序不能乱：

```
block
pass
run
dribble
shoot
ball in hand
defense
pick
no_action
walk
```

本项目默认 10 个类别（如上）。若更换数据集，请同步修改此文件。

### 3. 动作文本描述 `action_descriptions_enhance.txt`

用于生成 CLIP 文本特征，格式为 `类名:描述文本`：

```
block:a basketball player without possession of the ball jumping vertically ...
pass:a basketball player using one or both hands to thrust the ball horizontally ...
```

- 每行对应 `label_map.txt` 中的一个类
- 描述越具体、越有区分度，识别效果越好
- 训练/推理时通过 `--use-detailed-descriptions` 启用；不用则用 `--template` 模板生成简单描述

## 快速开始

### 训练

**单卡训练：**

```bash
python train_actionclip_spacejam.py \
    --train-anno data/spacejam/train.csv \
    --val-anno data/spacejam/val.csv \
    --label-map data/spacejam/label_map.txt \
    --model ViT-B/16-16 \
    --clip-len 16 \
    --epochs 10 \
    --batch-size 32 \
    --lr 5e-6 \
    --output-dir work_dirs/actionclip_spacejam \
    --use-detailed-descriptions \
    --action-descriptions data/spacejam/action_descriptions_enhance.txt
```

**多卡训练（推荐，用 train_spacejam.sh）：**

```bash
bash train_spacejam.sh
# 或自定义参数
bash train_spacejam.sh --epochs 20 --batch-size 16 --lr 2e-6 --model ViT-B/32-8 --clip-len 8 --gpus 4
```

> ⚠️ 使用前请修改 `train_spacejam.sh` 中的 conda 环境名、`CUDA_VISIBLE_DEVICES` 和数据路径。

### 推理

```bash
python actionclip_yolo_sliding_window.py \
    --video /path/to/your_video.mp4 \
    --start-frame 0 --end-frame 1800 \
    --model work_dirs/actionclip_spacejam/<timestamp>/best_model.pth \
    --yolo-model /path/to/yolov8n.pt \
    --window-len 32 --stride 14 \
    --label-map data/spacejam/label_map.txt \
    --use-detailed-descriptions \
    --action-descriptions data/spacejam/action_descriptions_enhance.txt \
    --out-filename output.mp4 \
    --fps 15
```

推理流程：YOLO 逐帧检测人物 → 跨帧追踪关联同一球员 → 滑动窗口采样 → ActionCLIP 识别 → 输出带动作标签的可视化视频。

## 训练参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-anno` | `.../data/spacejam/train.csv` | 训练集标注文件 |
| `--val-anno` | `.../data/spacejam/val.csv` | 验证集标注文件 |
| `--label-map` | `.../data/spacejam/label_map.txt` | 标签映射文件 |
| `--model` | `ViT-B/16-16` | ActionCLIP 预训练模型，可选 `ViT-B/32-8`/`ViT-B/16-8`/`ViT-B/16-16`/`ViT-B/16-32` |
| `--clip-len` | `16` | 每个视频采样帧数，**必须与模型名末尾数字一致**（如 `ViT-B/16-16` → 16） |
| `--epochs` | `5` | 训练轮数 |
| `--batch-size` | `32` | 批次大小（多卡时为每卡 batch） |
| `--lr` | `5e-6` | adapter 学习率；不冻结时 backbone 学习率 = `lr × 0.1` |
| `--weight-decay` | `0.01` | 权重衰减 |
| `--freeze-backbone` | False | 冻结 CLIP 骨干，只训练 adapter（显存更省、训练更快，但精度上限低） |
| `--output-dir` | `.../work_dirs/actionclip_spacejam` | 输出目录（每次训练自动建时间戳子目录） |
| `--resume` | None | 从 checkpoint 恢复训练 |
| `--template` | `The basketball player is {}` | 文本模板（未启用详细描述时使用） |
| `--use-detailed-descriptions` | False | 启用详细动作描述（推荐） |
| `--action-descriptions` | `.../action_descriptions_enhance.txt` | 描述文件路径 |
| `--use-distributed` | False | 多卡训练时由 torchrun 自动触发，一般无需手动指定 |
| `--workers` | `8` | DataLoader 线程数 |
| `--device` | `cuda` | 训练设备 |
| `--seed` | `42` | 随机种子 |

## 推理参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | `...` | 输入视频路径 |
| `--start-frame` | `1200` | 起始帧（0-based，含） |
| `--end-frame` | `1800` | 结束帧（含，-1 到视频末尾） |
| `--model` | `.../epoch_5.pth` | 训练好的模型权重路径（`.pth`） |
| `--yolo-model` | `.../yolov8n.pt` | YOLOv8 模型路径 |
| `--window-len` | `32` | 滑动窗口长度（帧） |
| `--stride` | `14` | 滑动步长（帧），越小窗口越密集 |
| `--input-size` | `224` | 模型输入尺寸 |
| `--expand-ratio` | `1.5` | 裁剪模式下检测框扩展比例 |
| `--padding-mode` | False | 填充模式（左右补黑边保持高度）vs 默认裁剪模式（扩展成正方形） |
| `--conf-thres` | `0.3` | YOLO 置信度阈值 |
| `--template` | `The basketball player is {}` | 文本模板（需与训练时一致） |
| `--use-detailed-descriptions` | False | 启用详细描述（需与训练时一致） |
| `--action-descriptions` | `.../action_descriptions.txt` | 描述文件路径 |
| `--label-map` | `.../label_map.txt` | 标签映射文件 |
| `--out-filename` | `...` | 输出视频路径 |
| `--fps` | `15` | 输出视频帧率 |
| `--device` | `cuda` | 推理设备 |

> **关键**：推理时的 `--template`、`--use-detailed-descriptions`、`--label-map` 必须与训练时一致，否则文本特征不匹配会导致识别错误。

## 训练策略说明

1. **分层学习率**：不冻结时 backbone 用 `lr×0.1`，adapter 用 `lr`，避免预训练 CLIP 权重被破坏
2. **Focal Loss**：`gamma=2.0`，`alpha` 为类别权重（按 `sqrt(total/(n×count))` 计算），缓解类别不平衡
3. **CosineAnnealingLR**：学习率余弦退火到 `1e-7`
4. **混合精度**：`GradScaler` + `autocast` 节省显存
5. **文本特征预计算**：训练开始前一次性编码所有类别的文本特征，不重复计算；文本编码器参数不更新
6. **多卡 DDP**：`find_unused_parameters=True`（文本编码器参数不参与前向但 requires_grad=True）

## 输出文件

训练输出在 `work_dirs/actionclip_spacejam/<时间戳>/` 下：

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 验证准确率最高的模型（含 state_dict/optimizer/scheduler/epoch/acc） |
| `epoch_{N}.pth` | 每轮保存的模型 |
| `latest_model.pth` | 最新一轮模型（用于 resume） |
| `training.log` | 训练日志（同时输出到终端） |
| `config.json` | 本次训练的全部参数 |

## ⚠️ 重要注意事项

1. **类别权重硬编码**：`train_actionclip_spacejam.py` 中 `class_counts`（约第 489 行）硬编码了 SpaceJam 10 类的训练样本数。**更换数据集后必须修改**为你的实际类别样本数，否则 Focal Loss 权重错误。统计方法：
   ```python
   # 统计 train.csv 每类样本数
   from collections import Counter
   with open('data/spacejam/train.csv') as f:
       labels = [int(line.strip().split(',')[1]) for line in f if line.strip()]
   print(Counter(labels))  # 按 0,1,2... 顺序填入 class_counts
   ```

2. **`--clip-len` 必须与模型匹配**：
   - `ViT-B/32-8` → `--clip-len 8`
   - `ViT-B/16-8` → `--clip-len 8`
   - `ViT-B/16-16` → `--clip-len 16`
   - `ViT-B/16-32` → `--clip-len 32`
   
   不匹配会导致 adapter 位置编码维度对不上，报错。

3. **路径修改**：代码中默认路径为原作者环境（`/data/ljy23/...`）。使用前请将 `train_actionclip_spacejam.py`、`actionclip_yolo_sliding_window.py`、`train_spacejam.sh` 中的数据路径、模型路径、输出路径、conda 环境名、GPU 编号改为你自己的。

4. **推理与训练一致性**：推理时的文本编码方式（`--template` / `--use-detailed-descriptions` + `--action-descriptions`）必须与训练时完全一致，否则文本特征空间不对齐。

5. **YOLO 模型**：推理依赖 YOLOv8 做人物检测，默认 `yolov8n.pt`。可换成更大的 `yolov8s.pt`/`yolov8m.pt` 提升检测精度。

6. **预训练模型下载**：首次训练会从 openmmlab 下载 ActionCLIP 预训练权重（约 1GB），需联网。
