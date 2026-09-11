#!/bin/bash

# ActionCLIP 微调脚本 - SpaceJam 数据集
#
# 使用方法:
#   bash train_spacejam.sh
#
# 或者自定义参数:
#   bash train_spacejam.sh --epochs 20 --batch-size 16

# set -x

# 激活 conda 环境
source /data/ljy23/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab
export CUDA_VISIBLE_DEVICES=5,6,7,8
# 基础配置
PYTHON="python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_actionclip_spacejam.py"

# 训练参数
EPOCHS=10
BATCH_SIZE=32
LR=2e-6
MODEL="ViT-B/32-8"
CLIP_LEN=8
GPUS=4
FREEZE_BACKBONE="false"

# 数据路径
TRAIN_ANNO="${SCRIPT_DIR}/data/spacejam/train.csv"
VAL_ANNO="${SCRIPT_DIR}/data/spacejam/val.csv"
LABEL_MAP="${SCRIPT_DIR}/data/spacejam/label_map.txt"

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/work_dirs/actionclip_spacejam"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --clip-len)
            CLIP_LEN="$2"
            shift 2
            ;;
        --freeze-backbone)
            FREEZE_BACKBONE=true
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 构建冻结骨干的参数
if [[ "${FREEZE_BACKBONE}" == "true" ]]; then
    FREEZE_ARG="--freeze-backbone"
    echo "Backbone: 冻结（只训练adapter）"
else
    FREEZE_ARG=""
    echo "Backbone lr: ${LR} * 0.1 = $(echo "${LR} * 0.1" | bc)"
fi

echo "========================================"
echo "ActionCLIP 微调 - SpaceJam 数据集"
echo "========================================"
echo "模型: ${MODEL}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "GPUs: ${GPUS}"
echo "Freeze:${FREEZE_BACKBONE}"
echo "========================================"

# 运行训练
if [[ "${GPUS}" -gt 1 ]]; then
    echo "使用 torchrun 启动多卡训练"
    torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS} ${TRAIN_SCRIPT} \
        --train-anno ${TRAIN_ANNO} \
        --val-anno ${VAL_ANNO} \
        --label-map ${LABEL_MAP} \
        --model ${MODEL} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --clip-len ${CLIP_LEN} \
        --output-dir ${OUTPUT_DIR} \
        --workers 8 \
        --device cuda \
        --use-detailed-descriptions \
        --use-distributed \
        ${FREEZE_ARG}
else
    ${PYTHON} ${TRAIN_SCRIPT} \
        --train-anno ${TRAIN_ANNO} \
        --val-anno ${VAL_ANNO} \
        --label-map ${LABEL_MAP} \
        --model ${MODEL} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --clip-len ${CLIP_LEN} \
        --output-dir ${OUTPUT_DIR} \
        --workers 8 \
        --device cuda \
        --use-detailed-descriptions \
        ${FREEZE_ARG}
fi

echo "========================================"
echo "训练完成!"
echo "========================================"