#!/usr/bin/env python3
"""
Fine-tune FROSTER (TemporalClipVideo) on Spacejam dataset.

Usage (single GPU):
    python train_spacejam.py --config configs/Spacejam/TemporalCLIP_vitb16_spacejam.yaml

Usage (multi-GPU):
    python train_spacejam.py --config configs/Spacejam/TemporalCLIP_vitb16_spacejam.yaml --gpus 2

Or using torchrun:
    torchrun --nproc_per_node=2 train_spacejam.py --config configs/Spacejam/TemporalCLIP_vitb16_spacejam.yaml
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add FROSTER to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slowfast.models.optimizer as optim_module
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import EpochTimer, TrainMeter, ValMeter


def parse_args():
    parser = argparse.ArgumentParser(description="FROSTER fine-tuning on Spacejam")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/Spacejam/TemporalCLIP_vitb16_spacejam.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use-teacher",
        action="store_true",
        help="蒸馏"
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=1.0,
        help="disill weight",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load and merge config from YAML file."""
    from slowfast.config.defaults import get_cfg
    from slowfast.config.custom_config import add_custom_config

    cfg = get_cfg()
    # Ensure custom configs are registered
    add_custom_config(cfg)
    cfg.merge_from_file(config_path)
    cfg = assert_and_infer_cfg(cfg)
    return cfg


def compute_class_weights(cfg):
    """Load class weights for imbalanced data."""
    weights_path = os.path.join(
        os.path.dirname(cfg.DATA.INDEX_LABEL_MAPPING_FILE),
        "class_weights.json",
    )
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            weights_dict = json.load(f)
        num_classes = cfg.MODEL.NUM_CLASSES
        weights = torch.ones(num_classes)
        for k, v in weights_dict.items():
            idx = int(k)
            if idx < num_classes:
                weights[idx] = v
        return weights
    return None


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
    class_weights=None,
    distill_weight=0.0,
):
    """Train for one epoch."""
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    # Loss function with optional class weights
    if class_weights is not None:
        class_weights = class_weights.to(next(model.parameters()).device)
        loss_fun = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fun = nn.CrossEntropyLoss()

    for cur_iter, (inputs, labels, index, time_info, meta) in enumerate(train_loader):
        # Transfer data to GPU
        if cfg.NUM_GPUS:
            if isinstance(inputs, list):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], list):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            time_info = time_info.cuda(non_blocking=True)

        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )

        # Update learning rate
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim_module.get_epoch_lr(epoch_exact, cfg)
        optim_module.set_lr(optimizer, lr)

        train_meter.data_toc()

        # Forward pass
        optimizer.zero_grad()
        distill_loss_val = 0.0

        def _compute_loss(outputs):
            """计算loss, 支持普通模式和教师蒸馏模式"""
            nonlocal distill_loss_val
            if (
                isinstance(outputs, (list, tuple))
                and len(outputs) == 2
                and isinstance(outputs[1], (list, tuple))
            ):
                # 蒸馏模式: forward 返回 [student_list], [teacher_list]
                student_out, teacher_out = outputs
                preds = student_out[0]
                ce_loss = loss_fun(preds, labels)
                if distill_weight > 0:
                    s_feat, t_feat = student_out[1], teacher_out[1]
                    s_cls, t_cls = student_out[2], teacher_out[2]
                    feat_loss = 1.0 - F.cosine_similarity(
                        s_feat, t_feat.detach(), dim=-1
                    ).mean()
                    cls_loss = 1.0 - F.cosine_similarity(
                        s_cls, t_cls.detach(), dim=-1
                    ).mean()
                    distill_loss_val = (feat_loss + cls_loss).item()
                    loss = ce_loss + distill_weight * (feat_loss + cls_loss)
                else:
                    loss = ce_loss
                return preds, loss
            else:
                preds = outputs
                loss = loss_fun(preds, labels)
                return preds, loss

        if cfg.TRAIN.MIXED_PRECISION:
            with autocast():
                outputs = model(inputs)
                preds, loss = _compute_loss(outputs)
        else:
            outputs = model(inputs)
            preds, loss = _compute_loss(outputs)

        # Backward pass
        if cfg.TRAIN.MIXED_PRECISION:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Gradient clipping
        if cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim_module.get_grad_norm_(model.parameters())

        optimizer.step()
        if cfg.TRAIN.MIXED_PRECISION:
            scaler.step()
            scaler.update()

        # Compute metrics
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        top1_err, top5_err = None, None
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        # Gather metrics across GPUs
        if cfg.NUM_GPUS > 1:
            loss, grad_norm, top1_err, top5_err = du.all_reduce(
                [loss.detach(), grad_norm, top1_err, top5_err]
            )

        loss, grad_norm, top1_err, top5_err = (
            loss.item(),
            grad_norm.item(),
            top1_err.item(),
            top5_err.item(),
        )

        # Update stats
        train_meter.update_stats(
            top1_err, top5_err, loss, lr, grad_norm,
            batch_size * max(cfg.NUM_GPUS, 1),
        )

        # Tensorboard logging
        if writer is not None:
            scalars = {
                "Train/loss": loss,
                "Train/lr": lr,
                "Train/Top1_err": top1_err,
                "Train/Top5_err": top5_err,
            }
            if distill_loss_val > 0:
                scalars["Train/distill_loss"] = distill_loss_val
            writer.add_scalars(
                scalars,
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.iter_toc()
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    del inputs
    torch.cuda.empty_cache()
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    val_loader,
    model,
    val_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """Evaluate the model on validation set."""
    model.eval()
    val_meter.iter_tic()

    # 每类准确率统计(延迟初始化, 需从首个batch获取类别数)
    class_correct = None
    class_total = None
    logger = logging.get_logger(__name__)

    for cur_iter, (inputs, labels, index, time_info, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            if isinstance(inputs, list):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            index = index.cuda()
            time_info = time_info.cuda()

        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        val_meter.data_toc()

        preds = model(inputs)
        if isinstance(preds, (list, tuple)):
            if isinstance(preds[0], (list, tuple)):
                preds = preds[0][0]
            else:
                preds = preds[0]

        # 每类正确数/总数累加(向量化scatter_add, 高效)
        if class_correct is None:
            num_classes = preds.size(1)
            class_correct = torch.zeros(num_classes, device=preds.device)
            class_total = torch.zeros(num_classes, device=preds.device)
        pred_label = preds.argmax(dim=1)
        correct_mask = (pred_label == labels).float()
        class_correct.scatter_add_(0, labels, correct_mask)
        class_total.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float))

        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])

        top1_err, top5_err = top1_err.item(), top5_err.item()

        val_meter.iter_toc()
        val_meter.update_stats(
            top1_err, top5_err, batch_size * max(cfg.NUM_GPUS, 1),
        )

        if writer is not None:
            writer.add_scalars(
                {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                global_step=len(val_loader) * cur_epoch + cur_iter,
            )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()

    # 输出每类准确率
    if class_correct is not None:
        if cfg.NUM_GPUS > 1:
            class_correct, class_total = du.all_reduce(
                [class_correct, class_total], average=False
            )
        # 尝试读取类名(便于定位弱类)
        names = None
        try:
            mp = cfg.DATA.INDEX_LABEL_MAPPING_FILE
            if mp and os.path.exists(mp):
                with open(mp) as f:
                    lm = json.load(f)
                names = [lm[str(i)].split(":")[0].strip() for i in range(len(lm))]
        except Exception:
            names = None

        n_classes = len(class_correct)
        total_samples = int(class_total.sum().item())
        logger.info(
            f"===== Validation per-class accuracy (epoch {cur_epoch}) ====="
        )
        for c in range(n_classes):
            t = int(class_total[c].item())
            name = names[c] if names and c < len(names) else f"class_{c}"
            if t > 0:
                acc = class_correct[c].item() / t * 100.0
                logger.info(
                    f"  [{c:3d}] {name:<20s}: {acc:6.2f}% "
                    f"({int(class_correct[c].item())}/{t})"
                )
            else:
                logger.info(f"  [{c:3d}] {name:<20s}:   N/A (0 samples)")
        mean_acc = (
            class_correct.sum().item() / max(total_samples, 1) * 100.0
        )
        logger.info(
            f"  Overall(mean): {mean_acc:6.2f}% ({total_samples} samples)"
        )
        logger.info("=" * 60)


def train(args):
    """Main training function."""
    # Load config
    cfg = load_config(args.config)

    # Override num GPUs if specified
    if args.gpus is not None:
        cfg.NUM_GPUS = args.gpus

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize distributed training
    du.init_distributed_training(cfg)

    # Setup logging
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger = logging.get_logger(__name__)

    logger.info("=" * 60)
    logger.info("FROSTER Fine-tuning on Spacejam")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {cfg.OUTPUT_DIR}")
    logger.info(f"GPUs: {cfg.NUM_GPUS}")
    logger.info(f"Batch size: {cfg.TRAIN.BATCH_SIZE}")
    logger.info(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    logger.info(f"Max epochs: {cfg.SOLVER.MAX_EPOCH}")
    logger.info(f"Num classes: {cfg.MODEL.NUM_CLASSES}")
    logger.info("=" * 60)

    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 教师模型蒸馏
    if args.use_teacher:
        cfg.MODEL.KEEP_RAW_MODEL = True
        cfg.MODEL.RAW_MODEL_DISTILLATION = True
        logger.info(
            f"启用教师模型蒸馏: KEEP_RAW_MODEL=True, "
            f"RAW_MODEL_DISTILLATION=True, distill_weight={args.distill_weight}"
        )

    # Build model
    logger.info("Building model...")
    model = build_model(cfg)

    # Log trainable parameters
    if du.is_master_proc():
        trainable_params = []
        for k, v in model.named_parameters():
            if v.requires_grad:
                trainable_params.append(k)
        logger.info(f"Trainable parameters ({len(trainable_params)}):")
        for name in trainable_params[:20]:
            logger.info(f"  {name}")
        if len(trainable_params) > 20:
            logger.info(f"  ... and {len(trainable_params) - 20} more")

    # Load pre-trained weights
    if cfg.TRAIN.CUSTOM_LOAD:
        custom_load_file = cfg.TRAIN.CUSTOM_LOAD_FILE
        if os.path.exists(custom_load_file):
            logger.info(f"Loading pre-trained weights from {custom_load_file}")
            checkpoint = torch.load(custom_load_file, map_location="cpu")
            checkpoint_model = checkpoint.get("model_state", checkpoint)
            state_dict = model.state_dict()

            # Handle DDP module prefix
            if "module" in list(state_dict.keys())[0]:
                new_checkpoint_model = {}
                for key, value in checkpoint_model.items():
                    new_checkpoint_model["module." + key] = value
                checkpoint_model = new_checkpoint_model

            # Load with strict=False to handle class size mismatch
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint_model, strict=False
            )
            logger.info(f"Missing keys: {len(missing_keys)}")
            if len(missing_keys) <= 10:
                for k in missing_keys:
                    logger.info(f"  {k}")
            logger.info(f"Unexpected keys: {len(unexpected_keys)}")
        else:
            logger.warning(f"Pre-trained weights not found: {custom_load_file}")
            logger.warning("Training from scratch!")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        logger.info(f"Resuming from {args.resume}")
        checkpoint_epoch = cu.load_checkpoint(
            args.resume,
            model,
            cfg.NUM_GPUS > 1,
            epoch_reset=False,
        )
        start_epoch = checkpoint_epoch + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    elif cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Auto-resuming from last checkpoint")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
            )
            start_epoch = checkpoint_epoch + 1

    # Create optimizer
    optimizer = optim_module.construct_optimizer(model, cfg)

    # Create GradScaler for mixed precision
    scaler = GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Class weights for imbalanced data
    class_weights = compute_class_weights(cfg)
    if class_weights is not None:
        logger.info(f"Using class weights: {class_weights.tolist()}")

    # Create meters
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # Tensorboard writer
    writer = None
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, "tensorboard"))

    # Eval-only mode
    if args.eval_only:
        logger.info("Running evaluation only...")
        eval_epoch(val_loader, model, val_meter, 0, cfg, writer)
        return

    # Training loop
    logger.info(f"Starting training from epoch {start_epoch + 1}")
    epoch_timer = EpochTimer()
    best_top1 = 0.0

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        epoch_timer.epoch_tic()

        # Shuffle dataset
        loader.shuffle_dataset(train_loader, cur_epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(cur_epoch)

        # Train
        train_epoch(
            train_loader, model, optimizer, scaler,
            train_meter, cur_epoch, cfg, writer, class_weights,
            distill_weight=args.distill_weight if args.use_teacher else 0.0,
        )

        epoch_timer.epoch_toc()

        # Note: train_epoch already calls log_epoch_stats (which sets
        # top1_epoch, loss_epoch, etc.) followed by reset(). The reset()
        # clears num_samples but not the *_epoch attributes, so we can
        # safely access them here.
        logger.info(
            f"Epoch {cur_epoch + 1}: "
            f"train_top1={train_meter.top1_epoch:.2f}%, "
            f"train_loss={train_meter.loss_epoch:.4f}, "
            f"time={epoch_timer.last_epoch_time():.1f}s"
        )

        # Evaluate
        is_eval_epoch = (
            cfg.TRAIN.EVAL_PERIOD > 0
            and (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0
        )

        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
            # Note: eval_epoch already calls log_epoch_stats followed by reset()
            logger.info(
                f"Epoch {cur_epoch + 1} validation: "
                f"val_top1={val_meter.top1_epoch:.2f}%, "
                f"val_top5={val_meter.top5_epoch:.2f}%"
            )

            # Track best
            if val_meter.top1_epoch > best_top1:
                best_top1 = val_meter.top1_epoch
                # Save best model
                best_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pyth")
                if du.is_master_proc():
                    torch.save(
                        {
                            "model_state": model.module.state_dict()
                            if cfg.NUM_GPUS > 1
                            else model.state_dict(),
                            "epoch": cur_epoch,
                            "top1": best_top1,
                        },
                        best_path,
                    )
                logger.info(f"New best model saved: {best_path} (top1={best_top1:.2f}%)")

        # Save checkpoint
        is_checkpoint_epoch = (
            cfg.TRAIN.CHECKPOINT_PERIOD > 0
            and (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0
        )

        if is_checkpoint_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            logger.info(f"Checkpoint saved at epoch {cur_epoch + 1}")

    # Save final model
    final_path = os.path.join(cfg.OUTPUT_DIR, "final_model.pyth")
    if du.is_master_proc():
        torch.save(
            {
                "model_state": model.module.state_dict()
                if cfg.NUM_GPUS > 1
                else model.state_dict(),
                "epoch": cfg.SOLVER.MAX_EPOCH,
                "top1": best_top1,
            },
            final_path,
        )
    logger.info(f"Training complete! Final model saved to {final_path}")
    logger.info(f"Best top-1 accuracy: {best_top1:.2f}%")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
