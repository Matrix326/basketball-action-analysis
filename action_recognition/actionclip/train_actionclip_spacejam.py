"""
ActionCLIP 微调脚本（针对 SpaceJam 数据集）

使用方法:
python train_actionclip_spacejam.py --config configs/actionclip_spacejam_finetune.py

训练策略:
1. 不冻结 CLIP backbone，使用分层学习率
2. Backbone 学习率: 1e-6, Adapter 学习率: 1e-5
3. 使用 Focal Loss 处理类别不平衡
4. 使用 cosine annealing 学习率调度
5. 使用混合精度训练（节省显存）
6. 每轮保存模型
"""

import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime

import clip
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# 导入 ActionCLIP 模型
from models.load import init_actionclip
from mmaction.utils import register_all_modules

# 注册 MMAction2 所有模块（必须在 init_actionclip 之前）
register_all_modules(True)


class Tee:
    """同时输出到终端和日志文件的流包装器"""
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        if not data:
            return
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()


class FocalLoss(nn.Module):
    """Focal Loss 处理类别不平衡"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SpaceJamDataset(Dataset):
    """SpaceJam 数据集加载器"""

    def __init__(self, annotation_file, clip_len=16, transform=None):
        """
        Args:
            annotation_file: CSV标注文件路径
            clip_len: 每个视频采样帧数
            transform: 数据增强变换
        """
        self.samples = []
        self.clip_len = clip_len
        self.transform = transform

        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    video_path, label = line.split(',')
                    if os.path.exists(video_path):
                        self.samples.append((video_path, int(label)))

        print(f"加载数据集: {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        try:
            video_anno = dict(filename=video_path, start_index=0)
            video_tensor = self.transform(video_anno)
            return {
                'video': video_tensor,
                'label': label,
                'video_path': video_path
            }
        except Exception as e:
            print(f"加载视频失败: {video_path}, 错误: {e}")
            return {
                'video': torch.zeros(3, self.clip_len, 224, 224),
                'label': -1,
                'video_path': video_path
            }


def get_model_module(model):
    """Return the underlying model for DDP-wrapped modules."""
    return model.module if isinstance(model, DDP) else model


def load_pretrained_text_features(model, label_map_path, device, template='The basketball player is {}',
                                   use_detailed_descriptions=False, action_descriptions_path=None):
    """加载预训练的文本特征"""
    with open(label_map_path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]

    if use_detailed_descriptions and action_descriptions_path:
        print(f"使用详细动作描述: {action_descriptions_path}")
        text_descriptions = []
        label_to_desc = {}
        with open(action_descriptions_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    action_name, desc = line.split(':', 1)
                    label_to_desc[action_name.strip()] = desc.strip()

        for label in labels:
            if label in label_to_desc:
                text_descriptions.append(label_to_desc[label])
                print(f"  {label}: {label_to_desc[label][:80]}...")
            else:
                text_descriptions.append(template.format(label))
                print(f"  {label}: (使用简单描述) {template.format(label)}")
    else:
        text_descriptions = [template.format(label) for label in labels]

    text = clip.tokenize(text_descriptions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print(f"\n加载文本特征: {len(labels)} 个类别")
    return text_features


def train_one_epoch(model, dataloader, optimizer, scheduler, text_features,
                    device, epoch, scaler=None, clip_len=16, criterion=None):
    """训练一个 epoch"""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    skipped_batches = 0

    is_ddp = isinstance(model, DDP)
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        videos = batch['video'].to(device)
        labels = batch['label'].to(device)

        if (labels < 0).any():
            skipped_batches += 1
            continue

        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            # 必须通过DDP模型调用forward，才能触发梯度同步
            # 直接调用 model.module.encode_video() 会绕过DDP，导致梯度不同步
            if is_ddp:
                # DDP forward: 调用 model(videos, mode='tensor') -> ActionClip.forward(inputs, mode='tensor')
                video_features = model(videos, mode='tensor')
            else:
                video_features = model.encode_video(videos)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * video_features @ text_features.T)
            if criterion is not None:
                loss = criterion(similarity, labels)
            else:
                loss = nn.functional.cross_entropy(similarity, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = similarity.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    scheduler.step()

    if skipped_batches > 0:
        print(f"  跳过 {skipped_batches} 个失败batch")

    return total_loss / len(dataloader), 100. * correct / total


@torch.no_grad()
def validate(model, dataloader, text_features, device, criterion=None,
             is_distributed=False):
    """验证函数"""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # 验证时直接用module即可，不需要梯度同步
    model_module = get_model_module(model)
    for batch in tqdm(dataloader, desc='Validating'):
        videos = batch['video'].to(device)
        labels = batch['label'].to(device)

        if (labels < 0).any():
            continue

        video_features = model_module.encode_video(videos)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * video_features @ text_features.T)
        if criterion is not None:
            loss = criterion(similarity, labels)
        else:
            loss = nn.functional.cross_entropy(similarity, labels)

        total_loss += loss.item()
        _, predicted = similarity.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        del videos, labels, video_features, similarity, loss
        torch.cuda.empty_cache()

    # 多卡训练时汇总验证结果
    if is_distributed:
        # 将 correct 和 total 汇总到所有rank
        correct_tensor = torch.tensor(correct, dtype=torch.float64, device=device)
        total_tensor = torch.tensor(total, dtype=torch.float64, device=device)
        loss_tensor = torch.tensor(total_loss, dtype=torch.float64, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        correct = correct_tensor.item()
        total = total_tensor.item()
        total_loss = loss_tensor.item() / world_size

    avg_loss = total_loss / len(dataloader)
    val_acc = 100. * correct / total if total > 0 else 0.0
    return avg_loss, val_acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='ActionCLIP Fine-tuning on SpaceJam')
    parser.add_argument('--train-anno', type=str,
                        default='/data/ljy23/project/stal/mm/mmaction2/projects/actionclip/data/spacejam/train.csv',
                        help='训练集标注文件')
    parser.add_argument('--val-anno', type=str,
                        default='/data/ljy23/project/stal/mm/mmaction2/projects/actionclip/data/spacejam/val.csv',
                        help='验证集标注文件')
    parser.add_argument('--label-map', type=str,
                        default='/data/ljy23/project/stal/mm/mmaction2/projects/actionclip/data/spacejam/label_map.txt',
                        help='标签映射文件')
    parser.add_argument('--model', type=str, default='ViT-B/16-16',
                        help='ActionCLIP模型')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='学习率（adapter）')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--use-distributed', action='store_true',
                        help='使用 torchrun 启动的多卡训练')
    parser.add_argument('--clip-len', type=int, default=16,
                        help='每个视频采样帧数')
    parser.add_argument('--output-dir', type=str,
                        default='/data/ljy23/project/stal/mm/mmaction2/projects/actionclip/work_dirs/actionclip_spacejam',
                        help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的checkpoint路径')
    parser.add_argument('--template', type=str, default='The basketball player is {}',
                        help='文本模板')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--use-detailed-descriptions', action='store_true',
                        help='使用详细动作描述而不是简单类别名')
    parser.add_argument('--action-descriptions', type=str,
                        default='/data/ljy23/project/stal/mm/mmaction2/projects/actionclip/data/spacejam/action_descriptions_enhance.txt',
                        help='详细动作描述文件路径')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='冻结CLIP骨干网络，只训练adapter层')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    is_distributed = args.use_distributed or int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = 0
    local_rank = 0
    world_size = 1
    if is_distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

    if args.device == 'cuda':
        if is_distributed:
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device(args.device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(work_dir, exist_ok=True)

    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if rank == 0:
        log_path = os.path.join(work_dir, 'training.log')
        log_file = open(log_path, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)

    try:
        with open(os.path.join(work_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

        print('=' * 60)
        print('ActionCLIP 微调 - SpaceJam 数据集')
        print('=' * 60)
        print(f'模型: {args.model}')
        print(f'训练集: {args.train_anno}')
        print(f'验证集: {args.val_anno}')
        print(f'输出目录: {work_dir}')
        print('训练参数:')
        print(f'  Epochs: {args.epochs}')
        print(f'  Batch size: {args.batch_size}')
        print(f'  Learning rate: {args.lr}')

        print(f'\n加载 ActionCLIP 模型...')
        model, transform, _ = init_actionclip(args.model, device=device)

        print(f'\n加载文本特征...')
        text_features = load_pretrained_text_features(
            get_model_module(model), args.label_map, device, args.template,
            use_detailed_descriptions=args.use_detailed_descriptions,
            action_descriptions_path=args.action_descriptions
        )

        print(f'\n加载数据集...')
        train_dataset = SpaceJamDataset(args.train_anno, args.clip_len, transform)
        val_dataset = SpaceJamDataset(args.val_anno, args.clip_len, transform)

        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True)
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, 16),
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=True
        )

        # DDP包装必须在创建optimizer之前完成
        # 因为DDP在构造时会broadcast参数，确保所有GPU初始参数一致
        if is_distributed:
            # find_unused_parameters=True: 因为训练时只调用encode_video，
            # CLIP文本编码器的参数不参与前向传播（或被冻结），但requires_grad=True
            model = DDP(model, device_ids=[local_rank],
                        find_unused_parameters=True)

        backbone_params = []
        adapter_params = []
        for name, param in model.named_parameters():
            if 'adapter' in name:
                adapter_params.append(param)
            else:
                backbone_params.append(param)

        if args.freeze_backbone:
            # 冻结骨干网络：只训练adapter
            for param in backbone_params:
                param.requires_grad = False
            optimizer = AdamW([
                {'params': adapter_params, 'lr': args.lr}
            ], weight_decay=args.weight_decay)
            print(f'\n训练模式: 冻结骨干网络（只训练adapter）')
            print(f'  Adapter 学习率: {args.lr}')
            print(f'  Backbone: 冻结（不更新）')
        else:
            # 不冻结：分层学习率
            optimizer = AdamW([
                {'params': backbone_params, 'lr': args.lr * 0.1},
                {'params': adapter_params, 'lr': args.lr}
            ], weight_decay=args.weight_decay)
            print(f'\n训练模式: 分层学习率（骨干+adapter）')
            print(f'  Backbone 学习率: {args.lr * 0.1}')
            print(f'  Adapter 学习率: {args.lr}')

        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        scaler = GradScaler()

        class_counts = [4381, 4715, 4744, 1431, 1876, 969, 3074, 3129, 5167, 9369]
        total = sum(class_counts)
        class_weights = [np.sqrt(total / (len(class_counts) * count)) for count in class_counts]
        print(f'\n类别权重:')
        for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
            print(f'  类别 {i}: {count} 样本, 权重: {weight:.2f}')

        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        criterion.to(device)

        def get_model_state_dict(model_module):
            if isinstance(model_module, DDP):
                return model_module.module.state_dict()
            return model_module.state_dict()

        best_val_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            print(f'\n{"=" * 60}')
            print(f'Epoch {epoch}/{args.epochs}')
            print(f'{"=" * 60}')

            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, text_features,
                device, epoch, scaler, args.clip_len, criterion
            )

            val_loss, val_acc, val_preds, val_labels = validate(
                model, val_loader, text_features, device, criterion,
                is_distributed=is_distributed
            )

            print(f'\n训练结果:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            if is_distributed:
                dist.barrier()

            if rank == 0:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoint = {
                        'epoch': epoch,
                        'state_dict': get_model_state_dict(model),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_acc': best_val_acc,
                        'val_acc': val_acc,
                        'train_acc': train_acc
                    }
                    torch.save(checkpoint, os.path.join(work_dir, 'best_model.pth'))
                    print(f'  保存最佳模型: Val Acc {val_acc:.2f}%')

                epoch_checkpoint = {
                    'epoch': epoch,
                    'state_dict': get_model_state_dict(model),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc
                }
                torch.save(epoch_checkpoint, os.path.join(work_dir, f'epoch_{epoch}.pth'))
                print(f'  保存 Epoch {epoch} 模型')

                torch.save({
                    'epoch': epoch,
                    'state_dict': get_model_state_dict(model),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(work_dir, 'latest_model.pth'))

        print(f'\n{"=" * 60}')
        print('训练完成!')
        print(f'{"=" * 60}')
        print(f'最佳验证准确率: {best_val_acc:.2f}%')
        print(f'模型保存在: {work_dir}')
    finally:
        if rank == 0:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if log_file is not None:
                log_file.close()
        if is_distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    print('cuda is available:', torch.cuda.is_available())
    main()
