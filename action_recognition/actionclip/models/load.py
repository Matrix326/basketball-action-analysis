import os
import torch
from mmengine.dataset import Compose
from mmengine.runner.checkpoint import _load_checkpoint
from torchvision.transforms import Normalize

from .actionclip import ActionClip

_MODELS = {
    'ViT-B/32-8':
    'https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb/vit-b-32-8f.pth',  # noqa: E501
    'ViT-B/16-8':
    'https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x8_k400-rgb/vit-b-16-8f.pth',  # noqa: E501
    'ViT-B/16-16':
    'https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x16_k400-rgb/vit-b-16-16f.pth',  # noqa: E501
    'ViT-B/16-32':
    'https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/actionclip_vit-base-p16-res224-clip-pre_1x1x32_k400-rgb/vit-b-16-32f.pth',  # noqa: E501
}


def available_models():
    """Returns the names of available ActionCLIP models."""
    return list(_MODELS.keys())


def _transform(num_segs):
    pipeline = [
        dict(type='DecordInit'),
        dict(
            type='SampleFrames',
            clip_len=1,
            frame_interval=1,
            num_clips=num_segs,
            test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='FormatShape', input_format='NCHW'),
        lambda x: torch.tensor(x['imgs']).div(255),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ]
    return Compose(pipeline)


def init_actionclip(name_or_path, device):
    if name_or_path in _MODELS:
        # 加载预定义的预训练模型
        model_path = _MODELS[name_or_path]
        checkpoint = _load_checkpoint(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        clip_arch = name_or_path.split('-')[0] + '-' + name_or_path.split('-')[1]
        num_adapter_segs = int(name_or_path.split('-')[2])
    else:
        # 加载本地检查点（微调后的模型）
        if os.path.exists(name_or_path):
            checkpoint = torch.load(name_or_path, map_location='cpu')
            # 处理不同格式的checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 从state_dict推断模型参数
            if 'adapter.frame_position_embeddings.weight' in state_dict:
                num_adapter_segs = state_dict['adapter.frame_position_embeddings.weight'].shape[0]
            else:
                num_adapter_segs = 8
            
            # 从state_dict推断clip架构
            # 通过conv1.weight的形状判断patch_size
            # ViT-B/32: conv1.weight shape = [768, 3, 32, 32]
            # ViT-B/16: conv1.weight shape = [768, 3, 16, 16]
            if 'clip.visual.conv1.weight' in state_dict:
                conv1_weight = state_dict['clip.visual.conv1.weight']
                patch_size = conv1_weight.shape[2]  # 获取patch_size
                if patch_size == 32:
                    clip_arch = 'ViT-B/32'
                elif patch_size == 16:
                    print('加载vit16-16')
                    clip_arch = 'ViT-B/16'
                else:
                    raise ValueError(f"Unknown patch size: {patch_size}")
            else:
                # 默认使用ViT-B/32
                clip_arch = 'ViT-B/32'
        else:
            raise ValueError(f"Model {name_or_path} not found. Available models: {available_models()}")
    
    assert num_adapter_segs == \
           state_dict['adapter.frame_position_embeddings.weight'].shape[0]
    num_adapter_layers = len([
        k for k in state_dict.keys()
        if k.startswith('adapter.') and k.endswith('.attn.in_proj_weight')
    ])

    model = ActionClip(
        clip_arch=clip_arch,
        num_adapter_segs=num_adapter_segs,
        num_adapter_layers=num_adapter_layers)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, _transform(num_adapter_segs), num_adapter_segs
