# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional

from mmcv.transforms import BaseTransform
from mmengine.dist import get_dist_info

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import bbox_xyxy2cs


@TRANSFORMS.register_module()
class GetBBoxCenterScale(BaseTransform):
    """Convert XYXY bounding boxes to padded center/scale values."""

    def __init__(self, padding: float = 1.25) -> None:
        super().__init__()
        self.padding = padding

    def transform(self, results: Dict) -> Optional[dict]:
        if 'bbox_center' in results and 'bbox_scale' in results:
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('Using existing bbox_center and bbox_scale; '
                              'padding is still applied.')
            results['bbox_scale'] = results['bbox_scale'] * self.padding
        else:
            center, scale = bbox_xyxy2cs(
                results['bbox'], padding=self.padding)
            results['bbox_center'] = center
            results['bbox_scale'] = scale
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(padding={self.padding})'
