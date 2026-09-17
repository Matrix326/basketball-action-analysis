"""Static field declarations for MMEngine's dynamically populated pose records.

At runtime these names are the original MMEngine classes. The type-only views
record fields populated by the pose pipeline and preserve the concrete return
class of MMEngine conversion methods.
"""

from typing import TYPE_CHECKING, Any, Sequence

from mmengine.structures import InstanceData as _InstanceData, PixelData as _PixelData

if TYPE_CHECKING:
    from torch import Tensor
    from typing_extensions import Self

    class InstanceData(_InstanceData):
        # Instance coordinates can be NumPy arrays or tensors, depending on the
        # transform stage; MMEngine intentionally permits both representations.
        keypoints: Any
        keypoint_scores: Any
        bboxes: Any
        bbox_scores: Any
        keypoint_x_labels: Any
        keypoint_y_labels: Any
        keypoint_weights: Tensor

        @staticmethod
        def cat(instances_list: Sequence[_InstanceData]) -> "InstanceData": ...

        def to_tensor(self) -> Self: ...

        def numpy(self) -> Self: ...

    class PixelData(_PixelData):
        heatmaps: Any

        def to_tensor(self) -> Self: ...

        def numpy(self) -> Self: ...

else:
    InstanceData = _InstanceData
    PixelData = _PixelData
