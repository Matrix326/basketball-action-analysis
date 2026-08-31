"""Batched TensorRT and ONNX post-processing used by the main pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
import torchvision.transforms.functional as tv_f
from torchvision.ops import batched_nms
from basketball_repro.detection_runtime import detection_point, point_in_polygon

PERSON_CLASS_ID = 1

BALL_CLASS_ID = 37

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)

def lowres_mask_nms_keep(
    masks_low: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    count = int(masks_low.shape[0])
    if count <= 1:
        return torch.ones((count,), device=masks_low.device, dtype=torch.bool)

    masks_binary = masks_low > 0.0
    flat = masks_binary.flatten(1).to(dtype=torch.float32)
    intersection = flat @ flat.T
    areas = flat.sum(dim=1)
    union = areas[:, None] + areas[None, :] - intersection
    ious = intersection / union.clamp_min(1.0)

    order = torch.argsort(scores, descending=True)
    keep = torch.zeros((count,), device=masks_low.device, dtype=torch.bool)
    removed = torch.zeros((count,), device=masks_low.device, dtype=torch.bool)
    for index_tensor in order:
        index = int(index_tensor.item())
        if bool(removed[index]):
            continue
        keep[index] = True
        suppress = (ious[index] > threshold) & (labels == labels[index]) & (~keep)
        removed |= suppress
    return keep

def preprocess_frame(
    frame_bgr: np.ndarray,
    *,
    device: torch.device,
    resolution: int,
    means: Iterable[float] = IMAGENET_MEAN,
    stds: Iterable[float] = IMAGENET_STD,
) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().to(device)
    tensor = tensor.to(dtype=torch.float32).div_(255.0)
    tensor = tv_f.resize(tensor, [resolution, resolution])
    tensor = tv_f.normalize(tensor, list(means), list(stds))
    return tensor

def detections_from_raw_tensors(
    frame_bgr: np.ndarray,
    *,
    logits: torch.Tensor,
    boxes_cxcywh: torch.Tensor,
    masks_low: torch.Tensor,
    threshold: float,
    ball_threshold: float,
    roi_polygon: np.ndarray | None = None,
    roi_mode: str = "bottom-center",
    ball_min_size: float = 8.0,
    ball_max_size: float = 80.0,
    ball_min_aspect: float = 0.50,
    ball_max_aspect: float = 2.00,
    nms_threshold: float = 0.5,
    lowres_mask_nms: bool = True,
):
    import supervision as sv

    class_ids = torch.tensor([PERSON_CLASS_ID, BALL_CLASS_ID], device=logits.device)
    class_names = {PERSON_CLASS_ID: "person", BALL_CLASS_ID: "sports ball"}
    height, width = frame_bgr.shape[:2]
    selected_scores = logits[:, class_ids].sigmoid()
    query_idx, local_class_idx = torch.nonzero(selected_scores >= min(threshold, ball_threshold), as_tuple=True)
    if query_idx.numel() == 0:
        detections = sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            mask=np.empty((0, height, width), dtype=bool),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int64),
        )
        detections.data["class_name"] = np.empty((0,), dtype=object)
        detections.data["source_shape"] = np.empty((0, 2), dtype=np.int64)
        return detections

    labels = class_ids[local_class_idx]
    scores = selected_scores[query_idx, local_class_idx]
    keep = ((labels == PERSON_CLASS_ID) & (scores >= threshold)) | (
        (labels == BALL_CLASS_ID) & (scores >= ball_threshold)
    )
    query_idx = query_idx[keep]
    labels = labels[keep]
    scores = scores[keep]
    if query_idx.numel() == 0:
        detections = sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            mask=np.empty((0, height, width), dtype=bool),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int64),
        )
        detections.data["class_name"] = np.empty((0,), dtype=object)
        detections.data["source_shape"] = np.empty((0, 2), dtype=np.int64)
        return detections

    boxes = box_cxcywh_to_xyxy(boxes_cxcywh[query_idx])
    scale = torch.tensor([width, height, width, height], device=boxes.device, dtype=boxes.dtype)
    boxes = boxes * scale
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)

    person_keep = labels == PERSON_CLASS_ID
    ball_keep = labels == BALL_CLASS_ID
    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    size = torch.maximum(box_w, box_h)
    aspect = box_w / torch.clamp(box_h, min=1e-6)
    ball_keep = ball_keep & (size >= ball_min_size) & (size <= ball_max_size)
    ball_keep = ball_keep & (aspect >= ball_min_aspect) & (aspect <= ball_max_aspect)

    if roi_polygon is not None and person_keep.any():
        boxes_np = boxes.detach().cpu().numpy()
        roi_keep = torch.zeros_like(person_keep)
        for index in torch.nonzero(person_keep, as_tuple=False).flatten().tolist():
            foot = detection_point(boxes_np[index], mode=roi_mode)
            if point_in_polygon(foot, roi_polygon):
                roi_keep[index] = True
        person_keep = person_keep & roi_keep

    pre_keep = person_keep | ball_keep
    query_idx = query_idx[pre_keep]
    labels = labels[pre_keep]
    scores = scores[pre_keep]
    boxes = boxes[pre_keep]
    if query_idx.numel() == 0:
        detections = sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            mask=np.empty((0, height, width), dtype=bool),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int64),
        )
        detections.data["class_name"] = np.empty((0,), dtype=object)
        detections.data["source_shape"] = np.empty((0, 2), dtype=np.int64)
        return detections

    if query_idx.numel() > 1:
        try:
            keep_nms = batched_nms(boxes.float(), scores.float(), labels, nms_threshold)
            query_idx = query_idx[keep_nms]
            labels = labels[keep_nms]
            scores = scores[keep_nms]
            boxes = boxes[keep_nms]
        except Exception:
            pass

    if lowres_mask_nms and query_idx.numel() > 1:
        keep_mask_nms = lowres_mask_nms_keep(
            masks_low[query_idx],
            scores,
            labels,
            threshold=nms_threshold,
        )
        query_idx = query_idx[keep_mask_nms]
        labels = labels[keep_mask_nms]
        scores = scores[keep_mask_nms]
        boxes = boxes[keep_mask_nms]

    selected_masks = masks_low[query_idx].unsqueeze(1)
    masks = torch_f.interpolate(selected_masks, size=(height, width), mode="bilinear", align_corners=False) > 0.0

    class_ids_np = labels.detach().cpu().numpy().astype(np.int64)
    detections = sv.Detections(
        xyxy=boxes.detach().cpu().numpy().astype(np.float32),
        mask=masks.squeeze(1).detach().cpu().numpy(),
        confidence=scores.detach().cpu().numpy().astype(np.float32),
        class_id=class_ids_np,
    )
    detections.data["class_name"] = np.array([class_names[int(cid)] for cid in class_ids_np], dtype=object)
    detections.data["source_shape"] = np.tile(np.array([height, width], dtype=np.int64), (len(detections), 1))
    return detections

class TensorRTRunner:
    def __init__(self, engine_path: str | Path) -> None:
        import tensorrt as trt

        self.trt = trt
        self.device = torch.device("cuda")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)
        self.buffers: dict[str, torch.Tensor] = {}
        self.input_name = ""
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.float32 if dtype.__name__ == "float32" else torch.float16
            self.buffers[name] = torch.empty(shape, device=self.device, dtype=torch_dtype)
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
        if not self.input_name:
            raise RuntimeError(f"TensorRT engine has no input tensor: {engine_path}")
        input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.batch_size = int(input_shape[0])
        self.resolution = int(input_shape[-1])

    def predict_batch(
        self,
        frames_bgr: list[np.ndarray],
        *,
        threshold: float,
        ball_threshold: float,
        **postprocess_kwargs,
    ) -> list:
        real_count = len(frames_bgr)
        if real_count == 0:
            return []
        if real_count > self.batch_size:
            raise ValueError(f"TensorRT batch has {self.batch_size} slots, got {real_count} frames")
        padded_frames = list(frames_bgr)
        if real_count < self.batch_size:
            padded_frames.extend([frames_bgr[-1]] * (self.batch_size - real_count))
        input_tensor = torch.stack(
            [preprocess_frame(frame, device=self.device, resolution=self.resolution) for frame in padded_frames],
            dim=0,
        )
        with torch.cuda.stream(self.stream):
            self.buffers[self.input_name].copy_(input_tensor)
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        self.stream.synchronize()
        detections = []
        for index, frame_bgr in enumerate(frames_bgr):
            detections.append(
                detections_from_raw_tensors(
                    frame_bgr,
                    logits=self.buffers["labels"][index],
                    boxes_cxcywh=self.buffers["dets"][index],
                    masks_low=self.buffers["masks"][index],
                    threshold=threshold,
                    ball_threshold=ball_threshold,
                    **postprocess_kwargs,
                )
            )
        return detections

def trt_predict_batch(
    runner: TensorRTRunner,
    frames_bgr: list[np.ndarray],
    *,
    threshold: float,
    ball_threshold: float,
    **postprocess_kwargs,
) -> list:
    outputs = []
    for start in range(0, len(frames_bgr), runner.batch_size):
        outputs.extend(
            runner.predict_batch(
                frames_bgr[start : start + runner.batch_size],
                threshold=threshold,
                ball_threshold=ball_threshold,
                **postprocess_kwargs,
            )
        )
    return outputs
