"""RF-DETR TensorRT/ONNX backends and detector batching."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from config import Config
from basketball_repro.inference_runtime import (
    BALL_CLASS_ID,
    PERSON_CLASS_ID,
    TensorRTRunner,
    detections_from_raw_tensors,
    preprocess_frame,
    trt_predict_batch,
)

class OnnxRunner:
    """Static-batch RF-DETR ONNX runner with the same post-processing as TRT."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        preferred = (
            ("CUDAExecutionProvider", "CPUExecutionProvider")
            if torch.cuda.is_available()
            else ("CPUExecutionProvider",)
        )
        providers = [name for name in preferred if name in available]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.batch_size = int(model_input.shape[0])
        self.resolution = int(model_input.shape[-1])
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.provider = self.session.get_providers()[0]

    def predict_batch(
        self,
        frames_bgr: list[np.ndarray],
        *,
        threshold: float,
        ball_threshold: float,
        **postprocess_kwargs: Any,
    ) -> list[Any]:
        real_count = len(frames_bgr)
        if real_count == 0:
            return []
        outputs: list[Any] = []
        for start in range(0, real_count, self.batch_size):
            chunk = frames_bgr[start : start + self.batch_size]
            padded = list(chunk)
            padded.extend([chunk[-1]] * (self.batch_size - len(chunk)))
            tensors = [
                preprocess_frame(frame, device=torch.device("cpu"), resolution=self.resolution).numpy()
                for frame in padded
            ]
            raw_values = self.session.run(self.output_names, {self.input_name: np.stack(tensors).astype(np.float32)})
            raw = dict(zip(self.output_names, raw_values))
            for index, frame in enumerate(chunk):
                outputs.append(
                    detections_from_raw_tensors(
                        frame,
                        logits=torch.from_numpy(raw["labels"][index]),
                        boxes_cxcywh=torch.from_numpy(raw["dets"][index]),
                        masks_low=torch.from_numpy(raw["masks"][index]),
                        threshold=threshold,
                        ball_threshold=ball_threshold,
                        **postprocess_kwargs,
                    )
                )
        return outputs

class RfdetrPkgRunner:
    """Fine-tuned RFDETRBase (rfdetr package, PyTorch) inference backend.

    Emits the same supervision Detections contract as the TRT/ONNX backends:
    class ids are remapped (ball -> BALL_CLASS_ID, player -> PERSON_CLASS_ID)
    and each box gets a rectangular pseudo-mask == its bbox, so downstream
    mask-based geometry (foot_xy, roi_ratio, mask features) keeps working.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        batch_size: int = 8,
        resolution: int | None = None,
    ) -> None:
        from rfdetr.variants import RFDETRBase

        self.model = RFDETRBase.from_checkpoint(str(checkpoint_path))
        names = getattr(self.model, "class_names", None)
        if names and list(names[:2]) != ["ball", "player"]:
            print(f"[warn] checkpoint classes {names}; expecting ['ball', 'player']")
        self.name = "rfdetr-pkg-ft"
        self.batch_size = int(batch_size)
        self.resolution = int(
            resolution or getattr(self.model, "resolution", 560) or 560
        )

    def predict_batch(
        self,
        frames_bgr: list[np.ndarray],
        *,
        threshold: float,
        ball_threshold: float,
        **postprocess_kwargs: Any,
    ) -> list[Any]:
        import cv2

        imgs_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        preds = self.model.predict(
            imgs_rgb,
            threshold=min(threshold, ball_threshold),
            include_source_image=False,
        )
        if not isinstance(preds, list):
            preds = [preds]
        out = []
        for frame, dets in zip(frames_bgr, preds):
            out.append(
                _ft_detections_to_pipeline(
                    frame,
                    dets,
                    threshold=threshold,
                    ball_threshold=ball_threshold,
                    **postprocess_kwargs,
                )
            )
        return out


def _ft_detections_to_pipeline(
    frame_bgr: np.ndarray,
    dets: Any,
    *,
    threshold: float,
    ball_threshold: float,
    ball_min_size: float = 5.0,
    ball_max_size: float = 100.0,
    ball_min_aspect: float = 0.35,
    ball_max_aspect: float = 2.8,
    **_ignored: Any,
):
    """Convert rfdetr-package Detections (0=ball, 1=player) to the pipeline
    sv.Detections format with rectangular pseudo-masks."""
    import supervision as sv

    height, width = frame_bgr.shape[:2]

    def empty() -> sv.Detections:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            mask=np.empty((0, height, width), dtype=bool),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int64),
            data={
                "class_name": np.empty((0,), dtype=object),
                "source_shape": np.empty((0, 2), dtype=np.int64),
            },
        )

    if dets is None or len(dets) == 0:
        return empty()

    keep: list[tuple[int, int, str]] = []
    for index in range(len(dets)):
        class_id = int(dets.class_id[index])
        score = float(dets.confidence[index])
        x1, y1, x2, y2 = [float(v) for v in dets.xyxy[index]]
        w, h = x2 - x1, y2 - y1
        if class_id == 0:  # ball
            if score < ball_threshold:
                continue
            size = max(w, h)
            aspect = w / h if h > 0 else 999.0
            if size < ball_min_size or size > ball_max_size:
                continue
            if aspect < ball_min_aspect or aspect > ball_max_aspect:
                continue
            keep.append((index, BALL_CLASS_ID, "sports ball"))
        elif class_id == 1:  # player
            if score < threshold:
                continue
            keep.append((index, PERSON_CLASS_ID, "person"))
    if not keep:
        return empty()

    indices = [k[0] for k in keep]
    xyxy = dets.xyxy[indices].astype(np.float32)
    confidence = dets.confidence[indices].astype(np.float32)
    class_id = np.array([k[1] for k in keep], dtype=np.int64)
    class_name = np.array([k[2] for k in keep], dtype=object)
    masks = np.zeros((len(keep), height, width), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if x2i > x1i and y2i > y1i:
            masks[i, y1i:y2i, x1i:x2i] = True
    return sv.Detections(
        xyxy=xyxy,
        mask=masks,
        confidence=confidence,
        class_id=class_id,
        data={
            "class_name": class_name,
            "source_shape": np.tile(
                np.array([height, width], dtype=np.int64), (len(keep), 1)
            ),
        },
    )



def _ft_detections_to_pipeline(
    frame_bgr: np.ndarray,
    dets: Any,
    *,
    threshold: float,
    ball_threshold: float,
    ball_min_size: float = 5.0,
    ball_max_size: float = 100.0,
    ball_min_aspect: float = 0.35,
    ball_max_aspect: float = 2.8,
    **_ignored: Any,
):
    """Convert rfdetr-package Detections (0=ball, 1=player) to the pipeline
    sv.Detections format with rectangular pseudo-masks."""
    import supervision as sv

    height, width = frame_bgr.shape[:2]

    def empty() -> sv.Detections:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            mask=np.empty((0, height, width), dtype=bool),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int64),
            data={
                "class_name": np.empty((0,), dtype=object),
                "source_shape": np.empty((0, 2), dtype=np.int64),
            },
        )

    if dets is None or len(dets) == 0:
        return empty()

    keep: list[tuple[int, int, str]] = []
    for index in range(len(dets)):
        class_id = int(dets.class_id[index])
        score = float(dets.confidence[index])
        x1, y1, x2, y2 = [float(v) for v in dets.xyxy[index]]
        w, h = x2 - x1, y2 - y1
        if class_id == 0:  # ball
            if score < ball_threshold:
                continue
            size = max(w, h)
            aspect = w / h if h > 0 else 999.0
            if size < ball_min_size or size > ball_max_size:
                continue
            if aspect < ball_min_aspect or aspect > ball_max_aspect:
                continue
            keep.append((index, BALL_CLASS_ID, "sports ball"))
        elif class_id == 1:  # player
            if score < threshold:
                continue
            keep.append((index, PERSON_CLASS_ID, "person"))
    if not keep:
        return empty()

    indices = [k[0] for k in keep]
    xyxy = dets.xyxy[indices].astype(np.float32)
    confidence = dets.confidence[indices].astype(np.float32)
    class_id = np.array([k[1] for k in keep], dtype=np.int64)
    class_name = np.array([k[2] for k in keep], dtype=object)
    masks = np.zeros((len(keep), height, width), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if x2i > x1i and y2i > y1i:
            masks[i, y1i:y2i, x1i:x2i] = True
    return sv.Detections(
        xyxy=xyxy,
        mask=masks,
        confidence=confidence,
        class_id=class_id,
        data={
            "class_name": class_name,
            "source_shape": np.tile(
                np.array([height, width], dtype=np.int64), (len(keep), 1)
            ),
        },
    )



class HybridRunner:
    """人检测用 2XL (真 mask: foot_xy 精确 + ReID mask 特征完整),
    球检测用微调模型 (球检测/追踪质量更好: 跳变 2.86% vs 2XL 3.52%)。

    合并两份 sv.Detections: 人的框/mask 来自 2XL, 球的框来自微调模型。
    """

    def __init__(
        self,
        config: Config,
        *,
        person_checkpoint: str | None = None,
        ball_checkpoint: str | None = None,
    ) -> None:
        # 2XL: ONNX 优先, TRT 回退 (复用现有后端)
        onnx_path = config.get("rfdetr.onnx_path")
        engine_path = config.get("rfdetr.engine_path")
        self.person_runner: Optional[OnnxRunner] = None
        if onnx_path and Path(onnx_path).exists():
            self.person_runner = OnnxRunner(onnx_path)
        if self.person_runner is None and engine_path and Path(engine_path).exists():
            from basketball_repro.inference_runtime import TensorRTRunner
            self.person_runner = TensorRTRunner(engine_path)  # type: ignore[assignment]
        if self.person_runner is None:
            raise RuntimeError("HybridRunner: no 2XL backend (onnx/tensorrt) for person detection")
        # 微调模型: 球检测
        self.ball_runner = RfdetrPkgRunner(
            ball_checkpoint or config.get("rfdetr.checkpoint_path"),
            batch_size=int(config.get("rfdetr.batch_size", 8)),
        )
        self.name = f"hybrid-2xl+{self.ball_runner.name}"
        self.batch_size = 8

    def predict_batch(
        self,
        frames_bgr: list[np.ndarray],
        *,
        threshold: float,
        ball_threshold: float,
        **postprocess_kwargs: Any,
    ) -> list[Any]:
        import supervision as sv

        real_count = len(frames_bgr)
        if real_count == 0:
            return []
        # 人: 2XL (整帧推理, 与 OnnxRunner 相同批处理)
        person_outs = self.person_runner.predict_batch(
            frames_bgr,
            threshold=threshold,
            ball_threshold=ball_threshold,
            **postprocess_kwargs,
        )
        # 球: 微调模型
        ball_outs = self.ball_runner.predict_batch(
            frames_bgr,
            threshold=threshold,
            ball_threshold=ball_threshold,
            **postprocess_kwargs,
        )
        merged = []
        for pdet, bdet in zip(person_outs, ball_outs):
            pm = (pdet.class_id == PERSON_CLASS_ID) if len(pdet) else np.zeros(0, dtype=bool)
            bm = (bdet.class_id == BALL_CLASS_ID) if len(bdet) else np.zeros(0, dtype=bool)
            keep_p = pdet[pm]
            keep_b = bdet[bm]
            if len(keep_p) == 0:
                merged.append(keep_b)
                continue
            if len(keep_b) == 0:
                merged.append(keep_p)
                continue
            out = sv.Detections(
                xyxy=np.concatenate([keep_p.xyxy, keep_b.xyxy]),
                mask=np.concatenate([keep_p.mask, keep_b.mask]),
                confidence=np.concatenate([keep_p.confidence, keep_b.confidence]),
                class_id=np.concatenate([keep_p.class_id, keep_b.class_id]),
                data={
                    "class_name": np.concatenate([keep_p.data["class_name"], keep_b.data["class_name"]]),
                    "source_shape": np.concatenate([keep_p.data["source_shape"], keep_b.data["source_shape"]]),
                },
            )
            merged.append(out)
        return merged



class RFDetrSegmenter:
    """Shared TensorRT/ONNX RF-DETR-Seg 2XL inference facade."""

    def __init__(self, config: Config) -> None:
        engine_path = config.get("rfdetr.engine_path")
        onnx_path = config.get("rfdetr.onnx_path")
        backend = str(config.get("rfdetr.backend", "auto")).lower()
        self.runner: Optional[TensorRTRunner] = None
        self.onnx_runner: Optional[OnnxRunner] = None
        # Hybrid: 人=2XL(真mask) 球=微调模型
        if backend == "hybrid":
            self.runner = HybridRunner(config)  # type: ignore[assignment]
            self.name = self.runner.name
        can_use_engine = bool(engine_path and Path(engine_path).exists())
        if self.runner is None and backend in {"auto", "tensorrt"} and can_use_engine:
            try:
                self.runner = TensorRTRunner(engine_path)
                engine_precision = str(config.get("rfdetr.engine_precision", "unknown")).lower()
                self.name = f"tensorrt-{engine_precision}"
            except (ImportError, RuntimeError) as exc:
                if backend == "tensorrt":
                    raise
                print(f"[warn] TensorRT unavailable ({exc}); trying ONNX Runtime")

        if self.runner is None and backend in {"auto", "onnx"} and onnx_path and Path(onnx_path).exists():
            try:
                self.onnx_runner = OnnxRunner(onnx_path)
                self.name = f"onnxruntime-{self.onnx_runner.provider}"
            except (ImportError, RuntimeError) as exc:
                if backend == "onnx":
                    raise
                print(f"[warn] ONNX Runtime unavailable ({exc})")

        if self.runner is None and self.onnx_runner is None:
            raise RuntimeError("Neither the bundled TensorRT engine nor ONNX model could be loaded")

        self.threshold = float(config.get("rfdetr.person_threshold", 0.35))
        self.ball_threshold = float(config.get("rfdetr.ball_threshold", 0.16))
        self.ball_min_size = float(config.get("rfdetr.ball_min_size", 5.0))
        self.ball_max_size = float(config.get("rfdetr.ball_max_size", 100.0))
        self.ball_min_aspect = float(config.get("rfdetr.ball_min_aspect", 0.35))
        self.ball_max_aspect = float(config.get("rfdetr.ball_max_aspect", 2.8))
        if self.runner is not None:
            self.backend_batch_size = int(self.runner.batch_size)
        elif self.onnx_runner is not None:
            self.backend_batch_size = int(self.onnx_runner.batch_size)
        else:
            raise AssertionError("RF-DETR backend was not initialized")
        self.inference_calls = 0
        self.inference_slots = 0

    def predict(self, frames_bgr: list[np.ndarray], roi_polygons: list[Optional[np.ndarray]]) -> list[Any]:
        common = dict(
            threshold=self.threshold,
            ball_threshold=self.ball_threshold,
            roi_mode="bottom-center",
            ball_min_size=self.ball_min_size,
            ball_max_size=self.ball_max_size,
            ball_min_aspect=self.ball_min_aspect,
            ball_max_aspect=self.ball_max_aspect,
            lowres_mask_nms=True,
        )
        outputs: list[Any] = [None] * len(frames_bgr)
        # ROI is part of the fast pre-filter, so frames with different camera
        # ROIs are grouped separately. Same-ROI views retain batched 2XL/TRT
        # inference instead of paying one forward pass per camera.
        groups: dict[bytes, list[int]] = {}
        for index, polygon in enumerate(roi_polygons):
            key = b"none" if polygon is None else np.asarray(polygon, dtype=np.float32).tobytes()
            groups.setdefault(key, []).append(index)
        for indices in groups.values():
            batch = [frames_bgr[index] for index in indices]
            kwargs = {**common, "roi_polygon": roi_polygons[indices[0]]}
            if self.runner is not None:
                calls = math.ceil(len(batch) / self.runner.batch_size)
                self.inference_calls += calls
                self.inference_slots += calls * self.runner.batch_size
                if isinstance(self.runner, (RfdetrPkgRunner, HybridRunner)):
                    batch_outputs = self.runner.predict_batch(batch, **kwargs)
                else:
                    batch_outputs = trt_predict_batch(self.runner, batch, **kwargs)
            elif self.onnx_runner is not None:
                calls = math.ceil(len(batch) / self.onnx_runner.batch_size)
                self.inference_calls += calls
                self.inference_slots += calls * self.onnx_runner.batch_size
                batch_outputs = self.onnx_runner.predict_batch(batch, **kwargs)
            else:
                raise AssertionError("RF-DETR backend was not initialized")
            for index, result in zip(indices, batch_outputs):
                outputs[index] = result
        return outputs

def _temporal_batch_length(detector_batch_capacity: int, view_count: int) -> int:
    """Number of synchronized timestamps that fit in one detector batch."""
    if detector_batch_capacity <= 0:
        raise ValueError("detector_batch_capacity must be positive")
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    return max(1, detector_batch_capacity // view_count)

