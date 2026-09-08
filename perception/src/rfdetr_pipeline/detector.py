"""RF-DETR TensorRT/ONNX backends and detector batching."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from config import Config
from basketball_repro.inference_runtime import (
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

class RFDetrSegmenter:
    """Shared TensorRT/ONNX RF-DETR-Seg 2XL inference facade."""

    def __init__(self, config: Config) -> None:
        engine_path = config.get("rfdetr.engine_path")
        onnx_path = config.get("rfdetr.onnx_path")
        backend = str(config.get("rfdetr.backend", "auto")).lower()
        self.runner: Optional[TensorRTRunner] = None
        self.onnx_runner: Optional[OnnxRunner] = None
        can_use_engine = bool(engine_path and Path(engine_path).exists())
        if backend in {"auto", "tensorrt"} and can_use_engine:
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

