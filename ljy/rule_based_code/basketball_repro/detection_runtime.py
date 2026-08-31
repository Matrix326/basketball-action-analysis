"""Detection, contour and ball-tracking primitives used by the main pipeline."""

from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np

def parse_points(value: str | None, width: int, height: int, *, min_points: int) -> np.ndarray | None:
    if value is None or not value.strip():
        return None

    points: list[tuple[float, float]] = []
    for token in value.replace(";", " ").split():
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid point {token!r}; expected x,y")
        points.append((float(parts[0]), float(parts[1])))

    if len(points) < min_points:
        raise ValueError(f"Expected at least {min_points} points, got {len(points)}")

    is_normalized = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)
    if is_normalized:
        points = [(x * width, y * height) for x, y in points]

    return np.array(points, dtype=np.float32)

def make_roi_mask(shape: tuple[int, int], polygon: np.ndarray | None) -> np.ndarray | None:
    if polygon is None:
        return None
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask

def point_in_polygon(point: tuple[float, float], polygon: np.ndarray | None) -> bool:
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0

def detection_point(xyxy: np.ndarray, *, mode: str) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x = (x1 + x2) / 2.0
    y = y2 if mode == "bottom-center" else (y1 + y2) / 2.0
    return x, y

def mask_foot_point(
    binary: np.ndarray,
    fallback: tuple[float, float],
    *,
    mode: str,
    bounds: tuple[int, int, int, int] | None,
) -> tuple[float, float]:
    if mode != "bottom-center":
        return fallback
    if bounds is None:
        return fallback
    x, y, width, height = bounds
    crop = binary[y : y + height, x : x + width]
    ys, xs = np.nonzero(crop)
    if len(xs) == 0:
        return fallback
    y_max = int(ys.max())
    y_min = int(ys.min())
    band_height = max(4, int(round((y_max - y_min + 1) * 0.035)))
    band = ys >= y_max - band_height
    if not np.any(band):
        return fallback
    return float(x + np.mean(xs[band])), float(y + y_max)

def class_name_at(detections, index: int) -> str:
    names = getattr(detections, "data", {}).get("class_name")
    if names is not None and len(names) > index:
        return str(names[index])
    return str(int(detections.class_id[index]))

def color_for_id(track_id: int) -> tuple[int, int, int]:
    palette = [
        (52, 211, 153),
        (96, 165, 250),
        (251, 191, 36),
        (244, 114, 182),
        (167, 139, 250),
        (248, 113, 113),
        (45, 212, 191),
        (250, 204, 21),
    ]
    return palette[track_id % len(palette)]

def nms_keep_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    keep: list[int] = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        order = class_indices[np.argsort(scores[class_indices])[::-1]]
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break

            rest = order[1:]
            current_box = boxes[current]
            rest_boxes = boxes[rest]
            xx1 = np.maximum(current_box[0], rest_boxes[:, 0])
            yy1 = np.maximum(current_box[1], rest_boxes[:, 1])
            xx2 = np.minimum(current_box[2], rest_boxes[:, 2])
            yy2 = np.minimum(current_box[3], rest_boxes[:, 3])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            area_current = max(0.0, float(current_box[2] - current_box[0])) * max(
                0.0, float(current_box[3] - current_box[1])
            )
            area_rest = np.maximum(0.0, rest_boxes[:, 2] - rest_boxes[:, 0]) * np.maximum(
                0.0, rest_boxes[:, 3] - rest_boxes[:, 1]
            )
            denom = area_current + area_rest - inter
            iou = np.divide(inter, denom, out=np.zeros_like(inter, dtype=np.float32), where=denom > 0)
            order = rest[iou <= threshold]

    return np.array(keep, dtype=np.int64)

def mask_nms_keep_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    rows = len(masks)
    sort_index = scores.argsort()[::-1]
    sorted_masks = masks[sort_index].astype(np.uint8, copy=False)
    sorted_labels = labels[sort_index]

    areas = np.empty((rows,), dtype=np.float32)
    bounds: list[tuple[int, int, int, int] | None] = []
    for index, mask in enumerate(sorted_masks):
        area = int(cv2.countNonZero(mask))
        areas[index] = float(area)
        bounds.append(cv2.boundingRect(mask) if area > 0 else None)

    keep_sorted = np.ones((rows,), dtype=bool)
    for row_index in range(rows):
        if not keep_sorted[row_index] or bounds[row_index] is None or areas[row_index] <= 0:
            continue
        x1, y1, w1, h1 = bounds[row_index]
        ax2 = x1 + w1
        ay2 = y1 + h1
        for other_index in range(row_index + 1, rows):
            if not keep_sorted[other_index] or sorted_labels[row_index] != sorted_labels[other_index]:
                continue
            if bounds[other_index] is None or areas[other_index] <= 0:
                continue
            x2, y2, w2, h2 = bounds[other_index]
            bx2 = x2 + w2
            by2 = y2 + h2
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            intersection = int(
                np.count_nonzero(
                    sorted_masks[row_index, iy1:iy2, ix1:ix2] & sorted_masks[other_index, iy1:iy2, ix1:ix2]
                )
            )
            if intersection <= 0:
                continue
            union = areas[row_index] + areas[other_index] - float(intersection)
            if union > 0 and float(intersection) / union > threshold:
                keep_sorted[other_index] = False

    keep = np.zeros((rows,), dtype=bool)
    keep[sort_index] = keep_sorted
    return keep

def fast_detections_nms(detections, *, threshold: float = 0.5, class_agnostic: bool = False):
    boxes = np.asarray(detections.xyxy, dtype=np.float32)
    if len(boxes) <= 1 or getattr(detections, "confidence", None) is None:
        return detections

    scores = np.asarray(detections.confidence, dtype=np.float32)
    if class_agnostic or getattr(detections, "class_id", None) is None:
        labels = np.zeros((len(boxes),), dtype=np.int64)
    else:
        labels = np.asarray(detections.class_id, dtype=np.int64)

    if getattr(detections, "mask", None) is not None:
        keep_mask = mask_nms_keep_mask(np.asarray(detections.mask), scores, labels, threshold=threshold)
        if bool(keep_mask.all()):
            return detections
        return detections[keep_mask]

    keep = nms_keep_indices(boxes, scores, labels, threshold=threshold)
    if keep.size == len(boxes):
        return detections
    return detections[keep]

def mask_histogram_from_hsv(frame_hsv: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    binary = mask.astype(np.uint8)
    if int(binary.sum()) < 20:
        return None
    hist = cv2.calcHist([frame_hsv], [0, 1], binary, [18, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)

def mask_geometry(mask: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int, int, int] | None, list[np.ndarray]]:
    binary = mask.astype(np.uint8, copy=False)
    mask_pixels = int(cv2.countNonZero(binary))
    if mask_pixels <= 0:
        return binary, 0, None, []

    x, y, width, height = cv2.boundingRect(binary)
    crop = binary[y : y + height, x : x + width]
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        offset = np.array([[[x, y]]], dtype=contours[0].dtype)
        contours = [contour + offset for contour in contours]
    return binary, mask_pixels, (x, y, width, height), contours

def orange_score(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray, mask: np.ndarray | None) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    pad = 3
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame_bgr.shape[1], x2 + pad)
    y2 = min(frame_bgr.shape[0], y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if mask is None:
        object_mask = np.ones(hsv.shape[:2], dtype=np.uint8)
    else:
        object_mask = mask[y1:y2, x1:x2].astype(np.uint8)
        if int(object_mask.sum()) < 8:
            object_mask = np.ones(hsv.shape[:2], dtype=np.uint8)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    orange = ((hue >= 2) & (hue <= 23) & (sat >= 55) & (val >= 45) & (object_mask > 0))
    denom = int((object_mask > 0).sum())
    return float(orange.sum() / denom) if denom else 0.0

@dataclass
class PlayerDetection:
    bbox_xyxy: np.ndarray
    mask: np.ndarray
    contours: list[np.ndarray]
    confidence: float | None
    foot_xy: tuple[float, float]
    center_xy: tuple[float, float]
    histogram: np.ndarray | None
    mask_pixels: int
    roi_ratio: float | None

@dataclass
class BallDetection:
    bbox_xyxy: np.ndarray
    mask: np.ndarray | None
    contours: list[np.ndarray]
    confidence: float | None
    center_xy: tuple[float, float]
    size: float
    orange_score: float

class BallTracker:
    def __init__(
        self,
        *,
        max_distance: float,
        max_jump: float,
        switch_margin: float,
        min_jump_quality: float,
        max_missed: int,
        trail_length: int,
        stationary_switch_frames: int,
    ) -> None:
        self.max_distance = max_distance
        self.max_jump = max_jump
        self.switch_margin = switch_margin
        self.min_jump_quality = min_jump_quality
        self.max_missed = max_missed
        self.stationary_switch_frames = stationary_switch_frames
        self.trail: deque[tuple[float, float]] = deque(maxlen=trail_length)
        self.last_center: tuple[float, float] | None = None
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.last_detection: BallDetection | None = None
        self.missed = 0
        self.stationary_frames = 0

    def quality(self, detection: BallDetection) -> float:
        confidence = float(detection.confidence or 0.0)
        size_score = min(1.0, detection.size / 80.0)
        return confidence + 0.95 * detection.orange_score + 0.25 * size_score

    def update(self, candidates: list[BallDetection]) -> BallDetection | None:
        if not candidates:
            self.missed += 1
            if self.missed > self.max_missed:
                self.last_center = None
                self.velocity = (0.0, 0.0)
                self.last_detection = None
                self.trail.clear()
                self.stationary_frames = 0
            return None

        if self.last_center is None:
            selected = max(candidates, key=self.quality)
            reset_trail = False
        else:
            reset_trail = False
            predicted_center = (
                self.last_center[0] + self.velocity[0],
                self.last_center[1] + self.velocity[1],
            )
            last_speed = float(np.hypot(self.velocity[0], self.velocity[1]))
            jump_gate = min(
                self.max_distance,
                self.max_jump + 0.75 * last_speed + 45.0 * min(self.missed, 3),
            )
            nearby: list[BallDetection] = []
            for det in candidates:
                predicted_distance = float(
                    np.hypot(det.center_xy[0] - predicted_center[0], det.center_xy[1] - predicted_center[1])
                )
                direct_distance = float(
                    np.hypot(det.center_xy[0] - self.last_center[0], det.center_xy[1] - self.last_center[1])
                )
                is_plausible_jump = direct_distance <= self.max_jump or self.quality(det) >= self.min_jump_quality
                if predicted_distance <= jump_gate and direct_distance <= self.max_distance and is_plausible_jump:
                    nearby.append(det)
            if not nearby:
                self.missed += 1
                if self.missed > self.max_missed:
                    self.last_center = None
                    self.velocity = (0.0, 0.0)
                    self.last_detection = None
                    self.trail.clear()
                return None

            def cost(det: BallDetection) -> float:
                predicted_distance = float(
                    np.hypot(det.center_xy[0] - predicted_center[0], det.center_xy[1] - predicted_center[1])
                )
                direct_distance = float(
                    np.hypot(det.center_xy[0] - self.last_center[0], det.center_xy[1] - self.last_center[1])
                )
                return (
                    predicted_distance
                    + 0.35 * direct_distance
                    - 70.0 * float(det.confidence or 0.0)
                    - 170.0 * det.orange_score
                    - 0.30 * det.size
                )

            selected = min(nearby, key=cost)
            if self.stationary_frames >= self.stationary_switch_frames:
                best = max(candidates, key=self.quality)
                best_distance = float(
                    np.hypot(best.center_xy[0] - self.last_center[0], best.center_xy[1] - self.last_center[1])
                )
                if (
                    best is not selected
                    and best.orange_score >= 0.08
                    and best_distance <= self.max_jump
                    and self.quality(best) >= self.quality(selected) + self.switch_margin
                ):
                    selected = best
                    reset_trail = True

        previous_center = self.last_center
        self.last_center = selected.center_xy
        self.last_detection = selected
        self.missed = 0
        if reset_trail:
            self.trail.clear()
        if previous_center is not None:
            movement = float(np.hypot(selected.center_xy[0] - previous_center[0], selected.center_xy[1] - previous_center[1]))
            measured_velocity = (
                selected.center_xy[0] - previous_center[0],
                selected.center_xy[1] - previous_center[1],
            )
            self.velocity = (
                0.65 * self.velocity[0] + 0.35 * measured_velocity[0],
                0.65 * self.velocity[1] + 0.35 * measured_velocity[1],
            )
            self.stationary_frames = self.stationary_frames + 1 if movement < 4.0 else 0
        else:
            self.velocity = (0.0, 0.0)
            self.stationary_frames = 0
        self.trail.append(selected.center_xy)
        return selected

def extract_detections(
    frame_bgr: np.ndarray,
    detections,
    *,
    person_class: str,
    ball_class: str,
    threshold: float,
    ball_threshold: float,
    roi_polygon: np.ndarray | None,
    roi_mask: np.ndarray | None,
    roi_mode: str,
    min_mask_roi_ratio: float,
    min_mask_pixels: int,
    ball_min_size: float,
    ball_max_size: float,
    ball_min_aspect: float,
    ball_max_aspect: float,
    run_nms: bool = True,
) -> tuple[list[PlayerDetection], list[BallDetection]]:
    if run_nms:
        try:
            detections = fast_detections_nms(detections, threshold=0.5, class_agnostic=False)
        except Exception:
            try:
                detections = detections.with_nms(threshold=0.5, class_agnostic=False)
            except Exception:
                pass

    masks = getattr(detections, "mask", None)
    if masks is None:
        raise RuntimeError("RF-DETR-Seg did not return masks; use a segmentation model size.")

    player_detections: list[PlayerDetection] = []
    ball_detections: list[BallDetection] = []
    for index, mask in enumerate(masks):
        class_name = class_name_at(detections, index)
        xyxy = np.asarray(detections.xyxy[index], dtype=np.float32)
        score = float(detections.confidence[index]) if detections.confidence is not None else None
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        width, height = x2 - x1, y2 - y1
        center = detection_point(xyxy, mode="center")

        binary, mask_pixels, mask_bounds, contours = mask_geometry(mask)

        if class_name == ball_class:
            if score is not None and score < ball_threshold:
                continue
            size = max(width, height)
            aspect = width / height if height > 0 else 999.0
            if size < ball_min_size or size > ball_max_size:
                continue
            if aspect < ball_min_aspect or aspect > ball_max_aspect:
                continue
            ball_detections.append(
                BallDetection(
                    bbox_xyxy=xyxy,
                    mask=binary if mask_pixels > 0 else None,
                    contours=contours,
                    confidence=score,
                    center_xy=center,
                    size=float(size),
                    orange_score=orange_score(frame_bgr, xyxy, binary if mask_pixels > 0 else None),
                )
            )
            continue

        if class_name != person_class:
            continue
        if score is not None and score < threshold:
            continue

        if mask_pixels < min_mask_pixels:
            continue

        foot = mask_foot_point(
            binary,
            detection_point(xyxy, mode=roi_mode),
            mode=roi_mode,
            bounds=mask_bounds,
        )
        if not point_in_polygon(foot, roi_polygon):
            continue

        roi_ratio = None
        if roi_mask is not None:
            if mask_bounds is None:
                roi_pixels = 0
            else:
                mx, my, mw, mh = mask_bounds
                roi_pixels = int((binary[my : my + mh, mx : mx + mw] * roi_mask[my : my + mh, mx : mx + mw]).sum())
            roi_ratio = roi_pixels / float(mask_pixels)
            if roi_ratio < min_mask_roi_ratio:
                continue

        if mask_bounds is None:
            histogram = None
        else:
            mx, my, mw, mh = mask_bounds
            hsv_crop = cv2.cvtColor(frame_bgr[my : my + mh, mx : mx + mw], cv2.COLOR_BGR2HSV)
            histogram = mask_histogram_from_hsv(hsv_crop, binary[my : my + mh, mx : mx + mw])

        player_detections.append(
            PlayerDetection(
                bbox_xyxy=xyxy,
                mask=binary,
                contours=contours,
                confidence=score,
                foot_xy=foot,
                center_xy=center,
                histogram=histogram,
                mask_pixels=mask_pixels,
                roi_ratio=roi_ratio,
            )
        )

    return player_detections, ball_detections

class AsyncVideoWriter:
    def __init__(self, writer, *, max_queue: int = 64) -> None:
        self.writer = writer
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(1, max_queue))
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="async-video-writer", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            frame = self.queue.get()
            try:
                if frame is None:
                    return
                self.writer.write(frame)
            except BaseException as exc:
                self.error = exc
                return
            finally:
                self.queue.task_done()

    def write(self, frame: np.ndarray) -> None:
        if self.error is not None:
            raise RuntimeError("Async video writer failed") from self.error
        self.queue.put(frame)

    def release(self) -> None:
        self.queue.put(None)
        self.thread.join()
        self.writer.release()
        if self.error is not None:
            raise RuntimeError("Async video writer failed") from self.error
