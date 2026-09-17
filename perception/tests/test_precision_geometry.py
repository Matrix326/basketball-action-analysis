from __future__ import annotations

from types import SimpleNamespace
from typing import cast
import unittest

import numpy as np

from src.rfdetr_pipeline.observations import ObservationGroup, PoseObservation
from src.rfdetr_pipeline.temporal import refine_pose_bone_lengths
from src.track.trajectory_utils import (
    ground_to_court_pixel,
    prepare_court_canvas,
    repair_isolated_jumps,
)


class PrecisionGeometryTests(unittest.TestCase):
    def test_bone_refinement_stabilizes_length_without_moving_center(self) -> None:
        poses = {}
        quality = {}
        original_centers = {}
        for frame in range(40):
            pose = np.full((17, 3), np.nan, dtype=np.float32)
            length = 0.4 if frame % 2 == 0 else 0.8
            pose[5] = [-length / 2, 0.0, 1.5]
            pose[7] = [length / 2, 0.0, 1.5]
            poses[frame] = {1: pose.tolist()}
            quality[frame] = {
                1: {"predicted_3d": False, "mean_reprojection_error_px": 5.0}
            }
            original_centers[frame] = np.mean(pose[[5, 7]], axis=0)

        stats = refine_pose_bone_lengths(
            poses,
            quality,
            [(5, 7)],
            strength=0.5,
            iterations=6,
            min_reference_samples=20,
        )
        lengths = []
        for frame, tracks in poses.items():
            pose = np.asarray(tracks[1])
            lengths.append(float(np.linalg.norm(pose[5] - pose[7])))
            np.testing.assert_allclose(
                np.mean(pose[[5, 7]], axis=0), original_centers[frame], atol=1e-6
            )

        self.assertEqual(stats["reference_bones"], 1)
        self.assertLess(np.ptp(lengths), 0.02)

    def test_ground_fusion_rejects_clear_outlier(self) -> None:
        observations = {
            "view1": SimpleNamespace(
                ground_position=np.array([1.0, 1.0, 0.0]),
                overlap_ratio=0.0,
                quality=0.9,
            ),
            "view2": SimpleNamespace(
                ground_position=np.array([1.1, 0.9, 0.0]),
                overlap_ratio=0.0,
                quality=0.8,
            ),
            "view3": SimpleNamespace(
                ground_position=np.array([7.0, 7.0, 0.0]),
                overlap_ratio=0.0,
                quality=1.0,
            ),
        }

        point = ObservationGroup(
            cast(dict[str, PoseObservation], observations)
        ).ground_position

        assert point is not None
        np.testing.assert_allclose(point, [1.047, 0.953, 0.0], atol=0.01)

    def test_isolated_jump_is_interpolated(self) -> None:
        points = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (4.0, 4.0), (0.4, 0.0)]
        repaired, indices = repair_isolated_jumps(
            points,
            range(len(points)),
            jump_distance_threshold=3.0,
            speed_ratio_threshold=8.0,
            frame_rate=30.0,
            lookback_frames=15,
        )

        self.assertEqual(indices, [3])
        np.testing.assert_allclose(repaired[3], [0.3, 0.0], atol=1e-6)

    def test_court_mapping_preserves_metric_aspect(self) -> None:
        background = np.zeros((773, 416, 3), dtype=np.uint8)
        canvas, scale, offset_x, offset_y = prepare_court_canvas(
            background, 800, 1400, 15.0, 28.0, 50.0
        )

        self.assertEqual(canvas.shape, (1400, 800, 3))
        self.assertEqual((scale, offset_x, offset_y), (50.0, 25, 0))
        self.assertEqual(
            ground_to_court_pixel(
                7.5,
                14.0,
                court_width_m=15.0,
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
            ),
            (400, 700),
        )


if __name__ == "__main__":
    unittest.main()
