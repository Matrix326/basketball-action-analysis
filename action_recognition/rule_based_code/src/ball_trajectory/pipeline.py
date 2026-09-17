"""Orchestrator: raw ball observations -> continuous smoothed 3D trajectory.

Stage order:
    load -> outlier rejection -> bounces -> ballistic segments ->
    flight/dribble split -> kinematic gap fill -> RTS smoothing ->
    state classification -> interface JSON assembly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional as Optional, cast

import numpy as np

from .ballistics import GRAVITY, fit_ballistic
from .fill import fill_gaps
from .io import (
    BallObservations,
    TrajectoryInput,
    build_input,
    load_poses_json,
    save_interface,
)
from .segmentation import (
    BallisticSegment,
    classify_ballistic_segments,
    detect_ballistic_segments,
    detect_bounces,
)
from .smoother import rts_smooth
from .states import (
    STATE_DRIBBLE,
    STATE_FLIGHT,
    STATE_GROUND,
    STATE_HELD,
    STATE_UNKNOWN,
    classify_states,
    parse_hand_centers,
)

DEFAULTS: dict[str, Any] = {
    "fps": 30,
    "gravity_m_s2": GRAVITY,
    "ball_radius_m": 0.12,
    "max_speed_m_s": 30.0,
    "max_height_m": 10.0,
    "window_seconds": 0.5,
    "residual_threshold_m": 0.10,
    "accel_z_range_m_s2": [-13.0, -6.0],
    "min_window_observations": 8,
    "gap_bridge_frames": 10,
    "min_fit_observations": 6,
    "flight_min_apex_m": 2.0,
    "flight_min_horizontal_m": 2.0,
    "dribble_max_apex_m": 1.4,
    "dribble_max_player_distance_m": 1.5,
    "max_fill_gap_frames": 24,
    "max_extrapolate_frames": 12,
    "flank_frames_back": 15,
    "bounce_restitution": 0.8,
    "use_drag_fit": False,
    "measurement_noise_m": 0.03,
    "process_noise_flight_m_s2": 1.5,
    "process_noise_other_m_s2": 6.0,
    "bounce_q_multiplier": 10.0,
    "hold_reach_m": 0.45,
    "hold_min_frames": 6,
    "hold_max_speed_m_s": 1.5,
    "ground_z_m": 0.15,
    "static_speed_m_s": 0.05,
    "player_distance_for_static_m": 1.5,
    "margin_frames": 60,
    "poses_json": "",
    "output_path": "",
}


class BallTrajectoryPostProcessor:
    """End-to-end offline ball-trajectory post-processor."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def _cfg(self, key: str, default: Any = None) -> Any:
        if self.config is None:
            return DEFAULTS.get(key, default)
        if hasattr(self.config, "get"):
            try:
                return self.config.get(
                    f"ball_trajectory.{key}", DEFAULTS.get(key, default)
                )
            except Exception:
                return DEFAULTS.get(key, default)
        section = (
            self.config.get("ball_trajectory", {})
            if isinstance(self.config, dict)
            else {}
        )
        return section.get(key, DEFAULTS.get(key, default))

    # ------------------------------------------------------------------ stage

    def preprocess_observations(
        self, obs: BallObservations, fps: float
    ) -> BallObservations:
        """Outlier rejection: isolated spikes (two consecutive huge speeds),
        physically impossible heights, view-combination jump runs, and
        locked-on-wrong-target runs (few views, still, far from the last
        valid point)."""
        positions = obs.positions
        frames = obs.frame_indices
        mask = np.zeros(len(positions), dtype=bool)
        if len(positions) < 3:
            return BallObservations(
                frame_indices=frames,
                positions=positions,
                views=obs.views,
                outlier_mask=mask,
                measured_positions=positions.copy(),
            )

        max_speed = float(self._cfg("max_speed_m_s", 30.0))
        max_height = float(self._cfg("max_height_m", 10.0))
        delta_t = np.diff(frames).astype(np.float64) / fps
        speeds = np.zeros(len(positions))
        speeds[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.maximum(
            delta_t, 1e-6
        )

        # View-combination switches leave a RUN of 15-25 m/s points; a real
        # basketball never exceeds ~12 m/s.
        flight_max_speed = float(self._cfg("flight_max_speed_m_s", 14.0))
        for i in range(1, len(positions) - 1):
            if speeds[i] > flight_max_speed and (
                speeds[i - 1] > flight_max_speed or speeds[i + 1] > flight_max_speed
            ):
                mask[i] = True

        last_valid = -1
        for i in range(len(positions)):
            speed_before = speeds[i] if i > 0 else 0.0
            speed_after = speeds[i + 1] if i + 1 < len(positions) else 0.0
            if mask[i]:
                continue
            if speed_before > max_speed and speed_after > max_speed:
                mask[i] = True
            if not mask[i] and last_valid >= 0:
                gap_s = max(1e-6, (frames[i] - frames[last_valid]) / fps)
                displacement = float(
                    np.linalg.norm(positions[i] - positions[last_valid])
                )
                if displacement > 2.0 and displacement / gap_s > 13.0:
                    mask[i] = True
            # spike: jumps away from the a->c segment
            if 0 < i < len(positions) - 1 and not (mask[i - 1] or mask[i + 1]):
                a, b, c = positions[i - 1], positions[i], positions[i + 1]
                disp_before = float(np.linalg.norm(b - a))
                disp_after = float(np.linalg.norm(c - b))
                if max(disp_before, disp_after) > 0.6:
                    ab, ac = b - a, c - a
                    t = float(
                        np.clip(np.dot(ab, ac) / max(np.dot(ac, ac), 1e-9), 0.0, 1.0)
                    )
                    distance = float(np.linalg.norm(ab - t * ac))
                    if distance > 0.4:
                        mask[i] = True
            if positions[i, 2] > max_height or positions[i, 2] < -0.15:
                mask[i] = True
            if not mask[i]:
                last_valid = i

        # Locked-on-wrong-target: few views + still + far from the last valid
        # point (triangulation latched onto a wrong target and sits there).
        last_valid_geo = -1
        for i in range(len(positions)):
            if mask[i]:
                continue
            if obs.views[i] <= 2 and speeds[i] < 5.0 and last_valid_geo >= 0:
                disp = float(np.linalg.norm(positions[i] - positions[last_valid_geo]))
                if disp > 1.5:
                    mask[i] = True
                    continue
            last_valid_geo = i

        # Outlier frames keep their raw position in `positions` (fill/RTS
        # decide what to do with them), but downstream consumers must not
        # treat an outlier position as a measurement: NaN it out so the
        # finite checks in fill/RTS/state machine skip it.
        measured = positions.copy()
        measured[mask] = np.nan
        cleaned = BallObservations(
            frame_indices=frames,
            positions=positions,
            views=obs.views,
            outlier_mask=mask,
            measured_positions=measured,
        )
        return cleaned

    # ------------------------------------------------------------------ core

    def process_data(self, data: dict) -> dict:
        fps = float(self._cfg("fps") or 30.0)
        ball_radius = float(self._cfg("ball_radius_m"))
        input_data = build_input(data, fps=fps)
        fps = input_data.fps

        obs = self.preprocess_observations(input_data.observations, fps)

        # ---- bounces, segments, split ----
        bounces = detect_bounces(
            obs,
            fps,
            ball_radius=ball_radius,
            min_impact_speed=float(self._cfg("bounce_min_impact_speed_m_s", 1.0)),
        )
        segments = detect_ballistic_segments(
            obs,
            fps,
            window_seconds=float(self._cfg("window_seconds")),
            residual_threshold=float(self._cfg("residual_threshold_m")),
            accel_z_range=cast(
                tuple[float, float],
                tuple(float(value) for value in self._cfg("accel_z_range_m_s2")),
            ),
            min_window_observations=int(self._cfg("min_window_observations")),
            gap_bridge_frames=int(self._cfg("gap_bridge_frames")),
            min_fit_observations=int(self._cfg("min_fit_observations")),
            use_drag_fit=bool(self._cfg("use_drag_fit")),
            bounces=bounces,
        )
        # Semantic z-peak scan supplements the ballistic segments for runs
        # the window stage rejects (noisy arcs); the segments it adds start
        # at the first z>1.5 frame, which is later than the true release.
        min_z = float(self._cfg("fallback_launch_z_m", 1.5))
        peak_z = float(self._cfg("fallback_peak_z_m", 2.2))
        runs_z: list[tuple[int, int]] = []
        start = None
        prev = None
        gap = 0
        max_gap = int(self._cfg("fallback_launch_gap_frames", 5))
        for i, f in enumerate(obs.frame_indices):
            if obs.positions[i, 2] > min_z:
                if start is None:
                    start = f
                gap = 0
                prev = f
            else:
                if start is not None:
                    gap += 1
                    if gap > max_gap:
                        if prev is not None and prev - start >= 5:
                            runs_z.append((start, prev))
                        start, prev, gap = None, None, 0
        if start is not None and prev is not None and prev - start >= 5:
            runs_z.append((start, prev))
        from .segmentation import BallisticSegment

        existing_ranges = {(int(s.start), int(s.end)) for s in segments}
        for rs, re_ in runs_z:
            sel = (
                (obs.frame_indices >= rs)
                & (obs.frame_indices <= re_)
                & (~obs.outlier_mask)
            )
            zs = obs.positions[sel, 2]
            if len(zs) >= 5 and float(np.max(zs)) >= peak_z:
                covered = sum(
                    max(0, min(re_, es) - max(rs, ss) + 1) for ss, es in existing_ranges
                )
                if covered >= 0.5 * (re_ - rs + 1):
                    continue
                pf = obs.frame_indices[sel]
                pp = obs.positions[sel]
                try:
                    p0, v0, rms = fit_ballistic(pf, pp, float(pf[0]), fps)
                except ValueError:
                    continue
                segments.append(
                    BallisticSegment(
                        start=int(pf[0]),
                        end=int(pf[-1]),
                        p0=p0.astype(np.float64),
                        v0=v0.astype(np.float64),
                        rms=float(rms),
                        observed_frames=int(len(pf)),
                    )
                )
        hand_centers = parse_hand_centers(input_data.skeleton)
        classify_ballistic_segments(
            segments,
            bounces,
            hand_centers,
            fps,
            flight_min_apex_m=float(self._cfg("flight_min_apex_m")),
            flight_min_horizontal_m=float(self._cfg("flight_min_horizontal_m")),
            dribble_max_apex_m=float(self._cfg("dribble_max_apex_m")),
            dribble_max_player_distance_m=float(
                self._cfg("dribble_max_player_distance_m")
            ),
            residual_threshold=float(self._cfg("residual_threshold_m")),
        )

        # ---- gap filling ----
        filled = fill_gaps(
            obs,
            segments,
            bounces,
            fps,
            ball_radius=ball_radius,
            max_fill_gap_frames=int(self._cfg("max_fill_gap_frames")),
            max_extrapolate_frames=int(self._cfg("max_extrapolate_frames")),
            flank_frames_back=int(self._cfg("flank_frames_back")),
            residual_threshold=float(self._cfg("residual_threshold_m")),
            restitution=float(self._cfg("bounce_restitution")),
            max_height=float(self._cfg("max_height_m")),
            max_speed_m_s=float(self._cfg("max_speed_m_s")),
            use_drag_fit=bool(self._cfg("use_drag_fit")),
        )

        # ---- extend grid with margin and map everything onto it ----
        margin = int(self._cfg("margin_frames"))
        grid_start = max(0, int(filled.frames[0]) - margin) if len(filled.frames) else 0
        grid_end = int(filled.frames[-1]) + margin if len(filled.frames) else 0
        grid = np.arange(grid_start, grid_end + 1, dtype=np.int64)
        grid_to_filled = {int(frame): i for i, frame in enumerate(filled.frames)}

        positions_grid = np.full((len(grid), 3), np.nan)
        observed_grid = np.zeros(len(grid), dtype=bool)
        outlier_grid = np.zeros(len(grid), dtype=bool)
        filled_grid = np.zeros(len(grid), dtype=bool)
        filled_source_grid = np.full(len(grid), None, dtype=object)
        views_grid = np.zeros(len(grid), dtype=np.int64)
        for i, frame in enumerate(grid):
            idx = grid_to_filled.get(int(frame))
            if idx is None:
                continue
            positions_grid[i] = filled.positions[idx]
            # Any outlier frame keeps a raw (bad) position even when the
            # filler marked it filled — NaN it out so RTS/state machine
            # never see it (the smoothed output bridges the gap).
            if filled.outlier[idx]:
                positions_grid[i] = np.nan
            observed_grid[i] = filled.observed[idx]
            outlier_grid[i] = filled.outlier[idx]
            filled_grid[i] = filled.filled[idx]
            filled_source_grid[i] = filled.filled_source[idx]
            if observed_grid[i] or outlier_grid[i]:
                obs_idx = int(np.argmin(np.abs(obs.frame_indices - frame)))
                if int(obs.frame_indices[obs_idx]) == int(frame):
                    views_grid[i] = int(obs.views[obs_idx])

        # ---- RTS smoothing (measurements = observed only) ----
        seg_kinds = np.full(len(grid), "", dtype=object)
        seg_confidence = np.zeros(len(grid))
        for seg in segments:
            sel = (grid >= seg.start) & (grid <= seg.end)
            seg_kinds[sel] = seg.kind
            seg_confidence[sel] = seg.confidence

        bounce_frames = {int(b.frame_index) for b in bounces}
        smoothed_pos, smoothed_vel = rts_smooth(
            grid,
            positions_grid,
            observed_grid,
            views_grid,
            seg_kinds,
            bounce_frames,
            fps,
            measurement_noise_m=float(self._cfg("measurement_noise_m")),
            process_noise_flight_m_s2=float(self._cfg("process_noise_flight_m_s2")),
            process_noise_other_m_s2=float(self._cfg("process_noise_other_m_s2")),
            bounce_q_multiplier=float(self._cfg("bounce_q_multiplier")),
        )

        # ---- state classification ----
        states, state_conf = classify_states(
            grid,
            smoothed_pos,
            smoothed_vel,
            seg_kinds,
            seg_confidence,
            bounce_frames,
            hand_centers,
            fps,
            hold_reach_m=float(self._cfg("hold_reach_m")),
            hold_min_frames=int(self._cfg("hold_min_frames")),
            hold_max_speed_m_s=float(self._cfg("hold_max_speed_m_s")),
            ground_z_m=float(self._cfg("ground_z_m")),
            static_speed_m_s=float(self._cfg("static_speed_m_s")),
            static_min_frames=int(float(self._cfg("static_min_frames_s", 2.0)) * fps),
            player_distance_for_static_m=float(
                self._cfg("player_distance_for_static_m")
            ),
        )

        # ---- null out long unsupported runs ----
        max_fill_gap = int(self._cfg("max_fill_gap_frames"))
        run_start = None
        nulled = np.zeros(len(grid), dtype=bool)
        for i in range(len(grid) + 1):
            unobserved = i < len(grid) and not observed_grid[i]
            if unobserved and run_start is None:
                run_start = i
            elif not unobserved and run_start is not None:
                if i - run_start > max_fill_gap:
                    nulled[run_start:i] = True
                run_start = None

        # ---- final median smoothing (window 13: removes detector jump
        # excursions while preserving genuine fast flight arcs) ----
        smooth_window = int(self._cfg("final_median_window", 13))
        positions_final = _median_smooth_positions(
            grid, smoothed_pos, window=smooth_window
        )
        # ---- assemble interface ----
        return self._assemble(
            grid,
            positions_grid,
            observed_grid,
            outlier_grid,
            filled_grid,
            filled_source_grid,
            views_grid,
            positions_final,
            smoothed_vel,
            states,
            state_conf,
            nulled,
            segments,
            bounces,
            input_data,
            fps,
            ball_radius,
            obs,
        )

    # ------------------------------------------------------------- assemble

    def _assemble(
        self,
        grid: np.ndarray,
        positions_grid: np.ndarray,
        observed_grid: np.ndarray,
        outlier_grid: np.ndarray,
        filled_grid: np.ndarray,
        filled_source_grid: np.ndarray,
        views_grid: np.ndarray,
        smoothed_pos: np.ndarray,
        smoothed_vel: np.ndarray,
        states: np.ndarray,
        state_conf: np.ndarray,
        nulled: np.ndarray,
        segments: list[BallisticSegment],
        bounces: list[Any],
        input_data: TrajectoryInput,
        fps: float,
        ball_radius: float,
        obs: BallObservations,
    ) -> dict:
        frames_out: dict[str, dict[str, Any]] = {}
        obs_by_frame = {int(frame): i for i, frame in enumerate(obs.frame_indices)}
        _null_position = None

        for i, frame in enumerate(grid):
            record: dict[str, Any] = {
                "frame_index": int(frame),
                "position": None,
                "velocity": None,
                "state": str(states[i]),
                "state_confidence": round(float(state_conf[i]), 4),
                "observed": bool(observed_grid[i]),
                "measured": None,
                "outlier": bool(outlier_grid[i]),
                "filled": False,
                "filled_source": None,
                "views": int(views_grid[i]) if views_grid[i] > 0 else None,
            }

            raw_idx = obs_by_frame.get(int(frame))
            if raw_idx is not None and (observed_grid[i] or outlier_grid[i]):
                record["measured"] = [
                    float(value) for value in obs.measured_positions[raw_idx]
                ]

            if nulled[i]:
                record["state"] = STATE_UNKNOWN
                record["state_confidence"] = 0.3
                record["filled"] = False
                record["filled_source"] = None
                frames_out[str(frame)] = record
                continue

            support = np.isfinite(smoothed_pos[i]).all()
            if observed_grid[i]:
                if support:
                    record["position"] = [float(value) for value in smoothed_pos[i]]
                    record["velocity"] = [float(value) for value in smoothed_vel[i]]
                record["filled"] = False
                record["filled_source"] = None
            elif filled_grid[i]:
                if np.isfinite(positions_grid[i]).all():
                    record["position"] = [float(value) for value in positions_grid[i]]
                    record["velocity"] = (
                        [float(value) for value in smoothed_vel[i]] if support else None
                    )
                record["filled"] = True
                record["filled_source"] = str(filled_source_grid[i])
            elif support:
                record["position"] = [float(value) for value in smoothed_pos[i]]
                record["velocity"] = [float(value) for value in smoothed_vel[i]]
                record["filled"] = True
                record["filled_source"] = "rts"

            frames_out[str(frame)] = record

        segments_out = [
            {
                "type": seg.kind,
                "start_frame": seg.start,
                "end_frame": seg.end,
                "confidence": round(float(seg.confidence), 4),
                "params": {
                    "p0": [float(value) for value in seg.p0],
                    "v0": [float(value) for value in seg.v0],
                    "g": [0.0, 0.0, -float(self._cfg("gravity_m_s2"))],
                    "fitted_frames": int(seg.observed_frames),
                    "rms_residual_m": round(float(seg.rms), 4),
                },
                "filled_gap_frames": int(seg.filled_gap_frames),
            }
            for seg in segments
        ]
        bounces_out = [
            {
                "frame_index": int(b.frame_index),
                "position": [float(value) for value in b.position],
                "impact_speed_m_s": round(float(b.impact_speed_m_s), 3),
            }
            for b in bounces
        ]

        # ---- stats ----
        _activity = (grid >= int(grid[0]) + int(self._cfg("margin_frames", 60))) & (
            grid <= int(grid[-1]) - int(self._cfg("margin_frames", 60))
        )
        non_null = np.array(
            [
                not nulled[i]
                and (
                    (observed_grid[i] and np.isfinite(smoothed_pos[i]).all())
                    or (filled_grid[i] and np.isfinite(positions_grid[i]).all())
                    or (
                        not observed_grid[i]
                        and not filled_grid[i]
                        and np.isfinite(smoothed_pos[i]).all()
                    )
                )
                for i in range(len(grid))
            ],
            dtype=bool,
        )

        unobserved = ~observed_grid & ~outlier_grid
        gap_count = 0
        gap_frames_filled = 0
        longest_gap = 0
        run_start = None
        for i in range(len(grid) + 1):
            is_gap = i < len(grid) and unobserved[i] and not nulled[i]
            if is_gap and run_start is None:
                run_start = i
            elif not is_gap and run_start is not None:
                length = i - run_start
                gap_count += 1
                gap_frames_filled += int(np.sum(filled_grid[run_start:i]))
                longest_gap = max(longest_gap, length)
                run_start = None

        state_counts: dict[str, int] = {}
        for state in (
            STATE_FLIGHT,
            STATE_HELD,
            STATE_DRIBBLE,
            STATE_GROUND,
            STATE_UNKNOWN,
        ):
            state_counts[state] = int(np.sum(states == state))

        stats = {
            "first_frame": int(grid[0]),
            "last_frame": int(grid[-1]),
            "total_frames": int(len(grid)),
            "observed_frames": int(np.sum(observed_grid)),
            "outlier_count": int(np.sum(outlier_grid)),
            "coverage": round(float(np.mean(non_null)), 4) if len(grid) else 0.0,
            "gap_count": gap_count,
            "gap_frames_filled": gap_frames_filled,
            "longest_gap_frames": longest_gap,
            "filled_by": {
                "ballistic": int(np.sum(filled_source_grid == "ballistic")),
                "rts": int(np.sum(filled_source_grid == "rts")),
            },
            "segments": {
                "flight": sum(1 for seg in segments if seg.kind == STATE_FLIGHT),
                "dribble": sum(1 for seg in segments if seg.kind == STATE_DRIBBLE),
            },
            "state_counts": state_counts,
        }

        return {
            "schema_version": "ball-trajectory/v1",
            "source": {
                "poses_3d_json": str(self._cfg("poses_json", "")),
                "fps": float(fps),
                "gravity_m_s2": float(self._cfg("gravity_m_s2")),
                "ball_radius_m": ball_radius,
                "coordinate_system": "calibrated world metres, z up, origin at court floor",
            },
            "frames": frames_out,
            "segments": segments_out,
            "bounces": bounces_out,
            "stats": stats,
        }

    # ------------------------------------------------------------------ cli

    def process(
        self, poses_json_path: str | Path, output_path: str | Path | None = None
    ) -> dict:
        poses_json_path = Path(poses_json_path)
        if output_path is None:
            output_path = Path(self._cfg("output_path", "")) or (
                poses_json_path.parent / "ball_trajectory.json"
            )
        data = load_poses_json(poses_json_path)
        result = self.process_data(data)
        save_interface(result, output_path)
        print(f"[ok] ball trajectory interface -> {Path(output_path)}")
        return result


def _central_velocity(
    frames: np.ndarray, positions: np.ndarray, fps: float
) -> np.ndarray:
    """3-point central-difference velocity (m/s) from a smoothed trajectory."""
    n = len(positions)
    velocity = np.full_like(positions, np.nan)
    if n < 2:
        return velocity
    for i in range(n):
        if not np.isfinite(positions[i]).all():
            continue
        left = max(0, i - 1)
        right = min(n - 1, i + 1)
        if left == right:
            continue
        dt = float(frames[right] - frames[left]) / fps
        velocity[i] = (positions[right] - positions[left]) / dt
    return velocity


def _median_smooth_positions(
    frames: np.ndarray, positions: np.ndarray, window: int = 5
) -> np.ndarray:
    """Replace each position with the median of the neighbouring frames
    (only over frames that have a finite position)."""
    n = len(positions)
    if n < window:
        return positions
    result = positions.copy()
    half = window // 2
    for i in range(n):
        if not np.isfinite(positions[i]).all():
            continue
        left = max(0, i - half)
        right = min(n, i + half + 1)
        segment = positions[left:right]
        valid = np.array([np.isfinite(p).all() for p in segment])
        if np.count_nonzero(valid) < 3:
            continue
        result[i] = np.median(segment[valid], axis=0)
    return result


def _interpolate_switches(
    frames: np.ndarray,
    positions: np.ndarray,
    observed: np.ndarray,
    max_jump: float = 2.0,
) -> np.ndarray:
    """Observations that leap more than max_jump from the recent median are
    detector-switch residuals (side-to-side jumps in the video). Replace them
    with linear interpolation between the observed neighbours."""
    n = len(positions)
    result = positions.copy()
    for i in range(n):
        if not observed[i] or not np.isfinite(positions[i]).all():
            continue
        window = [
            j
            for j in range(max(0, i - 7), i)
            if observed[j] and np.isfinite(positions[j]).all()
        ]
        if len(window) < 3:
            continue
        median_pos = np.median(positions[window], axis=0)
        if float(np.linalg.norm(positions[i] - median_pos)) <= max_jump:
            continue
        prev_idx = window[-1]
        next_idx = None
        for j in range(i + 1, n):
            if observed[j]:
                next_idx = j
                break
        if next_idx is None:
            continue
        prev_pos, next_pos = positions[prev_idx], positions[next_idx]
        if not np.isfinite(next_pos).all():
            continue
        span = max(1e-9, float(frames[next_idx] - frames[prev_idx]))
        t = float(frames[i] - frames[prev_idx]) / span
        result[i] = prev_pos + t * (next_pos - prev_pos)
    return result
