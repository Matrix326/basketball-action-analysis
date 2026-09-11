"""Kinematic gap filling: ballistic bridges through occlusion gaps.

Strategy per gap (consecutive unobserved frames between observed a and b):

1. Gap fully inside a ballistic segment -> reuse the segment's fitted arc.
2. Else try a double-sided gravity fit over both flanks (flight through the
   gap with no bounce in it).
3. Else try a bounce bridge: descending left arc meets the floor at z = r,
   ascending right arc continues with restitution e.
4. Else leave NaN — the RTS smoother's gravity process model coasts through
   (marked "rts" in the output), and gaps longer than ``max_fill_gap_frames``
   stay NaN in the final interface (state=unknown).

Frames beyond the last observed frame are extrapolated from a covering
ballistic segment, capped at ``max_extrapolate_frames`` while z stays above
the floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .ballistics import fit_ballistic, simulate_ballistic
from .io import BallObservations
from .segmentation import BallisticSegment, Bounce


@dataclass

class FilledTrajectory:
    frames: np.ndarray                 # int64 (M,)
    positions: np.ndarray              # (M, 3), NaN where unsupported
    observed: np.ndarray               # bool
    outlier: np.ndarray                # bool — measurement present but rejected
    filled: np.ndarray                 # bool
    filled_source: np.ndarray          # object array: None | "ballistic" | "rts"


def _quadratic_roots_z(p0z: float, v0z: float, t0: float, target: float, fps: float) -> list[float]:
    """Roots tau of z(t0+tau) = target under gravity: 0.5*a*tau^2 + v0z*tau + (p0z-target) = 0."""
    a = -0.5 * 9.81
    b = v0z
    c = p0z - target
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return []
    sqrt_disc = np.sqrt(disc)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    return sorted(roots)


def _velocity_at(frames: np.ndarray, p0: np.ndarray, v0: np.ndarray, t0: float, fps: float) -> np.ndarray:
    """Velocity vectors at the given frames under the ballistic model -> (M, 3)."""
    tau = (frames - t0) / fps
    gravity = np.zeros(3)
    gravity[2] = -9.81
    return v0[None, :] + gravity[None, :] * tau[:, None]


def _try_bounce_bridge(
    left_frames: np.ndarray,
    left_pos: np.ndarray,
    right_frames: np.ndarray,
    right_pos: np.ndarray,
    gap_start: int,
    gap_end: int,
    fps: float,
    ball_radius: float,
    restitution: float,
    impact_time_tolerance_frames: float = 3.0,
    velocity_tolerance_m_s: float = 1.5,
) -> Optional[tuple[np.ndarray, str]]:
    """Fit descending-left + ascending-right arcs meeting at z = ball_radius.

    Returns (positions for frames [gap_start, gap_end], "ballistic") on success.
    """
    if len(left_frames) < 3 or len(right_frames) < 3:
        return None
    try:
        p0_l, v0_l, _ = fit_ballistic(left_frames, left_pos, float(left_frames[0]), fps)
        p0_r, v0_r, _ = fit_ballistic(right_frames, right_pos, float(right_frames[0]), fps)
    except ValueError:
        return None

    # Downward root of the left arc (tau after the apex) at z = r.
    roots_l = _quadratic_roots_z(float(p0_l[2]), float(v0_l[2]), 0.0, ball_radius, fps)
    if not roots_l:
        return None
    tau_apex_l = float(v0_l[2]) / 9.81
    tau_impact_l = None
    for root in roots_l:
        if root >= tau_apex_l - 1e-9:
            tau_impact_l = root
            break
    if tau_impact_l is None:
        return None

    # Upward root of the right arc at z = r.
    roots_r = _quadratic_roots_z(float(p0_r[2]), float(v0_r[2]), 0.0, ball_radius, fps)
    if not roots_r:
        return None
    tau_apex_r = float(v0_r[2]) / 9.81
    tau_impact_r = None
    for root in roots_r:
        if root <= tau_apex_r + 1e-9:
            tau_impact_r = root
            break
    if tau_impact_r is None:
        return None

    impact_l = float(left_frames[0]) + tau_impact_l * fps
    impact_r = float(right_frames[0]) + tau_impact_r * fps
    if abs(impact_l - impact_r) > impact_time_tolerance_frames:
        return None

    vz_l = float(v0_l[2] - 9.81 * tau_impact_l)
    vz_r = float(v0_r[2] - 9.81 * tau_impact_r)
    if abs(vz_r - (-restitution * vz_l)) > velocity_tolerance_m_s:
        return None

    impact_frame = int(round((impact_l + impact_r) / 2.0))
    if not (gap_start <= impact_frame <= gap_end):
        return None

    gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
    left_part = gap_frames[gap_frames < impact_frame]
    right_part = gap_frames[gap_frames >= impact_frame]
    result = np.full((len(gap_frames), 3), np.nan)
    if len(left_part):
        result[gap_frames < impact_frame] = simulate_ballistic(
            left_part, p0_l, v0_l, float(left_frames[0]), fps
        )
    if len(right_part):
        result[gap_frames >= impact_frame] = simulate_ballistic(
            right_part, p0_r, v0_r, float(right_frames[0]), fps
        )
    return result, "ballistic"


def _try_double_sided(
    left_frames: np.ndarray,
    left_pos: np.ndarray,
    right_frames: np.ndarray,
    right_pos: np.ndarray,
    gap_start: int,
    gap_end: int,
    fps: float,
    residual_threshold: float,
    max_height: float,
    max_speed_m_s: float,
) -> Optional[tuple[np.ndarray, str]]:
    """Single parabolic fit through both flanks of a gap."""
    flank_frames = np.concatenate([left_frames, right_frames])
    flank_pos = np.concatenate([left_pos, right_pos])
    if len(flank_frames) < 6:
        return None
    try:
        p0, v0, rms = fit_ballistic(flank_frames, flank_pos, float(flank_frames[0]), fps)
    except ValueError:
        return None
    if rms > 1.5 * residual_threshold:
        return None
    gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
    sim = simulate_ballistic(gap_frames, p0, v0, float(flank_frames[0]), fps)
    if np.any(sim[:, 2] < -0.05) or np.any(sim[:, 2] > max_height):
        return None
    speeds = _velocity_at(gap_frames, p0, v0, float(flank_frames[0]), fps)
    if np.any(np.linalg.norm(speeds, axis=1) > max_speed_m_s):
        return None
    return sim, "ballistic"


def fill_gaps(
    obs: BallObservations,
    segments: list[BallisticSegment],
    bounces: list[Bounce],
    fps: float,
    ball_radius: float = 0.12,
    max_fill_gap_frames: int = 24,
    max_extrapolate_frames: int = 12,
    flank_frames_back: int = 15,
    residual_threshold: float = 0.10,
    restitution: float = 0.8,
    max_height: float = 10.0,
    max_speed_m_s: float = 30.0,
    use_drag_fit: bool = False,
) -> FilledTrajectory:
    """Fill observation gaps with ballistic arcs; everything else stays NaN."""
    frames = obs.frame_indices
    positions = obs.positions

    if len(frames) == 0:
        return FilledTrajectory(
            frames=np.array([], dtype=np.int64),
            positions=np.zeros((0, 3)),
            observed=np.zeros(0, dtype=bool),
            outlier=obs.outlier_mask,
            filled=np.zeros(0, dtype=bool),
            filled_source=np.array([], dtype=object),
        )

    first, last = int(frames[0]), int(frames[-1])
    grid = np.arange(first, last + 1, dtype=np.int64)
    n = len(grid)
    observed = np.zeros(n, dtype=bool)
    outlier = np.zeros(n, dtype=bool)
    positions_full = np.full((n, 3), np.nan)
    obs_index = 0
    for i, frame in enumerate(grid):
        if obs_index < len(frames) and int(frames[obs_index]) == int(frame):
            if not obs.outlier_mask[obs_index]:
                observed[i] = True
                positions_full[i] = positions[obs_index]
            else:
                outlier[i] = True
            obs_index += 1

    filled = np.zeros(n, dtype=bool)
    filled_source = np.full(n, None, dtype=object)

    bounce_frames = {b.frame_index for b in bounces}

    # ---- interior gaps ----
    gap_start_idx = None
    for i in range(n):
        if not observed[i] and gap_start_idx is None:
            gap_start_idx = i
        elif observed[i] and gap_start_idx is not None:
            gap_end_idx = i - 1
            gap_start = int(grid[gap_start_idx])
            gap_end = int(grid[gap_end_idx])
            length = gap_end - gap_start + 1
            if length <= max_fill_gap_frames:
                _fill_one_gap(
                    grid, positions_full, filled, filled_source,
                    gap_start_idx, gap_end_idx, segments, bounce_frames,
                    frames, positions, fps, ball_radius, restitution,
                    flank_frames_back, residual_threshold, max_height,
                    max_speed_m_s, use_drag_fit,
                )
            gap_start_idx = None
    if gap_start_idx is not None:
        gap_end_idx = n - 1
        gap_start = int(grid[gap_start_idx])
        gap_end = int(grid[gap_end_idx])
        if gap_end - gap_start + 1 <= max_fill_gap_frames:
            _fill_one_gap(
                grid, positions_full, filled, filled_source,
                gap_start_idx, gap_end_idx, segments, bounce_frames,
                frames, positions, fps, ball_radius, restitution,
                flank_frames_back, residual_threshold, max_height,
                max_speed_m_s, use_drag_fit,
            )

    # ---- trailing / leading extrapolation from a covering segment ----
    _extrapolate_edges(
        grid, positions_full, filled, filled_source, segments, frames,
        fps, max_extrapolate_frames, ball_radius,
    )

    return FilledTrajectory(
        frames=grid,
        positions=positions_full,
        observed=observed,
        outlier=outlier,
        filled=filled,
        filled_source=filled_source,
    )


def _fill_one_gap(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    gap_start_idx: int,
    gap_end_idx: int,
    segments: list[BallisticSegment],
    bounce_frames: set[int],
    obs_frames: np.ndarray,
    obs_pos: np.ndarray,
    fps: float,
    ball_radius: float,
    restitution: float,
    flank_frames_back: int,
    residual_threshold: float,
    max_height: float,
    max_speed_m_s: float,
    use_drag_fit: bool,
) -> None:
    gap_start = int(grid[gap_start_idx])
    gap_end = int(grid[gap_end_idx])

    # Case 1: gap fully inside a ballistic segment, no bounce in it.
    for seg in segments:
        if seg.start <= gap_start and gap_end <= seg.end:
            if any(bf >= gap_start and bf <= gap_end for bf in bounce_frames):
                pass
            if not any(gap_start <= bf <= gap_end for bf in bounce_frames):
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                if (np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height)):
                    speeds = _velocity_at(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                    if np.all(np.linalg.norm(speeds, axis=1) <= max_speed_m_s):
                        positions_full[gap_start_idx:gap_end_idx + 1] = sim
                        filled[gap_start_idx:gap_end_idx + 1] = True
                        filled_source[gap_start_idx:gap_end_idx + 1] = "ballistic"
                        seg.filled_gap_frames += gap_end - gap_start + 1
                        return

    # Case 2/3: double-sided fit or bounce bridge using flanking observations.
    left_sel = obs_frames <= grid[gap_start_idx - 1] if gap_start_idx > 0 else None
    # Observed frames before the gap (bounded window).
    before = obs_frames[obs_frames <= grid[gap_start_idx - 1]] if gap_start_idx > 0 else obs_frames[:0]
    before = before[-flank_frames_back:]
    after = obs_frames[obs_frames >= grid[gap_end_idx + 1]] if gap_end_idx + 1 < len(grid) else obs_frames[len(obs_frames):]
    after = after[:flank_frames_back]

    before_pos = obs_pos[np.isin(obs_frames, before)]
    after_pos = obs_pos[np.isin(obs_frames, after)]

    best: Optional[tuple[np.ndarray, str]] = None
    if len(before) >= 3 and len(after) >= 3:
        best = _try_bounce_bridge(
            before, before_pos, after, after_pos,
            gap_start, gap_end, fps, ball_radius, restitution,
        )
        if best is None:
            best = _try_double_sided(
                before, before_pos, after, after_pos,
                gap_start, gap_end, fps, residual_threshold, max_height, max_speed_m_s,
            )
    elif len(before) >= 6:
        # Only a left flank (gap at the very end) — covered by extrapolation.
        pass
    elif len(after) >= 6:
        # Only a right flank (gap at the very start) — leading edge handled by
        # _extrapolate_edges via segments; try a one-sided fit as well.
        try:
            p0, v0, rms = fit_ballistic(after, after_pos, float(after[0]), fps)
            if rms <= residual_threshold:
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, p0, v0, float(after[0]), fps)
                if np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height):
                    best = (sim, "ballistic")
        except ValueError:
            pass

    if best is not None:
        sim, source = best
        positions_full[gap_start_idx:gap_end_idx + 1] = sim
        filled[gap_start_idx:gap_end_idx + 1] = True
        filled_source[gap_start_idx:gap_end_idx + 1] = source


def _extrapolate_edges(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    segments: list[BallisticSegment],
    obs_frames: np.ndarray,
    fps: float,
    max_extrapolate_frames: int,
    ball_radius: float,
) -> None:
    if len(segments) == 0 or len(grid) == 0:
        return
    last_obs = int(obs_frames[-1])
    first_obs = int(obs_frames[0])

    for seg in segments:
        # Trailing: segment contains the last observed frame.
        if seg.start <= last_obs <= seg.end:
            tail_start = last_obs + 1
            if tail_start <= grid[-1]:
                tail = np.arange(tail_start, min(int(grid[-1]), tail_start + max_extrapolate_frames) + 1, dtype=np.float64)
                sim = simulate_ballistic(tail, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for i, frame in enumerate(tail.astype(int)):
                    if not keep[i]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[i]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"
        # Leading: segment contains the first observed frame.
        if seg.start <= first_obs <= seg.end:
            head_end = first_obs - 1
            if head_end >= grid[0]:
                head = np.arange(max(int(grid[0]), head_end - max_extrapolate_frames + 1), head_end + 1, dtype=np.float64)
                sim = simulate_ballistic(head, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for j, frame in enumerate(head.astype(int)):
                    if not keep[j]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[j]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"

def _quadratic_roots_z(p0z: float, v0z: float, t0: float, target: float, fps: float) -> list[float]:
    """Roots tau of z(t0+tau) = target under gravity: 0.5*a*tau^2 + v0z*tau + (p0z-target) = 0."""
    a = -0.5 * 9.81
    b = v0z
    c = p0z - target
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return []
    sqrt_disc = np.sqrt(disc)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    return sorted(roots)


def _velocity_at(frames: np.ndarray, p0: np.ndarray, v0: np.ndarray, t0: float, fps: float) -> np.ndarray:
    """Velocity vectors at the given frames under the ballistic model -> (M, 3)."""
    tau = (frames - t0) / fps
    gravity = np.zeros(3)
    gravity[2] = -9.81
    return v0[None, :] + gravity[None, :] * tau[:, None]


def _try_bounce_bridge(
    left_frames: np.ndarray,
    left_pos: np.ndarray,
    right_frames: np.ndarray,
    right_pos: np.ndarray,
    gap_start: int,
    gap_end: int,
    fps: float,
    ball_radius: float,
    restitution: float,
    impact_time_tolerance_frames: float = 3.0,
    velocity_tolerance_m_s: float = 1.5,
) -> Optional[tuple[np.ndarray, str]]:
    """Fit descending-left + ascending-right arcs meeting at z = ball_radius.

    Returns (positions for frames [gap_start, gap_end], "ballistic") on success.
    """
    if len(left_frames) < 3 or len(right_frames) < 3:
        return None
    try:
        p0_l, v0_l, _ = fit_ballistic(left_frames, left_pos, float(left_frames[0]), fps)
        p0_r, v0_r, _ = fit_ballistic(right_frames, right_pos, float(right_frames[0]), fps)
    except ValueError:
        return None

    # Downward root of the left arc (tau after the apex) at z = r.
    roots_l = _quadratic_roots_z(float(p0_l[2]), float(v0_l[2]), 0.0, ball_radius, fps)
    if not roots_l:
        return None
    tau_apex_l = float(v0_l[2]) / 9.81
    tau_impact_l = None
    for root in roots_l:
        if root >= tau_apex_l - 1e-9:
            tau_impact_l = root
            break
    if tau_impact_l is None:
        return None

    # Upward root of the right arc at z = r.
    roots_r = _quadratic_roots_z(float(p0_r[2]), float(v0_r[2]), 0.0, ball_radius, fps)
    if not roots_r:
        return None
    tau_apex_r = float(v0_r[2]) / 9.81
    tau_impact_r = None
    for root in roots_r:
        if root <= tau_apex_r + 1e-9:
            tau_impact_r = root
            break
    if tau_impact_r is None:
        return None

    impact_l = float(left_frames[0]) + tau_impact_l * fps
    impact_r = float(right_frames[0]) + tau_impact_r * fps
    if abs(impact_l - impact_r) > impact_time_tolerance_frames:
        return None

    vz_l = float(v0_l[2] - 9.81 * tau_impact_l)
    vz_r = float(v0_r[2] - 9.81 * tau_impact_r)
    if abs(vz_r - (-restitution * vz_l)) > velocity_tolerance_m_s:
        return None

    impact_frame = int(round((impact_l + impact_r) / 2.0))
    if not (gap_start <= impact_frame <= gap_end):
        return None

    gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
    left_part = gap_frames[gap_frames < impact_frame]
    right_part = gap_frames[gap_frames >= impact_frame]
    result = np.full((len(gap_frames), 3), np.nan)
    if len(left_part):
        result[gap_frames < impact_frame] = simulate_ballistic(
            left_part, p0_l, v0_l, float(left_frames[0]), fps
        )
    if len(right_part):
        result[gap_frames >= impact_frame] = simulate_ballistic(
            right_part, p0_r, v0_r, float(right_frames[0]), fps
        )
    return result, "ballistic"


def _try_double_sided(
    left_frames: np.ndarray,
    left_pos: np.ndarray,
    right_frames: np.ndarray,
    right_pos: np.ndarray,
    gap_start: int,
    gap_end: int,
    fps: float,
    residual_threshold: float,
    max_height: float,
    max_speed_m_s: float,
) -> Optional[tuple[np.ndarray, str]]:
    """Single parabolic fit through both flanks of a gap."""
    flank_frames = np.concatenate([left_frames, right_frames])
    flank_pos = np.concatenate([left_pos, right_pos])
    if len(flank_frames) < 6:
        return None
    try:
        p0, v0, rms = fit_ballistic(flank_frames, flank_pos, float(flank_frames[0]), fps)
    except ValueError:
        return None
    if rms > 1.5 * residual_threshold:
        return None
    gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
    sim = simulate_ballistic(gap_frames, p0, v0, float(flank_frames[0]), fps)
    if np.any(sim[:, 2] < -0.05) or np.any(sim[:, 2] > max_height):
        return None
    speeds = _velocity_at(gap_frames, p0, v0, float(flank_frames[0]), fps)
    if np.any(np.linalg.norm(speeds, axis=1) > max_speed_m_s):
        return None
    return sim, "ballistic"


def fill_gaps(
    obs: BallObservations,
    segments: list[BallisticSegment],
    bounces: list[Bounce],
    fps: float,
    ball_radius: float = 0.12,
    max_fill_gap_frames: int = 24,
    max_extrapolate_frames: int = 12,
    flank_frames_back: int = 15,
    residual_threshold: float = 0.10,
    restitution: float = 0.8,
    max_height: float = 10.0,
    max_speed_m_s: float = 30.0,
    use_drag_fit: bool = False,
) -> FilledTrajectory:
    """Fill observation gaps with ballistic arcs; everything else stays NaN."""
    frames = obs.frame_indices
    positions = obs.positions

    if len(frames) == 0:
        return FilledTrajectory(
            frames=np.array([], dtype=np.int64),
            positions=np.zeros((0, 3)),
            observed=np.zeros(0, dtype=bool),
            outlier=obs.outlier_mask,
            filled=np.zeros(0, dtype=bool),
            filled_source=np.array([], dtype=object),
        )

    first, last = int(frames[0]), int(frames[-1])
    grid = np.arange(first, last + 1, dtype=np.int64)
    n = len(grid)
    observed = np.zeros(n, dtype=bool)
    outlier = np.zeros(n, dtype=bool)
    positions_full = np.full((n, 3), np.nan)
    obs_index = 0
    for i, frame in enumerate(grid):
        if obs_index < len(frames) and int(frames[obs_index]) == int(frame):
            if not obs.outlier_mask[obs_index]:
                observed[i] = True
                positions_full[i] = positions[obs_index]
            else:
                outlier[i] = True
            obs_index += 1

    filled = np.zeros(n, dtype=bool)
    filled_source = np.full(n, None, dtype=object)

    bounce_frames = {b.frame_index for b in bounces}

    # ---- interior gaps ----
    gap_start_idx = None
    for i in range(n):
        if not observed[i] and gap_start_idx is None:
            gap_start_idx = i
        elif observed[i] and gap_start_idx is not None:
            gap_end_idx = i - 1
            gap_start = int(grid[gap_start_idx])
            gap_end = int(grid[gap_end_idx])
            length = gap_end - gap_start + 1
            if length <= max_fill_gap_frames:
                _fill_one_gap(
                    grid, positions_full, filled, filled_source,
                    gap_start_idx, gap_end_idx, segments, bounce_frames,
                    frames, positions, fps, ball_radius, restitution,
                    flank_frames_back, residual_threshold, max_height,
                    max_speed_m_s, use_drag_fit,
                )
            gap_start_idx = None
    if gap_start_idx is not None:
        gap_end_idx = n - 1
        gap_start = int(grid[gap_start_idx])
        gap_end = int(grid[gap_end_idx])
        if gap_end - gap_start + 1 <= max_fill_gap_frames:
            _fill_one_gap(
                grid, positions_full, filled, filled_source,
                gap_start_idx, gap_end_idx, segments, bounce_frames,
                frames, positions, fps, ball_radius, restitution,
                flank_frames_back, residual_threshold, max_height,
                max_speed_m_s, use_drag_fit,
            )

    # ---- trailing / leading extrapolation from a covering segment ----
    _extrapolate_edges(
        grid, positions_full, filled, filled_source, segments, frames,
        fps, max_extrapolate_frames, ball_radius,
    )

    return FilledTrajectory(
        frames=grid,
        positions=positions_full,
        observed=observed,
        outlier=outlier,
        filled=filled,
        filled_source=filled_source,
    )


def _fill_one_gap(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    gap_start_idx: int,
    gap_end_idx: int,
    segments: list[BallisticSegment],
    bounce_frames: set[int],
    obs_frames: np.ndarray,
    obs_pos: np.ndarray,
    fps: float,
    ball_radius: float,
    restitution: float,
    flank_frames_back: int,
    residual_threshold: float,
    max_height: float,
    max_speed_m_s: float,
    use_drag_fit: bool,
) -> None:
    gap_start = int(grid[gap_start_idx])
    gap_end = int(grid[gap_end_idx])

    # Case 1: gap fully inside a ballistic segment, no bounce in it.
    for seg in segments:
        if seg.start <= gap_start and gap_end <= seg.end:
            if any(bf >= gap_start and bf <= gap_end for bf in bounce_frames):
                pass
            if not any(gap_start <= bf <= gap_end for bf in bounce_frames):
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                if (np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height)):
                    speeds = _velocity_at(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                    if np.all(np.linalg.norm(speeds, axis=1) <= max_speed_m_s):
                        positions_full[gap_start_idx:gap_end_idx + 1] = sim
                        filled[gap_start_idx:gap_end_idx + 1] = True
                        filled_source[gap_start_idx:gap_end_idx + 1] = "ballistic"
                        seg.filled_gap_frames += gap_end - gap_start + 1
                        return

    # Case 2/3: double-sided fit or bounce bridge using flanking observations.
    left_sel = obs_frames <= grid[gap_start_idx - 1] if gap_start_idx > 0 else None
    # Observed frames before the gap (bounded window).
    before = obs_frames[obs_frames <= grid[gap_start_idx - 1]] if gap_start_idx > 0 else obs_frames[:0]
    before = before[-flank_frames_back:]
    after = obs_frames[obs_frames >= grid[gap_end_idx + 1]] if gap_end_idx + 1 < len(grid) else obs_frames[len(obs_frames):]
    after = after[:flank_frames_back]

    before_pos = obs_pos[np.isin(obs_frames, before)]
    after_pos = obs_pos[np.isin(obs_frames, after)]

    best: Optional[tuple[np.ndarray, str]] = None
    if len(before) >= 3 and len(after) >= 3:
        best = _try_bounce_bridge(
            before, before_pos, after, after_pos,
            gap_start, gap_end, fps, ball_radius, restitution,
        )
        if best is None:
            best = _try_double_sided(
                before, before_pos, after, after_pos,
                gap_start, gap_end, fps, residual_threshold, max_height, max_speed_m_s,
            )
    elif len(before) >= 6:
        # Only a left flank (gap at the very end) — covered by extrapolation.
        pass
    elif len(after) >= 6:
        # Only a right flank (gap at the very start) — leading edge handled by
        # _extrapolate_edges via segments; try a one-sided fit as well.
        try:
            p0, v0, rms = fit_ballistic(after, after_pos, float(after[0]), fps)
            if rms <= residual_threshold:
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, p0, v0, float(after[0]), fps)
                if np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height):
                    best = (sim, "ballistic")
        except ValueError:
            pass

    if best is not None:
        sim, source = best
        positions_full[gap_start_idx:gap_end_idx + 1] = sim
        filled[gap_start_idx:gap_end_idx + 1] = True
        filled_source[gap_start_idx:gap_end_idx + 1] = source


def _extrapolate_edges(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    segments: list[BallisticSegment],
    obs_frames: np.ndarray,
    fps: float,
    max_extrapolate_frames: int,
    ball_radius: float,
) -> None:
    if len(segments) == 0 or len(grid) == 0:
        return
    last_obs = int(obs_frames[-1])
    first_obs = int(obs_frames[0])

    for seg in segments:
        # Trailing: segment contains the last observed frame.
        if seg.start <= last_obs <= seg.end:
            tail_start = last_obs + 1
            if tail_start <= grid[-1]:
                tail = np.arange(tail_start, min(int(grid[-1]), tail_start + max_extrapolate_frames) + 1, dtype=np.float64)
                sim = simulate_ballistic(tail, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for i, frame in enumerate(tail.astype(int)):
                    if not keep[i]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[i]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"
        # Leading: segment contains the first observed frame.
        if seg.start <= first_obs <= seg.end:
            head_end = first_obs - 1
            if head_end >= grid[0]:
                head = np.arange(max(int(grid[0]), head_end - max_extrapolate_frames + 1), head_end + 1, dtype=np.float64)
                sim = simulate_ballistic(head, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for j, frame in enumerate(head.astype(int)):
                    if not keep[j]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[j]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"

def _velocity_at(frames: np.ndarray, p0: np.ndarray, v0: np.ndarray, t0: float, fps: float) -> np.ndarray:
    """Velocity vectors at the given frames under the ballistic model -> (M, 3)."""
    tau = (frames - t0) / fps
    gravity = np.zeros(3)
    gravity[2] = -9.81
    return v0[None, :] + gravity[None, :] * tau[:, None]


def _try_bounce_bridge(
    left_frames: np.ndarray,
    left_pos: np.ndarray,
    right_frames: np.ndarray,
    right_pos: np.ndarray,
    gap_start: int,
    gap_end: int,
    fps: float,
    ball_radius: float,
    restitution: float,
    impact_time_tolerance_frames: float = 3.0,
    velocity_tolerance_m_s: float = 1.5,
) -> Optional[tuple[np.ndarray, str]]:
    """Fit descending-left + ascending-right arcs meeting at z = ball_radius.

    Returns (positions for frames [gap_start, gap_end], "ballistic") on success.
    """
    if len(left_frames) < 3 or len(right_frames) < 3:
        return None
    try:
        p0_l, v0_l, _ = fit_ballistic(left_frames, left_pos, float(left_frames[0]), fps)
        p0_r, v0_r, _ = fit_ballistic(right_frames, right_pos, float(right_frames[0]), fps)
    except ValueError:
        return None

    # Downward root of the left arc (tau after the apex) at z = r.
    roots_l = _quadratic_roots_z(float(p0_l[2]), float(v0_l[2]), 0.0, ball_radius, fps)
    if not roots_l:
        return None
    tau_apex_l = float(v0_l[2]) / 9.81
    tau_impact_l = None
    for root in roots_l:
        if root >= tau_apex_l - 1e-9:
            tau_impact_l = root
            break
    if tau_impact_l is None:
        return None

    # Upward root of the right arc at z = r.
    roots_r = _quadratic_roots_z(float(p0_r[2]), float(v0_r[2]), 0.0, ball_radius, fps)
    if not roots_r:
        return None
    tau_apex_r = float(v0_r[2]) / 9.81
    tau_impact_r = None
    for root in roots_r:
        if root <= tau_apex_r + 1e-9:
            tau_impact_r = root
            break
    if tau_impact_r is None:
        return None

    impact_l = float(left_frames[0]) + tau_impact_l * fps
    impact_r = float(right_frames[0]) + tau_impact_r * fps
    if abs(impact_l - impact_r) > impact_time_tolerance_frames:
        return None

    vz_l = float(v0_l[2] - 9.81 * tau_impact_l)
    vz_r = float(v0_r[2] - 9.81 * tau_impact_r)
    if abs(vz_r - (-restitution * vz_l)) > velocity_tolerance_m_s:
        return None

    impact_frame = int(round((impact_l + impact_r) / 2.0))
    if not (gap_start <= impact_frame <= gap_end):
        return None

    gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
    left_part = gap_frames[gap_frames < impact_frame]
    right_part = gap_frames[gap_frames >= impact_frame]
    result = np.full((len(gap_frames), 3), np.nan)
    if len(left_part):
        result[gap_frames < impact_frame] = simulate_ballistic(
            left_part, p0_l, v0_l, float(left_frames[0]), fps
        )
    if len(right_part):
        result[gap_frames >= impact_frame] = simulate_ballistic(
            right_part, p0_r, v0_r, float(right_frames[0]), fps
        )
    return result, "ballistic"


def _try_double_sided(
    left_frames: np.ndarray,
    left_pos: np.ndarray,
    right_frames: np.ndarray,
    right_pos: np.ndarray,
    gap_start: int,
    gap_end: int,
    fps: float,
    residual_threshold: float,
    max_height: float,
    max_speed_m_s: float,
) -> Optional[tuple[np.ndarray, str]]:
    """Single parabolic fit through both flanks of a gap."""
    flank_frames = np.concatenate([left_frames, right_frames])
    flank_pos = np.concatenate([left_pos, right_pos])
    if len(flank_frames) < 6:
        return None
    try:
        p0, v0, rms = fit_ballistic(flank_frames, flank_pos, float(flank_frames[0]), fps)
    except ValueError:
        return None
    if rms > 1.5 * residual_threshold:
        return None
    gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
    sim = simulate_ballistic(gap_frames, p0, v0, float(flank_frames[0]), fps)
    if np.any(sim[:, 2] < -0.05) or np.any(sim[:, 2] > max_height):
        return None
    speeds = _velocity_at(gap_frames, p0, v0, float(flank_frames[0]), fps)
    if np.any(np.linalg.norm(speeds, axis=1) > max_speed_m_s):
        return None
    return sim, "ballistic"


def fill_gaps(
    obs: BallObservations,
    segments: list[BallisticSegment],
    bounces: list[Bounce],
    fps: float,
    ball_radius: float = 0.12,
    max_fill_gap_frames: int = 24,
    max_extrapolate_frames: int = 12,
    flank_frames_back: int = 15,
    residual_threshold: float = 0.10,
    restitution: float = 0.8,
    max_height: float = 10.0,
    max_speed_m_s: float = 30.0,
    use_drag_fit: bool = False,
) -> FilledTrajectory:
    """Fill observation gaps with ballistic arcs; everything else stays NaN."""
    frames = obs.frame_indices
    positions = obs.positions

    if len(frames) == 0:
        return FilledTrajectory(
            frames=np.array([], dtype=np.int64),
            positions=np.zeros((0, 3)),
            observed=np.zeros(0, dtype=bool),
            outlier=obs.outlier_mask,
            filled=np.zeros(0, dtype=bool),
            filled_source=np.array([], dtype=object),
        )

    first, last = int(frames[0]), int(frames[-1])
    grid = np.arange(first, last + 1, dtype=np.int64)
    n = len(grid)
    observed = np.zeros(n, dtype=bool)
    outlier = np.zeros(n, dtype=bool)
    positions_full = np.full((n, 3), np.nan)
    obs_index = 0
    for i, frame in enumerate(grid):
        if obs_index < len(frames) and int(frames[obs_index]) == int(frame):
            if not obs.outlier_mask[obs_index]:
                observed[i] = True
                positions_full[i] = positions[obs_index]
            else:
                outlier[i] = True
            obs_index += 1

    filled = np.zeros(n, dtype=bool)
    filled_source = np.full(n, None, dtype=object)

    bounce_frames = {b.frame_index for b in bounces}

    # ---- interior gaps ----
    gap_start_idx = None
    for i in range(n):
        if not observed[i] and gap_start_idx is None:
            gap_start_idx = i
        elif observed[i] and gap_start_idx is not None:
            gap_end_idx = i - 1
            gap_start = int(grid[gap_start_idx])
            gap_end = int(grid[gap_end_idx])
            length = gap_end - gap_start + 1
            if length <= max_fill_gap_frames:
                _fill_one_gap(
                    grid, positions_full, filled, filled_source,
                    gap_start_idx, gap_end_idx, segments, bounce_frames,
                    frames, positions, fps, ball_radius, restitution,
                    flank_frames_back, residual_threshold, max_height,
                    max_speed_m_s, use_drag_fit,
                )
            gap_start_idx = None
    if gap_start_idx is not None:
        gap_end_idx = n - 1
        gap_start = int(grid[gap_start_idx])
        gap_end = int(grid[gap_end_idx])
        if gap_end - gap_start + 1 <= max_fill_gap_frames:
            _fill_one_gap(
                grid, positions_full, filled, filled_source,
                gap_start_idx, gap_end_idx, segments, bounce_frames,
                frames, positions, fps, ball_radius, restitution,
                flank_frames_back, residual_threshold, max_height,
                max_speed_m_s, use_drag_fit,
            )

    # ---- trailing / leading extrapolation from a covering segment ----
    _extrapolate_edges(
        grid, positions_full, filled, filled_source, segments, frames,
        fps, max_extrapolate_frames, ball_radius,
    )

    return FilledTrajectory(
        frames=grid,
        positions=positions_full,
        observed=observed,
        outlier=outlier,
        filled=filled,
        filled_source=filled_source,
    )


def _fill_one_gap(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    gap_start_idx: int,
    gap_end_idx: int,
    segments: list[BallisticSegment],
    bounce_frames: set[int],
    obs_frames: np.ndarray,
    obs_pos: np.ndarray,
    fps: float,
    ball_radius: float,
    restitution: float,
    flank_frames_back: int,
    residual_threshold: float,
    max_height: float,
    max_speed_m_s: float,
    use_drag_fit: bool,
) -> None:
    gap_start = int(grid[gap_start_idx])
    gap_end = int(grid[gap_end_idx])

    # Case 1: gap fully inside a ballistic segment, no bounce in it.
    for seg in segments:
        if seg.start <= gap_start and gap_end <= seg.end:
            if any(bf >= gap_start and bf <= gap_end for bf in bounce_frames):
                pass
            if not any(gap_start <= bf <= gap_end for bf in bounce_frames):
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                if (np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height)):
                    speeds = _velocity_at(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                    if np.all(np.linalg.norm(speeds, axis=1) <= max_speed_m_s):
                        positions_full[gap_start_idx:gap_end_idx + 1] = sim
                        filled[gap_start_idx:gap_end_idx + 1] = True
                        filled_source[gap_start_idx:gap_end_idx + 1] = "ballistic"
                        seg.filled_gap_frames += gap_end - gap_start + 1
                        return

    # Case 2/3: double-sided fit or bounce bridge using flanking observations.
    left_sel = obs_frames <= grid[gap_start_idx - 1] if gap_start_idx > 0 else None
    # Observed frames before the gap (bounded window).
    before = obs_frames[obs_frames <= grid[gap_start_idx - 1]] if gap_start_idx > 0 else obs_frames[:0]
    before = before[-flank_frames_back:]
    after = obs_frames[obs_frames >= grid[gap_end_idx + 1]] if gap_end_idx + 1 < len(grid) else obs_frames[len(obs_frames):]
    after = after[:flank_frames_back]

    before_pos = obs_pos[np.isin(obs_frames, before)]
    after_pos = obs_pos[np.isin(obs_frames, after)]

    best: Optional[tuple[np.ndarray, str]] = None
    if len(before) >= 3 and len(after) >= 3:
        best = _try_bounce_bridge(
            before, before_pos, after, after_pos,
            gap_start, gap_end, fps, ball_radius, restitution,
        )
        if best is None:
            best = _try_double_sided(
                before, before_pos, after, after_pos,
                gap_start, gap_end, fps, residual_threshold, max_height, max_speed_m_s,
            )
    elif len(before) >= 6:
        # Only a left flank (gap at the very end) — covered by extrapolation.
        pass
    elif len(after) >= 6:
        # Only a right flank (gap at the very start) — leading edge handled by
        # _extrapolate_edges via segments; try a one-sided fit as well.
        try:
            p0, v0, rms = fit_ballistic(after, after_pos, float(after[0]), fps)
            if rms <= residual_threshold:
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, p0, v0, float(after[0]), fps)
                if np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height):
                    best = (sim, "ballistic")
        except ValueError:
            pass

    if best is not None:
        sim, source = best
        positions_full[gap_start_idx:gap_end_idx + 1] = sim
        filled[gap_start_idx:gap_end_idx + 1] = True
        filled_source[gap_start_idx:gap_end_idx + 1] = source


def _extrapolate_edges(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    segments: list[BallisticSegment],
    obs_frames: np.ndarray,
    fps: float,
    max_extrapolate_frames: int,
    ball_radius: float,
) -> None:
    if len(segments) == 0 or len(grid) == 0:
        return
    last_obs = int(obs_frames[-1])
    first_obs = int(obs_frames[0])

    for seg in segments:
        # Trailing: segment contains the last observed frame.
        if seg.start <= last_obs <= seg.end:
            tail_start = last_obs + 1
            if tail_start <= grid[-1]:
                tail = np.arange(tail_start, min(int(grid[-1]), tail_start + max_extrapolate_frames) + 1, dtype=np.float64)
                sim = simulate_ballistic(tail, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for i, frame in enumerate(tail.astype(int)):
                    if not keep[i]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[i]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"
        # Leading: segment contains the first observed frame.
        if seg.start <= first_obs <= seg.end:
            head_end = first_obs - 1
            if head_end >= grid[0]:
                head = np.arange(max(int(grid[0]), head_end - max_extrapolate_frames + 1), head_end + 1, dtype=np.float64)
                sim = simulate_ballistic(head, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for j, frame in enumerate(head.astype(int)):
                    if not keep[j]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[j]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"

def _try_bounce_bridge(
    left_frames: np.ndarray,
    left_pos: np.ndarray,
    right_frames: np.ndarray,
    right_pos: np.ndarray,
    gap_start: int,
    gap_end: int,
    fps: float,
    ball_radius: float,
    restitution: float,
    impact_time_tolerance_frames: float = 3.0,
    velocity_tolerance_m_s: float = 1.5,
) -> Optional[tuple[np.ndarray, str]]:
    """Fit descending-left + ascending-right arcs meeting at z = ball_radius.

    Returns (positions for frames [gap_start, gap_end], "ballistic") on success.
    """
    if len(left_frames) < 3 or len(right_frames) < 3:
        return None
    try:
        p0_l, v0_l, _ = fit_ballistic(left_frames, left_pos, float(left_frames[0]), fps)
        p0_r, v0_r, _ = fit_ballistic(right_frames, right_pos, float(right_frames[0]), fps)
    except ValueError:
        return None

    # Downward root of the left arc (tau after the apex) at z = r.
    roots_l = _quadratic_roots_z(float(p0_l[2]), float(v0_l[2]), 0.0, ball_radius, fps)
    if not roots_l:
        return None
    tau_apex_l = float(v0_l[2]) / 9.81
    tau_impact_l = None
    for root in roots_l:
        if root >= tau_apex_l - 1e-9:
            tau_impact_l = root
            break
    if tau_impact_l is None:
        return None

    # Upward root of the right arc at z = r.
    roots_r = _quadratic_roots_z(float(p0_r[2]), float(v0_r[2]), 0.0, ball_radius, fps)
    if not roots_r:
        return None
    tau_apex_r = float(v0_r[2]) / 9.81
    tau_impact_r = None
    for root in roots_r:
        if root <= tau_apex_r + 1e-9:
            tau_impact_r = root
            break
    if tau_impact_r is None:
        return None

    impact_l = float(left_frames[0]) + tau_impact_l * fps
    impact_r = float(right_frames[0]) + tau_impact_r * fps
    if abs(impact_l - impact_r) > impact_time_tolerance_frames:
        return None

    vz_l = float(v0_l[2] - 9.81 * tau_impact_l)
    vz_r = float(v0_r[2] - 9.81 * tau_impact_r)
    if abs(vz_r - (-restitution * vz_l)) > velocity_tolerance_m_s:
        return None

    impact_frame = int(round((impact_l + impact_r) / 2.0))
    if not (gap_start <= impact_frame <= gap_end):
        return None

    gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
    left_part = gap_frames[gap_frames < impact_frame]
    right_part = gap_frames[gap_frames >= impact_frame]
    result = np.full((len(gap_frames), 3), np.nan)
    if len(left_part):
        result[gap_frames < impact_frame] = simulate_ballistic(
            left_part, p0_l, v0_l, float(left_frames[0]), fps
        )
    if len(right_part):
        result[gap_frames >= impact_frame] = simulate_ballistic(
            right_part, p0_r, v0_r, float(right_frames[0]), fps
        )
    return result, "ballistic"


def _try_double_sided(
    left_frames: np.ndarray,
    left_pos: np.ndarray,
    right_frames: np.ndarray,
    right_pos: np.ndarray,
    gap_start: int,
    gap_end: int,
    fps: float,
    residual_threshold: float,
    max_height: float,
    max_speed_m_s: float,
) -> Optional[tuple[np.ndarray, str]]:
    """Single parabolic fit through both flanks of a gap."""
    flank_frames = np.concatenate([left_frames, right_frames])
    flank_pos = np.concatenate([left_pos, right_pos])
    if len(flank_frames) < 6:
        return None
    try:
        p0, v0, rms = fit_ballistic(flank_frames, flank_pos, float(flank_frames[0]), fps)
    except ValueError:
        return None
    if rms > 1.5 * residual_threshold:
        return None
    gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
    sim = simulate_ballistic(gap_frames, p0, v0, float(flank_frames[0]), fps)
    if np.any(sim[:, 2] < -0.05) or np.any(sim[:, 2] > max_height):
        return None
    speeds = _velocity_at(gap_frames, p0, v0, float(flank_frames[0]), fps)
    if np.any(np.linalg.norm(speeds, axis=1) > max_speed_m_s):
        return None
    return sim, "ballistic"


def fill_gaps(
    obs: BallObservations,
    segments: list[BallisticSegment],
    bounces: list[Bounce],
    fps: float,
    ball_radius: float = 0.12,
    max_fill_gap_frames: int = 24,
    max_extrapolate_frames: int = 12,
    flank_frames_back: int = 15,
    residual_threshold: float = 0.10,
    restitution: float = 0.8,
    max_height: float = 10.0,
    max_speed_m_s: float = 30.0,
    use_drag_fit: bool = False,
) -> FilledTrajectory:
    """Fill observation gaps with ballistic arcs; everything else stays NaN."""
    frames = obs.frame_indices
    positions = obs.positions

    if len(frames) == 0:
        return FilledTrajectory(
            frames=np.array([], dtype=np.int64),
            positions=np.zeros((0, 3)),
            observed=np.zeros(0, dtype=bool),
            outlier=obs.outlier_mask,
            filled=np.zeros(0, dtype=bool),
            filled_source=np.array([], dtype=object),
        )

    first, last = int(frames[0]), int(frames[-1])
    grid = np.arange(first, last + 1, dtype=np.int64)
    n = len(grid)
    observed = np.zeros(n, dtype=bool)
    outlier = np.zeros(n, dtype=bool)
    positions_full = np.full((n, 3), np.nan)
    obs_index = 0
    for i, frame in enumerate(grid):
        if obs_index < len(frames) and int(frames[obs_index]) == int(frame):
            if not obs.outlier_mask[obs_index]:
                observed[i] = True
                positions_full[i] = positions[obs_index]
            else:
                outlier[i] = True
            obs_index += 1

    filled = np.zeros(n, dtype=bool)
    filled_source = np.full(n, None, dtype=object)

    bounce_frames = {b.frame_index for b in bounces}

    # ---- interior gaps ----
    gap_start_idx = None
    for i in range(n):
        if not observed[i] and gap_start_idx is None:
            gap_start_idx = i
        elif observed[i] and gap_start_idx is not None:
            gap_end_idx = i - 1
            gap_start = int(grid[gap_start_idx])
            gap_end = int(grid[gap_end_idx])
            length = gap_end - gap_start + 1
            if length <= max_fill_gap_frames:
                _fill_one_gap(
                    grid, positions_full, filled, filled_source,
                    gap_start_idx, gap_end_idx, segments, bounce_frames,
                    frames, positions, fps, ball_radius, restitution,
                    flank_frames_back, residual_threshold, max_height,
                    max_speed_m_s, use_drag_fit,
                )
            gap_start_idx = None
    if gap_start_idx is not None:
        gap_end_idx = n - 1
        gap_start = int(grid[gap_start_idx])
        gap_end = int(grid[gap_end_idx])
        if gap_end - gap_start + 1 <= max_fill_gap_frames:
            _fill_one_gap(
                grid, positions_full, filled, filled_source,
                gap_start_idx, gap_end_idx, segments, bounce_frames,
                frames, positions, fps, ball_radius, restitution,
                flank_frames_back, residual_threshold, max_height,
                max_speed_m_s, use_drag_fit,
            )

    # ---- trailing / leading extrapolation from a covering segment ----
    _extrapolate_edges(
        grid, positions_full, filled, filled_source, segments, frames,
        fps, max_extrapolate_frames, ball_radius,
    )

    return FilledTrajectory(
        frames=grid,
        positions=positions_full,
        observed=observed,
        outlier=outlier,
        filled=filled,
        filled_source=filled_source,
    )


def _fill_one_gap(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    gap_start_idx: int,
    gap_end_idx: int,
    segments: list[BallisticSegment],
    bounce_frames: set[int],
    obs_frames: np.ndarray,
    obs_pos: np.ndarray,
    fps: float,
    ball_radius: float,
    restitution: float,
    flank_frames_back: int,
    residual_threshold: float,
    max_height: float,
    max_speed_m_s: float,
    use_drag_fit: bool,
) -> None:
    gap_start = int(grid[gap_start_idx])
    gap_end = int(grid[gap_end_idx])

    # Case 1: gap fully inside a ballistic segment, no bounce in it.
    for seg in segments:
        if seg.start <= gap_start and gap_end <= seg.end:
            if any(bf >= gap_start and bf <= gap_end for bf in bounce_frames):
                pass
            if not any(gap_start <= bf <= gap_end for bf in bounce_frames):
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                if (np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height)):
                    speeds = _velocity_at(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                    if np.all(np.linalg.norm(speeds, axis=1) <= max_speed_m_s):
                        positions_full[gap_start_idx:gap_end_idx + 1] = sim
                        filled[gap_start_idx:gap_end_idx + 1] = True
                        filled_source[gap_start_idx:gap_end_idx + 1] = "ballistic"
                        seg.filled_gap_frames += gap_end - gap_start + 1
                        return

    # Case 2/3: double-sided fit or bounce bridge using flanking observations.
    left_sel = obs_frames <= grid[gap_start_idx - 1] if gap_start_idx > 0 else None
    # Observed frames before the gap (bounded window).
    before = obs_frames[obs_frames <= grid[gap_start_idx - 1]] if gap_start_idx > 0 else obs_frames[:0]
    before = before[-flank_frames_back:]
    after = obs_frames[obs_frames >= grid[gap_end_idx + 1]] if gap_end_idx + 1 < len(grid) else obs_frames[len(obs_frames):]
    after = after[:flank_frames_back]

    before_pos = obs_pos[np.isin(obs_frames, before)]
    after_pos = obs_pos[np.isin(obs_frames, after)]

    best: Optional[tuple[np.ndarray, str]] = None
    if len(before) >= 3 and len(after) >= 3:
        best = _try_bounce_bridge(
            before, before_pos, after, after_pos,
            gap_start, gap_end, fps, ball_radius, restitution,
        )
        if best is None:
            best = _try_double_sided(
                before, before_pos, after, after_pos,
                gap_start, gap_end, fps, residual_threshold, max_height, max_speed_m_s,
            )
    elif len(before) >= 6:
        # Only a left flank (gap at the very end) — covered by extrapolation.
        pass
    elif len(after) >= 6:
        # Only a right flank (gap at the very start) — leading edge handled by
        # _extrapolate_edges via segments; try a one-sided fit as well.
        try:
            p0, v0, rms = fit_ballistic(after, after_pos, float(after[0]), fps)
            if rms <= residual_threshold:
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, p0, v0, float(after[0]), fps)
                if np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height):
                    best = (sim, "ballistic")
        except ValueError:
            pass

    if best is not None:
        sim, source = best
        positions_full[gap_start_idx:gap_end_idx + 1] = sim
        filled[gap_start_idx:gap_end_idx + 1] = True
        filled_source[gap_start_idx:gap_end_idx + 1] = source


def _extrapolate_edges(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    segments: list[BallisticSegment],
    obs_frames: np.ndarray,
    fps: float,
    max_extrapolate_frames: int,
    ball_radius: float,
) -> None:
    if len(segments) == 0 or len(grid) == 0:
        return
    last_obs = int(obs_frames[-1])
    first_obs = int(obs_frames[0])

    for seg in segments:
        # Trailing: segment contains the last observed frame.
        if seg.start <= last_obs <= seg.end:
            tail_start = last_obs + 1
            if tail_start <= grid[-1]:
                tail = np.arange(tail_start, min(int(grid[-1]), tail_start + max_extrapolate_frames) + 1, dtype=np.float64)
                sim = simulate_ballistic(tail, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for i, frame in enumerate(tail.astype(int)):
                    if not keep[i]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[i]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"
        # Leading: segment contains the first observed frame.
        if seg.start <= first_obs <= seg.end:
            head_end = first_obs - 1
            if head_end >= grid[0]:
                head = np.arange(max(int(grid[0]), head_end - max_extrapolate_frames + 1), head_end + 1, dtype=np.float64)
                sim = simulate_ballistic(head, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for j, frame in enumerate(head.astype(int)):
                    if not keep[j]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[j]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"


def fill_gaps(
    obs: BallObservations,
    segments: list[BallisticSegment],
    bounces: list[Bounce],
    fps: float,
    ball_radius: float = 0.12,
    max_fill_gap_frames: int = 24,
    max_extrapolate_frames: int = 12,
    flank_frames_back: int = 15,
    residual_threshold: float = 0.10,
    restitution: float = 0.8,
    max_height: float = 10.0,
    max_speed_m_s: float = 30.0,
    use_drag_fit: bool = False,
) -> FilledTrajectory:
    """Fill observation gaps with ballistic arcs; everything else stays NaN."""
    # Outlier frames carry bad positions; exclude them from the flank
    # observations used for gap fitting (they would bias the arc).
    clean_obs_frames = obs.frame_indices[~obs.outlier_mask]
    clean_obs_pos = obs.positions[~obs.outlier_mask]
    frames = obs.frame_indices
    positions = obs.positions

    if len(frames) == 0:
        return FilledTrajectory(
            frames=np.array([], dtype=np.int64),
            positions=np.zeros((0, 3)),
            observed=np.zeros(0, dtype=bool),
            outlier=obs.outlier_mask,
            filled=np.zeros(0, dtype=bool),
            filled_source=np.array([], dtype=object),
        )

    first, last = int(frames[0]), int(frames[-1])
    grid = np.arange(first, last + 1, dtype=np.int64)
    n = len(grid)
    observed = np.zeros(n, dtype=bool)
    outlier = np.zeros(n, dtype=bool)
    positions_full = np.full((n, 3), np.nan)
    obs_index = 0
    for i, frame in enumerate(grid):
        if obs_index < len(frames) and int(frames[obs_index]) == int(frame):
            if not obs.outlier_mask[obs_index]:
                observed[i] = True
                positions_full[i] = positions[obs_index]
            else:
                outlier[i] = True
            obs_index += 1

    filled = np.zeros(n, dtype=bool)
    filled_source = np.full(n, None, dtype=object)

    bounce_frames = {b.frame_index for b in bounces}

    # ---- interior gaps ----
    gap_start_idx = None
    for i in range(n):
        if not observed[i] and gap_start_idx is None:
            gap_start_idx = i
        elif observed[i] and gap_start_idx is not None:
            gap_end_idx = i - 1
            gap_start = int(grid[gap_start_idx])
            gap_end = int(grid[gap_end_idx])
            length = gap_end - gap_start + 1
            if length <= max_fill_gap_frames:
                _fill_one_gap(
                    grid, positions_full, filled, filled_source,
                    gap_start_idx, gap_end_idx, segments, bounce_frames,
                    clean_obs_frames, clean_obs_pos, fps, ball_radius, restitution,
                    flank_frames_back, residual_threshold, max_height,
                    max_speed_m_s, use_drag_fit,
                )
            gap_start_idx = None
    if gap_start_idx is not None:
        gap_end_idx = n - 1
        gap_start = int(grid[gap_start_idx])
        gap_end = int(grid[gap_end_idx])
        if gap_end - gap_start + 1 <= max_fill_gap_frames:
            _fill_one_gap(
                grid, positions_full, filled, filled_source,
                gap_start_idx, gap_end_idx, segments, bounce_frames,
                frames, positions, fps, ball_radius, restitution,
                flank_frames_back, residual_threshold, max_height,
                max_speed_m_s, use_drag_fit,
            )

    # ---- trailing / leading extrapolation from a covering segment ----
    _extrapolate_edges(
        grid, positions_full, filled, filled_source, segments,
        clean_obs_frames,
        fps, max_extrapolate_frames, ball_radius,
    )

    return FilledTrajectory(
        frames=grid,
        positions=positions_full,
        observed=observed,
        outlier=outlier,
        filled=filled,
        filled_source=filled_source,
    )


def _fill_one_gap(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    gap_start_idx: int,
    gap_end_idx: int,
    segments: list[BallisticSegment],
    bounce_frames: set[int],
    obs_frames: np.ndarray,
    obs_pos: np.ndarray,
    fps: float,
    ball_radius: float,
    restitution: float,
    flank_frames_back: int,
    residual_threshold: float,
    max_height: float,
    max_speed_m_s: float,
    use_drag_fit: bool,
) -> None:
    gap_start = int(grid[gap_start_idx])
    gap_end = int(grid[gap_end_idx])

    # Case 1: gap fully inside a ballistic segment, no bounce in it.
    for seg in segments:
        if seg.start <= gap_start and gap_end <= seg.end:
            if any(bf >= gap_start and bf <= gap_end for bf in bounce_frames):
                pass
            if not any(gap_start <= bf <= gap_end for bf in bounce_frames):
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                if (np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height)):
                    speeds = _velocity_at(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                    if np.all(np.linalg.norm(speeds, axis=1) <= max_speed_m_s):
                        positions_full[gap_start_idx:gap_end_idx + 1] = sim
                        filled[gap_start_idx:gap_end_idx + 1] = True
                        filled_source[gap_start_idx:gap_end_idx + 1] = "ballistic"
                        seg.filled_gap_frames += gap_end - gap_start + 1
                        return

    # Case 2/3: double-sided fit or bounce bridge using flanking observations.
    left_sel = obs_frames <= grid[gap_start_idx - 1] if gap_start_idx > 0 else None
    # Observed frames before the gap (bounded window).
    before = obs_frames[obs_frames <= grid[gap_start_idx - 1]] if gap_start_idx > 0 else obs_frames[:0]
    before = before[-flank_frames_back:]
    after = obs_frames[obs_frames >= grid[gap_end_idx + 1]] if gap_end_idx + 1 < len(grid) else obs_frames[len(obs_frames):]
    after = after[:flank_frames_back]

    before_pos = obs_pos[np.isin(obs_frames, before)]
    after_pos = obs_pos[np.isin(obs_frames, after)]

    best: Optional[tuple[np.ndarray, str]] = None
    if len(before) >= 3 and len(after) >= 3:
        best = _try_bounce_bridge(
            before, before_pos, after, after_pos,
            gap_start, gap_end, fps, ball_radius, restitution,
        )
        if best is None:
            best = _try_double_sided(
                before, before_pos, after, after_pos,
                gap_start, gap_end, fps, residual_threshold, max_height, max_speed_m_s,
            )
    elif len(before) >= 6:
        # Only a left flank (gap at the very end) — covered by extrapolation.
        pass
    elif len(after) >= 6:
        # Only a right flank (gap at the very start) — leading edge handled by
        # _extrapolate_edges via segments; try a one-sided fit as well.
        try:
            p0, v0, rms = fit_ballistic(after, after_pos, float(after[0]), fps)
            if rms <= residual_threshold:
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, p0, v0, float(after[0]), fps)
                if np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height):
                    best = (sim, "ballistic")
        except ValueError:
            pass

    if best is not None:
        sim, source = best
        positions_full[gap_start_idx:gap_end_idx + 1] = sim
        filled[gap_start_idx:gap_end_idx + 1] = True
        filled_source[gap_start_idx:gap_end_idx + 1] = source


def _extrapolate_edges(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    segments: list[BallisticSegment],
    obs_frames: np.ndarray,
    fps: float,
    max_extrapolate_frames: int,
    ball_radius: float,
) -> None:
    if len(segments) == 0 or len(grid) == 0:
        return
    last_obs = int(obs_frames[-1])
    first_obs = int(obs_frames[0])

    for seg in segments:
        # Trailing: segment contains the last observed frame.
        if seg.start <= last_obs <= seg.end:
            tail_start = last_obs + 1
            if tail_start <= grid[-1]:
                tail = np.arange(tail_start, min(int(grid[-1]), tail_start + max_extrapolate_frames) + 1, dtype=np.float64)
                sim = simulate_ballistic(tail, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for i, frame in enumerate(tail.astype(int)):
                    if not keep[i]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[i]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"
        # Leading: segment contains the first observed frame.
        if seg.start <= first_obs <= seg.end:
            head_end = first_obs - 1
            if head_end >= grid[0]:
                head = np.arange(max(int(grid[0]), head_end - max_extrapolate_frames + 1), head_end + 1, dtype=np.float64)
                sim = simulate_ballistic(head, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for j, frame in enumerate(head.astype(int)):
                    if not keep[j]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[j]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"

def _fill_one_gap(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    gap_start_idx: int,
    gap_end_idx: int,
    segments: list[BallisticSegment],
    bounce_frames: set[int],
    obs_frames: np.ndarray,
    obs_pos: np.ndarray,
    fps: float,
    ball_radius: float,
    restitution: float,
    flank_frames_back: int,
    residual_threshold: float,
    max_height: float,
    max_speed_m_s: float,
    use_drag_fit: bool,
) -> None:
    gap_start = int(grid[gap_start_idx])
    gap_end = int(grid[gap_end_idx])

    # Case 1: gap fully inside a ballistic segment, no bounce in it.
    for seg in segments:
        if seg.start <= gap_start and gap_end <= seg.end:
            if any(bf >= gap_start and bf <= gap_end for bf in bounce_frames):
                pass
            if not any(gap_start <= bf <= gap_end for bf in bounce_frames):
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                if (np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height)):
                    speeds = _velocity_at(gap_frames, seg.p0, seg.v0, float(seg.start), fps)
                    if np.all(np.linalg.norm(speeds, axis=1) <= max_speed_m_s):
                        positions_full[gap_start_idx:gap_end_idx + 1] = sim
                        filled[gap_start_idx:gap_end_idx + 1] = True
                        filled_source[gap_start_idx:gap_end_idx + 1] = "ballistic"
                        seg.filled_gap_frames += gap_end - gap_start + 1
                        return

    # Case 2/3: double-sided fit or bounce bridge using flanking observations.
    left_sel = obs_frames <= grid[gap_start_idx - 1] if gap_start_idx > 0 else None
    # Observed frames before the gap (bounded window).
    before = obs_frames[obs_frames <= grid[gap_start_idx - 1]] if gap_start_idx > 0 else obs_frames[:0]
    before = before[-flank_frames_back:]
    after = obs_frames[obs_frames >= grid[gap_end_idx + 1]] if gap_end_idx + 1 < len(grid) else obs_frames[len(obs_frames):]
    after = after[:flank_frames_back]

    before_pos = obs_pos[np.isin(obs_frames, before)]
    after_pos = obs_pos[np.isin(obs_frames, after)]

    best: Optional[tuple[np.ndarray, str]] = None
    if len(before) >= 3 and len(after) >= 3:
        best = _try_bounce_bridge(
            before, before_pos, after, after_pos,
            gap_start, gap_end, fps, ball_radius, restitution,
        )
        if best is None:
            best = _try_double_sided(
                before, before_pos, after, after_pos,
                gap_start, gap_end, fps, residual_threshold, max_height, max_speed_m_s,
            )
    elif len(before) >= 6:
        # Only a left flank (gap at the very end) — covered by extrapolation.
        pass
    elif len(after) >= 6:
        # Only a right flank (gap at the very start) — leading edge handled by
        # _extrapolate_edges via segments; try a one-sided fit as well.
        try:
            p0, v0, rms = fit_ballistic(after, after_pos, float(after[0]), fps)
            if rms <= residual_threshold:
                gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
                sim = simulate_ballistic(gap_frames, p0, v0, float(after[0]), fps)
                if np.all(sim[:, 2] >= -0.05) and np.all(sim[:, 2] <= max_height):
                    best = (sim, "ballistic")
        except ValueError:
            pass

    if best is not None:
        sim, source = best
        positions_full[gap_start_idx:gap_end_idx + 1] = sim
        filled[gap_start_idx:gap_end_idx + 1] = True
        filled_source[gap_start_idx:gap_end_idx + 1] = source


def _extrapolate_edges(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    segments: list[BallisticSegment],
    obs_frames: np.ndarray,
    fps: float,
    max_extrapolate_frames: int,
    ball_radius: float,
) -> None:
    if len(segments) == 0 or len(grid) == 0:
        return
    last_obs = int(obs_frames[-1])
    first_obs = int(obs_frames[0])

    for seg in segments:
        # Trailing: segment contains the last observed frame.
        if seg.start <= last_obs <= seg.end:
            tail_start = last_obs + 1
            if tail_start <= grid[-1]:
                tail = np.arange(tail_start, min(int(grid[-1]), tail_start + max_extrapolate_frames) + 1, dtype=np.float64)
                sim = simulate_ballistic(tail, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for i, frame in enumerate(tail.astype(int)):
                    if not keep[i]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[i]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"
        # Leading: segment contains the first observed frame.
        if seg.start <= first_obs <= seg.end:
            head_end = first_obs - 1
            if head_end >= grid[0]:
                head = np.arange(max(int(grid[0]), head_end - max_extrapolate_frames + 1), head_end + 1, dtype=np.float64)
                sim = simulate_ballistic(head, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for j, frame in enumerate(head.astype(int)):
                    if not keep[j]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[j]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"

def _extrapolate_edges(
    grid: np.ndarray,
    positions_full: np.ndarray,
    filled: np.ndarray,
    filled_source: np.ndarray,
    segments: list[BallisticSegment],
    obs_frames: np.ndarray,
    fps: float,
    max_extrapolate_frames: int,
    ball_radius: float,
) -> None:
    if len(segments) == 0 or len(grid) == 0:
        return
    last_obs = int(obs_frames[-1])
    first_obs = int(obs_frames[0])

    for seg in segments:
        # Trailing: segment contains the last observed frame.
        if seg.start <= last_obs <= seg.end:
            tail_start = last_obs + 1
            if tail_start <= grid[-1]:
                tail = np.arange(tail_start, min(int(grid[-1]), tail_start + max_extrapolate_frames) + 1, dtype=np.float64)
                sim = simulate_ballistic(tail, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for i, frame in enumerate(tail.astype(int)):
                    if not keep[i]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[i]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"
        # Leading: segment contains the first observed frame.
        if seg.start <= first_obs <= seg.end:
            head_end = first_obs - 1
            if head_end >= grid[0]:
                head = np.arange(max(int(grid[0]), head_end - max_extrapolate_frames + 1), head_end + 1, dtype=np.float64)
                sim = simulate_ballistic(head, seg.p0, seg.v0, float(seg.start), fps)
                keep = (sim[:, 2] >= ball_radius - 0.02)
                for j, frame in enumerate(head.astype(int)):
                    if not keep[j]:
                        break
                    idx = int(frame) - int(grid[0])
                    if 0 <= idx < len(grid) and not filled[idx]:
                        positions_full[idx] = sim[j]
                        filled[idx] = True
                        filled_source[idx] = "ballistic"

def _try_floor_landing(
    left_frames: np.ndarray,
    left_pos: np.ndarray,
    right_frames: np.ndarray,
    right_pos: np.ndarray,
    gap_start: int,
    gap_end: int,
    fps: float,
    ball_radius: float,
) -> Optional[tuple[np.ndarray, str]]:
    """Fill a gap where the left arc lands inside it and the ball then stays
    on the floor (right flank on the ground): arc until impact, floor after."""
    if len(left_frames) < 3 or len(right_frames) < 3:
        return None
    if float(np.mean(np.abs(right_pos[:, 2] - ball_radius))) > 0.15:
        return None
    right_speeds = np.linalg.norm(np.diff(right_pos, axis=0), axis=1) / (
        np.diff(right_frames).astype(np.float64) / fps
    )
    if len(right_speeds) and float(np.max(right_speeds)) > 5.0:
        return None
    try:
        p0, v0, _ = fit_ballistic(left_frames, left_pos, float(left_frames[0]), fps)
    except ValueError:
        return None
    roots = _quadratic_roots_z(float(p0[2]), float(v0[2]), 0.0, ball_radius, fps)
    if not roots:
        return None
    tau_apex = float(v0[2]) / 9.81
    tau_impact = None
    for root in roots:
        if root >= tau_apex - 1e-9:
            tau_impact = root
            break
    if tau_impact is None:
        return None
    impact_frame = int(round(float(left_frames[0]) + tau_impact * fps))
    if not (gap_start <= impact_frame <= gap_end):
        return None

    gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
    result = np.full((len(gap_frames), 3), np.nan)
    arc_part = gap_frames[gap_frames < impact_frame]
    floor_part = gap_frames[gap_frames >= impact_frame]
    if len(arc_part):
        result[gap_frames < impact_frame] = simulate_ballistic(
            arc_part, p0, v0, float(left_frames[0]), fps
        )
    impact_pos = simulate_ballistic(
        np.array([impact_frame], dtype=np.float64), p0, v0, float(left_frames[0]), fps
    )[0]
    if len(floor_part):
        vx, vy = v0[0], v0[1]
        for k, frame in enumerate(floor_part):
            t = (float(frame) - impact_frame) / fps
            result[len(arc_part) + k] = [impact_pos[0] + vx * t, impact_pos[1] + vy * t, ball_radius]
    return result, "ballistic"



def _try_drop_fill(
    left_frames: np.ndarray,
    left_pos: np.ndarray,
    right_frames: np.ndarray,
    right_pos: np.ndarray,
    gap_start: int,
    gap_end: int,
    fps: float,
    ball_radius: float,
    max_gap_frames: int = 4,
) -> Optional[tuple[np.ndarray, str]]:
    """Fill a short gap where the ball was put down: left flank held high and
    static, right flank on the floor. The ball is on the floor throughout the
    gap, with horizontal position interpolated between the flanks."""
    length = gap_end - gap_start + 1
    if length > max_gap_frames:
        return None
    if len(left_frames) < 2 or len(right_frames) < 2:
        return None
    left_speed = float(np.linalg.norm(left_pos[-1] - left_pos[0])) / (
        max(1e-6, (left_frames[-1] - left_frames[0]) / fps)
    )
    if left_speed > 2.0 or float(np.mean(left_pos[:, 2])) < 0.5:
        return None
    if float(np.mean(np.abs(right_pos[:, 2] - ball_radius))) > 0.15:
        return None
    right_speed = float(np.linalg.norm(right_pos[-1] - right_pos[0])) / (
        max(1e-6, (right_frames[-1] - right_frames[0]) / fps)
    )
    if right_speed > 3.0:
        return None

    left_anchor = left_pos[-1]
    right_anchor = right_pos[0]
    gap_frames = np.arange(gap_start, gap_end + 1, dtype=np.float64)
    total = float(gap_end + 1 - gap_start)
    result = np.full((len(gap_frames), 3), np.nan)
    for k, frame in enumerate(gap_frames):
        t = (float(frame) - gap_start + 1.0) / (total + 1.0)  # 0..1 between anchors
        result[k] = [
            left_anchor[0] + t * (right_anchor[0] - left_anchor[0]),
            left_anchor[1] + t * (right_anchor[1] - left_anchor[1]),
            ball_radius,
        ]
    return result, "ballistic"


