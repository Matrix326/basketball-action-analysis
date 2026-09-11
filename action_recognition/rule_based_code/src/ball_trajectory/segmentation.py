"""Ballistic flight-segment detection, bounce detection, flight/dribble split.

A frame is *ballistic-consistent* when a constant-gravity fit over the
observed points in its sliding window is tight (small RMS residual) and the
window's vertical acceleration is close to -g. Contiguous consistent runs
become ballistic segments; their apex height / horizontal span / player-hand
proximity / bounce count then decide flight vs dribble.

A carried ball (held, ~0 acceleration) does NOT pass: the gravity fit leaves
~0.3 m of curvature across a 0.5 s window, well above the residual gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from .ballistics import fit_ballistic, fit_free_g, fit_with_drag, simulate_ballistic
from .io import BallObservations


@dataclass

class Bounce:
    frame_index: int
    position: np.ndarray        # (3,) observed position at impact
    impact_speed_m_s: float


@dataclass

class BallisticSegment:
    start: int                  # inclusive frame index
    end: int                    # inclusive frame index
    p0: np.ndarray              # fit at t0 = start
    v0: np.ndarray
    rms: float                  # residual over observed frames in [start, end]
    observed_frames: int
    kind: str = "flight"        # set by classify
    confidence: float = 0.0     # set by classify
    filled_gap_frames: int = 0  # filled by fill.py


# --------------------------------------------------------------------------
# deterministic RANSAC-style trim

def _lcg_triples(n: int, trials: int, seed: int = 1234567):
    """Yield deterministic pseudo-random index triples (0..n-1)."""
    state = seed
    for _ in range(trials):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        first = state % n
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        second = state % n
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        third = state % n
        if len({first, second, third}) == 3:
            yield first, second, third


def _ransac_trim(
    wf: np.ndarray,
    wp: np.ndarray,
    fps: float,
    threshold: float,
    trials: int = 48,
    min_inliers: int = 8,
    inlier_tolerance: float = 1.5,
    min_spread: float = 0.3,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Largest subset of wf/wp fitting the fixed-g model within tolerance.

    The inlier subset must also span >= min_spread metres — a static (held)
    window's parabolic crossing region is spatially clustered and must not
    pass as ballistic. Returns (subset_frames, subset_pos, rms) or None.
    """
    n = len(wf)
    best: Optional[tuple[int, float, np.ndarray]] = None
    for first, second, third in _lcg_triples(n, trials):
        idx = np.array([first, second, third])
        try:
            p0, v0, _ = fit_ballistic(wf[idx], wp[idx], float(wf[idx][0]), fps)
        except ValueError:
            continue
        model = simulate_ballistic(wf, p0, v0, float(wf[0]), fps)
        residuals = np.linalg.norm(wp - model, axis=1)
        inliers = residuals <= threshold * inlier_tolerance
        count = int(np.count_nonzero(inliers))
        if count < min_inliers:
            continue
        inlier_pos = wp[inliers]
        spread = float(np.max(np.linalg.norm(inlier_pos - inlier_pos[0], axis=1)))
        if spread < min_spread:
            continue
        score = (count, -float(np.sum(residuals[inliers])))
        if best is None or score > best[0]:
            best = (score, count, inliers)
    if best is None:
        return None
    _, count, inliers = best
    wf_sub = wf[inliers]
    wp_sub = wp[inliers]
    p0, v0, rms = fit_ballistic(wf_sub, wp_sub, float(wf_sub[0]), fps)
    return wf_sub, wp_sub, rms


# --------------------------------------------------------------------------
# bounce detection

def detect_bounces(
    obs: BallObservations,
    fps: float,
    ball_radius: float,
    min_impact_speed: float = 1.0,
    max_flank_gap: int = 6,
) -> list[Bounce]:
    """Bounce = z near the floor AND fit-based vz flips negative -> positive.

    Only observed frames qualify; bounces inside gaps are handled by fill.py's
    bounce-bridge. Requires both flanks observed (ascending + descending).
    """
    frames = obs.frame_indices[~obs.outlier_mask]
    positions = obs.positions[~obs.outlier_mask]
    bounces: list[Bounce] = []

    for idx, (frame, pos) in enumerate(zip(frames, positions)):
        if pos[2] > ball_radius + 0.05:
            continue

        left = positions[max(0, idx - max_flank_gap):idx]
        left_frames = frames[max(0, idx - max_flank_gap):idx]
        right = positions[idx + 1:idx + 1 + max_flank_gap]
        right_frames = frames[idx + 1:idx + 1 + max_flank_gap]

        if len(left) < 3 or len(right) < 3:
            continue

        try:
            _, v0_left, _ = fit_ballistic(left_frames, left, float(left_frames[-1]), fps)
            _, v0_right, _ = fit_ballistic(right_frames, right, float(right_frames[0]), fps)
        except ValueError:
            continue

        tau = (float(frame) - float(left_frames[-1])) / fps
        vz_before = v0_left[2] - 9.81 * tau
        tau_after = (float(right_frames[0]) - float(frame)) / fps
        vz_after = v0_right[2] - 9.81 * tau_after

        if vz_before < -min_impact_speed and vz_after > min_impact_speed:
            bounces.append(
                Bounce(
                    frame_index=int(frame),
                    position=positions[idx].astype(float).copy(),
                    impact_speed_m_s=float(-vz_before),
                )
            )
    # A contact frame and the launch frame right after it can both pass the
    # test; keep the stronger impact within a 2-frame neighbourhood.
    bounces.sort(key=lambda b: (b.frame_index, -b.impact_speed_m_s))
    deduped: list[Bounce] = []
    for bounce in bounces:
        if deduped and abs(bounce.frame_index - deduped[-1].frame_index) <= 2:
            if bounce.impact_speed_m_s > deduped[-1].impact_speed_m_s:
                deduped[-1] = bounce
            continue
        deduped.append(bounce)
    return deduped


# --------------------------------------------------------------------------
# flight segment detection

def detect_ballistic_segments(
    obs: BallObservations,
    fps: float,
    window_seconds: float = 0.5,
    residual_threshold: float = 0.10,
    accel_z_range: tuple[float, float] = (-13.0, -6.0),
    min_window_observations: int = 8,
    gap_bridge_frames: int = 25,
    min_fit_observations: int = 8,
    use_drag_fit: bool = False,
    bounces: Optional[list[Bounce]] = None,
) -> list[BallisticSegment]:
    """Find contiguous frame runs consistent with free flight."""
    if bounces is None:
        bounces = detect_bounces(obs, fps, ball_radius=0.12)
    bounce_frames = {b.frame_index for b in bounces}
    frames = obs.frame_indices[~obs.outlier_mask]
    positions = obs.positions[~obs.outlier_mask]
    if len(frames) < min_fit_observations:
        return []

    half = int(round(window_seconds * fps / 2.0))
    half = max(3, half)

    first, last = int(frames[0]), int(frames[-1])
    consistency = np.zeros(last - first + 1, dtype=bool)
    observed_in_range = np.zeros(last - first + 1, dtype=bool)
    offset = first

    for idx, frame in enumerate(frames):
        observed_in_range[int(frame) - offset] = True
        window_frames = frames[
            (frames >= frame - half) & (frames <= frame + half)
        ]
        if len(window_frames) < min_window_observations:
            continue
        window_pos = positions[np.isin(frames, window_frames)]
        try:
            p0, v0, rms = fit_ballistic(window_frames, window_pos, float(window_frames[0]), fps)
            if rms > residual_threshold:
                trimmed = _ransac_trim(window_frames, window_pos, fps, residual_threshold)
                if trimmed is None:
                    continue
                window_frames, window_pos, rms = trimmed
        except ValueError:
            continue
        if rms > residual_threshold:
            continue
        consistency[int(frame) - offset] = True

    # Contiguous runs of consistent frames.
    runs: list[tuple[int, int]] = []
    start = None
    for k in range(len(consistency)):
        if consistency[k]:
            if start is None:
                start = k
        elif start is not None:
            runs.append((start + offset, k - 1 + offset))
            start = None
    if start is not None:
        runs.append((start + offset, len(consistency) - 1 + offset))

    # Merge runs separated by short holes (unobserved OR observed-but-
    # inconsistent, e.g. apex frames); the union fit validates the merge.
    merged: list[tuple[int, int]] = []
    for run in runs:
        if not merged:
            merged.append(run)
            continue
        prev_start, prev_end = merged[-1]
        if run[0] - prev_end - 1 <= gap_bridge_frames:
            try:
                span_frames = frames[(frames >= prev_start) & (frames <= run[1])]
                span_pos = positions[np.isin(frames, span_frames)]
                _, _, rms = fit_ballistic(span_frames, span_pos, float(prev_start), fps)
                if rms <= 1.25 * residual_threshold:
                    merged[-1] = (prev_start, run[1])
                    continue
            except ValueError:
                pass
        merged.append(run)

    segments: list[BallisticSegment] = []
    for run_start, run_end in merged:
        sel = (frames >= run_start) & (frames <= run_end)
        run_frames = frames[sel]
        run_pos = positions[sel]
        # Detected bounces cut runs into single arcs; the recursive split
        # then only handles missed bounces.
        for piece_frames, piece_pos in _split_at_bounces(
            run_frames, run_pos, bounce_frames
        ):
            for piece_frames, piece_pos in _split_until_fits(
                piece_frames, piece_pos, fps, residual_threshold, min_fit_observations
            ):
                p0, v0, rms = fit_ballistic(piece_frames, piece_pos, float(piece_frames[0]), fps)
                if use_drag_fit:
                    p0_d, v0_d, _, rms_d, _ = fit_with_drag(piece_frames, piece_pos, float(piece_frames[0]), fps)
                    if p0_d is not None:
                        p0, v0, rms = p0_d, v0_d, rms_d
                if rms > 1.25 * residual_threshold:
                    continue
                # A static piece (carried/still ball) fits a parabola trivially;
                # real arcs move. Reject segments whose max speed is negligible.
                tau = (piece_frames - float(piece_frames[0])) / fps
                gravity = np.zeros(3)
                gravity[2] = -9.81
                speeds = np.linalg.norm(v0[None, :] + gravity[None, :] * tau[:, None], axis=1)
                if float(np.max(speeds)) < 1.0:
                    continue
                # Segment-level free-fall gate (well-conditioned on >=12 points).
                if not _passes_free_fall_gate(piece_frames, piece_pos, fps, accel_z_range):
                    continue
                segments.append(
                    BallisticSegment(
                        start=int(piece_frames[0]),
                        end=int(piece_frames[-1]),
                        p0=p0.astype(np.float64),
                        v0=v0.astype(np.float64),
                        rms=float(rms),
                        observed_frames=int(len(piece_frames)),
                    )
                )
    return segments


def _passes_free_fall_gate(
    piece_frames: np.ndarray,
    piece_pos: np.ndarray,
    fps: float,
    accel_z_range: tuple[float, float],
) -> bool:
    """Segment-level vertical-acceleration check, length-adaptive."""
    n = len(piece_frames)
    if n < 8:
        return True  # too few points to identify g reliably
    _, _, a_z, _ = fit_free_g(piece_frames, piece_pos, float(piece_frames[0]), fps)
    if n >= 12:
        return bool(accel_z_range[0] <= a_z <= accel_z_range[1])
    return bool(-20.0 <= a_z <= -2.0)


def _split_at_bounces(
    run_frames: np.ndarray,
    run_pos: np.ndarray,
    bounce_frames: set[int],
):
    """Yield run pieces cut at detected bounce frames (excluded from pieces)."""
    if not bounce_frames:
        yield run_frames, run_pos
        return
    cut_indices = [i for i, frame in enumerate(run_frames) if int(frame) in bounce_frames]
    if not cut_indices:
        yield run_frames, run_pos
        return
    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for cut in cut_indices:
        if cut > start:
            pieces.append((run_frames[start:cut], run_pos[start:cut]))
        start = cut + 1
    if start < len(run_frames):
        pieces.append((run_frames[start:], run_pos[start:]))
    for piece in pieces:
        yield piece


def _split_until_fits(
    run_frames: np.ndarray,
    run_pos: np.ndarray,
    fps: float,
    threshold: float,
    min_points: int,
    depth: int = 0,
):
    """Recursively split a run until each piece fits the ballistic model.

    Leading/trailing frames that do not fit are trimmed first (otherwise the
    split point sits at the edge and only one frame is peeled per level).
    Yields (frames, pos) pieces with rms <= 1.5*threshold, or >= min_points.
    """
    n = len(run_frames)
    if n < min_points:
        return
    _, _, rms = fit_ballistic(run_frames, run_pos, float(run_frames[0]), fps)
    if rms <= 1.25 * threshold or depth >= 6 or n <= 2 * min_points:
        yield run_frames, run_pos
        return
    p0, v0, _ = fit_ballistic(run_frames, run_pos, float(run_frames[0]), fps)
    residuals = np.linalg.norm(run_pos - simulate_ballistic(run_frames, p0, v0, float(run_frames[0]), fps), axis=1)
    bad = residuals > 1.25 * threshold
    keep_start, keep_end = 0, n
    if bad[0]:
        keep_start = int(np.argmax(~bad))
    if bad[-1] and keep_end - keep_start > 1:
        keep_end = n - int(np.argmax(~bad[::-1]))
    if keep_end - keep_start >= min_points and (keep_start > 0 or keep_end < n):
        yield from _split_until_fits(
            run_frames[keep_start:keep_end], run_pos[keep_start:keep_end],
            fps, threshold, min_points, depth + 1,
        )
        return
    split_at = int(np.argmax(residuals))
    left = slice(0, split_at + 1)
    right = slice(split_at + 1, n)
    yield from _split_until_fits(run_frames[left], run_pos[left], fps, threshold, min_points, depth + 1)
    yield from _split_until_fits(run_frames[right], run_pos[right], fps, threshold, min_points, depth + 1)


# --------------------------------------------------------------------------
# flight vs dribble split

def _apex_parameters(p0: np.ndarray, v0: np.ndarray, start: int, end: int, fps: float) -> tuple[float, float]:
    """(apex_z_m, horizontal_displacement_m) of the fitted arc over [start, end]."""
    grid = np.arange(start, end + 1, dtype=np.float64)
    sim = simulate_ballistic(grid, p0, v0, float(start), fps)
    apex_z = float(np.max(sim[:, 2]))
    dx, dy = sim[-1, 0] - sim[0, 0], sim[-1, 1] - sim[0, 1]
    return apex_z, float(np.hypot(dx, dy))


def classify_ballistic_segments(
    segments: list[BallisticSegment],
    bounces: list[Bounce],
    hand_centers: Optional[dict[int, np.ndarray]],
    fps: float,
    flight_min_apex_m: float = 2.0,
    flight_min_horizontal_m: float = 2.0,
    dribble_max_apex_m: float = 1.4,
    dribble_max_player_distance_m: float = 1.5,
    residual_threshold: float = 0.10,
) -> None:
    """In-place: set segment.kind ("flight"|"dribble") and confidence."""
    bounce_frames = {b.frame_index for b in bounces}

    for seg in segments:
        apex_z, horizontal = _apex_parameters(seg.p0, seg.v0, seg.start, seg.end, fps)
        bounce_count = sum(1 for frame in bounce_frames if seg.start <= frame <= seg.end)
        span = max(1, seg.end - seg.start + 1)
        coverage = seg.observed_frames / span

        min_hand_distance: Optional[float] = None
        if hand_centers:
            distances = []
            for frame in range(seg.start, seg.end + 1):
                ball = simulate_ballistic(
                    np.array([frame], dtype=np.float64), seg.p0, seg.v0, float(seg.start), fps
                )[0]
                hands = hand_centers.get(frame)
                if hands is None or len(hands) == 0:
                    continue
                distances.append(float(np.min(np.linalg.norm(hands - ball, axis=1))))
            if distances:
                min_hand_distance = min(distances)

        # the lowest z reached by the arc — dribbling runs low (a dribble arc
        # split at the bounce may only contain its upper part, hence 0.9)
        grid_sim = simulate_ballistic(
            np.arange(seg.start, seg.end + 1, dtype=np.float64), seg.p0, seg.v0, float(seg.start), fps
        )
        min_z = float(np.min(grid_sim[:, 2]))
        touches_floor = min_z <= 0.9

        is_flight = apex_z >= flight_min_apex_m or horizontal >= flight_min_horizontal_m
        if is_flight:
            seg.kind = "flight"
        else:
            is_dribble = touches_floor and apex_z <= dribble_max_apex_m and (
                bounce_count >= 1
                or (min_hand_distance is not None and min_hand_distance <= dribble_max_player_distance_m)
            )
            if is_dribble:
                seg.kind = "dribble"
            else:
                # Dead zone (1.4 < apex < 2.0): resolve by hand proximity, else
                # by bounce count; without skeleton a low arc with <2 bounces
                # is conservatively called flight.
                if min_hand_distance is not None and min_hand_distance <= dribble_max_player_distance_m:
                    seg.kind = "dribble"
                elif bounce_count >= 2:
                    seg.kind = "dribble"
                else:
                    seg.kind = "flight"

        if seg.kind == "flight":
            seg.confidence = float(np.clip(1.0 - seg.rms / max(residual_threshold, 1e-6), 0.0, 1.0) * coverage)
        else:
            proximity = 1.0 if min_hand_distance is None else float(
                np.clip(1.0 - min_hand_distance / max(dribble_max_player_distance_m, 1e-6), 0.0, 1.0)
            )
            seg.confidence = float(
                np.clip(1.0 - seg.rms / max(residual_threshold, 1e-6), 0.0, 1.0)
                * (0.5 + 0.5 * proximity)
            )

def _estimate_z_accel(z: np.ndarray, dt: float) -> np.ndarray:
    """Windowed vertical acceleration (m/s^2) via Savitzky-Golay 2nd derivative.

    Retained for diagnostics; the segmentation gate now uses fit_free_g, which
    is robust to uneven frame sampling (dropped frames break savgol's uniform
    sampling assumption).
    """
    n = len(z)
