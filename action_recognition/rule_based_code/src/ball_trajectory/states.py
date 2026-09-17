"""Rule-based per-frame ball state classification (for downstream action recognition).

States: flight | held | dribble | ground | unknown.

Precedence:
1. Inside a fitted ballistic segment -> its kind (flight/dribble).
2. Held: ball within ``hold_reach_m`` of a player's hands, moving slower than
   ``hold_max_speed_m_s``, sustained >= ``hold_min_frames`` frames.
3. Static residue far from players (likely background false positive) ->
   unknown.
4. Ground: z <= ``ground_z_m`` and slow (rolling ball).
5. Otherwise unknown.

Hands come from the 3D skeleton: wrist midpoint (COCO-17 kpts 9,10), single
wrist fallback, shoulder midpoint (5,6) minus 0.3 m fallback. Without a
skeleton, held/static degrade to unknown; dribble still works via bounce
counts inside segments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

STATE_FLIGHT = "flight"
STATE_HELD = "held"
STATE_DRIBBLE = "dribble"
STATE_GROUND = "ground"
STATE_UNKNOWN = "unknown"

_STATES = (STATE_FLIGHT, STATE_HELD, STATE_DRIBBLE, STATE_GROUND, STATE_UNKNOWN)
_STATE_INDEX = {state: index for index, state in enumerate(_STATES)}


def parse_hand_centers(
    skeleton: Optional[dict[int, dict[int, np.ndarray]]],
) -> Optional[dict[int, np.ndarray]]:
    """skeleton: frame -> player_id -> (17,3) (NaN for missing keypoints).

    Returns frame -> (K, 3) estimated hand centers, or None without skeleton.
    """
    if not skeleton:
        return None
    centers: dict[int, np.ndarray] = {}
    for frame, players in skeleton.items():
        hands = []
        for kpts in players.values():
            kpts = np.asarray(kpts, dtype=np.float64)
            if kpts.shape != (17, 3):
                continue
            left, right = kpts[9], kpts[10]
            left_ok, right_ok = np.isfinite(left).all(), np.isfinite(right).all()
            if left_ok and right_ok:
                hands.append((left + right) / 2.0)
            elif left_ok:
                hands.append(left)
            elif right_ok:
                hands.append(right)
            else:
                left_s, right_s = kpts[5], kpts[6]
                if np.isfinite(left_s).all() and np.isfinite(right_s).all():
                    hand = (left_s + right_s) / 2.0
                    hand[2] -= 0.3
                    hands.append(hand)
        if hands:
            centers[frame] = np.asarray(hands, dtype=np.float64)
    return centers if centers else None


def classify_states(
    frames: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    seg_kinds: np.ndarray,
    seg_confidence: np.ndarray,
    bounce_frames: set[int],
    hand_centers: Optional[dict[int, np.ndarray]],
    fps: float,
    hold_reach_m: float = 0.45,
    hold_min_frames: int = 6,
    hold_max_speed_m_s: float = 1.5,
    ground_z_m: float = 0.15,
    static_speed_m_s: float = 0.05,
    static_min_frames: int = 60,
    player_distance_for_static_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (state strings (N,), confidence (N,))."""
    n = len(frames)
    states = np.full(n, STATE_UNKNOWN, dtype=object)
    confidence = np.full(n, 0.3)

    speed = np.linalg.norm(velocities, axis=1)
    _dt = 1.0 / float(fps)

    for i in range(n):
        if not np.isfinite(positions[i]).all() or not np.isfinite(velocities[i]).all():
            continue
        kind = seg_kinds[i]
        if kind in (STATE_FLIGHT, STATE_DRIBBLE):
            states[i] = kind
            confidence[i] = float(seg_confidence[i])
            continue

        pos = positions[i]
        hands = hand_centers.get(int(frames[i])) if hand_centers else None
        if hands is not None and len(hands):
            distance = float(np.min(np.linalg.norm(hands - pos, axis=1)))
        else:
            distance = float("inf")

        # Held candidate (sustained check below).
        if distance <= hold_reach_m and speed[i] <= hold_max_speed_m_s:
            states[i] = STATE_HELD
            confidence[i] = float(
                np.clip(1.0 - distance / max(hold_reach_m, 1e-6), 0.0, 1.0)
                * np.clip(1.0 - speed[i] / max(hold_max_speed_m_s, 1e-6), 0.0, 1.0)
            )
            continue

        if pos[2] <= ground_z_m and speed[i] <= 3.0:
            states[i] = STATE_GROUND
            confidence[i] = float(
                np.clip(1.0 - pos[2] / max(ground_z_m, 1e-6), 0.0, 1.0)
                * np.clip(1.0 - speed[i] / 3.0, 0.0, 1.0)
            )
            continue

        states[i] = STATE_UNKNOWN
        confidence[i] = 0.3

    # Held must be sustained — short spurts revert to unknown.
    if hold_min_frames > 1:
        run_start = None
        for i in range(n):
            if states[i] == STATE_HELD and run_start is None:
                run_start = i
            elif states[i] != STATE_HELD and run_start is not None:
                if i - run_start < hold_min_frames:
                    states[run_start:i] = STATE_UNKNOWN
                    confidence[run_start:i] = 0.3
                run_start = None
        if run_start is not None and n - run_start < hold_min_frames:
            states[run_start:] = STATE_UNKNOWN
            confidence[run_start:] = 0.3

    # Static residue far from players -> unknown (background false positives).
    run_start = None
    for i in range(n + 1):
        is_static = (
            i < n
            and states[i] not in (STATE_FLIGHT, STATE_DRIBBLE)
            and np.isfinite(speed[i])
            and speed[i] <= static_speed_m_s
        )
        if is_static and run_start is None:
            run_start = i
        elif not is_static and run_start is not None:
            if i - run_start >= static_min_frames:
                for j in range(run_start, i):
                    if int(frames[j]) in bounce_frames:
                        continue
                    hands = hand_centers.get(int(frames[j])) if hand_centers else None
                    far = True
                    if hands is not None and len(hands):
                        d = float(np.min(np.linalg.norm(hands - positions[j], axis=1)))
                        far = d > player_distance_for_static_m
                    if far:
                        states[j] = STATE_UNKNOWN
                        confidence[j] = 0.3
            run_start = None

    # Majority filter over 5 frames, preserving bounce frames.
    states = _majority_filter(states, int(frames[0]), bounce_frames)

    # Recompute confidence sanity: unknown outside segment = 0.3, segments keep theirs.
    for i in range(n):
        if seg_kinds[i] in (STATE_FLIGHT, STATE_DRIBBLE) and states[i] == seg_kinds[i]:
            confidence[i] = float(seg_confidence[i])
    return states, confidence


def _majority_filter(
    states: np.ndarray, first_frame: int, bounce_frames: set[int]
) -> np.ndarray:
    n = len(states)
    if n < 3:
        return states
    result = states.copy()
    for i in range(n):
        if first_frame + i in bounce_frames:
            continue
        left = max(0, i - 2)
        right = min(n, i + 3)
        window = states[left:right]
        counts: dict[str, int] = {}
        for state in window:
            counts[str(state)] = counts.get(str(state), 0) + 1
        majority = max(counts, key=counts.__getitem__)
        if counts[majority] >= 3:
            result[i] = majority
    return result


def extend_segment_states(
    frames: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    states: np.ndarray,
    confidence: np.ndarray,
    seg_kinds: np.ndarray,
    seg_confidence: np.ndarray,
    extension_frames: int = 4,
) -> None:
    """Extend flight/dribble states a few frames past segment boundaries.

    Only frames whose motion is consistent with the segment's velocity at the
    boundary are extended — a catch (velocity ~0) right after a flight segment
    is NOT relabelled, while the trailing frame of the arc is. This recovers
    release/contact frames that the windowed detector cannot assign.
    """
    n = len(frames)
    for i in range(n):
        kind = seg_kinds[i]
        if kind not in (STATE_FLIGHT, STATE_DRIBBLE):
            continue
        boundary_vel = velocities[i]
        if not np.isfinite(boundary_vel).all():
            continue
        # extend left from segment start
        j = i - 1
        while j >= 0 and i - j <= extension_frames:
            if seg_kinds[j] in (STATE_FLIGHT, STATE_DRIBBLE):
                break
            if not np.isfinite(velocities[j]).all():
                break
            if float(np.dot(velocities[j], boundary_vel)) <= 0.0:
                break
            states[j] = kind
            confidence[j] = min(float(seg_confidence[i]), 0.5)
            j -= 1
        # extend right from segment end
        j = i + 1
        while j < n and j - i <= extension_frames:
            if seg_kinds[j] in (STATE_FLIGHT, STATE_DRIBBLE):
                break
            if not np.isfinite(velocities[j]).all():
                break
            if float(np.dot(velocities[j], boundary_vel)) <= 0.0:
                break
            states[j] = kind
            confidence[j] = min(float(seg_confidence[i]), 0.5)
            j += 1


def _motion_consistent(a: np.ndarray, b: np.ndarray) -> bool:
    """Velocity consistency for state extension: same heading AND vertical
    direction. The horizontal component alone must not mask a vertical
    direction change (e.g. a flight arc vs the ground roll that follows it)."""
    if float(np.dot(a, b)) <= 0.0:
        return False
    if abs(b[2]) < 0.3:
        return True  # boundary velocity is nearly horizontal
    return float(a[2] * b[2]) > 0.0
