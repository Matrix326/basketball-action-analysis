"""Rule engine: possession state machine + basketball action detection.

Events emitted (actions.json):
    dribble_start / dribble_end  — ball in low bounce cycles near a player
    release                      — ball leaves the handler's hands
    catch                        — ball comes into a player's hands
    pass                         — release(A) -> flight -> catch(B), A != B
    shoot / layup                — release -> flight ending near the hoop
    rebound                      — ball obtained near the hoop after a shot

Inputs (all frame-aligned with the same frame indices):
    ball_trajectory.json, poses_3d.json, hoop_3d.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

# ---------------------------------------------------------------------------
# defaults (tuned on synthetic + one real clip; adjust per dataset)

DEFAULTS: dict[str, Any] = {
    "hold_reach_m": 0.45,  # ball in hand distance
    "release_distance_m": 0.6,  # ball left the hand
    "catch_reach_m": 0.5,  # ball caught within this distance
    "dribble_reach_m": 1.2,  # handler stays within this during dribble
    "possess_switch_frames": 5,  # handler switches only after N consistent frames
    "possess_hold_frames": 30,  # keep possession up to 1 s after losing touch
    "flight_min_seconds": 0.3,  # a flight shorter than this is not an event
    "pass_min_horizontal_m": 1.5,
    "shoot_hoop_distance_m": 2.0,  # min horizontal dist to hoop during flight
    "shoot_min_apex_m": 2.0,  # a real shot arcs above ~2 m
    "layup_release_dist_m": 1.6,  # release close to the hoop -> layup
    "layup_max_seconds": 0.8,
    "rebound_hoop_distance_m": 2.0,
    "hand_fallback_shoulder_drop_m": 0.3,
    # fallback dribble detection when the trajectory classifier missed it
    "dribble_fallback_max_z": 1.2,  # ball must run low
    "dribble_fallback_min_speed": 0.5,
    "dribble_fallback_confirm_frames": 5,
    # a flight segment followed by someone gaining the ball counts as a pass
    "pass_post_segment_frames": 12,
    "pass_post_segment_reach_m": 0.8,
    # shot result: ball crossing the rim plane (z down through 3.05 m) within
    # the rim circle -> make; also a make when the ball passed over the rim
    # centre at rim height (reconstruction distorts the crossing point)
    "make_rim_height_m": 3.05,
    "make_horizontal_reach_m": 0.3,
    "make_overhead_reach_m": 0.7,
    "make_landing_z_m": 2.9,
    "make_landing_reach_m": 1.0,
    "make_2d_min_z_m": 3.0,
    "make_fall_through_reach_m": 0.5,  # after flying over the rim, the ball
    # must fall THROUGH the opening
    "make_probe_frames": 40,
    # flight arcs: walk a flight segment backward while the ball is still in
    # the air so release/apex/rim-crossing are all covered
    "flight_extend_min_z_m": 1.4,
    "flight_extend_min_speed_m_s": 2.0,
    "flight_extend_held_reach_m": 0.9,
    # a sudden vertical pop at flight start = a tip/deflection, not a shot
    "deflect_z_jump_m": 0.8,
    # shoot vs layup from the shooter's hands at the release: two hands on the
    # ball raised above the shoulders = jump shot; one hand = layup
    "hand_on_ball_reach_m": 1.35,
    "hand_shoulder_lift_m": 0.25,
    "one_hand_near_m": 0.85,
    "one_hand_far_m": 1.25,
    # low passes the flight classifier misses (ground/bounce passes)
    "low_pass_release_speed_m_s": 3.0,
    "low_pass_hand_reach_m": 0.95,
    "low_pass_catch_reach_m": 0.9,
    "low_pass_max_z_m": 1.7,
    "low_pass_max_frames": 60,
    # follow-up (补篮): a short tip-arc at the rim right after a MISSED shot
    "follow_up_max_gap_frames": 90,
    "follow_up_max_seconds": 4.0,  # the arc includes the grab-and-raise phase
    "follow_up_max_release_dist_m": 2.0,
    # block (盖帽): a deflection with a defender's hand at the contact point
    "block_hand_reach_m": 0.5,
    "rebound_probe_frames": 40,
    "launch_min_z_m": 1.8,  # semantic arc scan: a ball reaching this height
    # was LAUNCHED (a dribble never gets that high)
    "shot_near_rim_reach_m": 1.2,
    "shot_approach_min_fraction": 0.15,
    "shot_min_rim_apex_m": 2.9,
    "three_point_radius_m": 6.54,
    "layup_max_release_dist_m": 1.5,
    "layup_drive_speed_m_s": 2.5,
    "layup_fallback_min_apex_m": 1.8,
    "dribble_fallback_leave_speed_m_s": 2.5,
    "carry_reach_m": 1.6,
    "carry_max_speed_m_s": 2.0,
    "carry_confirm_frames": 4,
}


@dataclass
class HandPositions:
    """frame -> player_id -> hand centre (3,), NaN where unavailable."""

    data: dict[int, dict[int, np.ndarray]]

    def nearest(self, frame: int, point: np.ndarray) -> tuple[Optional[int], float]:
        players = self.data.get(frame, {})
        if not players:
            return None, float("inf")
        best_id, best_d = None, float("inf")
        for player_id, hand in players.items():
            if not np.isfinite(hand).all():
                continue
            d = float(np.linalg.norm(hand - point))
            if d < best_d:
                best_id, best_d = player_id, d
        return best_id, best_d


def parse_hand_positions(poses_3d: dict) -> HandPositions:
    """poses_3d: frame -> player_id -> (17,3) with NaN for missing kpts."""
    data: dict[int, dict[int, np.ndarray]] = {}
    for key, players in poses_3d.items():
        frame = int(float(key))
        frame_hands: dict[int, np.ndarray] = {}
        for player_key, kpts in players.items():
            kpts = np.asarray(kpts, dtype=np.float64).reshape(17, 3)
            left, right = kpts[9], kpts[10]
            left_ok, right_ok = np.isfinite(left).all(), np.isfinite(right).all()
            if left_ok and right_ok:
                hand = (left + right) / 2.0
            elif left_ok:
                hand = left
            elif right_ok:
                hand = right
            else:
                left_s, right_s = kpts[5], kpts[6]
                if not (np.isfinite(left_s).all() and np.isfinite(right_s).all()):
                    continue
                hand = (left_s + right_s) / 2.0
                hand[2] -= DEFAULTS["hand_fallback_shoulder_drop_m"]
            frame_hands[int(player_key)] = hand
        if frame_hands:
            data[frame] = frame_hands
    return HandPositions(data)


# ---------------------------------------------------------------------------


@dataclass
class Action:
    type: str
    start_frame: int
    end_frame: int
    actor_id: Optional[int] = None
    receiver_id: Optional[int] = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        record: dict[str, Any] = {
            "type": self.type,
            "frame": self.start_frame,
            "end_frame": self.end_frame,
        }
        if self.actor_id is not None:
            record["actor_id"] = self.actor_id
        if self.receiver_id is not None:
            record["receiver_id"] = self.receiver_id
        record.update(
            {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in self.params.items()
            }
        )
        return record


class ActionRuleEngine:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def _cfg(self, key: str, default: Any = None) -> Any:
        if self.config is None:
            return DEFAULTS.get(key, default)
        if hasattr(self.config, "get"):
            try:
                return self.config.get(
                    f"action_rules.{key}", DEFAULTS.get(key, default)
                )
            except Exception:
                return DEFAULTS.get(key, default)
        section = (
            self.config.get("action_rules", {}) if isinstance(self.config, dict) else {}
        )
        return section.get(key, DEFAULTS.get(key, default))

    # ------------------------------------------------------------------ core

    # ------------------------------------------------------------------ motion

    def _radial_leave_speed(
        self, actor_id: int, start: int, hoop2: np.ndarray, poses_3d: dict, fps: float
    ) -> Optional[float]:
        """Player's radial speed away from/toward the hoop around release."""
        skel = poses_3d.get(str(start), {}).get(str(actor_id))
        if skel is None:
            return None
        k = np.asarray(skel, dtype=np.float64).reshape(17, 3)
        p0 = k[11] if np.isfinite(k[11]).all() else k[0]
        prev = poses_3d.get(str(max(start - 3, 0)), {}).get(str(actor_id))
        if prev is None:
            return None
        p1 = np.asarray(prev, dtype=np.float64).reshape(17, 3)[11]
        if not np.isfinite(p1).all():
            return None
        d0 = float(np.linalg.norm(p0[:2] - hoop2))
        d1 = float(np.linalg.norm(p1[:2] - hoop2))
        dt = 3.0 / fps
        return (d0 - d1) / dt if dt > 0 else None

    def _release_hand_feature(
        self, actor_id: int, start: int, ball_pos: dict | np.ndarray, poses_3d: dict
    ) -> Optional[str]:
        """One-hand vs two-hand release from the skeleton at the release frame."""
        skel = poses_3d.get(str(start), {}).get(str(actor_id))
        if skel is None:
            return None
        k = np.asarray(skel, dtype=np.float64).reshape(17, 3)
        if start not in ball_pos:
            return None
        ball = ball_pos[start]
        reach = float(self._cfg("hand_on_ball_reach_m", 1.35))
        near = 0
        for idx in (9, 10):
            if (
                np.isfinite(k[idx]).all()
                and float(np.linalg.norm(k[idx] - ball)) <= reach
            ):
                near += 1
        if near >= 2:
            return "two"
        if near == 1:
            return "one"
        return None

    def _approach_speed(
        self, actor_id: int, start: int, hoop2: np.ndarray, poses_3d: dict, fps: float
    ) -> Optional[float]:
        """Approach speed of the shooter toward the hoop over the frames
        before release — a driving layup closes distance quickly."""
        if start < 3:
            return None
        p_far = None
        for f in range(max(start - 10, 0), start + 1):
            skel = poses_3d.get(str(f), {}).get(str(actor_id))
            if skel is None:
                continue
            k = np.asarray(skel, dtype=np.float64).reshape(17, 3)
            foot = k[11] if np.isfinite(k[11]).all() else k[0]
            if not np.isfinite(foot).all():
                continue
            d = float(np.linalg.norm(foot[:2] - hoop2))
            if p_far is None or d > p_far[0]:
                p_far = (d, foot)
        if p_far is None:
            return None
        skel0 = poses_3d.get(str(start), {}).get(str(actor_id))
        if skel0 is None:
            return None
        k0 = np.asarray(skel0, dtype=np.float64).reshape(17, 3)
        foot0 = k0[11] if np.isfinite(k0[11]).all() else k0[0]
        if not np.isfinite(foot0).all():
            return None
        dt = (start - max(start - 10, 0)) / fps
        if dt <= 0:
            return None
        return (p_far[0] - float(np.linalg.norm(foot0[:2] - hoop2))) / dt

    def _shot_kind(
        self,
        release_actor: int,
        start: int,
        start_pos: np.ndarray,
        release_dist_hoop: float,
        duration_s: float,
        poses_3d: dict,
        fps: float,
        hoop2: np.ndarray,
    ) -> str:
        """shoot vs layup: a layup releases CLOSE to the rim while the
        shooter is DRIVING (approach speed >= threshold)."""
        near_rim = release_dist_hoop <= float(
            self._cfg("layup_max_release_dist_m", 1.5)
        )
        drive = self._approach_speed(release_actor, start, hoop2, poses_3d, fps)
        if (
            near_rim
            and drive is not None
            and drive >= float(self._cfg("layup_drive_speed_m_s", 2.5))
        ):
            return "layup"
        if near_rim and duration_s <= float(self._cfg("layup_max_seconds", 0.8)):
            hand = self._release_hand_feature(release_actor, start, start_pos, poses_3d)
            if hand == "one":
                return "layup"
        return "shoot"

    # ------------------------------------------------------------------ arcs

    def _scan_launch_arcs(
        self,
        ball_pos: dict[int, np.ndarray],
        ball_state: dict[int, str],
        hands: "HandPositions",
        grid: list[int],
        fps: float,
    ) -> list[dict]:
        """Semantic launch detection: a ball that rises to launch_min_z_m
        (~head height) and comes back down (or is caught) was LAUNCHED — the
        trajectory classifier's free-fall fit drops exactly these arcs when
        the ball hits the rim/backboard (the contact supports or deflects the
        ball, so it is not in free fall). Scanning the z profile finds them;
        the arc start is walked back to the release (the ball carried up in
        the shooter's hands during a drive). Merged with the classifier's
        flight segments, they go through the same classification."""
        min_z = float(self._cfg("launch_min_z_m", 1.8))
        end_z = min_z - 0.6
        arcs: list[dict] = []
        in_arc = False
        start = None
        prev_f = None
        for f in grid:
            if f not in ball_pos:
                if in_arc and prev_f is not None and f - prev_f > 8:
                    arcs.append(
                        {"type": "flight", "start_frame": start, "end_frame": prev_f}
                    )
                    in_arc, start = False, None
                continue
            prev_f = f
            z = float(ball_pos[f][2])
            if not in_arc:
                if z >= min_z:
                    in_arc, start = True, f
                continue
            if z < min_z:
                # ended: caught (2 frames within hand reach) or fell to the floor
                pid, d = hands.nearest(f, ball_pos[f])
                d_prev = float("inf")
                if f - 1 in ball_pos:
                    _, d_prev = hands.nearest(f - 1, ball_pos[f - 1])
                caught = pid is not None and d <= 0.5 and d_prev <= 0.5
                if caught or z < end_z:
                    arcs.append(
                        {"type": "flight", "start_frame": start, "end_frame": f}
                    )
                    in_arc, start = False, None
        if in_arc and start is not None and prev_f is not None:
            arcs.append({"type": "flight", "start_frame": start, "end_frame": prev_f})
        # walk each arc start back to the release (the carry in the shooter's
        # hands during a drive) — same walk as the flight-segment extension
        out: list[dict] = []
        for seg in arcs:
            s, e = int(seg["start_frame"]), int(seg["end_frame"])
            while s - 1 >= grid[0] and s - 1 in ball_pos and s in ball_pos:
                p = ball_pos[s - 1]
                _, d_now = hands.nearest(s - 1, p)
                d_prev = float("inf")
                if s - 2 in ball_pos:
                    _, d_prev = hands.nearest(s - 2, ball_pos[s - 2])
                speed = float(np.linalg.norm(ball_pos[s] - p)) * fps
                held2 = d_now <= 0.9 and d_prev <= 0.9
                if held2 or (p[2] < 1.2 and (speed < 2.0 or d_now <= 1.2)):
                    break
                s -= 1
            out.append({"type": "flight", "start_frame": s, "end_frame": e})
        return out

    def _extend_flight_segments(
        self,
        flight_segments: list[dict],
        ball_pos: dict,
        hands: "HandPositions",
        grid: list[int],
        fps: float,
        catch_frames: Optional[set] = None,
    ) -> list[dict]:
        """Walk each flight segment backward/forward while the ball is still
        in the air so release/apex/rim-crossing are all covered."""
        out: list[dict] = []
        for seg in flight_segments:
            s, e = int(seg["start_frame"]), int(seg["end_frame"])
            # extend backward to the release (ball carried up in the hands)
            min_z = float(self._cfg("flight_extend_min_z_m", 1.4))
            min_speed = float(self._cfg("flight_extend_min_speed_m_s", 2.0))
            held_reach = float(self._cfg("flight_extend_held_reach_m", 0.9))
            while s - 1 >= grid[0] and s - 1 in ball_pos and s in ball_pos:
                prev = ball_pos[s - 1]
                cur = ball_pos[s]
                v = float(np.linalg.norm(cur - prev) * fps)
                if prev[2] > min_z or v < min_speed:
                    break
                # stop when a hand holds the ball (release point)
                pid, d = hands.nearest(s - 1, prev)
                if pid is not None and d <= held_reach:
                    break
                s -= 1
            # extend forward while the ball keeps moving (rim contact bounce)
            while e + 1 < grid[-1] and e + 1 in ball_pos and e in ball_pos:
                cur = ball_pos[e + 1]
                prev = ball_pos[e]
                v = float(np.linalg.norm(cur - prev) * fps)
                if cur[2] > min_z + 0.5 and v > min_speed:
                    e += 1
                else:
                    break
            out.append({**seg, "start_frame": s, "end_frame": e})
        # merge overlapping extended segments
        merged: list[dict] = []
        for seg in out:
            if merged and int(seg["start_frame"]) <= int(merged[-1]["end_frame"]) + 1:
                merged[-1]["end_frame"] = max(
                    int(merged[-1]["end_frame"]), int(seg["end_frame"])
                )
            else:
                merged.append(dict(seg))
        return merged

    def _low_pass_events(
        self,
        ball_pos: dict,
        ball_state: dict,
        hands: "HandPositions",
        grid: list[int],
        fps: float,
    ) -> list[Action]:
        """Low passes the flight classifier misses (ground/bounce passes):
        the ball is released at speed from player A and settles near player B
        without ever reaching launch height."""
        events: list[Action] = []
        max_z = float(self._cfg("low_pass_max_z_m", 1.7))
        release_speed = float(self._cfg("low_pass_release_speed_m_s", 3.0))
        hand_reach = float(self._cfg("low_pass_hand_reach_m", 0.95))
        catch_reach_lp = float(self._cfg("low_pass_catch_reach_m", 0.9))
        max_frames = int(self._cfg("low_pass_max_frames", 60))
        in_run = False
        run_start = None
        prev_f = None
        prev_pos = None
        for f in grid:
            if f not in ball_pos:
                in_run, run_start, prev_f, prev_pos = False, None, None, None
                continue
            pos = ball_pos[f]
            if prev_f is not None and prev_pos is not None:
                v = float(np.linalg.norm(pos - prev_pos) * fps)
                if not in_run:
                    if (
                        pos[2] <= max_z
                        and v >= release_speed
                        and ball_state.get(f, "unknown") not in ("held",)
                    ):
                        in_run, run_start = True, f
                elif pos[2] > max_z:
                    in_run, run_start = False, None
                elif v < release_speed * 0.6:
                    # settled: ball decelerated — check whether a player
                    # received it, then close the run regardless.
                    pid, d = hands.nearest(f, pos)
                    if pid is not None and d <= catch_reach_lp:
                        rel = None
                        if run_start is not None and run_start - 1 in ball_pos:
                            rp, rd = hands.nearest(
                                run_start - 1, ball_pos[run_start - 1]
                            )
                            if rp is not None and rd <= hand_reach:
                                rel = rp
                        if rel is not None and rel != pid:
                            events.append(
                                Action(
                                    "pass",
                                    cast(int, run_start),
                                    f,
                                    actor_id=rel,
                                    receiver_id=pid,
                                    params={"low": True},
                                )
                            )
                    in_run, run_start = False, None
                elif f - cast(int, run_start) > max_frames:
                    in_run, run_start = False, None
            prev_f, prev_pos = f, pos
        return events

    def process_data(
        self,
        ball_traj: dict,
        poses_3d: dict,
        hoop_3d: dict,
    ) -> dict:
        frames_out = ball_traj["frames"]
        segments = ball_traj.get("segments", [])
        flight_segments = [s for s in segments if s.get("type") == "flight"]
        # Merge adjacent flight segments (a dropped frame can split one
        # flight arc into two segments — classification must see the whole arc)
        max_split_gap = int(self._cfg("flight_merge_gap_frames", 5))
        if flight_segments:
            # Sort by start (longest first) so a child segment nested inside
            # a parent arc (trajectory split artifact) merges into the parent
            # instead of spawning a duplicate shot event.
            flight_segments = sorted(
                flight_segments,
                key=lambda s: (int(s["start_frame"]), -int(s["end_frame"])),
            )
            merged: list[dict] = [dict(flight_segments[0])]
            for seg in flight_segments[1:]:
                if (
                    int(seg["start_frame"])
                    <= int(merged[-1]["end_frame"]) + max_split_gap
                ):
                    merged[-1]["end_frame"] = max(
                        int(merged[-1]["end_frame"]), int(seg["end_frame"])
                    )
                else:
                    merged.append(dict(seg))
            flight_segments = merged

        hoop = np.asarray(hoop_3d["hoop_center"], dtype=np.float64)
        hands = parse_hand_positions(poses_3d)
        ctx_2d = None  # optional per-view 2D rim context (loaded when available)

        grid = sorted(int(k) for k in frames_out)
        if not grid:
            raise ValueError("empty ball trajectory")

        hold_reach = float(self._cfg("hold_reach_m"))
        release_dist = float(self._cfg("release_distance_m"))
        catch_reach = float(self._cfg("catch_reach_m"))
        dribble_reach = float(self._cfg("dribble_reach_m"))
        switch_frames = int(self._cfg("possess_switch_frames"))
        hold_frames = int(self._cfg("possess_hold_frames"))

        ball_pos: dict[int, np.ndarray] = {}
        ball_state: dict[int, str] = {}
        ball_vel: dict[int, np.ndarray] = {}
        for key, record in frames_out.items():
            frame = int(key)
            if record.get("position") is not None:
                ball_pos[frame] = np.asarray(record["position"])
            if record.get("velocity") is not None:
                ball_vel[frame] = np.asarray(record["velocity"])
            ball_state[frame] = record.get("state", "unknown")

        actions: list[Action] = []
        possession: dict[int, Optional[int]] = {}
        handler: Optional[int] = None
        handler_missing = 0
        candidate: Optional[int] = None
        candidate_frames = 0
        _last_contact_frame = -(10**6)
        prev_hands: Optional[dict[int, np.ndarray]] = None
        prev_hands_history: list[
            tuple[int, dict[int, np.ndarray]]
        ] = []  # (frame, hands)
        dribble_tracking: Optional[int] = None  # fallback dribble in progress
        pending_catches: list[tuple[int, int]] = []  # (frame, player_id)

        def dist_to_player(frame: int, player_id: int) -> float:
            players = hands.data.get(frame, {})
            hand = players.get(player_id)
            if hand is None or not np.isfinite(hand).all():
                return float("inf")
            return float(np.linalg.norm(hand - ball_pos.get(frame, hand)))

        def prev_max_dist(player_id: int, lookback: int = 3) -> float:
            """Max ball-to-hand distance over the last few frames (using each
            historical frame's own ball position) — if ANY recent frame had
            the ball far from the hand, the previous contact was broken."""
            best = float("inf")
            for f, h in prev_hands_history[-lookback:]:
                hand = h.get(player_id)
                if hand is None or not np.isfinite(hand).all():
                    continue
                if f in ball_pos:
                    best = min(best, float(np.linalg.norm(hand - ball_pos[f])))
            return best

        for frame in grid:
            state = ball_state.get(frame, "unknown")
            point = None
            nearest_id, nearest_d = None, float("inf")
            if frame in ball_pos:
                point = ball_pos[frame]
                nearest_id, nearest_d = hands.nearest(frame, point)

            if state == "held" and point is not None and nearest_d <= hold_reach:
                # ball in hands. A pickup or a change of hands is a catch
                # (not on the very first frame of the clip). A change of
                # hands WITHOUT a flight in between is a handoff pass.
                prev_d = (
                    prev_max_dist(cast(int, nearest_id))
                    if prev_hands_history
                    else float("inf")
                )
                if handler != nearest_id and prev_d > hold_reach and prev_hands_history:
                    actions.append(Action("catch", frame, frame, actor_id=nearest_id))
                    if handler is not None and not any(
                        ball_state.get(f, "") == "flight"
                        for f in range(max(frame - 10, 0), frame)
                    ):
                        actions.append(
                            Action(
                                "pass",
                                frame,
                                frame,
                                actor_id=handler,
                                receiver_id=nearest_id,
                                params={"handoff": True},
                            )
                        )
                handler = nearest_id
                handler_missing = 0
                _last_contact_frame = frame
            elif (
                state == "dribble" and point is not None and nearest_d <= dribble_reach
            ):
                # dribbling: switch handler only after N consistent frames
                if candidate == nearest_id:
                    candidate_frames += 1
                else:
                    candidate, candidate_frames = nearest_id, 1
                if candidate_frames >= switch_frames:
                    if handler is not None and handler != candidate:
                        actions.append(
                            Action("catch", frame, frame, actor_id=candidate)
                        )
                    handler = candidate
                    handler_missing = 0
                    _last_contact_frame = frame
            elif state == "flight":
                candidate, candidate_frames = None, 0
                if handler is not None:
                    # the ball leaves the handler's hands -> release
                    if dist_to_player(frame, handler) > release_dist:
                        actions.append(
                            Action("release", frame, frame, actor_id=handler)
                        )
                        handler = None
                # a catch is the ball ENTERING someone's hands from afar and
                # STAYING (the next frame must not show the ball leaving again)
                if (
                    point is not None
                    and nearest_d <= catch_reach
                    and prev_hands is not None
                    and nearest_id is not None
                ):
                    prev_hand = prev_hands.get(nearest_id)
                    prev_d = float("inf")
                    if prev_hand is not None and np.isfinite(prev_hand).all():
                        prev_d = float(np.linalg.norm(prev_hand - point))
                    if prev_d > catch_reach + 0.1:
                        pending_catches.append((frame, nearest_id))
            else:
                # ground / unknown: keep the handler for a while, and detect
                # dribbling the trajectory classifier may have missed (ball
                # running low near the same player while moving).
                fallback_dribble = (
                    state in ("unknown", "ground")
                    and point is not None
                    and nearest_id is not None
                    and nearest_d <= dribble_reach
                    and point[2] < float(self._cfg("dribble_fallback_max_z"))
                    and ball_vel.get(frame) is not None
                    and float(np.linalg.norm(ball_vel[frame]))
                    > float(self._cfg("dribble_fallback_min_speed"))
                )
                if fallback_dribble:
                    if candidate == nearest_id:
                        candidate_frames += 1
                    else:
                        candidate, candidate_frames = nearest_id, 1
                    if candidate_frames >= int(
                        self._cfg("dribble_fallback_confirm_frames")
                    ):
                        if dribble_tracking is None:
                            actions.append(
                                Action(
                                    "dribble_start",
                                    frame,
                                    frame,
                                    actor_id=nearest_id,
                                    params={"fallback": True},
                                )
                            )
                            dribble_tracking = nearest_id
                        if handler != candidate:
                            prev_d = (
                                prev_max_dist(candidate)
                                if prev_hands_history
                                else float("inf")
                            )
                            if prev_d > hold_reach and not any(
                                ball_state.get(f, "") == "flight"
                                for f in range(max(frame - 10, 0), frame)
                            ):
                                actions.append(
                                    Action(
                                        "pass",
                                        frame,
                                        frame,
                                        actor_id=handler,
                                        receiver_id=candidate,
                                        params={"handoff": True},
                                    )
                                )
                        handler = candidate
                        handler_missing = 0
                        _last_contact_frame = frame
                else:
                    candidate, candidate_frames = None, 0
                    if dribble_tracking is not None:
                        actions.append(
                            Action(
                                "dribble_end", frame, frame, actor_id=dribble_tracking
                            )
                        )
                        dribble_tracking = None
                    handler_missing += 1
                    if handler_missing > hold_frames:
                        handler = None

            # Confirm pending catches 2 frames later: the ball must have
            # STAYED near that player (a fly-by within reach is not a catch).
            remaining: list[tuple[int, int]] = []
            for pf, pid in pending_catches:
                if frame >= pf + 2:
                    if point is not None and np.isfinite(point).all():
                        players = hands.data.get(frame, {})
                        hand = players.get(pid)
                        if hand is not None and np.isfinite(hand).all():
                            d_now = float(np.linalg.norm(hand - point))
                            already = any(
                                a.type == "catch"
                                and a.actor_id == pid
                                and pf <= a.start_frame <= pf + 2
                                for a in actions
                            )
                            if d_now <= catch_reach * 1.2 and not already:
                                actions.append(Action("catch", pf, pf, actor_id=pid))
                                handler = pid
                                handler_missing = 0
                                _last_contact_frame = pf
                                continue
                remaining.append((pf, pid))
            pending_catches = remaining

            possession[frame] = handler
            prev_hands = hands.data.get(frame)
            if prev_hands is not None:
                prev_hands_history.append((frame, prev_hands))
                if len(prev_hands_history) > 8:
                    prev_hands_history.pop(0)

        # ---- flight-segment events (pass / shoot / layup / steal / rebound) ----
        # semantic launch arcs: the classifier's free-fall fit drops arcs that
        # hit the rim/backboard — scan the z profile so those shots still
        # count. Arcs the classifier already covers are dropped (duplicates).
        fps = float(ball_traj.get("source", {}).get("fps", 30.0))
        scanned = self._scan_launch_arcs(ball_pos, ball_state, hands, grid, fps)
        seg_ranges = [
            (int(s["start_frame"]), int(s["end_frame"])) for s in flight_segments
        ]
        for arc in scanned:
            s, e = int(arc["start_frame"]), int(arc["end_frame"])
            covered = sum(max(0, min(e, se) - max(s, ss) + 1) for ss, se in seg_ranges)
            if covered < 0.6 * (e - s + 1):
                flight_segments.append(arc)
        catch_frames = {a.start_frame for a in actions if a.type == "catch"}
        flight_segments = self._extend_flight_segments(
            flight_segments, ball_pos, hands, grid, fps, catch_frames=catch_frames
        )
        for seg in flight_segments:
            start, end = int(seg["start_frame"]), int(seg["end_frame"])
            frames_in = [f for f in range(start, end + 1) if f in ball_pos]
            if len(frames_in) < 2:
                continue
            points = np.asarray([ball_pos[f] for f in frames_in])
            duration_s = (end - start) / fps
            if duration_s < float(self._cfg("flight_min_seconds")):
                continue
            apex_z = float(np.max(points[:, 2]))
            start_pos = points[0]
            end_pos = points[-1]
            horizontal_displacement = float(np.linalg.norm(end_pos[:2] - start_pos[:2]))
            hoop_horizontal = float(
                np.min(np.linalg.norm(points[:, :2] - hoop[:2], axis=1))
            )
            release_dist_hoop = float(np.linalg.norm(start_pos[:2] - hoop[:2]))

            # who released and who caught
            release_actor = possession.get(start - 1) or possession.get(start)
            if release_actor is not None:
                # pose noise can leave a stale handler (e.g. the carry
                # fallback); if the handler's hand is not near the ball at the
                # release, re-probe the hands directly.
                hand = hands.data.get(start, {}).get(release_actor)
                if hand is None:
                    hand = hands.data.get(start - 1, {}).get(release_actor)
                if hand is None or not np.isfinite(hand).all():
                    release_actor = None
                elif (
                    start in ball_pos
                    and float(np.linalg.norm(hand - ball_pos[start]))
                    > release_dist * 2.5
                ):
                    release_actor = None
            if release_actor is None:
                # catch-and-shoot: the ball leaves the handler right after the
                # previous possession; probe the few frames before the segment
                # (the ball is still near the shooter's hands when released).
                for probe in range(max(start - 8, 0), start + 1):
                    if probe not in ball_pos:
                        continue
                    pid, d = hands.nearest(probe, ball_pos[probe])
                    if d <= release_dist * 2.5:
                        release_actor = pid
                        break
            catch_actor, catch_frame, catch_d = None, None, float("inf")
            for f in frames_in:
                pid, d = hands.nearest(f, ball_pos[f])
                if pid == release_actor:
                    continue  # the releaser's own hand near the ball is the
                    # carry/release contact, not a catch
                if d < catch_d:
                    catch_actor, catch_frame, catch_d = pid, f, d

            action: Optional[Action] = None
            near_hoop = hoop_horizontal < float(self._cfg("shoot_hoop_distance_m"))
            near_rim = hoop_horizontal <= float(self._cfg("shot_near_rim_reach_m"))
            min_hoop_idx = int(
                np.argmin(np.linalg.norm(points[:, :2] - hoop[:2], axis=1))
            )
            approaches_hoop = min_hoop_idx >= max(
                2, int(float(self._cfg("shot_approach_min_fraction")) * len(frames_in))
            )
            deflected = start - 1 in ball_pos and abs(
                float(ball_pos[start][2]) - float(ball_pos[start - 1][2])
            ) > float(self._cfg("deflect_z_jump_m"))
            if deflected:
                # a sudden vertical pop at the arc start is a tip/deflection
                # (抢球), not a shot — a ball leaves a player's hands smoothly.
                # BLOCK: a deflection while the ball was on a shot toward the
                # hoop AND a defender's hand is at the contact point.
                action = None
                if (
                    release_actor is not None
                    and apex_z >= float(self._cfg("shoot_min_apex_m"))
                    and near_rim
                ):
                    for f in frames_in[:6]:
                        pid, d = hands.nearest(f, ball_pos[f])
                        if (
                            pid is not None
                            and pid != release_actor
                            and d <= float(self._cfg("block_hand_reach_m", 0.5))
                        ):
                            action = Action(
                                "block",
                                start,
                                end,
                                actor_id=pid,
                                params={
                                    "after": "deflection",
                                    "apex_z": apex_z,
                                    "hoop_distance_m": hoop_horizontal,
                                    "duration_s": duration_s,
                                },
                            )
                            break
                if action is None:
                    # otherwise a deflection a TEAMMATE cleanly catches is a
                    # scoop pass; a contested ball emits nothing.
                    catcher, cframe, cd = None, None, float("inf")
                    for f in frames_in:
                        pid, d = hands.nearest(f, ball_pos[f])
                        if d < cd:
                            catcher, cframe, cd = pid, f, d
                    if catcher is None or cd > catch_reach * 2.0:
                        for probe in range(
                            end + 1,
                            min(
                                end + 1 + int(self._cfg("pass_post_segment_frames")),
                                grid[-1] + 1,
                            ),
                        ):
                            if probe not in ball_pos:
                                continue
                            pid, d = hands.nearest(probe, ball_pos[probe])
                            if (
                                pid is not None
                                and pid != release_actor
                                and d <= float(self._cfg("pass_post_segment_reach_m"))
                            ):
                                catcher, cframe, cd = pid, probe, d
                                break
                    if catcher is not None and catcher != release_actor:
                        action = Action(
                            "pass",
                            start,
                            cframe or end,
                            actor_id=release_actor,
                            receiver_id=catcher,
                            params={
                                "apex_z": apex_z,
                                "horizontal_displacement_m": horizontal_displacement,
                                "duration_s": duration_s,
                                "scoop": True,
                            },
                        )

            # follow-up (补篮): a short tip-arc at the rim right after a
            # MISSED shot, without the ball having been controlled in between
            # Put-backs often start from the rim pop (no hand release detected),
            # so when release_actor is missing, fall back to the most recent
            # catch/rebound actor before the arc.
            fu_actor = release_actor
            if fu_actor is None:
                for prev_a in reversed(actions):
                    if (
                        prev_a.type in ("catch", "rebound")
                        and prev_a.end_frame <= start
                    ):
                        fu_actor = prev_a.actor_id
                        break
            follow_up = False
            if (
                fu_actor is not None
                and apex_z >= float(self._cfg("shoot_min_apex_m"))
                and release_dist_hoop
                <= float(self._cfg("follow_up_max_release_dist_m"))
                and duration_s <= float(self._cfg("follow_up_max_seconds"))
            ):
                for prev_a in reversed(actions):
                    if prev_a.type in ("shoot", "layup") and prev_a.end_frame < start:
                        gap = start - prev_a.end_frame
                        # a quick put-back right after a MISSED shot is a
                        # follow-up even when the ball was briefly grabbed in
                        # between (grab-and-putback still counts as 补篮).
                        # A DRIBBLE in between means possession restarted —
                        # that arc is a new play, not a follow-up. Excluded:
                        #  - fallback dribbles (low-confidence, dropped later)
                        #  - dribble starts adjacent to a rebound/catch (the
                        #    rim-pop misclassified as dribbling)
                        dribbled = any(
                            a2.type == "dribble_start"
                            and not a2.params.get("fallback")
                            and prev_a.end_frame <= a2.start_frame <= start
                            and not any(
                                a3.type in ("rebound", "catch")
                                and abs(a3.start_frame - a2.start_frame) <= 15
                                for a3 in actions
                            )
                            for a2 in actions
                        )
                        if (
                            prev_a.params.get("result") == "miss"
                            and gap <= int(self._cfg("follow_up_max_gap_frames"))
                            and not dribbled
                        ):
                            follow_up = True
                        break

            if follow_up:
                action = Action(
                    "follow_up",
                    start,
                    end,
                    actor_id=fu_actor,
                    params={
                        "apex_z": apex_z,
                        "hoop_distance_m": hoop_horizontal,
                        "release_hoop_distance_m": release_dist_hoop,
                        "duration_s": duration_s,
                    },
                )
                action.params["result"] = _shot_result(
                    ball_pos, frames_in, end, hoop, self._cfg, ctx_2d=ctx_2d
                )
            elif deflected:
                action = None

            elif (
                (
                    near_hoop
                    and near_rim
                    or release_dist_hoop >= float(self._cfg("three_point_radius_m"))
                )
                and approaches_hoop
                and (
                    hoop_horizontal
                    <= (
                        float(self._cfg("shot_high_arc_miss_distance_m", 1.6))
                        if apex_z >= 3.5
                        else float(self._cfg("shot_max_miss_distance_m", 0.7))
                    )
                )
                and apex_z > float(self._cfg("shoot_min_apex_m"))
                and release_actor is not None
                # a caught arc is a pass UNLESS it crossed the rim (shot
                # rebounded into a teammate's hands) or was released from
                # behind the three-point arc (nobody passes from there to
                # nobody — it is a shot attempt that fell short/wide). The
                # 2D signal additionally rejects arcs caught right at the
                # rim mouth (ball stops in the hoop's 2D projection).
                and (
                    apex_z >= float(self._cfg("shot_min_rim_apex_m", 2.9))
                    or not (
                        catch_actor is not None
                        and catch_actor != release_actor
                        and catch_d <= 0.5
                    )
                    or release_dist_hoop >= float(self._cfg("three_point_radius_m"))
                )
            ):
                # shoot / layup — requires someone who released the ball (a
                # ball falling from the rim without a shooter is NOT a shot),
                # the arc to cross near the rim, and the ball to travel TOWARD
                # the hoop (a toss leaving the rim outward is a pass). The
                # release motion (one hand vs two hands) picks layup vs shoot.
                kind = self._shot_kind(
                    release_actor,
                    start,
                    start_pos,
                    release_dist_hoop,
                    duration_s,
                    poses_3d,
                    fps,
                    hoop[:2],
                )
                # three-pointer: the ball came from behind the measured arc
                # (radius 6.54 m from the rim centre) — use the ball's
                # FARTHEST point; probe a short window before the arc start.
                arc_probe = [
                    float(np.linalg.norm(ball_pos[f][:2] - hoop[:2]))
                    for f in range(max(start - 6, 0), end + 1)
                    if f in ball_pos
                ]
                is_three = (
                    kind == "shoot"
                    and bool(arc_probe)
                    and max(arc_probe) >= float(self._cfg("three_point_radius_m"))
                )
                action = Action(
                    kind,
                    start,
                    end,
                    actor_id=release_actor,
                    params={
                        "apex_z": apex_z,
                        "hoop_distance_m": hoop_horizontal,
                        "release_hoop_distance_m": release_dist_hoop,
                        "duration_s": duration_s,
                        "three_point": is_three,
                    },
                )

            if action is not None and action.type in ("shoot", "layup"):
                action.params["result"] = _shot_result(
                    ball_pos, frames_in, end, hoop, self._cfg, ctx_2d=ctx_2d
                )
                # rebound: after a MISS, someone gains the ball near the hoop
                # within a short probe window (the rim contact can delay the
                # grab a few frames past the flight segment).
                if action.params.get("result") == "miss":
                    rb_actor, rb_frame = None, None

                    # A rebound is an AIR grab near the rim: the ball must be
                    # falling (not still rising), still airborne, and close to
                    # the hoop. A shot that missed wide and is picked up on
                    # the floor is not a rebound.
                    def _air_grab(f: int) -> bool:
                        if f not in ball_pos:
                            return False
                        return float(ball_pos[f][2]) >= 0.8 and float(
                            np.linalg.norm(ball_pos[f][:2] - hoop[:2])
                        ) <= float(self._cfg("rebound_hoop_distance_m", 1.2))

                    # in-segment catch near the segment end (ball falling off
                    # the rim) — a catch during the rising arc is a contest
                    if (
                        catch_actor is not None
                        and catch_actor != release_actor
                        and catch_frame is not None
                        and catch_frame >= end - 15
                        and catch_d <= float(self._cfg("catch_reach_m", 0.5)) * 1.5
                        and _air_grab(catch_frame)
                    ):
                        rb_actor, rb_frame = catch_actor, catch_frame
                    else:
                        # post-segment probe: ball still airborne near the rim
                        for probe_f in range(
                            end + 1,
                            end + 1 + int(self._cfg("rebound_probe_frames", 40)),
                        ):
                            if probe_f not in ball_pos:
                                continue
                            if float(ball_pos[probe_f][2]) < 0.4:
                                break  # ball hit the floor — no rebound after
                            pid, d = hands.nearest(probe_f, ball_pos[probe_f])
                            if (
                                pid is not None
                                and pid != release_actor
                                and d <= float(self._cfg("catch_reach_m", 0.5)) * 1.5
                                and _air_grab(probe_f)
                            ):
                                rb_actor, rb_frame = pid, probe_f
                                break
                    if rb_actor is not None and rb_actor != release_actor:
                        actions.append(
                            Action(
                                "rebound",
                                cast(int, rb_frame),
                                cast(int, rb_frame),
                                actor_id=rb_actor,
                                params={"after": "shoot"},
                            )
                        )

            elif action is None and (
                catch_actor is not None
                and catch_actor != release_actor
                and catch_d <= catch_reach * 2.0
                and horizontal_displacement > float(self._cfg("pass_min_horizontal_m"))
            ):
                action = Action(
                    "pass",
                    start,
                    end,
                    actor_id=release_actor,
                    receiver_id=catch_actor,
                    params={
                        "apex_z": apex_z,
                        "horizontal_displacement_m": horizontal_displacement,
                        "duration_s": duration_s,
                    },
                )
            elif (
                action is None
                and near_hoop
                and release_actor is not None
                and apex_z >= float(self._cfg("layup_min_apex_m", 1.5))
                and release_dist_hoop <= float(self._cfg("layup_release_dist_m", 2.5))
            ):
                # a real arc released close to the hoop with no receiver ->
                # layup attempt (a flat ground pass near the hoop is not one)
                action = Action(
                    "layup",
                    start,
                    end,
                    actor_id=release_actor,
                    params={
                        "apex_z": apex_z,
                        "hoop_distance_m": hoop_horizontal,
                        "duration_s": duration_s,
                    },
                )
                action.params["result"] = _shot_result(
                    ball_pos, frames_in, end, hoop, self._cfg, ctx_2d=ctx_2d
                )

            if action is not None:
                actions.append(action)

        # low passes the flight classifier misses (ground/bounce passes)
        for lp in self._low_pass_events(ball_pos, ball_state, hands, grid, fps):
            actions.append(lp)

        actions.sort(key=lambda a: (a.start_frame, a.type))
        stats = {
            "actions": len(actions),
            "by_type": _count_by_type(actions),
            "frames_with_possession": sum(
                1 for v in possession.values() if v is not None
            ),
        }
        return {
            "schema_version": "actions/v1",
            "actions": [a.to_dict() for a in actions],
            # per-frame handler (the possession state machine's current
            # player) — used by visualizations to draw the handler box
            "possession": {
                str(k): v for k, v in sorted(possession.items()) if v is not None
            },
            "stats": stats,
        }

    def process(
        self,
        ball_traj_path: Path,
        poses_3d_path: Path,
        hoop_3d_path: Path,
        output_path: Optional[Path] = None,
    ) -> dict:
        """File-based entry: load inputs, run process_data, optionally write."""
        with open(ball_traj_path, encoding="utf-8") as handle:
            ball_traj = json.load(handle)
        with open(poses_3d_path, encoding="utf-8") as handle:
            poses = json.load(handle)
        with open(hoop_3d_path, encoding="utf-8") as handle:
            hoop = json.load(handle)
        result = self.process_data(ball_traj, poses.get("poses_3d", {}), hoop)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=1)
            print(f"[ok] actions -> {output_path}")
        return result


def _count_by_type(actions: list[Action]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        counts[action.type] = counts.get(action.type, 0) + 1
    return counts


def _shot_result(
    ball_pos: dict[int, np.ndarray],
    segment_frames: list[int],
    segment_end: int,
    hoop: np.ndarray,
    cfg: Any,
    ctx_2d: Optional[dict] = None,
) -> str:
    """Make/miss: the ball crosses the rim plane (z down through ~3.05 m)
    within the rim circle. Probes the flight segment plus a few frames after."""
    rim_height = float(cfg("make_rim_height_m", 3.05))
    reach = float(cfg("make_horizontal_reach_m", 0.6))
    probe = int(cfg("make_probe_frames", 40))
    probe_frames = sorted(
        set(segment_frames)
        | {f for f in range(segment_end + 1, segment_end + 1 + probe) if f in ball_pos}
    )
    for index in range(len(probe_frames) - 1):
        f0, f1 = probe_frames[index], probe_frames[index + 1]
        z0 = float(ball_pos[f0][2])
        z1 = float(ball_pos[f1][2])
        drop = z0 - z1
        # crossing pair: the frame pair straddles the rim plane while
        # descending (the +0.03 margin previously let a frame at z=3.052
        # fall through the crack and miss clean makes)
        if z0 > rim_height and z1 <= rim_height and drop > 0.05:
            # interpolate the crossing point
            t = float(np.clip((rim_height - z0) / max(1e-9, z1 - z0), 0.0, 1.0))
            xy = ball_pos[f0][:2] + t * (ball_pos[f1][:2] - ball_pos[f0][:2])
            if float(np.linalg.norm(xy - hoop[:2])) <= reach:
                return "make"
    # fallback: flew over the rim centre, then fell THROUGH the rim mouth —
    # every frame while descending through the rim zone (z in the rim band)
    # must stay inside the rim opening.
    fall_through = float(cfg("make_fall_through_reach_m", 0.5))
    overhead = float(cfg("make_overhead_reach_m", 0.7))
    over_rim = [
        f
        for f in probe_frames
        if float(ball_pos[f][2]) >= rim_height - 0.02
        and float(np.linalg.norm(ball_pos[f][:2] - hoop[:2])) <= overhead
    ]
    if over_rim:
        # band = frames right after the rim overpass while the ball keeps
        # falling through the mouth (z 3.05 -> 2.5, monotonic down). Later
        # frames of unrelated arcs (ball passed out, next play) must not
        # enter the band.
        band: list[int] = []
        prev_z = None
        for f in probe_frames:
            if f <= over_rim[-1]:
                continue
            z = float(ball_pos[f][2])
            if z < 2.5 or z > rim_height + 0.05:
                break
            if prev_z is not None and z > prev_z + 0.01:
                break
            band.append(f)
            prev_z = z
        fall_through_ok = bool(band) and all(
            float(np.linalg.norm(ball_pos[f][:2] - hoop[:2])) <= fall_through
            for f in band
        )
        if fall_through_ok:
            return "make"
    return "miss"


def _shot_result_2d(
    ctx_2d: dict,
    ball_pos: dict,
    segment_frames: list[int],
    segment_end: int,
    hoop: np.ndarray,
    probe: int,
) -> str:
    """2D rim-line crossing (open-source style): in ANY view the ball
    descends through the hoop's 2D rim line within the rim's 2D width."""
    import numpy as _np

    for view, rec in ctx_2d.items():
        line = rec.get("rim_line")
        if not line:
            continue
        (x1, y1), (x2, y2) = line
        for i in range(len(segment_frames) - 1):
            f0, f1 = segment_frames[i], segment_frames[i + 1]
            if f0 not in ball_pos or f1 not in ball_pos:
                continue
            p0, p1 = ball_pos[f0], ball_pos[f1]
            # project both to this view
            P = rec.get("P")
            if P is None:
                continue
            pr0 = P @ _np.array([*p0, 1.0])
            pr1 = P @ _np.array([*p1, 1.0])
            if pr0[2] <= 1e-9 or pr1[2] <= 1e-9:
                continue
            u0, v0 = pr0[:2] / pr0[2]
            u1, v1 = pr1[:2] / pr1[2]
            # segment-line intersection
            dx, dy = u1 - u0, v1 - v0
            lx, ly = x2 - x1, y2 - y1
            denom = dx * (-ly) - (-lx) * dy
            if abs(denom) < 1e-9:
                continue
            t = ((u0 - x1) * (-ly) - (-lx) * (v0 - y1)) / denom
            if 0 <= t <= 1:
                # ball z near rim plane at crossing
                zc = p0[2] + t * (p1[2] - p0[2])
                if zc >= float(_cfg_stub("make_2d_min_z_m", 3.0)):
                    return "make"
    return "miss"


def _cfg_stub(key: str, default: Any) -> Any:
    return DEFAULTS.get(key, default)


def _load_2d_context(self, ball_traj_path: Path) -> Optional[dict]:
    """Load per-view 2D rim context for the open-source 2D make check."""
    try:
        ctx_path = ball_traj_path.parent / "hoop_2d_context.json"
        if not ctx_path.exists():
            return None
        import json as _json

        return _json.load(open(ctx_path, encoding="utf-8"))
    except Exception:
        return None
