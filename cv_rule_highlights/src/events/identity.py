"""Anonymous identities built only from automatically detected video tracks."""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..core.timeline import source_frame


def _torso_histogram(crop):
    if crop is None or crop.size == 0:
        return None
    height, width = crop.shape[:2]
    torso = crop[int(.18 * height):int(.70 * height),
                 int(.18 * width):int(.82 * width)]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    hist = cv2.GaussianBlur(hist, (3, 3), .8)
    norm = float(np.linalg.norm(hist))
    return (hist / norm).reshape(-1) if norm else None


class IdentityRegistry:
    """Conservatively reuse a person ID across segments; keep ambiguous tracks apart."""

    def __init__(self):
        self.players = {}
        self.track_to_player = {}
        self._next_id = 1

    def _new_player(self):
        result = f"anon_{self._next_id:04d}"
        self._next_id += 1
        return result

    def add_game(self, game, segment_id, output):
        portraits = Path(output) / "portraits"
        portraits.mkdir(parents=True, exist_ok=True)
        candidates = defaultdict(list)
        spans = defaultdict(lambda: [None, None, 0])
        observed_intervals = defaultdict(list)
        for key in sorted(game.data["poses_2d"], key=int):
            players = game.data["poses_2d"][key]
            frame = int(key)
            for track, observations in players.items():
                span = spans[str(track)]
                span[0] = frame if span[0] is None else min(frame, span[0])
                span[1] = frame if span[1] is None else max(frame, span[1])
                span[2] += 1
                intervals = observed_intervals[str(track)]
                if intervals and frame - intervals[-1][1] <= 45:
                    intervals[-1][1] = frame + 1
                else:
                    intervals.append([frame, frame + 1])
                if frame % 150:
                    continue
                for view, obs in observations.items():
                    bbox = obs["bbox"]
                    area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                    score = area * float(obs.get("detection_confidence", .5))
                    if score > 500:
                        candidates[str(track)].append((score, frame, view, bbox))
        captures = {}
        local = []
        try:
            for track in sorted(spans, key=lambda key: int(key) if key.isdigit() else key):
                ranked = sorted(candidates[track], reverse=True)
                view_counts = defaultdict(int)
                for _, _, view, _ in ranked:
                    view_counts[view] += 1
                preferred_view = max(view_counts, key=view_counts.get) if view_counts else None
                ranked = [item for item in ranked if item[2] == preferred_view]
                # Sample the whole segment, so one clear minute cannot define
                # an identity for all later footage after an ID swap.
                begin, last, _ = spans[track]
                width = max(1, (last - begin + 1) // 6)
                stratified = []
                for bin_id in range(6):
                    bucket = [item for item in ranked
                              if min(5, (item[1] - begin) // width) == bin_id]
                    if bucket:
                        stratified.append(bucket[0])
                ranked = stratified + [item for item in ranked if item not in stratified]
                samples = []
                used_frames = []
                for _, frame, view, bbox in ranked:
                    if len(samples) >= 6:
                        break
                    if any(abs(frame - previous) < 120 for previous in used_frames):
                        continue
                    if view not in captures:
                        captures[view] = cv2.VideoCapture(game.views[view]["path"])
                    cap = captures[view]
                    cap.set(cv2.CAP_PROP_POS_FRAMES,
                            source_frame(frame, game.views[view]["frame_zero"]))
                    ok, image = cap.read()
                    if not ok:
                        continue
                    height, width = image.shape[:2]
                    x1, y1, x2, y2 = map(round, bbox)
                    crop = image[max(0, y1):min(height, y2),
                                 max(0, x1):min(width, x2)]
                    hist = _torso_histogram(crop)
                    if hist is None:
                        continue
                    samples.append(hist)
                    used_frames.append(frame)
                    if len(samples) == 1:
                        cv2.imwrite(str(portraits / f"{segment_id}_track_{track}.jpg"), crop)
                representative = None
                if samples:
                    representative = np.median(np.stack(samples), axis=0)
                    representative /= max(float(np.linalg.norm(representative)), 1e-9)
                consistency = (float(np.median([np.dot(sample, representative)
                                                for sample in samples]))
                               if samples else None)
                local.append((track, representative, spans[track], len(samples),
                              consistency, preferred_view))
        finally:
            for cap in captures.values():
                cap.release()

        # Conservative greedy one-to-one assignment. Similar clothing alone
        # is insufficient when several players have near-identical uniforms.
        previous = [(pid, item) for pid, item in self.players.items()
                    if item["last_segment"] != segment_id and item["hist"] is not None]
        scores = []
        for track, hist, _, sample_count, _, _ in local:
            if hist is None or sample_count < 2:
                continue
            ranked = sorted(
                ((float(np.dot(hist, item["hist"])), pid)
                 for pid, item in previous),
                reverse=True,
            )
            if ranked and ranked[0][0] >= .94 and (
                len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= .08
            ):
                scores.append((ranked[0][0], track, ranked[0][1]))
        assigned_tracks, assigned_people = set(), set()
        matches = {}
        for score, track, person in sorted(scores, reverse=True):
            if track not in assigned_tracks and person not in assigned_people:
                matches[track] = (person, score)
                assigned_tracks.add(track)
                assigned_people.add(person)

        for track, hist, span, sample_count, consistency, preferred_view in local:
            if track in matches:
                player_id, similarity = matches[track]
                identity_status = "cross_segment_matched"
            else:
                player_id, similarity = self._new_player(), None
                identity_status = "track_local"
            start, last, observed_frames = span
            coverage = observed_frames / max(1, last + 1 - start)
            reliable = (sample_count >= 4 and coverage >= .7
                        and consistency is not None and consistency >= .72)
            record = {
                "segment_id": segment_id, "track_id": track,
                "player_id": player_id, "start_frame": start,
                "end_frame": last + 1, "observed_frames": observed_frames,
                "valid_intervals": observed_intervals[track],
                "perception_run_id": str(game.poses_path),
                "appearance_samples": sample_count,
                "appearance_view": preferred_view,
                "appearance_consistency": consistency,
                "identity_quality": round(coverage * max(0.0, consistency or 0.0), 4),
                "identity_status": (identity_status if reliable else "uncertain"),
                "cross_segment_similarity": similarity,
                "portrait": str(portraits / f"{segment_id}_track_{track}.jpg")
                if sample_count else None,
            }
            self.track_to_player[f"{segment_id}:{track}"] = record
            if player_id not in self.players:
                self.players[player_id] = {
                    "player_id": player_id, "hist": hist,
                    "last_segment": segment_id, "tracks": [record],
                }
            else:
                item = self.players[player_id]
                item["tracks"].append(record)
                item["last_segment"] = segment_id
                if hist is not None:
                    item["hist"] = hist

    def resolve(self, segment_id, track_id, frame=None):
        if track_id is None:
            return None
        record = self.track_to_player.get(f"{segment_id}:{track_id}")
        if not record:
            return None
        if frame is not None and not any(start <= frame < end
                                         for start, end in record["valid_intervals"]):
            return None
        return record["player_id"]

    def status(self, segment_id, track_id):
        record = self.track_to_player.get(f"{segment_id}:{track_id}", {})
        return "confirmed" if record.get("identity_status") in {
            "track_local", "cross_segment_matched"
        } else "uncertain"

    def export(self):
        players = {}
        for pid, data in self.players.items():
            players[pid] = {key: value for key, value in data.items() if key != "hist"}
        return {"players": players, "track_to_player": self.track_to_player}
