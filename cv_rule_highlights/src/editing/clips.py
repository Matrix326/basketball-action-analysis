"""Clip planning and constant-frame-rate MP4 rendering with source audio."""

import json
from pathlib import Path
import subprocess

import cv2

from ..core.data import save_json
from ..core.timeline import source_frame


def view_quality(game, view, start, end, events):
    def visible(frame):
        ball = float(view in game.balls(frame))
        actors = {e["actor_id"] for e in events if e["actor_id"] is not None}
        actor = sum(view in game.players(frame).get(p, {}) for p in actors) / len(actors) if actors else ball
        return 0.6 * ball + 0.4 * actor

    frames = range(start, end)
    coverage = sum(visible(f) for f in frames) / (end - start)
    critical = []
    for event in events:
        t = event["resolution_frame"] if event["resolution_frame"] is not None else event["anchor_frame"]
        critical.extend(range(max(start, t - 3), min(end, t + 4)))
    critical_coverage = sum(visible(f) for f in critical) / len(critical) if critical else 0.0
    # Critical visibility matters more than average coverage over padding.
    return 0.4 * coverage + 0.6 * critical_coverage


def plan_clips(game, report, review=False):
    selected = [e for e in report["events"] if review or
                (e["visual_outcome"] == "made" and e["confidence"] == "confirmed")]
    intervals = []
    for event in selected:
        anchor = event["anchor_frame"]
        start = max(game.start, event["lead_in_frame"] - round(game.rules.pre_seconds * game.fps))
        finish = event["resolution_frame"]
        if finish is None:
            finish = event["rim_window"][1] - 1 if event["rim_window"] else anchor + round(game.rules.flight_seconds * game.fps)
        end = min(game.end, finish + round(game.rules.post_seconds * game.fps) + 1)
        intervals.append({"start": start, "end": end, "events": [event]})
    intervals.sort(key=lambda c: c["start"])
    merged = []
    for clip in intervals:
        if merged and clip["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], clip["end"])
            merged[-1]["events"].extend(clip["events"])
        else:
            merged.append(clip)
    # Overlapping footage is one continuous clip, not a claim of one possession.
    ranked = sorted(merged, key=lambda c: (-max(e["highlight_score"] for e in c["events"]), c["end"] - c["start"]))
    remaining = round(game.rules.budget_seconds * game.fps) if game.rules.budget_seconds is not None else float("inf")
    accepted = []
    for clip in ranked:
        length = clip["end"] - clip["start"]
        if length <= remaining:
            accepted.append(clip)
            remaining -= length
    result = []
    for clip in sorted(accepted, key=lambda c: c["start"]):
        qualities = {view: view_quality(game, view, clip["start"], clip["end"], clip["events"]) for view in game.views}
        view = max(qualities, key=qualities.get)
        source = game.views[view]
        result.append({
            "id": f"clip_{len(result) + 1:04d}", "event_ids": [e["id"] for e in clip["events"]],
            "view": view, "source": source["path"],
            "sync_start_frame": clip["start"], "sync_end_frame": clip["end"],
            "source_start_frame": source_frame(clip["start"], source["frame_zero"]),
            "source_end_frame": source_frame(clip["end"], source["frame_zero"]),
            "view_scores": qualities,
        })
    return {
        "schema_version": "highlights-edl-0.1", "fps": game.fps,
        "mode": "review" if review else "confirmed_makes", "audio": False,
        "budget_seconds": game.rules.budget_seconds,
        "eligible_events": len(selected), "clips": result,
    }


def probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames", "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    for key in ("r_frame_rate", "avg_frame_rate"):
        numerator, denominator = map(int, stream[key].split("/"))
        stream[key] = numerator / denominator
    stream["nb_frames"] = int(stream["nb_frames"])
    return stream


def render(plan, output, size=(1280, 720)):
    output = Path(output)
    width, height = size
    if min(size) <= 0 or width % 2 or height % 2:
        raise ValueError("Output dimensions must be positive even numbers")
    fps = plan["fps"]
    sources = {}
    pieces = [piece for clip in plan["clips"] for piece in clip.get("shots", [clip])]
    for clip in pieces:
        source = clip["source"]
        if source not in sources:
            if not Path(source).is_file():
                raise FileNotFoundError(f"Video not found: {source}")
            sources[source] = probe(source)
        info = sources[source]
        if any(abs(info[k] - fps) > 1e-3 for k in ("r_frame_rate", "avg_frame_rate")):
            raise ValueError(f"Video FPS does not match perception: {source}")
        if not 0 <= clip["source_start_frame"] < clip["source_end_frame"] <= info["nb_frames"]:
            raise ValueError(f"Source interval in {source} lies outside video; check frame_zero")
    preserve_audio = plan.get("audio") != False
    source_audio = {}
    for source in sources:
        audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                                "-show_entries", "stream=index", "-of", "csv=p=0", source],
                               capture_output=True, text=True, check=True)
        source_audio[source] = bool(audio.stdout.strip())
    with_audio = preserve_audio and (any(source_audio.values()) or plan.get("force_audio", False))
    clips_dir = output / "clips"
    clips_dir.mkdir()
    files = []
    total_frames = 0
    for clip in plan["clips"]:
        shots = clip.get('shots', [clip])
        count = sum(s['source_end_frame'] - s['source_start_frame'] for s in shots)
        target = clips_dir / f"{clip['id']}.mp4"
        if len(shots) > 1:
            if any(a['sync_end_frame'] != b['sync_start_frame'] for a, b in zip(shots, shots[1:])):
                raise ValueError('Camera cuts must be contiguous')
            parts = output / 'camera_parts' / clip['id']
            parts.mkdir(parents=True)
            render({'fps': fps, 'audio': plan.get('audio'), 'force_audio': with_audio,
                    'clips': [{**shot, 'id': f'camera_{i:02d}'} for i, shot in enumerate(shots)]}, parts, size)
            (parts / 'highlights.mp4').rename(target)
            files.append(f"file 'clips/{target.name}'")
            total_frames += count
            continue
        clip = {**clip, **shots[0]}
        inputs = ["-ss", f"{clip['source_start_frame'] / fps:.9f}", "-i", clip["source"]]
        audio_args = ["-an"]
        if with_audio:
            if not source_audio[clip["source"]]:
                inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            stream = "0:a:0" if source_audio[clip["source"]] else "1:a:0"
            audio_args = ["-map", stream, "-af", f"apad,atrim=duration={count / fps:.9f},asetpts=PTS-STARTPTS",
                          "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-n", *inputs,
            "-map", "0:v:0", "-frames:v", str(count), "-t", f"{count / fps:.9f}", *audio_args,
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,setpts=PTS-STARTPTS",
            "-r", str(fps), "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "4",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
        ], check=True)
        if probe(target)["nb_frames"] != count:
            raise ValueError(f"Rendered frame count mismatch: {target}")
        files.append(f"file 'clips/{target.name}'")
        total_frames += count
    if files:
        concat = output / "concat.txt"
        concat.write_text("\n".join(files) + "\n")
        target = output / "highlights.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "concat", "-safe", "0", "-i", str(concat),
            "-map", "0", "-c", "copy", "-movflags", "+faststart", str(target),
        ], check=True)
        if probe(target)["nb_frames"] != total_frames:
            raise ValueError("Concatenated frame count mismatch")
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(target), "-f", "null", "-"], check=True)
    result = {"clips": len(files), "frames": total_frames, "duration_seconds": total_frames / fps,
              "audio": "source_audio_preserved" if with_audio else "no_audio",
              "empty_reason": None if files else "No eligible events"}
    save_json(output / "render.json", result)
    return result


def preview(game, output, frame):
    """Rim annotation check on source images, without running any detector."""
    for view, settings in game.views.items():
        cap = cv2.VideoCapture(settings["path"])
        local_frame = source_frame(frame, settings["frame_zero"])
        if local_frame < 0:
            cap.release()
            raise ValueError(f"Preview precedes video start: {view}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Cannot decode {view} at source frame {local_frame}")
        if image.shape[:2] != (settings["height"], settings["width"]):
            raise ValueError("Preview source dimensions must match perception coordinates")
        cx, cy, w, h = settings["rim"]
        cv2.ellipse(image, (round(cx), round(cy)), (round(w / 2), max(1, round(h / 2))), 0, 0, 360, (0, 255, 0), 2)
        cv2.line(image, (round(cx - w / 2), round(cy)), (round(cx + w / 2), round(cy)), (255, 255, 0), 1)
        cv2.putText(image, f"{view} sync={frame} source={local_frame}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(Path(output) / f"{view}.jpg"), image)
