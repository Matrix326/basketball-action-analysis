import cv2
import numpy as np
import pytest

from cv_rule_highlights.src.editing.clips import probe, render


def test_frame_offsets_and_concatenation(tmp_path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for frame in range(60):
        writer.write(np.full((120, 160, 3), frame * 4, dtype=np.uint8))
    writer.release()
    output = tmp_path / "render"
    output.mkdir()
    plan = {"fps": 30, "clips": [
        {"id": "clip_0001", "source": str(source), "source_start_frame": 11, "source_end_frame": 24},
        {"id": "clip_0002", "source": str(source), "source_start_frame": 40, "source_end_frame": 47},
    ]}
    result = render(plan, output, (160, 120))
    assert {key: result[key] for key in ("clips", "frames", "duration_seconds")} == {"clips": 2, "frames": 20, "duration_seconds": 20 / 30}
    assert result["audio"] == "no_audio"
    assert probe(output / "highlights.mp4")["nb_frames"] == 20
    cap = cv2.VideoCapture(str(output / "highlights.mp4"))
    values = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        values.append(frame.mean())
    cap.release()
    assert len(values) == 20
    # Lossy encoding changes brightness slightly, but not which source frame is used.
    np.testing.assert_allclose(values, np.array(list(range(11, 24)) + list(range(40, 47))) * 4, atol=7)


def test_empty_plan_does_not_fabricate_a_video(tmp_path):
    result = render({"fps": 30, "clips": []}, tmp_path)
    assert result["frames"] == 0
    assert not (tmp_path / "highlights.mp4").exists()


def test_missing_video_reports_the_source(tmp_path):
    plan = {"fps": 30, "clips": [{"source": str(tmp_path / "missing.mp4")}]}
    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        render(plan, tmp_path)


def test_source_audio_survives_clip_and_concat(tmp_path):
    import json
    import subprocess
    source = tmp_path / 'audio.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=160x120:r=30:d=2',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-c:v', 'libx264',
                    '-c:a', 'aac', '-shortest', str(source)], check=True)
    out = tmp_path / 'render_audio'; out.mkdir()
    plan = {'fps': 30, 'audio': 'preserve', 'clips': [
        {'id': 'a', 'source': str(source), 'source_start_frame': 0, 'source_end_frame': 30},
        {'id': 'b', 'source': str(source), 'source_start_frame': 30, 'source_end_frame': 60}]}
    result = render(plan, out, (160, 120))
    assert result['frames'] == 60 and result['audio'] == 'source_audio_preserved'
    probe_result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(out / 'highlights.mp4')],
                                  check=True, capture_output=True, text=True)
    assert any(s['codec_type'] == 'audio' for s in json.loads(probe_result.stdout)['streams'])


def test_dynamic_cut_preserves_source_frame_sequence(tmp_path):
    sources = []
    for offset in (0, 100):
        path = tmp_path / f'camera_{offset}.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (160, 120))
        for f in range(60):
            writer.write(np.full((120, 160, 3), offset + f, np.uint8))
        writer.release()
        sources.append(str(path))
    out = tmp_path / 'switch'; out.mkdir()
    plan = {'fps': 30, 'clips': [{'id': 'action', 'shots': [
        {'source': sources[0], 'source_start_frame': 10, 'source_end_frame': 30,
         'sync_start_frame': 10, 'sync_end_frame': 30},
        {'source': sources[1], 'source_start_frame': 30, 'source_end_frame': 50,
         'sync_start_frame': 30, 'sync_end_frame': 50}]}]}
    result = render(plan, out, (160, 120))
    assert result['frames'] == 40
    cap = cv2.VideoCapture(str(out / 'highlights.mp4'))
    values = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        values.append(frame.mean())
    cap.release()
    np.testing.assert_allclose(values, list(range(10, 30)) + list(range(130, 150)), atol=7)
