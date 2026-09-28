"""Behavior affected by moving implementations below src."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from cv_rule_highlights.src.perception import roi_redetect
from cv_rule_highlights.src.reporting.audit import code_digest


def test_code_digest_covers_nested_sources_only(tmp_path):
    (tmp_path / '__main__.py').write_text('pass\n')
    nested = tmp_path / 'src' / 'events'
    nested.mkdir(parents=True)
    module = nested / 'rules.py'
    module.write_text('threshold = 1\n')
    original = code_digest(tmp_path)
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'debug.py').write_text('irrelevant = True\n')
    assert code_digest(tmp_path) == original
    module.write_text('threshold = 2\n')
    assert code_digest(tmp_path) != original


def test_roi_redetect_resolves_upstream_after_move(monkeypatch, tmp_path):
    upstream = Path(__file__).resolve().parents[2] / 'perception'
    assert (upstream / 'src' / 'rfdetr_pipeline' / 'detector.py').is_file()
    monkeypatch.setattr(sys, 'path', [p for p in sys.path
        if p not in (str(upstream), str(upstream / 'src'))])
    config = ModuleType('config')
    config.load_config = lambda path: path
    detector = ModuleType('rfdetr_pipeline.detector')
    detector.RFDetrSegmenter = lambda config: SimpleNamespace(name='test')
    runtime = ModuleType('basketball_repro.inference_runtime')
    runtime.BALL_CLASS_ID = 1
    for module in (config, detector, runtime):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    game = SimpleNamespace(views={}, start=0, end=1)
    result = roi_redetect.redetect(game, {'visits': []}, 'unused.yaml', tmp_path)
    assert str(upstream) in sys.path and str(upstream / 'src') in sys.path
    assert Path(result['sidecar']).is_file()
    assert result['model'] == 'test'
