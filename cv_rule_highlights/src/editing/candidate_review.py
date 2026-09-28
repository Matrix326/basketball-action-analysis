"""Individual, playable defensive-event candidates with reasons and timestamps."""

import csv
from html import escape
from pathlib import Path

from .auto_clips import plan_segment
from .clips import render
from ..core.data import save_json
from ..reporting.reports import _page, LABELS


REASONS = {
    'CONTROL_TOO_BRIEF': '前后控制过短，可能是身份/手球抖动',
    'CONTROL_GAP_TOO_LONG': '控制权转移间隔超过1.2秒',
    'OPPOSITION_UNVERIFIED': '攻防关系未确认',
    'MULTIVIEW_TOUCH_UNVERIFIED': '缺少两视角局部触球证据',
    'RECEIVE_VS_STEAL_UNVERIFIED': '缺少争夺证据，不能区分接球/还球与抢断',
    'DEFLECTION_UNVERIFIED': '没有清晰的触球后变向',
    'POST_SHOT_OR_REBOUND_COLLECTION': '位于投篮后收球或篮板阶段',
    'DEFENSIVE_CONTACT_UNVERIFIED': '封盖触球、变向或攻防关系未确认',
    'SHOT_SCORED_AFTER_CONTACT': '该投篮最终进球，排除盖帽',
    'MISS_OR_STABLE_ACQUISITION_UNVERIFIED': '投失存在性或持续收球证据不足',
}


def export_candidate_review(games, events_by_segment, output, *, do_render=True, size=(1280, 720)):
    output = Path(output)
    records, counts = [], {}
    body = '<p><a href="index.html">返回总报告</a></p><p>包括确认、未决和排除项；每条单独成片，不合并相邻候选。人物编号来自自动跟踪。</p>'
    fps = next(iter(games.values())).fps
    for kind in ('rebound', 'steal', 'block'):
        events = [e for values in events_by_segment.values() for e in values if e['type'] == kind]
        clips = []
        for segment, game in games.items():
            clips.extend(plan_segment(game, segment, [e for e in events if e['segment_id'] == segment], review=True))
        for clip in clips:
            clip['id'] = clip['event_ids'][0]
        plan = {'fps': fps, 'audio': 'preserve', 'clips': clips,
                'selected_event_ids': [e['event_id'] for e in events], 'eligible_events': len(events)}
        folder = output / 'review' / kind
        folder.mkdir(parents=True)
        save_json(folder / 'edit_decision_list.json', plan)
        if do_render:
            render(plan, folder, size)
        counts[kind] = {status: sum(e['status'] == status for e in events)
                        for status in ('confirmed', 'uncertain', 'rejected')}
        body += f'<h2>{LABELS[kind]}：{len(events)} 条</h2>'
        if do_render and clips:
            body += f'<p><a href="../review/{kind}/highlights.mp4">该类全部核查片段</a></p>'
        body += '<table><tr><th>序号 / 时间</th><th>人物</th><th>状态与原因</th><th>视频及证据</th></tr>'
        by_id = {c['event_ids'][0]: c for c in clips}
        for i, e in enumerate(events, 1):
            clip = by_id[e['event_id']]
            roles = '；'.join(f"{name}: {r.get('player_id') or '?'} (track {r.get('track_id')})"
                              for name, r in e['actor_roles'].items() if isinstance(r, dict))
            reasons = '；'.join(REASONS.get(r, r) for r in e['uncertainty_reasons']) or '满足当前规则'
            frame = e['anchor_frame']
            record = {'category': kind, 'index': i, 'event_id': e['event_id'], 'segment': e['segment_id'],
                      'frame': frame, 'seconds': round(frame / fps, 3), 'status': e['status'],
                      'roles': roles, 'reason': reasons, 'start_seconds': round(clip['sync_start_frame'] / fps, 3),
                      'end_seconds': round(clip['sync_end_frame'] / fps, 3),
                      'video': f'review/{kind}/clips/{clip["id"]}.mp4'}
            records.append(record)
            link = f'<a href="../{record["video"]}">播放独立片段</a>' if do_render else '尚未渲染'
            evidence = '<details><summary>查看判据</summary><pre>' + escape(str(e['evidence'])) + '</pre></details>'
            body += f'<tr><td>{i} · {int(frame/fps)//60:02d}:{frame/fps%60:05.2f}<br>{escape(e["event_id"])}</td><td>{escape(roles)}</td><td>{LABELS[e["status"]]}<br>{escape(reasons)}</td><td>{link}{evidence}</td></tr>'
        body += '</table>'
    (output / 'reports').mkdir(exist_ok=True)
    (output / 'reports/candidate_review.html').write_text(_page('篮板、抢断、盖帽逐条核查', body), encoding='utf-8')
    save_json(output / 'review/candidates.json', records)
    save_json(output / 'review/summary.json', counts)
    with (output / 'review/candidates.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['category', 'index', 'event_id', 'segment', 'frame', 'seconds',
            'status', 'roles', 'reason', 'start_seconds', 'end_seconds', 'video'])
        writer.writeheader()
        writer.writerows(records)
    return counts
