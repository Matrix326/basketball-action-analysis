"""One unlimited confirmed reel per event type, using the existing event ledger."""

import json
from pathlib import Path

from .auto_clips import combine_plans
from .clips import render
from ..core.data import Game, save_json
from ..reporting.reports import _page, write_reports


CATEGORIES = {
    'made': ('进球', 'shot'),
    'rebound': ('篮板', 'rebound'),
    'successful_drive': ('突破 / 过人', 'successful_drive'),
    'assist_chain': ('视觉助攻链', 'assist_chain'),
    'steal': ('抢断', 'steal'),
    'block': ('盖帽', 'block'),
}


def category_plans(games, events_by_segment):
    return {
        category: {**combine_plans(games, {
            segment: [event for event in events if event['type'] == kind]
            for segment, events in events_by_segment.items()
        }), 'category': category}
        for category, (_, kind) in CATEGORIES.items()
    }


def export_event_clips(games, events_by_segment, output, *, do_render=True, size=(1280, 720)):
    output = Path(output)
    target = output / 'confirmed' / 'event_types'
    target.mkdir(parents=True)
    plans = category_plans(games, events_by_segment)
    summary = {}
    body = '<p><a href="index.html">返回全场与个人报告</a></p>'
    body += '<p>每类收录全部自动确认事件，不限总时长。突破与过人目前使用同一类规则，合为一个合集；未决候选不计入。</p>'
    for category, plan in plans.items():
        label = CATEGORIES[category][0]
        folder = target / category
        folder.mkdir()
        save_json(folder / 'edit_decision_list.json', plan)
        result = render(plan, folder, size) if do_render else None
        summary[category] = {'label': label, 'events': plan['eligible_events'],
                             'clips': len(plan['clips']), 'render': result}
        body += f'<h2>{label}：{plan["eligible_events"]} 次，{len(plan["clips"])} 段</h2>'
        if result and plan['clips']:
            url = f'../confirmed/event_types/{category}/highlights.mp4'
            body += f'<p>{result["duration_seconds"]:.2f} 秒 · <a href="{url}">打开视频</a></p><video controls preload="metadata" src="{url}"></video>'
        elif not plan['clips']:
            body += '<p>当前没有自动确认事件，不生成空视频；这不代表原视频中没有发生此类事件。</p>'
        else:
            body += '<p>剪辑清单已生成，视频尚未渲染。</p>'
    save_json(target / 'summary.json', summary)
    reports = output / 'reports'
    reports.mkdir(exist_ok=True)
    (reports / 'event_types.html').write_text(_page('按事件类型观看高光', body), encoding='utf-8')
    return summary


def export_run_event_clips(run, size=(1280, 720)):
    """Add category reels to a completed run without repeating any inference."""
    run = Path(run).resolve()
    events = json.loads((run / 'events' / 'all_events.json').read_text())
    groups = {}
    for event in events:
        groups.setdefault(event['segment_id'], []).append(event)
    games = {segment: Game.load(run / 'inputs' / f'{segment}.yaml') for segment in groups}
    summary = export_event_clips(games, groups, run, size=size)
    players = json.loads((run / 'identity' / 'players.json').read_text())
    stats = json.loads((run / 'stats' / 'all_players.json').read_text())
    global_plan = json.loads((run / 'confirmed' / 'global' / 'edit_decision_list.json').read_text())
    personal = {pid: json.loads((run / 'confirmed' / 'players' / pid / 'edit_decision_list.json').read_text())
                for pid in players}
    candidates = json.loads((run / 'candidates' / 'edit_decision_list.json').read_text())
    write_reports(events, stats, players, global_plan, personal, candidates, run)
    return summary
