"""Playable global and personal reports generated from the same event ledger."""

from html import escape
from pathlib import Path
import os



LABELS = {'shot': '投篮', 'rebound': '篮板', 'steal': '抢断', 'block': '盖帽',
          'successful_drive': '成功突破', 'assist_chain': '视觉助攻链', 'pass': '传球',
          'ball_transfer': '控球转移', 'made': '命中', 'missed': '未中', 'unknown': '未知',
          'confirmed': '确认', 'uncertain': '待核查', 'rejected': '排除'}
METRICS = {'shot_attempts_detected': '已检出投篮', 'made': '命中', 'missed': '未中',
           'outcome_unknown': '结果未知', 'rebounds_total': '篮板', 'steals': '抢断',
           'blocks': '盖帽', 'successful_drives': '成功突破', 'assist_chains': '视觉助攻链',
           'passes': '传球', 'receptions': '接球'}
STYLE = '<style>body{font:16px sans-serif;max-width:1150px;margin:36px auto;padding:0 18px;line-height:1.6}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ddd;padding:8px;text-align:left}video{max-width:100%;width:880px}img{height:180px;object-fit:contain}.cards{display:flex;gap:20px;flex-wrap:wrap}summary{cursor:pointer}code{overflow-wrap:anywhere}</style>'


def _page(title, body):
    return f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{escape(title)}</title>{STYLE}<body><h1>{escape(title)}</h1>{body}</body></html>'


def write_reports(events, stats, players, global_plan, personal_plans, candidates_plan, output):
    output = Path(output)
    reports = output / 'reports'
    (reports / 'players').mkdir(parents=True, exist_ok=True)
    clip_index = {}
    for category, plan in [('confirmed/global', global_plan), ('candidates', candidates_plan)]:
        for clip in plan['clips']:
            for eid in clip['event_ids']:
                clip_index[eid] = (category, clip)

    def rows(items, depth):
        result = ['<table><tr><th>事件</th><th>时间</th><th>类型 / 结果</th><th>状态</th><th>视频与证据</th></tr>']
        for event in items:
            f = event['outcome'].get('evidence_frame')
            f = event['anchor_frame'] if f is None else f
            seconds = f / global_plan['fps']
            proof = '<details><summary>判定依据</summary><pre>' + escape(str(event['evidence'])) + '</pre>' + escape(', '.join(event['uncertainty_reasons'])) + '</details>'
            video = ''
            if event['event_id'] in clip_index:
                category, clip = clip_index[event['event_id']]
                offset = max(0, (f - clip['sync_start_frame']) / global_plan['fps'] - 1)
                video = f'<a href="{depth}{category}/clips/{clip["id"]}.mp4#t={offset:.2f}">播放片段</a> '
            values = [event['event_id'], f'{event["segment_id"]} {int(seconds)//60:02d}:{seconds%60:05.2f}',
                      LABELS.get(event['type'], event['type']) + ' / ' + LABELS.get(event['outcome']['value'], event['outcome']['value']),
                      LABELS.get(event['status'], event['status'])]
            result.append('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in values) + '<td>' + video + proof + '</td></tr>')
        return '\n'.join(result) + '</table>'

    def summary(row):
        table = '<table><tr>' + ''.join(f'<th>{name}</th>' for name in METRICS.values()) + '</tr><tr>' + ''.join(f'<td>{row[key]}</td>' for key in METRICS) + '</tr></table>'
        pct = row['resolved_fg_pct']
        coverage = row['outcome_coverage']
        return table + f'<p>已判定结果命中率：{pct:.1%}；结果覆盖率：{coverage:.1%}。</p>' if pct is not None and coverage is not None else table + '<p>当前缺少足够的已判定投篮结果，命中率未计算。</p>'

    cards = []
    for pid, player in players.items():
        portrait = next((r.get('portrait') for r in player['tracks'] if r.get('portrait')), None)
        portrait_url = os.path.relpath(portrait, reports / 'players') if portrait else None
        image = f'<img src="{escape(portrait_url)}" alt="{pid}">' if portrait_url else ''
        row = stats['players'][pid]
        personal = personal_plans[pid]
        body = '<p><a href="../index.html">返回全场</a></p>' + image
        body += '<p>匿名编号来自本次视频跟踪。以下为视觉事件统计，未知结果未算作投失，分数与官方技术统计未推断。</p>' + summary(row)
        body += f'<p>有观测的跟踪时长：{row.get("tracked_frames", 0) / global_plan["fps"]:.1f} 秒（不是上场时间）。</p>'
        if personal['clips']:
            body += f'<video controls preload="metadata" src="../../confirmed/players/{pid}/highlights.mp4"></video>'
        else:
            body += '<p>没有角色及身份均确认的个人高光；保留空清单。</p>'
        own = [e for e in events if e['event_id'] in row['event_ids'] or any(isinstance(r, dict) and r.get('player_id') == pid and r.get('identity_status') == 'confirmed' for r in e['actor_roles'].values())]
        body += '<p>下方事件索引也包括防守、接球等关联角色；出现在索引中不等于计入正向高光。</p>' + rows(own, '../../')
        (reports / 'players' / f'{pid}.html').write_text(_page(pid + ' 个人报告', body), encoding='utf-8')
        md = [f'# {pid} 个人视觉数据', '', *[f'- {name}：{row[key]}' for key, name in METRICS.items()], '',
              '未决候选：' + str(row['uncertain_counts']), '', f'[可播放报告](../reports/players/{pid}.html)']
        (output / 'stats' / 'players' / f'{pid}.md').write_text('\n'.join(md).replace('../reports/', '../../reports/'), encoding='utf-8')
        card_image = f'<img src="{escape(os.path.relpath(portrait, reports))}" alt="{pid}">' if portrait else ''
        cards.append(f'<a href="players/{pid}.html">{card_image}<br>{pid}：{row["made"]} 命中，{len(personal["clips"])} 段高光</a>')
    body = '<p>第一段实际视频的自动分析结果。确认高光和待核查候选分开；以下次数不是官方技术统计，也不构成完整检出率证明。</p>'
    body += summary(stats['all_players'])
    if global_plan['clips']:
        body += '<video controls preload="metadata" src="../confirmed/global/highlights.mp4"></video>'
    if (reports / 'event_types.html').is_file():
        body += '<h2>按事件类型观看</h2><p><a href="event_types.html">进球、篮板、突破 / 过人、助攻、抢断、盖帽分类合集</a></p>'
    if (reports / 'candidate_review.html').is_file():
        body += '<h2>防守事件逐条核查</h2><p><a href="candidate_review.html">篮板、抢断、盖帽：时间、人物、原因与独立视频</a></p>'
    body += '<h2>个人报告与集锦</h2><div class="cards">' + ''.join(cards) + '</div>'
    body += f'<p>未归属投篮：{stats["unassigned"]["shot_attempts_detected"]}。<a href="../stats/unassigned.json">查看未归属统计</a></p>'
    body += '<h2>待核查候选</h2><p><a href="../candidates/candidates.mp4">独立候选合集</a>，不计入确认数量。</p>'
    body += '<h2>全部事件与证据</h2>' + rows(events, '../')
    (reports / 'index.html').write_text(_page('篮球高光与个人数据', body), encoding='utf-8')
