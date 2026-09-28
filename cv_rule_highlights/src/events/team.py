"""Conservative two-team relation inference from repeated visible transfers."""

from collections import Counter
from itertools import combinations


def infer_teams(transfers, tracks):
    """Return anonymous sides only when a 3v3 split is well separated.

    This uses the known first-segment game format, not jersey colour or the
    renderer's track palette. Contradictory evidence leaves every side unknown.
    """
    tracks = sorted(set(str(track) for track in tracks))
    weights = Counter(
        tuple(sorted((str(a), str(b)))) for a, b in transfers if a != b
    )
    report = {"schema_version": "team-relations-1", "teams": {},
              "pair_counts": {"/".join(pair): count
                              for pair, count in sorted(weights.items())},
              "status": "unknown", "reason": None}
    if len(tracks) != 6 or sum(weights.values()) < 18:
        report["reason"] = "INSUFFICIENT_STABLE_3V3_EVIDENCE"
        return report
    fixed = tracks[0]
    scores = []
    for others in combinations(tracks[1:], 2):
        left = {fixed, *others}
        score = sum(count if (a in left) == (b in left) else -count
                    for (a, b), count in weights.items())
        scores.append((score, tuple(sorted(left))))
    scores.sort(reverse=True)
    best, runner_up = scores[0][0], scores[1][0]
    total = sum(weights.values())
    if best < .15 * total or best - runner_up < max(6, .15 * total):
        report["reason"] = "CONTRADICTORY_OR_WEAK_TEAM_RELATIONS"
        report["best_score"] = best
        report["runner_up_score"] = runner_up
        return report
    left = set(scores[0][1])
    report.update(status="inferred", reason="REPEATED_TRANSFER_GRAPH",
                  best_score=best, runner_up_score=runner_up,
                  teams={track: "side_a" if track in left else "side_b"
                         for track in tracks})
    return report
