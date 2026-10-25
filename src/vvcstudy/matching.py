"""Rate-quality frontier and matching of measured files without interpolation."""

from __future__ import annotations


def frontier(rows: list[dict], metric: str) -> tuple[list[dict], list[dict]]:
    """Keep strictly improving quality/rate points; record dominated/equal points."""
    kept, removed = [], []
    for row in sorted(rows, key=lambda r: (-r[metric], r['bpp'], r['parameter'])):
        if kept and row['bpp'] >= kept[-1]['bpp']:
            reason = ('equivalent' if row[metric] == kept[-1][metric]
                      and row['bpp'] == kept[-1]['bpp'] else 'dominated')
            removed.append(dict(parameter=row['parameter'], reason=reason))
        else:
            kept.append(row)
    return list(reversed(kept)), removed


def actual_match(points: list[dict], metric: str, target: float) -> dict:
    """Return real frontier points bracketing the target, and the nearest one."""
    below = [r for r in points if r[metric] <= target]
    above = [r for r in points if r[metric] >= target]
    if not below or not above:
        return dict(status='outside_measured_range')
    lower = min(below, key=lambda r: (-r[metric], r['bpp'], r['parameter']))
    upper = min(above, key=lambda r: (r[metric], r['bpp'], r['parameter']))
    nearest = min((lower, upper), key=lambda r: (abs(r[metric] - target), r['bpp'], r['parameter']))
    result = dict(status='ok')
    for label, point in [('lower', lower), ('upper', upper), ('nearest', nearest)]:
        result.update({f'{label}_{key}': point[key] for key in ('parameter', 'bpp', 'file', 'sha256')})
        result[f'{label}_quality'] = point[metric]
        result[f'{label}_quality_delta'] = point[metric] - target
    return result
