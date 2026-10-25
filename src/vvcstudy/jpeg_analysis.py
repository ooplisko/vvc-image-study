"""Paired uncertainty and actual-file quality-matching sensitivity; CSV/JSON only."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .statistics import paired_spearman_bootstrap, percentile_intervals, spearman_last_axis
FEATURES = ('sobel_si', 'luma_sd', 'edge_fraction', 'glcm_contrast', 'glcm_entropy', 'glcm_homogeneity')
VARIANTS = ('nearest', 'lower', 'upper')
METRICS = ('psnr_rgb_db', 'msssim_rgb')
QPS = (22, 27, 32, 37)
PSNR_GAP_DB = 0.1
VERSION = 'jpeg-vvc-paired-uncertainty-v1'

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(paths: list[Path]) -> list[dict]:
    """Validate identities and measured values; retain unmatched rows explicitly."""
    rows, keys = [], set()
    numeric = (*FEATURES, 'target_quality', 'vtm_bpp',
               *(f'ratio_{v}' for v in VARIANTS),
               *(f'{v}_quality_delta' for v in VARIANTS))
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for raw in csv.DictReader(stream):
                row = {**raw, 'dataset': raw.get('dataset') or 'kodak',
                       'condition': raw.get('condition') or 'clean'}
                row['qp'] = int(row['qp'])
                if row['metric'] not in METRICS or row['qp'] not in QPS:
                    raise ValueError('Expected PSNR-RGB/MS-SSIM-RGB and QP 22/27/32/37')
                key = tuple(row[k] for k in ('dataset', 'condition', 'metric', 'qp', 'source'))
                if key in keys:
                    raise ValueError(f'Duplicate image/condition/metric/QP: {key}')
                keys.add(key)
                if row['status'] == 'ok':
                    for field in numeric:
                        row[field] = float(row[field])
                        if not np.isfinite(row[field]):
                            raise ValueError(f'Non-finite {field} in {key}')
                    if row['vtm_bpp'] <= 0 or any(row[f'ratio_{v}'] <= 0 for v in VARIANTS):
                        raise ValueError(f'Non-positive rate in {key}')
                    if row['lower_quality_delta'] > 1e-12 or row['upper_quality_delta'] < -1e-12:
                        raise ValueError(f'JPEG neighbours do not bracket quality in {key}')
                    nearest = row['nearest_quality_delta']
                    neighbours = [row[f'{v}_quality_delta'] for v in ('lower', 'upper')]
                    if not any(np.isclose(nearest, d, rtol=0, atol=1e-12) for d in neighbours):
                        raise ValueError(f'Nearest JPEG is not an actual neighbour in {key}')
                    if abs(nearest) > min(map(abs, neighbours)) + 1e-12:
                        raise ValueError(f'Nearest JPEG has a larger quality gap in {key}')
                rows.append(row)
    if not rows:
        raise ValueError('Empty input table')
    groups = defaultdict(set)
    for row in rows:
        groups[row['dataset'], row['condition']].add((row['metric'], row['qp']))
    expected = {(m, q) for m in METRICS for q in QPS}
    if any(cells != expected for cells in groups.values()):
        raise ValueError('Each dataset/condition must contain both metrics and all four QPs')
    return rows


@dataclass
class Cell:
    meta: dict
    sources: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray


def safe_rho(x, y) -> float:
    return float(spearman_last_axis(np.asarray(x), np.asarray(y))) if len(x) >= 3 else float('nan')


def prepare(rows: list[dict]) -> tuple[list[Cell], list[dict], list[dict], list[dict], list[dict]]:
    """Build complete image cohorts without treating QPs as independent images."""
    grouped, all_sources = defaultdict(dict), defaultdict(set)
    for row in rows:
        key = (row['dataset'], row['condition'], row['metric'], row['qp'])
        grouped[key][row['source']] = row
        all_sources[row['dataset']].add(row['source'])
    cells, coverage, quality, associations, contrasts = [], [], [], [], []
    indices = {}

    def add(meta, selected, feature, variant, subset):
        sources = tuple(sorted(selected))
        identity = (*meta.values(), feature, variant, sources)
        if identity in indices:
            return indices[identity]
        index = len(cells)
        indices[identity] = index
        cells.append(Cell({**meta, 'feature': feature, 'variant': variant,
                           'subset': subset, 'n_images': len(sources)}, sources,
                          np.array([selected[s][feature] for s in sources]),
                          np.array([selected[s][f'ratio_{variant}'] for s in sources])))
        return index

    for key, raw in sorted(grouped.items()):
        dataset, condition, metric, qp = key
        meta = dict(dataset=dataset, condition=condition, metric=metric, qp=qp)
        selected = {s: r for s, r in raw.items() if r['status'] == 'ok'}
        excluded = {s: r['status'] for s, r in raw.items() if r['status'] != 'ok'}
        missing = sorted(all_sources[dataset] - set(raw))
        coverage.append({**meta, 'n_source_images': len(all_sources[dataset]),
                         'n_rows': len(raw), 'n_matched': len(selected),
                         'excluded_json': json.dumps(excluded, sort_keys=True),
                         'missing_sources_json': json.dumps(missing)})
        for variant in VARIANTS:
            values = list(selected.values())
            gaps = np.array([r[f'{variant}_quality_delta'] for r in values])
            ratios = np.array([r[f'ratio_{variant}'] for r in values])
            quality.append({**meta, 'variant': variant, 'n_images': len(values),
                            'gap_unit': 'dB' if metric == 'psnr_rgb_db' else 'MS-SSIM',
                            'median_signed_gap': float(np.median(gaps)) if len(gaps) else np.nan,
                            'median_abs_gap': float(np.median(abs(gaps))) if len(gaps) else np.nan,
                            'max_abs_gap': float(max(abs(gaps))) if len(gaps) else np.nan,
                            'median_ratio': float(np.median(ratios)) if len(ratios) else np.nan,
                            'n_ratio_gt_1': int(np.sum(ratios > 1)),
                            'n_psnr_gap_le_0_1_db': int(np.sum(abs(gaps) <= PSNR_GAP_DB))
                            if metric == 'psnr_rgb_db' else ''})
            for feature in FEATURES:
                index = add(meta, selected, feature, variant, 'all_matched')
                if variant != 'nearest':
                    nearest = add(meta, selected, feature, 'nearest', 'all_matched')
                    contrasts.append({**meta, 'feature': feature, 'contrast': f'{variant}_minus_nearest',
                                      'n_images': len(selected), 'a': index, 'b': nearest,
                                      'kind': 'matching_choice'})
        for feature in (*FEATURES, 'ratio_nearest'):
            values = list(selected.values())
            associations.append({**meta, 'variable': feature, 'n_images': len(values),
                                 'rho_with_signed_quality_gap': safe_rho(
                                     [r[feature] for r in values], [r['nearest_quality_delta'] for r in values]),
                                 'rho_with_target_quality': safe_rho(
                                     [r[feature] for r in values], [r['target_quality'] for r in values])})
        if metric == 'psnr_rgb_db':
            narrow = {s: r for s, r in selected.items() if abs(r['nearest_quality_delta']) <= PSNR_GAP_DB}
            for feature in FEATURES:
                # Separate identity keeps the sensitivity rows even when no image is removed.
                copy_meta = {**meta, 'selection': 'psnr_gap_le_0_1_db'}
                add(copy_meta, narrow, feature, 'nearest', 'psnr_gap_le_0_1_db')

    for key, raw in sorted(grouped.items()):
        dataset, condition, metric, qp = key
        if condition == 'clean' or (dataset, 'clean', metric, qp) not in grouped:
            continue
        clean = grouped[dataset, 'clean', metric, qp]
        sources = sorted(s for s in raw.keys() & clean.keys()
                         if raw[s]['status'] == clean[s]['status'] == 'ok')
        for feature in FEATURES:
            pair = []
            for name, values in [('clean', clean), (condition, raw)]:
                meta = dict(dataset=dataset, condition=name, metric=metric, qp=qp)
                pair.append(add(meta, {s: values[s] for s in sources}, feature,
                                'nearest', 'paired_clean_noisy'))
            contrasts.append(dict(dataset=dataset, condition=condition, metric=metric, qp=qp,
                                  feature=feature, contrast='noisy_minus_clean', kind='noise_change',
                                  n_images=len(sources), a=pair[1], b=pair[0]))
    return cells, coverage, quality, associations, contrasts


def cohort_bootstrap(cells: list[Cell], output: Path, resamples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache each completed source cohort; invalid draws remain NaN, never redrawn."""
    observed = np.full(len(cells), np.nan)
    distribution = np.full((resamples, len(cells)), np.nan)
    cohorts = defaultdict(list)
    for i, cell in enumerate(cells):
        cohorts[cell.meta['dataset'], cell.sources].append(i)
    cache = output / 'bootstrap_cache'
    cache.mkdir(parents=True, exist_ok=True)
    for (dataset, sources), positions in cohorts.items():
        if len(sources) < 3:
            continue
        x = np.array([cells[i].x for i in positions])
        y = np.array([cells[i].y for i in positions])
        identity = dict(dataset=dataset, sources=sources, resamples=resamples, seed=seed,
                        cells=[cells[i].meta for i in positions],
                        x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                        y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                        helper_sha256=sha256(Path(__file__).with_name('statistics.py')), version=VERSION)
        encoded = json.dumps(identity, sort_keys=True)
        path = cache / f'{hashlib.sha256(encoded.encode()).hexdigest()}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved['identity'].item() != encoded or saved['distribution'].shape != (resamples, len(positions)):
                    raise ValueError(f'Bootstrap cache identity/shape mismatch: {path}')
                local_observed, draws = saved['observed'], saved['distribution']
                if not np.allclose(local_observed, spearman_last_axis(x, y), rtol=0, atol=1e-12, equal_nan=True):
                    raise ValueError(f'Bootstrap cache observations changed: {path}')
        else:
            print(f'Bootstrap {dataset}: {len(sources)} images, {len(positions)} cells, {resamples} draws', flush=True)
            local_observed, draws = paired_spearman_bootstrap(
                x, y, n_resamples=resamples, seed=seed, batch_size=128)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, identity=encoded, observed=local_observed, distribution=draws)
            temporary.replace(path)
        observed[positions], distribution[:, positions] = local_observed, draws
    return observed, distribution


def interval_record(ci: dict, i: int) -> dict:
    return {f'ci_{k}': v[i].item() for k, v in ci.items()}


def summarize(cells, observed, distribution, contrasts):
    ci = percentile_intervals(observed, distribution)
    correlations = [{**cell.meta, 'rho': float(observed[i]), **interval_record(ci, i),
                     'sources_json': json.dumps(cell.sources), 'interval_scope': 'pointwise_95_percent'}
                    for i, cell in enumerate(cells) if cell.meta['subset'] != 'paired_clean_noisy']
    differences = []
    for contrast in contrasts:
        a, b = contrast['a'], contrast['b']
        if cells[a].sources != cells[b].sources or cells[a].meta['dataset'] != cells[b].meta['dataset']:
            raise ValueError('A paired contrast must use identical source cohorts')
        value = np.array([observed[a] - observed[b]])
        sampled = (distribution[:, a] - distribution[:, b])[:, None]
        interval = percentile_intervals(value, sampled)
        differences.append({**{k: v for k, v in contrast.items() if k not in ('a', 'b')},
                            'rho_a': float(observed[a]), 'rho_b': float(observed[b]),
                            'rho_difference': float(value[0]), **interval_record(interval, 0),
                            'sources_json': json.dumps(cells[a].sources),
                            'interval_scope': 'pointwise_95_percent'})
    return correlations, differences


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze_jpeg(paths: list[Path], output: Path, *, resamples: int = 99_999,
                 seed: int = 20261022, check_only: bool = False) -> dict:
    """Analyze all quality/QP cells; never create prose or encode an image."""
    from .artifacts import write_json
    if resamples < 1 or seed < 0:
        raise ValueError("Positive resamples and nonnegative seed required.")
    rows = load_rows(paths)
    cells, coverage, quality, associations, contrasts = prepare(rows)
    report = {"protocol": VERSION, "input_sha256": [sha256(path) for path in paths],
              "analysis_sha256": sha256(Path(__file__)),
              "statistics_sha256": sha256(Path(__file__).with_name('statistics.py')),
              "resamples": resamples, "seed": seed, "n_rows": len(rows), "n_cells": len(cells),
              "interval_scope": "pointwise_95_percent", "interpolation": False,
              "sampling_unit": "source image within dataset; paired conditions and matching choices"}
    if check_only:
        return {**report, "coverage": coverage}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "analysis.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in report.items()):
            raise ValueError("Analysis configuration changed; use a different output folder.")
    write_json(manifest, report)
    observed, distribution = cohort_bootstrap(cells, output, resamples, seed)
    correlations, differences = summarize(cells, observed, distribution, contrasts)
    tables = {"correlations_with_ci.csv": correlations, "coverage.csv": coverage,
              "quality_sensitivity.csv": quality, "gap_associations.csv": associations,
              "matching_choice_differences.csv": [r for r in differences if r['kind'] == 'matching_choice'],
              "noise_correlation_changes.csv": [r for r in differences if r['kind'] == 'noise_change']}
    for name, values in tables.items():
        write_csv(output / name, values)
    report.update(complete=True, outputs={name: sha256(output / name) for name in tables})
    write_json(manifest, report)
    return report
