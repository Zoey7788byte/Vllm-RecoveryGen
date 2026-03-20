#!/usr/bin/env python3
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SeriesSpec:
    label: str
    run_dir: str
    color: str
    linestyle: str = '-'


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _to_float(v: str) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float('nan')


def _segment_dirs(run_dir: str) -> List[str]:
    out = []
    for entry in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, entry)
        if os.path.isdir(full):
            out.append(full)
    return out


def _load_preemption_series(run_dir: str, bin_s: float) -> Tuple[np.ndarray, np.ndarray]:
    bins_by_seg: List[np.ndarray] = []
    max_rel = 0.0

    for seg_dir in _segment_dirs(run_dir):
        p = os.path.join(seg_dir, 'metrics_timeseries.csv')
        if not os.path.isfile(p):
            continue
        rows = _read_csv(p)
        if not rows:
            continue

        rels: List[float] = []
        totals: List[float] = []
        deltas: List[float] = []

        for row in rows:
            t_rel = _to_float(row.get('t_rel_s', 'nan'))
            total = _to_float(row.get('preempt_total', 'nan'))
            delta = _to_float(row.get('on_preempt_count_delta', 'nan'))
            if math.isnan(t_rel) or math.isnan(total):
                continue
            rels.append(t_rel)
            totals.append(total)
            deltas.append(delta)

        if not rels:
            continue

        seg0 = min(rels)
        rels = [x - seg0 for x in rels]
        max_rel = max(max_rel, max(rels))

        events = np.zeros(len(rels), dtype=float)
        prev_total = None
        for i, total in enumerate(totals):
            delta = deltas[i]
            if math.isnan(delta):
                if prev_total is None:
                    delta = 0.0
                else:
                    delta = max(0.0, total - prev_total)
            prev_total = total
            events[i] = max(0.0, delta)

        n_bins = int(math.floor(max(rels) / bin_s)) + 1
        hist = np.zeros(n_bins, dtype=float)
        for rel, evt in zip(rels, events):
            idx = int(rel // bin_s)
            hist[idx] += evt
        bins_by_seg.append(hist)

    if not bins_by_seg:
        raise RuntimeError(f'no metrics_timeseries.csv found under {run_dir}')

    n_bins = max(len(x) for x in bins_by_seg)
    stacked = np.full((len(bins_by_seg), n_bins), np.nan, dtype=float)
    for i, hist in enumerate(bins_by_seg):
        stacked[i, :len(hist)] = hist

    mean_hist = np.nanmean(stacked, axis=0)
    xs = np.arange(n_bins, dtype=float) * bin_s
    return xs, mean_hist


def _load_stall_gaps(run_dir: str) -> np.ndarray:
    vals: List[float] = []
    for seg_dir in _segment_dirs(run_dir):
        p = os.path.join(seg_dir, 'client_results.csv')
        if not os.path.isfile(p):
            continue
        for row in _read_csv(p):
            if row.get('http_status', '').strip() != '200':
                continue
            gap = _to_float(row.get('stall_gap_max_s', 'nan'))
            if not math.isnan(gap):
                vals.append(gap)
    if not vals:
        raise RuntimeError(f'no stall_gap_max_s samples under {run_dir}')
    return np.array(vals, dtype=float)


def _plot_preemption(specs: Sequence[SeriesSpec], bin_s: float, out_prefix: str) -> List[Tuple[str, str, int, float]]:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    rows = []
    series: List[Tuple[SeriesSpec, np.ndarray, np.ndarray]] = []
    common_max_x = None
    for spec in specs:
        xs, ys = _load_preemption_series(spec.run_dir, bin_s)
        series.append((spec, xs, ys))
        series_max_x = float(xs[-1]) if len(xs) else 0.0
        common_max_x = series_max_x if common_max_x is None else min(common_max_x, series_max_x)

    if common_max_x is None:
        raise RuntimeError('no preemption series to plot')

    for spec, xs, ys in series:
        keep = xs <= common_max_x + 1e-9
        xs_clip = xs[keep]
        ys_clip = ys[keep]
        ax.plot(xs_clip, ys_clip, linewidth=2.4, color=spec.color, linestyle=spec.linestyle, label=spec.label)
        rows.append((spec.label, spec.run_dir, len(xs_clip), float(np.nanmax(ys_clip))))

    ax.set_xlim(0.0, common_max_x)
    ax.set_xlabel('Time since segment start (s)')
    ax.set_ylabel(f'Windowed preemptions / {bin_s:.0f}s')
    ax.set_title('Preemption Clustering Under Memory Pressure (lambda=1.0, common window)')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_prefix + '.png', dpi=220)
    fig.savefig(out_prefix + '.pdf')
    plt.close(fig)
    return rows


def _plot_stall_cdf(specs: Sequence[SeriesSpec], out_prefix: str) -> List[Tuple[str, str, int, float, float, float]]:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    rows = []
    positive_ms_all: List[float] = []
    series: List[Tuple[SeriesSpec, np.ndarray]] = []
    for spec in specs:
        vals = np.sort(_load_stall_gaps(spec.run_dir))
        series.append((spec, vals))
        positive_ms_all.extend((vals[vals > 0.0] * 1000.0).tolist())
        rows.append((
            spec.label,
            spec.run_dir,
            int(len(vals)),
            float(np.quantile(vals, 0.99)),
            float(np.quantile(vals, 0.999)),
            float(np.max(vals)),
        ))

    min_positive_ms = min(positive_ms_all) if positive_ms_all else 1.0
    floor_ms = max(min_positive_ms / 2.0, 1e-3)

    for spec, vals in series:
        vals_ms = np.maximum(vals * 1000.0, floor_ms)
        ys = np.arange(1, len(vals_ms) + 1, dtype=float) / len(vals_ms)
        ax.step(vals_ms, ys, where='post', linewidth=2.4, color=spec.color, linestyle=spec.linestyle, label=spec.label)

    ax.set_xscale('log')
    ax.set_xlim(left=floor_ms)
    ax.set_xlabel('Per-request max stall gap (ms, log scale)')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of Per-request Max Stall Gap (lambda=1.0)')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_prefix + '.png', dpi=220)
    fig.savefig(out_prefix + '.pdf')
    plt.close(fig)
    return rows


def _write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for row in rows:
            w.writerow(list(row))


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--nochunk-run', default=os.path.join(repo_root, 'logs/RecoveryGen/Baseline_no_chunked/client_logs/baseline_nochunk_lam1p0_mem0p75_retry_20260316_194026'))
    ap.add_argument('--baseline-run', default=os.path.join(repo_root, 'logs/RecoveryGen/Baseline/client_logs/baseline_lam1p0_mem0p75_20260316_151434_20260316_151435'))
    ap.add_argument('--recovergen-run', default=os.path.join(repo_root, 'logs/RecoveryGen/RecoverGen/client_logs/recovergen_lam1p0_mem0p75_retry_20260316_163608'))
    ap.add_argument('--plot-dir', required=True)
    ap.add_argument('--bin-s', type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    specs = [
        SeriesSpec('vllm (no_chunk)', args.nochunk_run, '#1F77B4', '--'),
        SeriesSpec('vllm', args.baseline_run, '#D62728', '-'),
        SeriesSpec('RecoverGen', args.recovergen_run, '#2CA02C', '-'),
    ]

    preempt_rows = _plot_preemption(specs, args.bin_s, os.path.join(args.plot_dir, 'fig_preemption_vs_time_lambda1p0'))
    stall_rows = _plot_stall_cdf(specs, os.path.join(args.plot_dir, 'fig_stall_gap_cdf_lambda1p0'))

    _write_csv(
        os.path.join(args.plot_dir, 'fig_preemption_vs_time_lambda1p0_summary.csv'),
        ['method', 'run_dir', 'n_time_bins', 'max_windowed_preemptions'],
        preempt_rows,
    )
    _write_csv(
        os.path.join(args.plot_dir, 'fig_stall_gap_cdf_lambda1p0_summary.csv'),
        ['method', 'run_dir', 'samples', 'stall_gap_p99_s', 'stall_gap_p999_s', 'stall_gap_max_s'],
        stall_rows,
    )

    print(f'[INFO] wrote plots to {args.plot_dir}')
    for row in preempt_rows:
        print(f'[INFO] preemption {row[0]} max_windowed_preemptions={row[3]:.3f} run={row[1]}')
    for row in stall_rows:
        print(f'[INFO] stall {row[0]} p99={row[3]:.4f}s p999={row[4]:.4f}s max={row[5]:.4f}s run={row[1]}')


if __name__ == '__main__':
    main()
