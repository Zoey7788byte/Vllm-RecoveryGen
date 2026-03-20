#!/usr/bin/env python3
import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MethodSpec:
    label: str
    ts_path: str
    color: str


def _read_rows(path: str):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _load_windows(ts_path: str, xmax_ms: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    rows = _read_rows(ts_path)
    ts_ms = np.array([_to_float(r.get('t_rel_s', '0')) * 1000.0 for r in rows], dtype=float)
    visible = np.array([_to_float(r.get('visible_tokens', '0')) for r in rows], dtype=float)
    preempt_delta = np.array([_to_float(r.get('on_preempt_count_delta', '0')) for r in rows], dtype=float)

    trigger_idx = np.where(preempt_delta > 0)[0]
    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx_pos, idx in enumerate(trigger_idx):
        t0 = ts_ms[idx]
        next_t0 = ts_ms[trigger_idx[idx_pos + 1]] if idx_pos + 1 < len(trigger_idx) else np.inf
        keep = (ts_ms >= t0) & (ts_ms <= min(t0 + xmax_ms, next_t0))
        xs = ts_ms[keep] - t0
        ys = visible[keep]
        if len(xs) == 0:
            continue
        windows.append((xs, ys))
    if not windows:
        raise RuntimeError(f'no preemption-triggered windows found in {ts_path}')
    return windows


def _window_on_grid(xs: np.ndarray, ys: np.ndarray, grid_ms: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid_ms, dtype=float)
    j = 0
    cur = 0.0
    for i, t in enumerate(grid_ms):
        while j < len(xs) and xs[j] <= t + 1e-12:
            cur = ys[j]
            j += 1
        out[i] = cur
    return out


def _aggregate_windows(
    windows: Sequence[Tuple[np.ndarray, np.ndarray]], xmax_ms: float, bin_ms: float
) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = int(np.floor(xmax_ms / bin_ms)) + 1
    grid = np.arange(n_bins, dtype=float) * bin_ms
    stacked = np.vstack([_window_on_grid(xs, ys, grid) for xs, ys in windows])
    return grid, stacked.mean(axis=0)


def _write_summary(path: str, rows: Sequence[Sequence[object]]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'method',
            'ts_path',
            'n_windows',
            'visible_tokens_mean_max',
            'first_nonzero_mean_ms',
        ])
        for row in rows:
            w.writerow(list(row))


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--baseline-ts',
        default=os.path.join(
            repo_root,
            'logs/RecoveryGen/Baseline/server_logs/20260318_124303/recovery/recovery_ts.csv',
        ),
    )
    ap.add_argument(
        '--recovergen-ts',
        default=os.path.join(
            repo_root,
            'logs/RecoveryGen/RecoverGen/server_logs/20260317_115210/recovery/recovery_ts.csv',
        ),
    )
    ap.add_argument('--plot-dir', required=True)
    ap.add_argument('--xmax-ms', type=float, default=5000.0)
    ap.add_argument('--bin-ms', type=float, default=20.0)
    args = ap.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    specs = [
        MethodSpec('vllm', args.baseline_ts, '#D62728'),
        MethodSpec('RecoverGen', args.recovergen_ts, '#2CA02C'),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    summary_rows = []

    for spec in specs:
        windows = _load_windows(spec.ts_path, args.xmax_ms)
        grid_ms, mean_visible = _aggregate_windows(windows, args.xmax_ms, args.bin_ms)
        ax.plot(grid_ms, mean_visible, linewidth=2.4, color=spec.color, label=spec.label)

        nz = np.where(mean_visible > 0)[0]
        first_nonzero_ms = '' if nz.size == 0 else f'{grid_ms[nz[0]]:.1f}'
        summary_rows.append([
            spec.label,
            spec.ts_path,
            len(windows),
            f'{mean_visible.max():.1f}',
            first_nonzero_ms,
        ])

    ax.set_xlabel('Time since preemption trigger (ms)')
    ax.set_ylabel('Mean visible tokens')
    ax.set_title('Cycle-Aligned Visible Token Coverage After Preemption (lambda=1.1)')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    out_prefix = os.path.join(args.plot_dir, 'fig_visible_tokens_lambda1p1')
    fig.savefig(out_prefix + '.png', dpi=220)
    fig.savefig(out_prefix + '.pdf')
    plt.close(fig)

    _write_summary(out_prefix + '_summary.csv', summary_rows)
    print(f'[INFO] wrote {out_prefix}.png/.pdf')
    for row in summary_rows:
        print(f'[INFO] {row[0]} n_windows={row[2]} mean_visible_max={row[3]} first_nonzero_ms={row[4]}')


if __name__ == '__main__':
    main()
