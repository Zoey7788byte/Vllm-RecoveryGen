#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SessionCurve:
    total_host_blocks: int
    xs_ms: np.ndarray
    ys_cov: np.ndarray


@dataclass
class MethodSpec:
    label: str
    events_path: str
    color: str


def _iter_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_curves(events_path: str) -> List[SessionCurve]:
    sessions: Dict[str, Dict[str, object]] = {}

    for rec in _iter_jsonl(events_path):
        event = rec.get('event')
        seq_id = rec.get('seq_id')
        if seq_id is None:
            continue
        key = str(seq_id)
        ts_ns = int(rec['ts_ns'])
        detail = rec.get('detail') or {}

        if event == 'PREEMPT_TRIGGERED':
            sess = sessions.setdefault(key, {
                'start_ns': ts_ns,
                'total_blocks': 0,
                'points': [(0.0, 0.0)],
                'restored_host_blocks': 0.0,
                'total_host_blocks': 0.0,
            })
            sess['start_ns'] = ts_ns
            continue

        if event == 'RECOVERY_STATE_CREATED':
            total_blocks = int(detail.get('num_blocks') or 0)
            sess = sessions.setdefault(key, {
                'start_ns': ts_ns,
                'total_blocks': 0,
                'points': [(0.0, 0.0)],
                'restored_host_blocks': 0.0,
                'total_host_blocks': 0.0,
            })
            sess['total_blocks'] = total_blocks
            sess.setdefault('points', [(0.0, 0.0)])
            continue

        sess = sessions.get(key)
        if not sess:
            continue

        if event == 'SWAP_OUT':
            sess['total_host_blocks'] = float(sess['total_host_blocks']) + float(detail.get('blocks_count') or 0.0)
            continue

        if event == 'SWAP_IN':
            total_host_blocks = float(sess['total_host_blocks'])
            if total_host_blocks <= 0:
                continue
            restored = float(sess['restored_host_blocks']) + float(detail.get('blocks_count') or 0.0)
            sess['restored_host_blocks'] = min(total_host_blocks, restored)
            rel_ms = (ts_ns - int(sess['start_ns'])) / 1e6
            cov = min(1.0, float(sess['restored_host_blocks']) / total_host_blocks)
            sess['points'].append((rel_ms, cov))
            continue

        if event == 'RECOVERY_REMAINING_HOST_BLOCKS':
            total_host_blocks = float(sess['total_host_blocks'])
            if total_host_blocks <= 0:
                continue
            n_host = float(detail.get('n_host') or 0.0)
            rel_ms = (ts_ns - int(sess['start_ns'])) / 1e6
            restored = max(0.0, total_host_blocks - n_host)
            sess['restored_host_blocks'] = max(float(sess['restored_host_blocks']), restored)
            cov = min(1.0, float(sess['restored_host_blocks']) / total_host_blocks)
            sess['points'].append((rel_ms, cov))
            continue

        if event == 'RECOVERY_PROGRESS_COMMIT':
            # Commit alone is not counted as restored progress here.
            # We only count host blocks that have actually been swapped back in,
            # or equivalently disappeared from remaining host blocks.
            continue

    curves: List[SessionCurve] = []
    for sess in sessions.values():
        total_host_blocks = int(round(float(sess['total_host_blocks'])))
        points = list(sess['points'])
        if total_host_blocks <= 0 or len(points) <= 1:
            continue
        points.sort(key=lambda x: x[0])
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)
        curves.append(SessionCurve(total_host_blocks=total_host_blocks, xs_ms=xs, ys_cov=ys))
    if not curves:
        raise RuntimeError(f'no recovery sessions parsed from {events_path}')
    return curves


def _coverage_on_grid(curve: SessionCurve, grid_ms: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid_ms, dtype=float)
    j = 0
    cur = 0.0
    for i, t in enumerate(grid_ms):
        while j < len(curve.xs_ms) and curve.xs_ms[j] <= t + 1e-12:
            cur = curve.ys_cov[j]
            j += 1
        out[i] = cur
    return out


def _aggregate_curve(curves: Sequence[SessionCurve], bin_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    max_ms = min(float(max(c.xs_ms[-1] for c in curves)), 5000.0)
    n_bins = int(math.floor(max_ms / bin_ms)) + 1
    grid = np.arange(n_bins, dtype=float) * bin_ms
    stacked = np.vstack([_coverage_on_grid(c, grid) for c in curves])
    return grid, stacked.mean(axis=0)


def _completion_ms(curve: SessionCurve) -> float:
    done = np.where(curve.ys_cov >= 0.999999)[0]
    if done.size:
        return float(curve.xs_ms[done[0]])
    return float(curve.xs_ms[-1])


def _write_summary(path: str, rows: Sequence[Sequence[object]]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'method',
            'events_path',
            'n_sessions',
            'median_total_blocks',
            'completion_p50_ms',
            'completion_p90_ms',
            'completion_max_ms',
        ])
        for row in rows:
            w.writerow(list(row))


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--baseline-events',
        default=os.path.join(
            repo_root,
            'logs/RecoveryGen/Baseline/server_logs/20260318_124303/recovery/recovery_events.jsonl',
        ),
    )
    ap.add_argument(
        '--recovergen-events',
        default=os.path.join(
            repo_root,
            'logs/RecoveryGen/RecoverGen/server_logs/20260317_115210/recovery/recovery_events.jsonl',
        ),
    )
    ap.add_argument('--plot-dir', required=True)
    ap.add_argument('--bin-ms', type=float, default=20.0)
    ap.add_argument('--xmax-ms', type=float, default=500.0)
    args = ap.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    specs = [
        MethodSpec('vllm', args.baseline_events, '#D62728'),
        MethodSpec('RecoverGen', args.recovergen_events, '#2CA02C'),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    summary_rows = []
    for spec in specs:
        curves = _load_curves(spec.events_path)
        grid_ms, mean_cov = _aggregate_curve(curves, args.bin_ms)
        ax.plot(grid_ms, mean_cov * 100.0, linewidth=2.4, color=spec.color, label=spec.label)

        total_blocks = np.array([c.total_host_blocks for c in curves], dtype=float)
        completion_ms = np.array([_completion_ms(c) for c in curves], dtype=float)
        summary_rows.append([
            spec.label,
            spec.events_path,
            len(curves),
            f'{np.median(total_blocks):.1f}',
            f'{np.quantile(completion_ms, 0.50):.1f}',
            f'{np.quantile(completion_ms, 0.90):.1f}',
            f'{np.max(completion_ms):.1f}',
        ])

    ax.set_xlabel('Time since preemption trigger (ms)')
    ax.set_ylabel('Strict restored host-KV coverage (%)')
    ax.set_title('Strict Host-KV Restore Progress Under Memory Pressure (lambda=1.1)')
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(0.0, args.xmax_ms)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    out_prefix = os.path.join(args.plot_dir, 'fig_recovery_progress_lambda1p1')
    fig.savefig(out_prefix + '.png', dpi=220)
    fig.savefig(out_prefix + '.pdf')
    plt.close(fig)

    _write_summary(out_prefix + '_summary.csv', summary_rows)
    print(f'[INFO] wrote {out_prefix}.png/.pdf')
    for row in summary_rows:
        print(f'[INFO] {row[0]} n_sessions={row[2]} completion_p50_ms={row[4]} completion_p90_ms={row[5]}')


if __name__ == '__main__':
    main()
