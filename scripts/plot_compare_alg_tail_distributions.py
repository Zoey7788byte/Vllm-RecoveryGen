#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHOD_SPECS = [
    ('vllm (no_chunk)', 'baseline_nochunk', '#4C72B0', '//'),
    ('vllm', 'baseline', '#DD8452', '\\\\'),
    ('RecoverGen (no_chunk)', 'recovergen_nochunk', '#55A868', '++'),
    ('RecoverGen', 'recovergen', '#C44E52', '**'),
]


@dataclass
class RunData:
    label: str
    run_dir: str
    ttft: np.ndarray
    tpot: np.ndarray


def _parse_lambda_tag_from_name(name: str) -> float:
    m = re.search(r'lam(\d+)p(\d+)', name)
    if not m:
        return float('nan')
    return float(f"{int(m.group(1))}.{m.group(2)}")


def _lambda_tag(lam: float) -> str:
    s = f"{lam:.2f}".rstrip('0').rstrip('.')
    if '.' not in s:
        s += '.0'
    return 'lam' + s.replace('.', 'p')


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


def _candidate_runs_with_lambda(root: str, target_lambda: float) -> List[str]:
    tag = _lambda_tag(target_lambda)
    candidates = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f'root not found: {root}')
    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        if tag not in entry:
            continue
        candidates.append(full)
    if not candidates:
        raise FileNotFoundError(f'no run matching {tag} under {root}')
    return sorted(candidates, reverse=True)


def _collect_segment_client_results(run_dir: str, target_lambda: float) -> List[str]:
    out = []
    for entry in os.listdir(run_dir):
        seg_dir = os.path.join(run_dir, entry)
        if not os.path.isdir(seg_dir):
            continue
        lam = _parse_lambda_tag_from_name(entry)
        if math.isnan(lam) or abs(lam - target_lambda) > 1e-9:
            continue
        p = os.path.join(seg_dir, 'client_results.csv')
        if os.path.isfile(p):
            out.append(p)
    return sorted(out)


def _load_request_arrays(run_dir: str, target_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    ttft_vals: List[float] = []
    tpot_vals: List[float] = []
    for path in _collect_segment_client_results(run_dir, target_lambda):
        for row in _read_csv(path):
            status = row.get('http_status', '').strip()
            if status and status != '200':
                continue
            ttft = _to_float(row.get('ttft_s', 'nan'))
            out_tok = _to_float(row.get('out_tok', 'nan'))
            t1 = _to_float(row.get('t_first_token_abs', 'nan'))
            t2 = _to_float(row.get('t_done_abs', 'nan'))
            if not math.isnan(ttft):
                ttft_vals.append(ttft)
            if (
                not math.isnan(out_tok)
                and out_tok > 1.0
                and not math.isnan(t1)
                and not math.isnan(t2)
                and t2 > t1
            ):
                tpot_vals.append((t2 - t1) / (out_tok - 1.0))
    if not ttft_vals:
        raise RuntimeError(f'no TTFT samples under {run_dir}')
    return np.array(ttft_vals, dtype=float), np.array(tpot_vals, dtype=float)


def _quantiles(arr: np.ndarray, probs: Sequence[float]) -> np.ndarray:
    if arr.size == 0:
        return np.full(len(probs), np.nan, dtype=float)
    return np.quantile(arr, probs)


def _ccdf_curve(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(arr)
    n = x.size
    y = 1.0 - np.arange(1, n + 1, dtype=float) / n
    y = np.clip(y, 1.0 / n, 1.0)
    return x, y


def _plot_ccdf(runs: List[RunData], metric_name: str, series_name: str, out_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for label, key, color, _hatch in METHOD_SPECS:
        run = next(r for r in runs if r.label == label)
        arr = getattr(run, series_name)
        x, y = _ccdf_curve(arr)
        ax.step(x, y, where='post', linewidth=2.0, color=color, label=f'{label} (n={arr.size})')
        p99 = float(np.quantile(arr, 0.99))
        p999 = float(np.quantile(arr, 0.999))
        ax.axvline(p99, color=color, linewidth=1.0, linestyle='--', alpha=0.45)
        ax.axvline(p999, color=color, linewidth=1.0, linestyle=':', alpha=0.45)
    ax.set_yscale('log')
    ax.set_xlabel(metric_name)
    ax.set_ylabel('P(X > x)')
    ax.set_title(f'{metric_name} CCDF @ lambda=0.9')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_prefix + '.png', dpi=200)
    fig.savefig(out_prefix + '.pdf')
    plt.close(fig)


def _plot_tail_quantile(runs: List[RunData], metric_name: str, series_name: str, out_prefix: str) -> None:
    qs = np.array([0.90, 0.95, 0.99, 0.995, 0.999, 0.9995], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for label, _key, color, _hatch in METHOD_SPECS:
        run = next(r for r in runs if r.label == label)
        arr = getattr(run, series_name)
        vals = _quantiles(arr, qs)
        ax.plot(qs, vals, marker='o', linewidth=2.0, color=color, label=label)
    ax.set_xlabel('Quantile q')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Tail Quantiles @ lambda=0.9')
    ax.set_xticks(qs)
    ax.set_xticklabels(['0.90', '0.95', '0.99', '0.995', '0.999', '0.9995'], rotation=20)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_prefix + '.png', dpi=200)
    fig.savefig(out_prefix + '.pdf')
    plt.close(fig)


def _write_tail_table(runs: List[RunData], out_csv: str) -> None:
    probs = [0.90, 0.95, 0.99, 0.999]
    headers = [
        'method', 'ttft_p90_s', 'ttft_p95_s', 'ttft_p99_s', 'ttft_p999_s',
        'tpot_p90_s', 'tpot_p95_s', 'tpot_p99_s', 'tpot_p999_s',
        'ttft_samples', 'tpot_samples',
    ]
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for label, _key, _color, _hatch in METHOD_SPECS:
            run = next(r for r in runs if r.label == label)
            tt = _quantiles(run.ttft, probs)
            tp = _quantiles(run.tpot, probs)
            w.writerow([
                label,
                *[f'{x:.6f}' for x in tt],
                *[f'{x:.6f}' for x in tp],
                run.ttft.size,
                run.tpot.size,
            ])


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--target-lambda', type=float, required=True)
    ap.add_argument('--baseline-root', default=os.path.join(repo_root, 'logs/RecoveryGen/Baseline/client_logs'))
    ap.add_argument('--recovergen-root', default=os.path.join(repo_root, 'logs/RecoveryGen/RecoverGen/client_logs'))
    ap.add_argument('--nochunk-root', default=os.path.join(repo_root, 'logs/RecoveryGen/Baseline_no_chunked/client_logs'))
    ap.add_argument('--recovergen-nochunk-root', default=os.path.join(repo_root, 'logs/RecoveryGen/RecoverGen_no_chunked/client_logs'))
    ap.add_argument('--plot-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    root_map = {
        'baseline_nochunk': args.nochunk_root,
        'baseline': args.baseline_root,
        'recovergen_nochunk': args.recovergen_nochunk_root,
        'recovergen': args.recovergen_root,
    }

    runs: List[RunData] = []
    for label, key, _color, _hatch in METHOD_SPECS:
        selected_run = None
        selected_ttft = None
        selected_tpot = None
        for run_dir in _candidate_runs_with_lambda(root_map[key], args.target_lambda):
            try:
                ttft, tpot = _load_request_arrays(run_dir, args.target_lambda)
            except Exception:
                continue
            if ttft.size > 0:
                selected_run = run_dir
                selected_ttft = ttft
                selected_tpot = tpot
                break
        if selected_run is None or selected_ttft is None or selected_tpot is None:
            raise RuntimeError(f'no valid run with samples for {label} under {root_map[key]}')
        runs.append(RunData(label=label, run_dir=selected_run, ttft=selected_ttft, tpot=selected_tpot))

    lam_tag = _lambda_tag(args.target_lambda)
    _plot_ccdf(runs, 'TTFT (s)', 'ttft', os.path.join(args.plot_dir, f'{lam_tag}_ttft_ccdf'))
    _plot_ccdf(runs, 'TPOT (s/token)', 'tpot', os.path.join(args.plot_dir, f'{lam_tag}_tpot_ccdf'))
    _plot_tail_quantile(runs, 'TTFT (s)', 'ttft', os.path.join(args.plot_dir, f'{lam_tag}_ttft_tail_quantiles'))
    _plot_tail_quantile(runs, 'TPOT (s/token)', 'tpot', os.path.join(args.plot_dir, f'{lam_tag}_tpot_tail_quantiles'))
    _write_tail_table(runs, os.path.join(args.plot_dir, f'{lam_tag}_tail_summary.csv'))

    print(f'[INFO] wrote plots to {args.plot_dir}')
    for run in runs:
        print(f'[INFO] {run.label}: run_dir={run.run_dir} ttft_n={run.ttft.size} tpot_n={run.tpot.size}')


if __name__ == '__main__':
    main()
