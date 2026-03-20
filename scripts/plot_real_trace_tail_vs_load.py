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
    ('vllm (no_chunk)', 'baseline_nochunk', '#4C72B0', 'o'),
    ('vllm', 'baseline', '#D62728', 's'),
    ('RecoverGen (no_chunk)', 'recovergen_nochunk', '#17BECF', '^'),
    ('RecoverGen', 'recovergen', '#2CA02C', 'D'),
]
METHOD_LOOKUP = {label: (label, key, color, marker) for label, key, color, marker in METHOD_SPECS}

@dataclass
class Point:
    lam: float
    ttft_p99: float
    tpot_p99: float
    run_dir: str
    n_ttft: int
    n_tpot: int


def _lambda_tag(lam: float) -> str:
    s = f"{lam:.2f}".rstrip('0').rstrip('.')
    if '.' not in s:
        s += '.0'
    return 'lam' + s.replace('.', 'p')


def _parse_lambda_from_seg(seg_name: str) -> float:
    m = re.search(r'lam(\d+)p(\d+)', seg_name)
    if not m:
        return float('nan')
    return float(f"{int(m.group(1))}.{m.group(2)}")


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


def _candidate_runs(root: str, lam: float) -> List[str]:
    tag = _lambda_tag(lam)
    if not os.path.isdir(root):
        return []
    out = []
    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and tag in entry:
            out.append(full)
    return sorted(out, reverse=True)


def _collect_request_arrays(run_dir: str, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    ttft_vals: List[float] = []
    tpot_vals: List[float] = []
    for entry in os.listdir(run_dir):
        seg_dir = os.path.join(run_dir, entry)
        if not os.path.isdir(seg_dir):
            continue
        seg_lam = _parse_lambda_from_seg(entry)
        if math.isnan(seg_lam) or abs(seg_lam - lam) > 1e-9:
            continue
        p = os.path.join(seg_dir, 'client_results.csv')
        if not os.path.isfile(p):
            continue
        for row in _read_csv(p):
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
    return np.array(ttft_vals, dtype=float), np.array(tpot_vals, dtype=float)


def _pick_point(root: str, lam: float) -> Point:
    for run_dir in _candidate_runs(root, lam):
        ttft, tpot = _collect_request_arrays(run_dir, lam)
        if ttft.size == 0 or tpot.size == 0:
            continue
        return Point(
            lam=lam,
            ttft_p99=float(np.quantile(ttft, 0.99)),
            tpot_p99=float(np.quantile(tpot, 0.99)),
            run_dir=run_dir,
            n_ttft=int(ttft.size),
            n_tpot=int(tpot.size),
        )
    raise RuntimeError(f'no valid run for lambda={lam} under {root}')


def _plot(points_by_method: Dict[str, List[Point]], method_specs: List[Tuple[str, str, str, str]], metric: str, ylabel: str, out_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, _key, color, marker in method_specs:
        pts = sorted(points_by_method[label], key=lambda x: x.lam)
        xs = [p.lam for p in pts]
        ys = [getattr(p, metric) for p in pts]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=2.2, markersize=7, label=label)
    ax.set_xlabel('Offered load λ (requests/s)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_prefix + '.png', dpi=220)
    fig.savefig(out_prefix + '.pdf')
    plt.close(fig)


def _write_csv(points_by_method: Dict[str, List[Point]], method_specs: List[Tuple[str, str, str, str]], out_csv: str) -> None:
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'lambda', 'ttft_p99_s_req', 'tpot_p99_s_req', 'ttft_samples', 'tpot_samples', 'run_dir'])
        for label, _key, _color, _marker in method_specs:
            for p in sorted(points_by_method[label], key=lambda x: x.lam):
                w.writerow([label, f'{p.lam:.2f}', f'{p.ttft_p99:.6f}', f'{p.tpot_p99:.6f}', p.n_ttft, p.n_tpot, p.run_dir])


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--lambdas', default='0.9,1.0')
    ap.add_argument('--baseline-root', default=os.path.join(repo_root, 'logs/RecoveryGen/Baseline/client_logs'))
    ap.add_argument('--recovergen-root', default=os.path.join(repo_root, 'logs/RecoveryGen/RecoverGen/client_logs'))
    ap.add_argument('--nochunk-root', default=os.path.join(repo_root, 'logs/RecoveryGen/Baseline_no_chunked/client_logs'))
    ap.add_argument('--recovergen-nochunk-root', default=os.path.join(repo_root, 'logs/RecoveryGen/RecoverGen_no_chunked/client_logs'))
    ap.add_argument('--plot-dir', required=True)
    ap.add_argument('--methods', default='vllm (no_chunk),vllm,RecoverGen (no_chunk),RecoverGen')
    args = ap.parse_args()

    lambdas = [float(x) for x in re.split(r'[ ,]+', args.lambdas.strip()) if x]
    selected_labels = [x.strip() for x in args.methods.split(',') if x.strip()]
    method_specs: List[Tuple[str, str, str, str]] = []
    for label in selected_labels:
        if label not in METHOD_LOOKUP:
            raise SystemExit(f'unknown method label: {label}')
        method_specs.append(METHOD_LOOKUP[label])
    os.makedirs(args.plot_dir, exist_ok=True)
    roots = {
        'vllm (no_chunk)': args.nochunk_root,
        'vllm': args.baseline_root,
        'RecoverGen (no_chunk)': args.recovergen_nochunk_root,
        'RecoverGen': args.recovergen_root,
    }

    points_by_method: Dict[str, List[Point]] = {label: [] for label, *_ in method_specs}
    for label, _key, _color, _marker in method_specs:
        for lam in lambdas:
            points_by_method[label].append(_pick_point(roots[label], lam))

    _plot(points_by_method, method_specs, 'ttft_p99', 'p99 TTFT (s)', os.path.join(args.plot_dir, 'fig_real_trace_tail_ttft_p99_vs_lambda'))
    _plot(points_by_method, method_specs, 'tpot_p99', 'p99 TPOT (s/token)', os.path.join(args.plot_dir, 'fig_real_trace_tail_tpot_p99_vs_lambda'))
    _write_csv(points_by_method, method_specs, os.path.join(args.plot_dir, 'fig_real_trace_tail_points.csv'))

    print(f'[INFO] wrote plots to {args.plot_dir}')
    for label, _key, _color, _marker in method_specs:
        for p in sorted(points_by_method[label], key=lambda x: x.lam):
            print(f'[INFO] {label} lambda={p.lam:.2f} ttft_p99={p.ttft_p99:.4f} tpot_p99={p.tpot_p99:.4f} run={p.run_dir}')


if __name__ == '__main__':
    main()
