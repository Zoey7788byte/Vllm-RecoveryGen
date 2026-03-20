#!/usr/bin/env python3
"""Plot real-trace tail percentiles for TTFT and TPOT.

Figure layout:
  - x-axis: Percentile (p50, p90, p99, p99.9)
  - subplot (a): TTFT (s)
  - subplot (b): TPOT (s/token)
  - methods: vllm (no_chunk), vllm, RecoverGen
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_SPECS = [
    ("vllm (no_chunk)", "baseline_nochunk", "#4C72B0", "o"),
    ("vllm", "baseline", "#D62728", "s"),
    ("RecoverGen", "recovergen", "#2CA02C", "D"),
]


@dataclass
class RunData:
    label: str
    run_dir: str
    ttft: np.ndarray
    tpot: np.ndarray


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: str) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float("nan")


def _lambda_tag(lam: float) -> str:
    s = f"{lam:.2f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return "lam" + s.replace(".", "p")


def _parse_lambda_from_seg(name: str) -> float:
    m = re.search(r"lam(\d+)p(\d+)", name)
    if not m:
        return float("nan")
    return float(f"{int(m.group(1))}.{m.group(2)}")


def _candidate_runs(root: str, target_lambda: float) -> List[str]:
    tag = _lambda_tag(target_lambda)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"root not found: {root}")
    out = []
    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and tag in entry:
            out.append(full)
    return sorted(out, reverse=True)


def _collect_arrays(run_dir: str, target_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    ttft_vals: List[float] = []
    tpot_vals: List[float] = []
    for entry in os.listdir(run_dir):
        seg_dir = os.path.join(run_dir, entry)
        if not os.path.isdir(seg_dir):
            continue
        lam = _parse_lambda_from_seg(entry)
        if math.isnan(lam) or abs(lam - target_lambda) > 1e-9:
            continue
        path = os.path.join(seg_dir, "client_results.csv")
        if not os.path.isfile(path):
            continue
        for row in _read_csv(path):
            status = row.get("http_status", "").strip()
            if status and status != "200":
                continue
            ttft = _to_float(row.get("ttft_s", "nan"))
            out_tok = _to_float(row.get("out_tok", "nan"))
            t1 = _to_float(row.get("t_first_token_abs", "nan"))
            t2 = _to_float(row.get("t_done_abs", "nan"))
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
        raise RuntimeError(f"no TTFT samples under {run_dir}")
    return np.array(ttft_vals, dtype=float), np.array(tpot_vals, dtype=float)


def _load_runs(target_lambda: float, repo_root: str) -> List[RunData]:
    root_map = {
        "baseline_nochunk": os.path.join(repo_root, "logs/RecoveryGen/Baseline_no_chunked/client_logs"),
        "baseline": os.path.join(repo_root, "logs/RecoveryGen/Baseline/client_logs"),
        "recovergen": os.path.join(repo_root, "logs/RecoveryGen/RecoverGen/client_logs"),
    }
    runs: List[RunData] = []
    for label, key, _color, _marker in METHOD_SPECS:
        selected_run = None
        selected_ttft = None
        selected_tpot = None
        for run_dir in _candidate_runs(root_map[key], target_lambda):
            try:
                ttft, tpot = _collect_arrays(run_dir, target_lambda)
            except Exception:
                continue
            if ttft.size > 0 and tpot.size > 0:
                selected_run = run_dir
                selected_ttft = ttft
                selected_tpot = tpot
                break
        if selected_run is None or selected_ttft is None or selected_tpot is None:
            raise RuntimeError(f"no valid run for {label} @ lambda={target_lambda}")
        runs.append(RunData(label=label, run_dir=selected_run, ttft=selected_ttft, tpot=selected_tpot))
    return runs


def _quantiles(arr: np.ndarray, probs: Sequence[float]) -> np.ndarray:
    if arr.size == 0:
        return np.full(len(probs), np.nan, dtype=float)
    return np.quantile(arr, probs)


def _write_csv(runs: List[RunData], probs: Sequence[float], out_csv: str) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "method",
            "ttft_p50_s",
            "ttft_p90_s",
            "ttft_p99_s",
            "ttft_p999_s",
            "tpot_p50_s_per_tok",
            "tpot_p90_s_per_tok",
            "tpot_p99_s_per_tok",
            "tpot_p999_s_per_tok",
            "ttft_samples",
            "tpot_samples",
            "run_dir",
        ])
        for run in runs:
            tt = _quantiles(run.ttft, probs)
            tp = _quantiles(run.tpot, probs)
            w.writerow([
                run.label,
                *[f"{x:.6f}" for x in tt],
                *[f"{x:.6f}" for x in tp],
                run.ttft.size,
                run.tpot.size,
                run.run_dir,
            ])


def _plot(runs: List[RunData], probs: Sequence[float], out_prefix: str) -> None:
    xlabels = ["p50", "p90", "p99", "p99.9"]
    xs = np.arange(len(probs))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)

    for label, _key, color, marker in METHOD_SPECS:
        run = next(r for r in runs if r.label == label)
        axes[0].plot(
            xs, _quantiles(run.ttft, probs), color=color, marker=marker,
            linewidth=2.2, markersize=7, label=label)
        axes[1].plot(
            xs, _quantiles(run.tpot, probs), color=color, marker=marker,
            linewidth=2.2, markersize=7, label=label)

    axes[0].set_ylabel("TTFT (s)")
    axes[1].set_ylabel("TPOT (s/token)")
    axes[0].set_title("TTFT")
    axes[1].set_title("TPOT")
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Percentile")
        ax.grid(True, alpha=0.25)
    axes[1].legend(frameon=True, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=220)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-lambda", type=float, default=0.9)
    ap.add_argument(
        "--plot-dir",
        default=os.path.join(repo_root, "logs/RecoveryGen/real_trace_tail_percentiles"),
    )
    args = ap.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    probs = [0.50, 0.90, 0.99, 0.999]
    runs = _load_runs(args.target_lambda, repo_root)

    lam_tag = _lambda_tag(args.target_lambda)
    out_prefix = os.path.join(args.plot_dir, f"fig1_real_trace_tail_{lam_tag}")
    _plot(runs, probs, out_prefix)
    _write_csv(runs, probs, out_prefix + "_points.csv")

    print(f"[INFO] wrote {out_prefix}.png/.pdf")
    print(f"[INFO] wrote {out_prefix}_points.csv")
    for run in runs:
        print(f"[INFO] {run.label}: ttft_n={run.ttft.size} tpot_n={run.tpot.size} run={run.run_dir}")


if __name__ == "__main__":
    main()
