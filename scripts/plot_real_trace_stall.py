#!/usr/bin/env python3
"""Plot real-trace streaming stall summary.

Default figure:
  - x-axis: Configuration
  - y-axis: long-gap frequency (%)
  - methods: vllm (no_chunk), vllm, RecoverGen

Also writes total long-gap time to CSV for optional alternate plotting.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_SPECS = [
    ("vllm (no_chunk)", "baseline_nochunk", "#4C72B0"),
    ("vllm", "baseline", "#D62728"),
    ("RecoverGen", "recovergen", "#2CA02C"),
]


@dataclass
class StallResult:
    label: str
    run_dir: str
    requests: int
    long_gap_threshold_s: float
    long_gap_frequency_pct: float
    total_long_gap_time_s: float
    stall_gap_p99_s: float
    stall_gap_p999_s: float
    stall_gap_max_s: float


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


def _percentile(values: List[float], q: float) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return float("nan")
    if q <= 0:
        return vals[0]
    if q >= 1:
        return vals[-1]
    k = (len(vals) - 1) * q
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def _load_stall_result(run_dir: str, target_lambda: float,
                       long_gap_threshold_s: float, label: str) -> StallResult:
    stall_gap_vals: List[float] = []
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
            stall_gap = _to_float(row.get("stall_gap_max_s", "nan"))
            if not math.isnan(stall_gap):
                stall_gap_vals.append(stall_gap)
    if not stall_gap_vals:
        raise RuntimeError(f"no stall_gap_max_s samples under {run_dir}")

    long_gaps = [x for x in stall_gap_vals if x >= long_gap_threshold_s]
    return StallResult(
        label=label,
        run_dir=run_dir,
        requests=len(stall_gap_vals),
        long_gap_threshold_s=long_gap_threshold_s,
        long_gap_frequency_pct=100.0 * len(long_gaps) / len(stall_gap_vals),
        total_long_gap_time_s=sum(long_gaps),
        stall_gap_p99_s=_percentile(stall_gap_vals, 0.99),
        stall_gap_p999_s=_percentile(stall_gap_vals, 0.999),
        stall_gap_max_s=max(stall_gap_vals),
    )


def _collect_results(repo_root: str, target_lambda: float,
                     long_gap_threshold_s: float) -> List[StallResult]:
    root_map = {
        "baseline_nochunk": os.path.join(repo_root, "logs/RecoveryGen/Baseline_no_chunked/client_logs"),
        "baseline": os.path.join(repo_root, "logs/RecoveryGen/Baseline/client_logs"),
        "recovergen": os.path.join(repo_root, "logs/RecoveryGen/RecoverGen/client_logs"),
    }

    out: List[StallResult] = []
    for label, key, _color in METHOD_SPECS:
        selected = None
        for run_dir in _candidate_runs(root_map[key], target_lambda):
            try:
                selected = _load_stall_result(
                    run_dir, target_lambda=target_lambda,
                    long_gap_threshold_s=long_gap_threshold_s, label=label)
            except Exception:
                continue
            break
        if selected is None:
            raise RuntimeError(f"no valid run for {label} @ lambda={target_lambda}")
        out.append(selected)
    return out


def _write_csv(results: List[StallResult], out_csv: str) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "method",
            "requests",
            "long_gap_threshold_s",
            "long_gap_frequency_pct",
            "total_long_gap_time_s",
            "stall_gap_p99_s",
            "stall_gap_p999_s",
            "stall_gap_max_s",
            "run_dir",
        ])
        for r in results:
            w.writerow([
                r.label,
                r.requests,
                f"{r.long_gap_threshold_s:.6f}",
                f"{r.long_gap_frequency_pct:.6f}",
                f"{r.total_long_gap_time_s:.6f}",
                f"{r.stall_gap_p99_s:.6f}",
                f"{r.stall_gap_p999_s:.6f}",
                f"{r.stall_gap_max_s:.6f}",
                r.run_dir,
            ])


def _plot(results: List[StallResult], metric: str, out_prefix: str) -> None:
    labels = [r.label for r in results]
    colors = [color for _label, _key, color in METHOD_SPECS]
    if metric == "total_long_gap_time":
        values = [r.total_long_gap_time_s for r in results]
        ylabel = "Total long-gap time (s)"
    else:
        values = [r.long_gap_frequency_pct for r in results]
        ylabel = "Long-gap frequency (%)"

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = list(range(len(labels)))
    bars = ax.bar(xs, values, width=0.68, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Configuration")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=220)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-lambda", type=float, default=0.9)
    ap.add_argument("--long-gap-threshold-s", type=float, default=1.0)
    ap.add_argument(
        "--metric",
        choices=["long_gap_frequency", "total_long_gap_time"],
        default="long_gap_frequency",
    )
    ap.add_argument(
        "--plot-dir",
        default=os.path.join(repo_root, "logs/RecoveryGen/real_trace_stall"),
    )
    args = ap.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    results = _collect_results(
        repo_root, target_lambda=args.target_lambda,
        long_gap_threshold_s=args.long_gap_threshold_s)
    lam_tag = _lambda_tag(args.target_lambda)
    metric_tag = "freq" if args.metric == "long_gap_frequency" else "time"
    out_prefix = os.path.join(args.plot_dir, f"fig2_real_trace_stall_{metric_tag}_{lam_tag}")
    _plot(results, args.metric, out_prefix)
    _write_csv(results, out_prefix + "_points.csv")

    print(f"[INFO] wrote {out_prefix}.png/.pdf")
    print(f"[INFO] wrote {out_prefix}_points.csv")
    for r in results:
        print(
            f"[INFO] {r.label}: freq_pct={r.long_gap_frequency_pct:.4f} "
            f"total_long_gap_s={r.total_long_gap_time_s:.4f} "
            f"p99={r.stall_gap_p99_s:.4f} run={r.run_dir}"
        )


if __name__ == "__main__":
    main()
