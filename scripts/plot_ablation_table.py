#!/usr/bin/env python3
"""Build a phase ablation table and compact metric plots.

Default rows:
  - Baseline
  - Micro-task only
  - Micro-task + Online-first
  - Full RecoverGen

Default comparison setup:
  - target lambda = 0.85
  - long-gap threshold = 1.0s

Outputs:
  - CSV
  - Markdown table
  - LaTeX table
  - compact multi-panel PNG/PDF
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_float(v: object, default: float = float("nan")) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _percentile(values: List[float], q: float) -> float:
    vals = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def _lambda_tag(lam: float) -> str:
    s = f"{lam:.2f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return "lam" + s.replace(".", "p")


def _parse_lambda_from_segment(name: str) -> float:
    m = re.search(r"lam(\d+)p(\d+)", name)
    if m:
        return float(f"{int(m.group(1))}.{m.group(2)}")
    m = re.search(r"lambda_(\d+\.\d+)", name)
    if m:
        return float(m.group(1))
    return float("nan")


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class Config:
    name: str
    client_root: str
    summary_path: str
    events_path: Optional[str] = None
    baseline_phase_md: Optional[str] = None


@dataclass
class Result:
    name: str
    ok_rows: int
    total_rows: int
    num_high_segments: int
    ttft_p99_s: float
    tpot_p99_s_per_tok: float
    long_gap_freq_pct: float
    preempt_total: float
    preempts_per_100req: float
    useful_rec_per_cycle: float
    repeat_ratio: float
    burst_length: float
    recovery_work_ratio: float
    coverage_ratio: float
    swapin_blocks: float
    swapout_blocks: float
    pinned_blocks: float
    committed_blocks: float
    recovery_actions: float


def _collect_client_metrics(client_root: str, target_lambda: float,
                            long_gap_threshold_s: float) -> Dict[str, float]:
    ttft: List[float] = []
    tpot: List[float] = []
    stall_gap: List[float] = []
    total_rows = 0
    ok_rows = 0

    for path in sorted(glob.glob(os.path.join(client_root, "*", "client_results.csv"))):
        seg_name = os.path.basename(os.path.dirname(path))
        seg_lambda = _parse_lambda_from_segment(seg_name)
        if math.isnan(seg_lambda) or abs(seg_lambda - target_lambda) > 1e-9:
            continue

        for row in _read_csv(path):
            total_rows += 1
            status = str(row.get("http_status", "")).strip()
            if status and status != "200":
                continue
            ok_rows += 1

            x_ttft = _to_float(row.get("ttft_s"))
            if not math.isnan(x_ttft):
                ttft.append(x_ttft)

            x_gap = _to_float(row.get("stall_gap_max_s"))
            if not math.isnan(x_gap):
                stall_gap.append(x_gap)

            out_tok = _to_float(row.get("out_tok"))
            t1 = _to_float(row.get("t_first_token_abs"))
            t2 = _to_float(row.get("t_done_abs"))
            if out_tok > 1.0 and not math.isnan(t1) and not math.isnan(t2) and t2 > t1:
                tpot.append((t2 - t1) / (out_tok - 1.0))

    long_gap_ratio = (
        sum(1 for x in stall_gap if x >= long_gap_threshold_s) / len(stall_gap)
        if stall_gap else float("nan")
    )
    return {
        "ok_rows": float(ok_rows),
        "total_rows": float(total_rows),
        "num_high_segments": float(sum(
            1 for path in glob.glob(os.path.join(client_root, "*", "client_results.csv"))
            if not math.isnan(_parse_lambda_from_segment(os.path.basename(os.path.dirname(path))))
            and abs(_parse_lambda_from_segment(os.path.basename(os.path.dirname(path))) - target_lambda) < 1e-9)),
        "ttft_p99_s": _percentile(ttft, 0.99),
        "tpot_p99_s_per_tok": _percentile(tpot, 0.99),
        "long_gap_freq_pct": 100.0 * long_gap_ratio if not math.isnan(long_gap_ratio) else float("nan"),
    }


def _summary_preempt_total(summary_path: str, target_lambda: float) -> float:
    total = 0.0
    for row in _read_csv(summary_path):
        lam = _to_float(row.get("lambda_rps"))
        if math.isnan(lam) or abs(lam - target_lambda) > 1e-9:
            continue
        total += _to_float(row.get("preempt_sum_delta"), 0.0)
    return total


def _baseline_phase_md_preempt_total(md_path: str, target_lambda: float) -> float:
    total = 0.0
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            if "cycle_" not in line or not line.startswith("|"):
                continue
            cols = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(cols) < 10:
                continue
            lam = _to_float(cols[4])
            if math.isnan(lam) or abs(lam - target_lambda) > 1e-9:
                continue
            total += _to_float(cols[9], 0.0)
    return total


def _baseline_phase_md_segment_count(md_path: str, target_lambda: float) -> int:
    total = 0
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            if "cycle_" not in line or not line.startswith("|"):
                continue
            cols = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(cols) < 10:
                continue
            lam = _to_float(cols[4])
            if not math.isnan(lam) and abs(lam - target_lambda) < 1e-9:
                total += 1
    return total


def _event_recovery_stats(events_path: Optional[str]) -> Dict[str, float]:
    if not events_path or not os.path.isfile(events_path):
        return {
            "swapin_blocks": float("nan"),
            "swapout_blocks": float("nan"),
            "pinned_blocks": float("nan"),
            "committed_blocks": float("nan"),
            "pin_events": float("nan"),
            "recovery_work_ratio": float("nan"),
            "coverage_ratio": float("nan"),
        }

    swapin_blocks = 0.0
    swapout_blocks = 0.0
    pinned_blocks = 0.0
    committed_blocks = 0.0
    pin_events = 0.0
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            event = obj.get("event")
            detail = obj.get("detail") or {}
            if event == "SWAP_IN":
                swapin_blocks += _to_float(detail.get("blocks_count"), 0.0)
            elif event == "SWAP_OUT":
                swapout_blocks += _to_float(detail.get("blocks_count"), 0.0)
            elif event == "MWS_PIN_APPLIED":
                pinned_blocks += _to_float(detail.get("pinned_blocks"), 0.0)
                pin_events += 1.0
            elif event == "RECOVERY_PROGRESS_COMMIT":
                committed_blocks += _to_float(detail.get("blocks_committed"), 0.0)

    work_ratio = (
        swapin_blocks / swapout_blocks if swapout_blocks > 0 else float("nan")
    )
    coverage_ratio = (
        (swapin_blocks + pinned_blocks) / swapout_blocks
        if swapout_blocks > 0 else float("nan")
    )
    return {
        "swapin_blocks": swapin_blocks,
        "swapout_blocks": swapout_blocks,
        "pinned_blocks": pinned_blocks,
        "committed_blocks": committed_blocks,
        "pin_events": pin_events,
        "recovery_work_ratio": work_ratio,
        "coverage_ratio": coverage_ratio,
    }


def _recovery_action_count(ts_path: Optional[str]) -> float:
    if not ts_path or not os.path.isfile(ts_path):
        return float("nan")
    count = 0.0
    with open(ts_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            swapin = _to_float(row.get("swapin_blocks"), 0.0)
            recompute = _to_float(row.get("recompute_tokens"), 0.0)
            if swapin > 0.0 or recompute > 0.0:
                count += 1.0
    return count


def _fmt(x: float, digits: int = 3, nan_text: str = "-") -> str:
    if x is None or math.isnan(x):
        return nan_text
    return f"{x:.{digits}f}"


def _write_csv(results: List[Result], out_csv: str) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "method",
            "ok_rows",
            "total_rows",
            "ttft_p99_s",
            "tpot_p99_s_per_tok",
            "long_gap_freq_pct_ge_1s",
            "preempt_total",
            "preempts_per_100req",
            "useful_rec_per_cycle",
            "repeat_ratio",
            "burst_length",
            "recovery_work_ratio_swapin_over_swapout",
            "coverage_ratio_swapin_plus_pinned_over_swapout",
            "num_high_segments",
            "swapin_blocks",
            "swapout_blocks",
            "pinned_blocks",
            "committed_blocks",
            "recovery_actions",
        ])
        for r in results:
            w.writerow([
                r.name,
                r.ok_rows,
                r.total_rows,
                _fmt(r.ttft_p99_s, 6),
                _fmt(r.tpot_p99_s_per_tok, 6),
                _fmt(r.long_gap_freq_pct, 6),
                _fmt(r.preempt_total, 6),
                _fmt(r.preempts_per_100req, 6),
                _fmt(r.useful_rec_per_cycle, 6),
                _fmt(r.repeat_ratio, 6),
                _fmt(r.burst_length, 6),
                _fmt(r.recovery_work_ratio, 6),
                _fmt(r.coverage_ratio, 6),
                r.num_high_segments,
                _fmt(r.swapin_blocks, 6),
                _fmt(r.swapout_blocks, 6),
                _fmt(r.pinned_blocks, 6),
                _fmt(r.committed_blocks, 6),
                _fmt(r.recovery_actions, 6),
            ])


def _write_md(results: List[Result], out_md: str, target_lambda: float,
              long_gap_threshold_s: float) -> None:
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"Ablation table at lambda={target_lambda:.2f}\n\n")
        f.write(
            f"Long-gap frequency uses `stall_gap_max_s >= {long_gap_threshold_s:.1f}s`.\n"
        )
        f.write(
            "Recovery work ratio is `swapin_blocks / swapout_blocks` from recovery events.\n\n"
        )
        f.write(
            "| Configuration | TTFT p99 | TPOT p99 | Long-gap freq. | Useful rec./cycle | "
            "Repeat ratio | Burst length |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r.name} | {_fmt(r.ttft_p99_s, 3)} | {_fmt(r.tpot_p99_s_per_tok, 3)} | "
                f"{_fmt(r.long_gap_freq_pct, 2)} | {_fmt(r.useful_rec_per_cycle, 2)} | "
                f"{_fmt(r.repeat_ratio, 3)} | {_fmt(r.burst_length, 2)} |\n"
            )


def _write_tex(results: List[Result], out_tex: str, target_lambda: float,
               long_gap_threshold_s: float) -> None:
    lines = [
        "% Auto-generated by scripts/plot_ablation_table.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation study of RecoverGen.}",
        "\\label{tab:ablation}",
        "\\vspace{2mm}",
        "\\begin{tabular}{l c c c c c c}",
        "\\hline",
        "\\textbf{Configuration} ",
        "& \\textbf{TTFT p99} ",
        "& \\textbf{TPOT p99} ",
        "& \\textbf{Long-gap freq.} ",
        "& \\textbf{Useful rec./cycle} ",
        "& \\textbf{Repeat ratio} ",
        "& \\textbf{Burst length} \\\\",
        "\\hline",
    ]
    for r in results:
        if r.name == "Baseline (blocking)":
            lines.append(
                f"{r.name} "
                f"& {_fmt(r.ttft_p99_s, 3)} "
                f"& {_fmt(r.tpot_p99_s_per_tok, 3)} "
                f"& {_fmt(r.long_gap_freq_pct, 2)} "
                f"& {_fmt(r.useful_rec_per_cycle, 2)} "
                f"& {_fmt(r.repeat_ratio, 3)} "
                f"& {_fmt(r.burst_length, 2)} \\\\"
            )
            lines.append("\\hline")
        else:
            lines.append(
                f"{r.name} "
                f"& {_fmt(r.ttft_p99_s, 3)} "
                f"& {_fmt(r.tpot_p99_s_per_tok, 3)} "
                f"& {_fmt(r.long_gap_freq_pct, 2)} "
                f"& {_fmt(r.useful_rec_per_cycle, 2)} "
                f"& {_fmt(r.repeat_ratio, 3)} "
                f"& {_fmt(r.burst_length, 2)} \\\\"
            )
    lines.extend(["\\hline", "\\end{tabular}", "\\vspace{-2mm}", "\\end{table}"])
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot_metric_panel(ax: plt.Axes, labels: List[str], values: List[float],
                       title: str, ylabel: str) -> None:
    xs = np.arange(len(labels))
    plot_vals = [0.0 if math.isnan(v) else v for v in values]
    colors = ["#9E9E9E", "#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(xs, plot_vals, color=colors[:len(labels)], width=0.72)
    for i, v in enumerate(values):
        if math.isnan(v):
            bars[i].set_hatch("//")
            bars[i].set_facecolor("#DDDDDD")
            ax.text(i, 0.02, "N/A", ha="center", va="bottom", fontsize=9, rotation=90)
        else:
            ax.text(i, v, _fmt(v, 2), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def _write_plot(results: List[Result], out_prefix: str) -> None:
    labels = [r.name for r in results]
    fig, axes = plt.subplots(1, 5, figsize=(19.5, 4.2))
    _plot_metric_panel(
        axes[0], labels, [r.ttft_p99_s for r in results], "TTFT Tail", "p99 TTFT (s)")
    _plot_metric_panel(
        axes[1], labels, [r.tpot_p99_s_per_tok for r in results], "TPOT Tail", "p99 TPOT (s/token)")
    _plot_metric_panel(
        axes[2], labels, [r.long_gap_freq_pct for r in results], "Stream Gaps", "Long-gap freq. (%)")
    _plot_metric_panel(
        axes[3], labels, [r.preempts_per_100req for r in results], "Preemption", "Preempts / 100 req")
    _plot_metric_panel(
        axes[4], labels, [r.recovery_work_ratio for r in results], "Recovery Work", "swapin / swapout")
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=220)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)


def build_results(configs: Iterable[Config], target_lambda: float,
                  long_gap_threshold_s: float,
                  ts_paths: Dict[str, str]) -> List[Result]:
    out: List[Result] = []
    for cfg in configs:
        client = _collect_client_metrics(
            cfg.client_root, target_lambda=target_lambda,
            long_gap_threshold_s=long_gap_threshold_s)

        if cfg.baseline_phase_md:
            preempt_total = _baseline_phase_md_preempt_total(
                cfg.baseline_phase_md, target_lambda=target_lambda)
            num_high_segments = _baseline_phase_md_segment_count(
                cfg.baseline_phase_md, target_lambda=target_lambda)
        else:
            preempt_total = _summary_preempt_total(
                cfg.summary_path, target_lambda=target_lambda)
            num_high_segments = int(client["num_high_segments"])

        rec = _event_recovery_stats(cfg.events_path)
        recovery_actions = _recovery_action_count(ts_paths.get(cfg.name))
        if not math.isnan(rec.get("pin_events", float("nan"))):
            recovery_actions = (
                recovery_actions + rec["pin_events"]
                if not math.isnan(recovery_actions) else rec["pin_events"]
            )
        total_rows = int(client["total_rows"])
        preempts_per_100req = (
            100.0 * preempt_total / total_rows if total_rows > 0 else float("nan")
        )
        useful_rec_per_cycle = float("nan")
        if not math.isnan(recovery_actions) and recovery_actions > 0:
            useful_rec_per_cycle = (
                rec["committed_blocks"] + rec["pinned_blocks"]) / recovery_actions
        burst_length = (
            preempt_total / num_high_segments if num_high_segments > 0 else float("nan")
        )
        out.append(Result(
            name=cfg.name,
            ok_rows=int(client["ok_rows"]),
            total_rows=total_rows,
            num_high_segments=num_high_segments,
            ttft_p99_s=client["ttft_p99_s"],
            tpot_p99_s_per_tok=client["tpot_p99_s_per_tok"],
            long_gap_freq_pct=client["long_gap_freq_pct"],
            preempt_total=preempt_total,
            preempts_per_100req=preempts_per_100req,
            useful_rec_per_cycle=useful_rec_per_cycle,
            repeat_ratio=rec["recovery_work_ratio"],
            burst_length=burst_length,
            recovery_work_ratio=rec["recovery_work_ratio"],
            coverage_ratio=rec["coverage_ratio"],
            swapin_blocks=rec["swapin_blocks"],
            swapout_blocks=rec["swapout_blocks"],
            pinned_blocks=rec["pinned_blocks"],
            committed_blocks=rec["committed_blocks"],
            recovery_actions=recovery_actions,
        ))
    return out


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-lambda", type=float, default=0.85)
    ap.add_argument("--long-gap-threshold-s", type=float, default=1.0)
    ap.add_argument(
        "--out-dir",
        default=os.path.join(repo_root, "logs/RecoveryGen/ablation_table"),
    )
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    configs = [
        Config(
            name="Baseline (blocking)",
            client_root=os.path.join(
                repo_root,
                "logs/RecoveryGen/Mot2_tail_instability/"
                "Mot2_tail_instability_20260227_200631_low0.60_high0.85_poisson-poisson_b4",
            ),
            summary_path=os.path.join(
                repo_root,
                "logs/RecoveryGen/Mot2_tail_instability/"
                "Mot2_tail_instability_20260227_200631_low0.60_high0.85_poisson-poisson_b4/summary.csv",
            ),
            baseline_phase_md=os.path.join(
                repo_root,
                "logs/RecoveryGen/Mot2_tail_instability/"
                "Mot2_tail_instability_20260227_200631_low0.60_high0.85_poisson-poisson_b4/"
                "artifacts_proof2/table_phase.md",
            ),
        ),
        Config(
            name="Phase 1: Micro-task Recovery",
            client_root=os.path.join(
                repo_root, "logs/RecoveryGen/Phase1_validation/client_logs/phase1_validate_20260228_212912"),
            summary_path=os.path.join(
                repo_root, "logs/RecoveryGen/Phase1_validation/client_logs/phase1_validate_20260228_212912/summary.csv"),
            events_path=os.path.join(
                repo_root, "logs/RecoveryGen/Phase1_validation/server_logs/20260228_212912/recovery/recovery_events.jsonl"),
        ),
        Config(
            name="Phase 2: +Online-first Scheduling",
            client_root=os.path.join(
                repo_root, "logs/RecoveryGen/Phase2_validation/client_logs/phase2_validate_20260301_162718"),
            summary_path=os.path.join(
                repo_root, "logs/RecoveryGen/Phase2_validation/client_logs/phase2_validate_20260301_162718/summary.csv"),
            events_path=os.path.join(
                repo_root, "logs/RecoveryGen/Phase2_validation/server_logs/20260301_162718/recovery/recovery_events.jsonl"),
        ),
        Config(
            name="Phase 3: Full RecoverGen",
            client_root=os.path.join(
                repo_root, "logs/RecoveryGen/Phase3_validation/client_logs/phase3_validate_20260305_231943"),
            summary_path=os.path.join(
                repo_root, "logs/RecoveryGen/Phase3_validation/client_logs/phase3_validate_20260305_231943/summary.csv"),
            events_path=os.path.join(
                repo_root, "logs/RecoveryGen/Phase3_validation/server_logs/20260305_231943/recovery/recovery_events.jsonl"),
        ),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    ts_paths = {
        "Phase 1: Micro-task Recovery": os.path.join(
            repo_root, "logs/RecoveryGen/Phase1_validation/server_logs/20260228_212912/recovery/recovery_ts.csv"),
        "Phase 2: +Online-first Scheduling": os.path.join(
            repo_root, "logs/RecoveryGen/Phase2_validation/server_logs/20260301_162718/recovery/recovery_ts.csv"),
        "Phase 3: Full RecoverGen": os.path.join(
            repo_root, "logs/RecoveryGen/Phase3_validation/server_logs/20260305_231943/recovery/recovery_ts.csv"),
    }
    results = build_results(
        configs, target_lambda=args.target_lambda,
        long_gap_threshold_s=args.long_gap_threshold_s,
        ts_paths=ts_paths)
    lam_tag = _lambda_tag(args.target_lambda)
    base = os.path.join(args.out_dir, f"ablation_{lam_tag}")

    _write_csv(results, base + ".csv")
    _write_md(results, base + ".md", args.target_lambda, args.long_gap_threshold_s)
    _write_tex(results, base + ".tex", args.target_lambda, args.long_gap_threshold_s)
    if not args.no_plots:
        _write_plot(results, base + "_metrics")

    print(f"[INFO] wrote {base}.csv")
    print(f"[INFO] wrote {base}.md")
    print(f"[INFO] wrote {base}.tex")
    if not args.no_plots:
        print(f"[INFO] wrote {base}_metrics.png/.pdf")
    for r in results:
        print(
            f"[INFO] {r.name}: ttft_p99={_fmt(r.ttft_p99_s, 4)} "
            f"tpot_p99={_fmt(r.tpot_p99_s_per_tok, 4)} "
            f"long_gap_pct={_fmt(r.long_gap_freq_pct, 2)} "
            f"preempts_per_100req={_fmt(r.preempts_per_100req, 2)} "
            f"recovery_work_ratio={_fmt(r.recovery_work_ratio, 4)}"
        )


if __name__ == "__main__":
    main()
