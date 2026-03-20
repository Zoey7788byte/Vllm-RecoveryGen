#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import re
import sys
from typing import List, Optional

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def require_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")


def safe_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_markdown_table(df: pd.DataFrame, out_md: str, max_rows: int = 200):
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(df2.columns) + " |\n")
        f.write("|" + "|".join(["---"] * len(df2.columns)) + "|\n")
        for _, row in df2.iterrows():
            vals = []
            for v in row.tolist():
                if isinstance(v, float):
                    if math.isnan(v):
                        vals.append("")
                    else:
                        vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def plot_series(df: pd.DataFrame, x: str, y: str, out_png: str, title: str):
    if x not in df.columns or y not in df.columns:
        raise KeyError(f"Missing column(s) for plot: x={x}, y={y}")

    d = df[[x, y]].dropna()
    if len(d) == 0:
        raise ValueError(f"No valid data for plot {y} vs {x}")

    plt.figure(figsize=(10, 3.2))
    plt.plot(d[x].to_numpy(), d[y].to_numpy())
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def add_phase_shading(ax, df: pd.DataFrame, x: str, phase_col: str = "phase"):
    if phase_col not in df.columns or x not in df.columns:
        return
    d = df[[x, phase_col]].dropna().copy()
    if len(d) == 0:
        return
    d = d.sort_values(x)

    def norm_phase(p):
        return str(p).strip().lower()

    phases = d[phase_col].map(norm_phase).tolist()
    xs = d[x].to_numpy()

    start = xs[0]
    cur = phases[0]
    for i in range(1, len(xs)):
        if phases[i] != cur:
            if cur == "low":
                ax.axvspan(start, xs[i], color="#e0e0e0", alpha=0.5, zorder=0)
            start = xs[i]
            cur = phases[i]
    if cur == "low":
        ax.axvspan(start, xs[-1], color="#e0e0e0", alpha=0.5, zorder=0)


def plot_series_with_phase(df: pd.DataFrame, x: str, y: str, out_png: str,
                           title: str):
    if x not in df.columns or y not in df.columns:
        raise KeyError(f"Missing column(s) for plot: x={x}, y={y}")

    d = df[[x, y, "phase"]].dropna(subset=[x, y])
    if len(d) == 0:
        raise ValueError(f"No valid data for plot {y} vs {x}")

    fig, ax = plt.subplots(figsize=(10, 3.2))
    add_phase_shading(ax, d, x, "phase")
    ax.plot(d[x].to_numpy(), d[y].to_numpy(), zorder=2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def has_nontrivial_signal(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return False
    return float(s.abs().max()) > 1e-12


def plot_recovery_instability(ts: pd.DataFrame, out_dir: str) -> Optional[str]:
    req_cols = ["t_rel_s", "phase", "preempt_rate_per_s", "req_waiting"]
    if not all(c in ts.columns for c in req_cols):
        return None

    d = ts.copy()
    d = safe_numeric(
        d,
        [
            "t_rel_s",
            "preempt_rate_per_s",
            "req_waiting",
            "restore_progress_stall_ms",
            "swapin_blocks",
            "recompute_tokens",
        ],
    )
    d = d.dropna(subset=["t_rel_s"]).sort_values("t_rel_s")
    if len(d) == 0:
        return None

    rec_col = None
    for c in ["restore_progress_stall_ms", "swapin_blocks", "recompute_tokens"]:
        if has_nontrivial_signal(d, c):
            rec_col = c
            break

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax0 = axes[0]
    add_phase_shading(ax0, d, "t_rel_s", "phase")
    ax0.plot(
        d["t_rel_s"].to_numpy(),
        d["preempt_rate_per_s"].to_numpy(),
        color="tab:blue",
        linewidth=1.1,
        label="preempt_rate_per_s",
    )
    ax0.set_ylabel("preempt_rate/s")
    ax0.set_title("Recovery-induced instability timeline")
    ax0.grid(alpha=0.2, linewidth=0.6)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    add_phase_shading(ax1, d, "t_rel_s", "phase")
    ax1.plot(
        d["t_rel_s"].to_numpy(),
        d["req_waiting"].to_numpy(),
        color="tab:orange",
        linewidth=1.1,
        label="req_waiting",
    )
    ax1.set_ylabel("waiting reqs")
    ax1.set_xlabel("t_rel_s")
    ax1.grid(alpha=0.2, linewidth=0.6)

    if rec_col is not None:
        ax1b = ax1.twinx()
        ax1b.plot(
            d["t_rel_s"].to_numpy(),
            d[rec_col].to_numpy(),
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label=rec_col,
        )
        ax1b.set_ylabel(rec_col)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1b.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    fig.tight_layout()
    png = os.path.join(out_dir, "recovery_instability.png")
    pdf = os.path.join(out_dir, "recovery_instability.pdf")
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return "recovery_instability"


def parse_cycle_phase(cond_dir: str):
    m = re.search(r"cycle_(\d+)_(low|high)_lambda_([0-9.]+)", cond_dir)
    if not m:
        return float("nan"), "", float("nan")
    return float(m.group(1)), m.group(2), float(m.group(3))


def build_tail_tables(summary_df: pd.DataFrame, out_dir: str) -> List[str]:
    produced = []
    if "cond_dir" not in summary_df.columns:
        return produced

    tail_cols = [
        "ttft_p99_s",
        "ttft_p999_s",
        "tpot_p99_s",
        "tpot_p999_s",
        "stall_gap_p99_s",
        "stall_gap_max_s",
    ]

    df = summary_df.copy()
    for c in tail_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    parsed = df["cond_dir"].map(parse_cycle_phase)
    df["cycle"] = parsed.map(lambda x: x[0])
    df["phase"] = parsed.map(lambda x: x[1])
    df["lambda_from_cond_dir"] = parsed.map(lambda x: x[2])

    keep = ["cycle", "phase", "lambda_rps", "lambda_from_cond_dir"] + [
        c for c in tail_cols if c in df.columns
    ]
    by_phase = df[keep].dropna(subset=["cycle"]).sort_values(["cycle", "phase"])

    phase_csv = os.path.join(out_dir, "table_tail_by_cycle_phase.csv")
    by_phase.to_csv(phase_csv, index=False)
    to_markdown_table(by_phase, os.path.join(out_dir, "table_tail_by_cycle_phase.md"))
    produced.append("table_tail_by_cycle_phase.csv / .md")

    low = by_phase[by_phase["phase"] == "low"].copy()
    high = by_phase[by_phase["phase"] == "high"].copy()
    pairs = low.merge(high, on="cycle", how="inner", suffixes=("_low", "_high"))

    for metric in ["ttft_p99_s", "ttft_p999_s", "tpot_p99_s", "tpot_p999_s", "stall_gap_p99_s"]:
        lcol = f"{metric}_low"
        hcol = f"{metric}_high"
        if lcol in pairs.columns and hcol in pairs.columns:
            pairs[f"{metric}_high_over_low"] = pairs[hcol] / pairs[lcol].replace(0, pd.NA)

    pair_cols = [
        "cycle",
        "lambda_rps_low",
        "lambda_rps_high",
        "ttft_p99_s_low",
        "ttft_p99_s_high",
        "ttft_p99_s_high_over_low",
        "ttft_p999_s_low",
        "ttft_p999_s_high",
        "ttft_p999_s_high_over_low",
        "tpot_p99_s_low",
        "tpot_p99_s_high",
        "tpot_p99_s_high_over_low",
        "tpot_p999_s_low",
        "tpot_p999_s_high",
        "tpot_p999_s_high_over_low",
        "stall_gap_p99_s_low",
        "stall_gap_p99_s_high",
        "stall_gap_p99_s_high_over_low",
    ]
    pair_cols = [c for c in pair_cols if c in pairs.columns]
    pairs = pairs[pair_cols].sort_values("cycle")

    pairs_csv = os.path.join(out_dir, "table_tail_pairs.csv")
    pairs.to_csv(pairs_csv, index=False)
    to_markdown_table(pairs, os.path.join(out_dir, "table_tail_pairs.md"))
    produced.append("table_tail_pairs.csv / .md")

    if "phase" in by_phase.columns:
        grp = by_phase.groupby("phase", dropna=True)
        agg = grp.agg(
            {
                c: ["median", "max"]
                for c in [
                    "ttft_p99_s",
                    "ttft_p999_s",
                    "tpot_p99_s",
                    "tpot_p999_s",
                    "stall_gap_p99_s",
                ]
                if c in by_phase.columns
            }
        )
        agg.columns = ["_".join(col).strip() for col in agg.columns.values]
        agg = agg.reset_index()
        agg.to_csv(os.path.join(out_dir, "table_tail_phase_aggregate.csv"), index=False)
        produced.append("table_tail_phase_aggregate.csv")

    return produced


def build_tail_plots(summary_df: pd.DataFrame, out_dir: str) -> List[str]:
    produced = []
    if "cond_dir" not in summary_df.columns:
        return produced

    cols = ["ttft_p99_s", "ttft_p999_s", "tpot_p99_s", "tpot_p999_s"]
    df = summary_df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    parsed = df["cond_dir"].map(parse_cycle_phase)
    df["cycle"] = parsed.map(lambda x: x[0])
    df["phase"] = parsed.map(lambda x: x[1])
    df = df.dropna(subset=["cycle"])
    if len(df) == 0:
        return produced

    keep_cols = ["cycle", "phase"] + [c for c in cols if c in df.columns]
    df = df[keep_cols].sort_values(["cycle", "phase"])
    low = df[df["phase"] == "low"].copy()
    high = df[df["phase"] == "high"].copy()
    pairs = low.merge(high, on="cycle", how="inner", suffixes=("_low", "_high"))
    if len(pairs) == 0:
        return produced

    pairs = pairs.sort_values("cycle")
    x = pairs["cycle"].to_numpy()

    def _plot_family(prefix: str, title: str, out_name: str) -> Optional[str]:
        m1 = f"{prefix}_p99_s"
        m2 = f"{prefix}_p999_s"
        req = [f"{m1}_low", f"{m1}_high", f"{m2}_low", f"{m2}_high"]
        if any(c not in pairs.columns for c in req):
            return None

        fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True)
        axes[0].plot(
            x, pairs[f"{m1}_low"].to_numpy(), marker="o", linewidth=1.2, label="low load"
        )
        axes[0].plot(
            x, pairs[f"{m1}_high"].to_numpy(), marker="o", linewidth=1.2, label="high load"
        )
        axes[0].set_ylabel(m1)
        axes[0].set_title(title)
        axes[0].grid(alpha=0.25, linewidth=0.6)
        axes[0].legend(loc="upper left")

        axes[1].plot(
            x, pairs[f"{m2}_low"].to_numpy(), marker="o", linewidth=1.2, label="low load"
        )
        axes[1].plot(
            x, pairs[f"{m2}_high"].to_numpy(), marker="o", linewidth=1.2, label="high load"
        )
        axes[1].set_ylabel(m2)
        axes[1].set_xlabel("cycle")
        axes[1].grid(alpha=0.25, linewidth=0.6)
        axes[1].legend(loc="upper left")

        fig.tight_layout()
        png = os.path.join(out_dir, f"{out_name}.png")
        pdf = os.path.join(out_dir, f"{out_name}.pdf")
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        plt.close(fig)
        return f"{out_name}.png / {out_name}.pdf"

    ttft_out = _plot_family(
        "ttft",
        "Mot2: TTFT tail by cycle (low/high load)",
        "fig_ttft_tail_by_cycle",
    )
    if ttft_out is not None:
        produced.append(ttft_out)

    tpot_out = _plot_family(
        "tpot",
        "Mot2: TPOT tail by cycle (low/high load)",
        "fig_tpot_tail_by_cycle",
    )
    if tpot_out is not None:
        produced.append(tpot_out)

    return produced


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 mot2_make_artifacts.py <RUN_DIR>")
        print(
            "Example: python3 mot2_make_artifacts.py "
            "logs/RecoveryGen/Mot2_tail_instability/"
            "Mot2_tail_instability_20260223_140000_low0.60_high0.85_poisson-poisson_b4"
        )
        sys.exit(1)

    run_dir = sys.argv[1].rstrip("/")
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(run_dir)

    f_all = os.path.join(run_dir, "metrics_timeseries_all.csv")
    f_cycle = os.path.join(run_dir, "proof2_cycle_summary.csv")
    f_phase = os.path.join(run_dir, "proof2_phase_summary.csv")
    f_summary = os.path.join(run_dir, "summary.csv")

    require_file(f_all)
    require_file(f_cycle)
    require_file(f_phase)

    out_dir = os.path.join(run_dir, "artifacts_proof2")
    os.makedirs(out_dir, exist_ok=True)

    ts = pd.read_csv(f_all)
    ts = safe_numeric(
        ts,
        [
            "t_rel_s",
            "preempt_rate_per_s",
            "preempt_total",
            "req_waiting",
            "req_running",
            "gpu_cache_usage_perc",
            "restore_progress_stall_ms",
            "swapin_blocks",
            "recompute_tokens",
        ],
    )

    plot_series_with_phase(
        ts,
        "t_rel_s",
        "preempt_rate_per_s",
        os.path.join(out_dir, "fig_preempt_rate_vs_time.png"),
        "Mot2: preempt_rate_per_s vs t_rel_s",
    )
    plot_series_with_phase(
        ts,
        "t_rel_s",
        "req_waiting",
        os.path.join(out_dir, "fig_waiting_vs_time.png"),
        "Mot2: req_waiting vs t_rel_s",
    )

    if "gpu_cache_usage_perc" in ts.columns:
        g = ts["gpu_cache_usage_perc"].dropna()
        if len(g) > 0 and (g.abs().max() > 1e-12):
            plot_series(
                ts,
                "t_rel_s",
                "gpu_cache_usage_perc",
                os.path.join(out_dir, "fig_gpu_cache_vs_time.png"),
                "Mot2: gpu_cache_usage_perc vs t_rel_s",
            )

    recovery_fig = plot_recovery_instability(ts, out_dir)

    cyc = pd.read_csv(f_cycle)
    phs = pd.read_csv(f_phase)

    cycle_cols = [
        c
        for c in [
            "cycle",
            "low_lambda_rps",
            "high_lambda_rps",
            "low_preempt_delta",
            "high_preempt_delta",
            "low_preempt_rate_avg",
            "high_preempt_rate_avg",
            "low_preempt_rate_max",
            "high_preempt_rate_max",
            "low_waiting_avg",
            "high_waiting_avg",
            "low_waiting_max",
            "high_waiting_max",
            "separation_ok",
        ]
        if c in cyc.columns
    ]
    cyc_out = cyc[cycle_cols].copy()

    phase_cols = [
        c
        for c in [
            "cond_dir",
            "cycle",
            "phase",
            "mode",
            "lambda_rps",
            "T_s",
            "rows",
            "preempt_min",
            "preempt_max",
            "preempt_delta",
            "preempt_rate_avg",
            "preempt_rate_max",
            "waiting_avg",
            "waiting_max",
            "running_avg",
            "running_max",
            "gpu_avg",
            "gpu_max",
            "parse_nan_rows",
        ]
        if c in phs.columns
    ]
    phs_out = phs[phase_cols].copy()

    cycle_csv = os.path.join(out_dir, "table_cycle.csv")
    phase_csv = os.path.join(out_dir, "table_phase.csv")
    cyc_out.to_csv(cycle_csv, index=False)
    phs_out.to_csv(phase_csv, index=False)

    to_markdown_table(cyc_out, os.path.join(out_dir, "table_cycle.md"))
    to_markdown_table(phs_out, os.path.join(out_dir, "table_phase.md"))

    if "phase" in phs_out.columns:
        num_cols = [
            c for c in phs_out.columns if c not in ("cond_dir", "phase", "mode")
        ]
        phs_num = phs_out.copy()
        for c in num_cols:
            phs_num[c] = pd.to_numeric(phs_num[c], errors="coerce")

        summary = phs_num.groupby("phase").agg(
            {
                "preempt_delta": ["mean", "median", "max"],
                "preempt_rate_max": ["mean", "median", "max"],
                "waiting_max": ["mean", "median", "max"],
                "waiting_avg": ["mean", "median", "max"],
            }
        )
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        summary.to_csv(os.path.join(out_dir, "table_phase_aggregate.csv"), index=False)

    tail_tables = []
    tail_figs = []
    if os.path.isfile(f_summary):
        summary_df = pd.read_csv(f_summary)
        tail_figs = build_tail_plots(summary_df, out_dir)
        tail_tables = build_tail_tables(summary_df, out_dir)

    print("[OK] Artifacts written to:", out_dir)
    print("  - Figures:")
    print("      fig_preempt_rate_vs_time.png")
    print("      fig_waiting_vs_time.png")
    print("      (optional) fig_gpu_cache_vs_time.png")
    if recovery_fig is not None:
        print("      recovery_instability.png / recovery_instability.pdf")
    if tail_figs:
        print("      " + "\n      ".join(tail_figs))
    print("  - Tables:")
    print("      table_cycle.csv / table_cycle.md")
    print("      table_phase.csv / table_phase.md")
    print("      (optional) table_phase_aggregate.csv")
    if tail_tables:
        print("      " + "\n      ".join(tail_tables))


if __name__ == "__main__":
    main()
