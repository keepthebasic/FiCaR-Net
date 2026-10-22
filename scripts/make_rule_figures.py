# -*- coding: utf-8 -*-
"""Publication figures for the rule-validation revision (submission style).

Style is matched to the original WR submission figures: sans-serif 9 pt,
top/right spines hidden, dashed light grid, bold left-aligned titles,
palette ORANGE #E67E22 / TEAL #1F9E89 / SLATE #2C3E50 / GREY #8A94A6 /
RED #C0392B, 300 dpi with tight bounding boxes.

Reads the artifacts produced by eval_rule_validation.py / analyze_rule_validation.py
under outputs/rule_validation/ (results_5mg/results_10mg, events_5mg,
partition_by_duration.csv) and writes PNGs to outputs/rule_validation/figs/:

    fig_capture_budget.png   capture vs sampling budget (5 vs 10 mg/L)
    fig_delay_budget52.png   detection-delay distribution at 52 samples/yr
    fig_short_events.png     short-event capture at 52 samples/yr

Usage::

    python -m src.experiments.make_rule_figures
    python -m src.experiments.make_rule_figures --dir "../outputs_server/rule_validation"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.paths import repo_root  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
})

SLATE = "#2C3E50"   # fixed schedules / dark neutrals
TEAL = "#1F9E89"    # concurrent flow trigger (primary adaptive)
ORANGE = "#E67E22"  # seasonal trigger / secondary bucket
GREY = "#8A94A6"    # logistic trigger / auxiliary
RED = "#C0392B"     # NWM forecast trigger / warnings

VAR_STYLE = {
    "fixed_1d":  (SLATE, "o", ":",  "fixed 1 d"),
    "fixed_7d":  (SLATE, "s", "-",  "fixed 7 d"),
    "fixed_14d": (SLATE, "^", "--", "fixed 14 d"),
    "fixed_30d": (SLATE, "v", "-.", "fixed 30 d"),
    "nowcast":   (TEAL, "o", "-",  "flow nowcast"),
    "nwm_fc":    (RED, "s", "-",   "NWM 7-d forecast"),
    "season":    (ORANGE, "^", "-", "seasonal"),
    "logit":     (GREY, "D", "-",  "logistic"),
}
ADAPTIVE = ["nowcast", "nwm_fc", "season", "logit"]
MODE_TITLE = {"temporal": "Temporal (pooled)",
              "logo": "Leave-one-gauge-out",
              "transfer": "Illinois \u2192 Iowa/Indiana"}


def _load(out_dir: Path, prefer: list[str], default: str) -> pd.DataFrame | None:
    for name in prefer + [default]:
        p = out_dir / name
        if p.exists():
            return pd.read_csv(p, dtype={"site_no": str})
    return None


def _title(ax, text: str) -> None:
    ax.set_title(text, fontweight="bold", loc="left", fontsize=10)


def fig_capture_budget(out_dir: Path, res5: pd.DataFrame | None, res10: pd.DataFrame | None) -> None:
    modes = [m for m in ["temporal", "logo", "transfer"]
             if (res5 is not None and m in set(res5["mode"])) or
                (res10 is not None and m in set(res10["mode"]))]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(9, 3.3), squeeze=False,
                             sharey=True)
    for ax, mode in zip(axes[0], modes):
        for res, filled, tag in [(res5, True, "5 mg/L"), (res10, False, "10 mg/L")]:
            if res is None:
                continue
            g = res[res["mode"] == mode]
            for variant, (color, marker, ls, label) in VAR_STYLE.items():
                v = g[g["variant"] == variant].sort_values("budget")
                if v.empty or v["capture_rate"].notna().sum() == 0:
                    continue
                fin = v[v["capture_rate"].notna()]
                ax.plot(fin["budget"], fin["capture_rate"], ls=ls, color=color,
                        marker=marker, ms=5,
                        lw=2.0 if variant in ADAPTIVE else 1.3,
                        mfc=color if filled else "white", mew=1.2,
                        alpha=0.95 if filled else 0.75,
                        label=label if (filled and variant != "nwm_fc") else None)
                if variant in ADAPTIVE:
                    ok = v[v["capture_ci_lo"].notna() & v["capture_ci_hi"].notna()]
                    if not ok.empty:
                        ax.fill_between(ok["budget"], ok["capture_ci_lo"],
                                        ok["capture_ci_hi"], color=color,
                                        alpha=0.10 if filled else 0.05, lw=0)
        ax.set_xscale("log")
        ax.set_xlim(8, 500)
        ax.set_xticks([12, 26, 52, 365])
        ax.set_xticklabels(["12", "26", "52", "365"])
        ax.set_ylim(0, 1.05)
        _title(ax, MODE_TITLE.get(mode, mode))
        ax.set_xlabel("Sampling budget (samples yr$^{-1}$)")
    axes[0][0].set_ylabel("Event capture rate")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=7.5)
    fig.text(0.995, 0.01, "solid markers: 5 mg/L   hollow: 10 mg/L",
             ha="right", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_dir / "figs" / "fig_capture_budget.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def fig_delay(out_dir: Path, ev: pd.DataFrame | None) -> None:
    if ev is None or "delay_days" not in ev:
        return
    wk = ev[(ev["budget"] == 52) & ev["captured"] & ev["delay_days"].notna()]
    if wk.empty:
        return
    variants = [v for v in ADAPTIVE if v in set(wk["variant"])]
    data = [wk.loc[wk["variant"] == v, "delay_days"].to_numpy() for v in variants]
    labels = [f"{VAR_STYLE[v][3]}\n(n={len(d)})" for v, d in zip(variants, data)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": SLATE, "lw": 1.2},
                    whiskerprops={"color": SLATE, "lw": 0.8},
                    capprops={"color": SLATE, "lw": 0.8})
    for patch, v in zip(bp["boxes"], variants):
        patch.set_facecolor(VAR_STYLE[v][0])
        patch.set_alpha(0.35)
        patch.set_edgecolor(VAR_STYLE[v][0])
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        x = rng.normal(i, 0.055, size=len(d))
        ax.scatter(x, d, s=9, color=VAR_STYLE[variants[i - 1]][0], alpha=0.55,
                   zorder=3, lw=0)
    ax.set_xticks(range(1, len(variants) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Detection delay after onset (days)")
    _title(ax, "Captured events at 52 samples yr$^{-1}$")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "fig_delay_budget52.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def fig_short_events(out_dir: Path) -> None:
    part = _load(out_dir, ["partition_by_duration.csv"],
                 "partition_by_duration.csv")
    if part is None:
        return
    wk = part[(part["budget"] == 52) & part["dur_bucket"].isin(["d1_3", "d4_7"])]
    if wk.empty:
        return
    modes = [m for m in ["temporal", "logo", "transfer"] if m in set(wk["mode"])]
    variants = [v for v in ["fixed_7d"] + ADAPTIVE if v in set(wk["variant"])]
    width = 0.38
    fig, axes = plt.subplots(1, len(modes), figsize=(9, 3.3), squeeze=False,
                             sharey=True)
    for ax, mode in zip(axes[0], modes):
        g = wk[wk["mode"] == mode]
        x = np.arange(len(variants))
        for j, (bucket, color, blabel) in enumerate(
                [("d1_3", TEAL, "1\u20133-d events"), ("d4_7", ORANGE, "4\u20137-d events")]):
            vals, ns = [], []
            for v in variants:
                row = g[(g["variant"] == v) & (g["dur_bucket"] == bucket)]
                vals.append(float(row["capture"].iloc[0]) if len(row) else np.nan)
                ns.append(int(row["n"].iloc[0]) if len(row) else 0)
            bars = ax.bar(x + (j - 0.5) * width, vals, width, color=color,
                          edgecolor="white", lw=0.5,
                          label=f"{blabel} (n={ns[0]})")
            for b, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(b.get_x() + b.get_width() / 2, val + 0.015,
                            f"{val:.2f}", ha="center", fontsize=6.5, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels([VAR_STYLE.get(v, (0, 0, 0, v))[3] for v in variants],
                           rotation=20, fontsize=7.5, ha="right")
        ax.set_ylim(0, 0.9)
        _title(ax, MODE_TITLE.get(mode, mode))
        ax.legend(fontsize=7)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Capture rate at 52 samples yr$^{-1}$")
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "fig_short_events.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None,
                    help="directory with rule-validation artifacts "
                         "(default: outputs/rule_validation)")
    args = ap.parse_args()
    out_dir = Path(args.dir) if args.dir else repo_root() / "outputs" / "rule_validation"
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    res5 = _load(out_dir, ["results_5mg.csv"], "rule_validation_results.csv")
    res10 = _load(out_dir, ["results_10mg.csv"], "__none__")
    ev = _load(out_dir, ["events_5mg.csv", "rule_validation_events.csv"],
               "rule_validation_events.csv")
    print(f"res5: {None if res5 is None else len(res5)} | "
          f"res10: {None if res10 is None else len(res10)} | "
          f"events: {None if ev is None else len(ev)}", flush=True)
    fig_capture_budget(out_dir, res5, res10)
    fig_delay(out_dir, ev)
    fig_short_events(out_dir)
    for p in sorted((out_dir / "figs").glob("*.png")):
        print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
