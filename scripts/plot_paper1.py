# -*- coding: utf-8 -*-
"""Paper-1 core figures from cascade CSVs.

Figs:
  1. E3 vs E4: miss rate & delta_miss vs lead (+ optional perturbed overlay)
  2. Cascade heatmap: lead × flow_quantile × exceed_miss_rate (annotate n_exceed)
  3. Flow skill: flow_nse (overall) and flow_nse_tail by quantile vs lead

Usage:
  python -m src.experiments.plot_paper1 \\
    --transfer outputs/cascade_ficar_transfer_v2 \\
    --perturbed outputs/cascade_perturbed \\
    --out outputs/paper1_figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.paths import repo_root


def _load_e3e4(d: Path) -> pd.DataFrame:
    fp = d / "e3_e4_summary.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_csv(fp)


def _load_cascade(d: Path) -> pd.DataFrame:
    fp = d / "cascade_error_table.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_csv(fp)


def fig_e3_e4(
    transfer: pd.DataFrame,
    out: Path,
    perturbed: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(transfer["lead"], transfer["oracle_miss"], "o-", color="#1f4e79", label="E3 oracle Q", lw=2)
    ax.plot(transfer["lead"], transfer["fc_miss"], "s-", color="#c45c26", label="E4 FiCaR residual", lw=2)
    if baseline is not None and "fc_miss" in baseline.columns:
        ax.plot(
            baseline["lead"],
            baseline["fc_miss"],
            "D-",
            color="#2a9d8f",
            label="E4 baseline residual",
            lw=1.8,
        )
    if perturbed is not None and "fc_miss" in perturbed.columns:
        ax.plot(
            perturbed["lead"],
            perturbed["fc_miss"],
            "^--",
            color="#6b6b6b",
            label="perturbed Q",
            lw=1.5,
        )
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel("Exceedance miss rate")
    ax.set_xticks(list(transfer["lead"]))
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("E3 vs E4 miss rate")

    ax = axes[1]
    leads = transfer["lead"].to_numpy()
    x = np.arange(len(leads))
    width = 0.25
    ax.bar(x - width, transfer["delta_miss"], width=width, color="#c45c26", label="FiCaR")
    if baseline is not None and "delta_miss" in baseline.columns:
        ax.bar(x, baseline["delta_miss"], width=width, color="#2a9d8f", label="baseline")
    if perturbed is not None and "delta_miss" in perturbed.columns:
        ax.bar(x + width, perturbed["delta_miss"], width=width, color="#9e9e9e", label="perturbed")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in leads])
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel(r"$\Delta$ miss (E4 $-$ E3)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Cascade miss amplification")

    fp = out / "fig_e3_e4_miss.png"
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    return fp


def fig_model_compare(
    ficar: pd.DataFrame,
    baseline: pd.DataFrame | None,
    perturbed: pd.DataFrame | None,
    out: Path,
) -> Path | None:
    """FiCaR vs baseline (E0) vs noise: delta_miss + flow_nse."""
    if baseline is None and perturbed is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(ficar["lead"], ficar["delta_miss"], "s-", color="#c45c26", lw=2, label="FiCaR")
    if baseline is not None:
        ax.plot(baseline["lead"], baseline["delta_miss"], "D-", color="#2a9d8f", lw=2, label="baseline LSTM")
    if perturbed is not None:
        ax.plot(perturbed["lead"], perturbed["delta_miss"], "^--", color="#6b6b6b", lw=1.5, label="perturbed")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel(r"$\Delta$ miss")
    ax.set_xticks(list(ficar["lead"]))
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("E0 vs E1f cascade (same nutrient head)")

    ax = axes[1]
    ax.plot(ficar["lead"], ficar["flow_nse"], "s-", color="#c45c26", lw=2, label="FiCaR")
    if baseline is not None:
        ax.plot(baseline["lead"], baseline["flow_nse"], "D-", color="#2a9d8f", lw=2, label="baseline LSTM")
    if perturbed is not None:
        ax.plot(perturbed["lead"], perturbed["flow_nse"], "^--", color="#6b6b6b", lw=1.5, label="perturbed")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel("Transferred flow NSE")
    ax.set_xticks(list(ficar["lead"]))
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Flow skill under residual transfer")

    fp = out / "fig_e0_e1f_compare.png"
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    return fp


def fig_cascade_heatmap(casc: pd.DataFrame, out: Path) -> Path:
    piv = casc.pivot(index="flow_quantile", columns="lead", values="exceed_miss_rate")
    piv = piv.sort_index(ascending=False)
    n_ex = None
    if "n_exceed" in casc.columns:
        n_ex = casc.pivot(index="flow_quantile", columns="lead", values="n_exceed")
        n_ex = n_ex.reindex(piv.index)

    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    vmax = max(0.35, float(np.nanmax(piv.to_numpy(dtype=float))))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="YlOrBr", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"Q{q:.2f}" for q in piv.index])
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel("Flow quantile stratum")
    ax.set_title("Exceedance miss rate (lead × flow quantile)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Miss rate")

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            if pd.isna(val):
                continue
            txt = f"{val:.2f}"
            if n_ex is not None and not pd.isna(n_ex.iloc[i, j]):
                txt += f"\nn={int(n_ex.iloc[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    fp = out / "fig_cascade_heatmap.png"
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    return fp


def fig_flow_skill(e3e4: pd.DataFrame, casc: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(e3e4["lead"], e3e4["flow_nse"], "o-", color="#1f4e79", lw=2)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel("Flow NSE (all days)")
    ax.set_xticks(list(e3e4["lead"]))
    ax.set_title("Transferred flow skill vs lead")

    ax = axes[1]
    for q, marker, color in [
        (0.5, "o", "#4c78a8"),
        (0.9, "s", "#f58518"),
        (0.95, "^", "#e45756"),
    ]:
        sub = casc[np.isclose(casc["flow_quantile"], q)]
        if sub.empty:
            continue
        ax.plot(sub["lead"], sub["flow_nse_tail"], marker=marker, color=color, lw=2, label=f"Q≥{q:.2f}")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel("Tail flow NSE")
    ax.set_xticks(sorted(casc["lead"].unique()))
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Tail flow degradation (H1)")

    fp = out / "fig_flow_skill.png"
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    return fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transfer",
        default="outputs/cascade_fair_transfer",
        help="FiCaR residual transfer dir",
    )
    parser.add_argument("--perturbed", default="", help="Optional noise ablation dir")
    parser.add_argument("--baseline", default="", help="Optional E0 baseline residual dir")
    parser.add_argument("--out", default="outputs/paper1_figs")
    args = parser.parse_args()

    root = repo_root()
    tdir = root / args.transfer
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    e3 = _load_e3e4(tdir)
    casc = _load_cascade(tdir)

    def _maybe(flag: str) -> pd.DataFrame | None:
        if not flag:
            return None
        pdir = root / flag
        if (pdir / "e3_e4_summary.csv").exists():
            return _load_e3e4(pdir)
        print(f"WARN: no e3_e4_summary in {pdir}, skip", flush=True)
        return None

    pert = _maybe(args.perturbed)
    base = _maybe(args.baseline)

    f1 = fig_e3_e4(e3, out, perturbed=pert, baseline=base)
    f2 = fig_cascade_heatmap(casc, out)
    f3 = fig_flow_skill(e3, casc, out)
    f4 = fig_model_compare(e3, base, pert, out)
    (out / "sources.txt").write_text(
        f"transfer={tdir}\nbaseline={args.baseline or 'none'}\nperturbed={args.perturbed or 'none'}\n",
        encoding="utf-8",
    )
    lines = [str(f1), str(f2), str(f3)]
    if f4 is not None:
        lines.append(str(f4))
    print("Wrote:\n  " + "\n  ".join(lines), flush=True)


if __name__ == "__main__":
    main()
