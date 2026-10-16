# -*- coding: utf-8 -*-
"""Summarize hydro + cascade ablation results into one table / bar chart.

Hydro rows from outputs/*/history.json (best val_nse).
Cascade rows from outputs/cascade_fair_abl_*/e3_e4_summary.csv (lead=3,7).

  python -m src.experiments.summarize_ablation --out outputs/ablation_summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.paths import repo_root

HYDRO_RUNS = [
    ("full", "outputs/hydro_ficar_full", True, 0.05),
    ("no_spectral", "outputs/hydro_ficar_abl_nospec", False, 0.05),
    ("no_mass_balance", "outputs/hydro_ficar_abl_nomb", True, 0.0),
    ("prod_v2_both_off", "outputs/hydro_ficar_v2", False, 0.0),
    ("baseline", "outputs/hydro_baseline_v2", None, 0.0),
]

CASCADE_RUNS = [
    ("full", "outputs/cascade_fair_abl_full"),
    ("no_spectral", "outputs/cascade_fair_abl_nospec"),
    ("no_mass_balance", "outputs/cascade_fair_abl_nomb"),
    ("prod_v2", "outputs/cascade_fair_transfer"),
    ("baseline", "outputs/cascade_fair_baseline"),
]


def best_from_history(hist_path: Path) -> dict | None:
    if not hist_path.exists():
        return None
    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    rows = [h for h in hist if "val_nse" in h]
    if not rows:
        return None
    return max(rows, key=lambda r: r["val_nse"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ablation_summary")
    args = parser.parse_args()
    root = repo_root()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    hydro_rows = []
    for name, rel, spec, wmb in HYDRO_RUNS:
        best = best_from_history(root / rel / "history.json")
        if best is None:
            hydro_rows.append(
                {
                    "variant": name,
                    "use_spectral": spec,
                    "w_mb": wmb,
                    "best_val_nse": None,
                    "best_val_kge": None,
                    "best_q95_bias": None,
                    "status": "missing",
                }
            )
            continue
        hydro_rows.append(
            {
                "variant": name,
                "use_spectral": spec,
                "w_mb": wmb,
                "best_val_nse": best.get("val_nse"),
                "best_val_kge": best.get("val_kge"),
                "best_q95_bias": best.get("val_q95_bias"),
                "best_peak_rel_err": best.get("val_peak_rel_err"),
                "best_epoch": best.get("epoch"),
                "status": "ok",
            }
        )
    hydro_df = pd.DataFrame(hydro_rows)
    hydro_df.to_csv(out / "hydro_ablation.csv", index=False)

    casc_rows = []
    for name, rel in CASCADE_RUNS:
        fp = root / rel / "e3_e4_summary.csv"
        if not fp.exists():
            continue
        e = pd.read_csv(fp)
        for lead in (3, 7):
            sub = e[e["lead"] == lead]
            if sub.empty:
                continue
            r = sub.iloc[0]
            casc_rows.append(
                {
                    "variant": name,
                    "lead": int(lead),
                    "delta_miss": float(r["delta_miss"]),
                    "flow_nse": float(r["flow_nse"]),
                    "fc_f1": float(r["fc_f1"]),
                    "oracle_f1": float(r["oracle_f1"]),
                }
            )
    casc_df = pd.DataFrame(casc_rows)
    if not casc_df.empty:
        casc_df.to_csv(out / "cascade_ablation.csv", index=False)

    # plots
    ok = hydro_df[hydro_df["status"] == "ok"]
    if not ok.empty:
        fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
        ax.bar(ok["variant"], ok["best_val_nse"], color="#1f4e79")
        ax.set_ylabel("Best val NSE (obs forcing)")
        ax.set_title("Hydro ablation")
        ax.tick_params(axis="x", rotation=20)
        fig.savefig(out / "fig_hydro_ablation_nse.png", dpi=200)
        plt.close(fig)

    if not casc_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
        for ax, lead in zip(axes, [3, 7]):
            sub = casc_df[casc_df["lead"] == lead]
            ax.bar(sub["variant"], sub["delta_miss"], color="#c45c26")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"Cascade Δmiss (lead={lead})")
            ax.set_ylabel(r"$\Delta$ miss")
            ax.tick_params(axis="x", rotation=20)
        fig.savefig(out / "fig_cascade_ablation_deltamiss.png", dpi=200)
        plt.close(fig)

    print(hydro_df.to_string(index=False), flush=True)
    if not casc_df.empty:
        print(casc_df.to_string(index=False), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
