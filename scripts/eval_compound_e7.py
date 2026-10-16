# -*- coding: utf-8 -*-
"""E7v2: reinforced compound-scenario cascade (Paper 2 failure typology).

Changes vs v1:
  - Report n / n_exceed for every cell (no silent zeros)
  - Multiple compound definitions: Q90×season, Q80×season, high-agri-basin×Q90
  - Contrast metrics use (Δmiss − Δmiss_avg) and fc_miss ratio (not Δmiss ratio when avg<0)
  - Typology summary: seasonal_cascade vs highflow_event_dense

Usage:
  python -m src.experiments.eval_compound_e7 --config configs/compound_e7_v2.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.paths import ensure_dirs, repo_root
from src.experiments.eval_cascade import (
    build_lead_residual_libs,
    ensure_hydro,
    ensure_wq,
    load_nutrient_head,
    set_seed,
    val_split,
)
from src.experiments.hydro_infer import apply_residual_transfer
from src.metrics import classification_scores, cost_sensitive_loss, nse


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def collect_sample_table(
    model,
    val_df: pd.DataFrame,
    *,
    lookback: int,
    lead: int,
    device: torch.device,
    residual_libs: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    model.eval()
    # basin-level agri threshold from full val set
    basin_agri = (
        val_df.groupby("basin_id")["frac_agriculture"].first().astype(float)
    )
    agri_cut = float(basin_agri.median())

    rows = []
    for bid, g in val_df.groupby("basin_id"):
        g = g.sort_values("date").reset_index(drop=True)
        q = g["discharge_mm"].to_numpy(np.float32)
        q_fc = apply_residual_transfer(q, residual_libs[lead], rng)
        ex = g["exceed"].to_numpy(np.float32)
        agri = float(g["frac_agriculture"].iloc[0])
        dates = pd.to_datetime(g["date"])
        doy = dates.dt.dayofyear.to_numpy(np.float32)
        months = dates.dt.month.to_numpy(np.int32)
        q80 = float(np.quantile(q, 0.80))
        q90 = float(np.quantile(q, 0.90))
        for t in range(lookback, len(g)):

            def pack(qv):
                x = np.stack(
                    [
                        qv[t - lookback : t],
                        np.sin(2 * np.pi * doy[t - lookback : t] / 365.25),
                        np.cos(2 * np.pi * doy[t - lookback : t] / 365.25),
                    ],
                    axis=-1,
                )
                return (
                    torch.from_numpy(x).unsqueeze(0).to(device),
                    torch.tensor([[agri]], dtype=torch.float32, device=device),
                )

            o0 = model(*pack(q))
            o1 = model(*pack(q_fc))
            rows.append(
                {
                    "basin_id": bid,
                    "date": dates.iloc[t],
                    "month": int(months[t]),
                    "q_true": float(q[t]),
                    "q_pred": float(q_fc[t]),
                    "q80": q80,
                    "q90": q90,
                    "frac_agriculture": agri,
                    "high_agri_basin": agri >= agri_cut,
                    "ex_true": float(ex[t]),
                    "ex_prob_oracle": float(o0["exceed_prob"]),
                    "ex_prob_fc": float(o1["exceed_prob"]),
                    "lead": lead,
                }
            )
    return pd.DataFrame(rows)


def label_flags(df: pd.DataFrame, agri_months: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    out["is_agri_season"] = out["month"].isin(list(agri_months))
    out["is_q90"] = out["q_true"] >= out["q90"]
    out["is_q80"] = out["q_true"] >= out["q80"]
    return out


def scenario_masks() -> dict:
    return {
        "average": lambda d: (~d["is_q90"]) & (~d["is_agri_season"]),
        "agri_only": lambda d: d["is_agri_season"] & (~d["is_q90"]),
        "high_flow_only_q90": lambda d: d["is_q90"] & (~d["is_agri_season"]),
        "agri_season": lambda d: d["is_agri_season"],
        "high_flow_q90": lambda d: d["is_q90"],
        "high_flow_q80": lambda d: d["is_q80"],
        # compound variants
        "compound_q90_season": lambda d: d["is_q90"] & d["is_agri_season"],
        "compound_q80_season": lambda d: d["is_q80"] & d["is_agri_season"],
        "compound_highagri_q90": lambda d: d["high_agri_basin"] & d["is_q90"],
        "compound_highagri_q90_season": lambda d: (
            d["high_agri_basin"] & d["is_q90"] & d["is_agri_season"]
        ),
    }


def score_slice(d: pd.DataFrame, thr: float = 0.5) -> dict:
    n = int(len(d))
    n_ex = int(d["ex_true"].sum()) if n else 0
    base = {
        "n": n,
        "n_exceed": n_ex,
        "exceed_rate": n_ex / n if n else np.nan,
        "oracle_miss": np.nan,
        "fc_miss": np.nan,
        "delta_miss": np.nan,
        "oracle_f1": np.nan,
        "fc_f1": np.nan,
        "cost_oracle": np.nan,
        "cost_fc": np.nan,
        "delta_cost": np.nan,
        "flow_nse": np.nan,
        "valid_classif": False,
    }
    if n == 0 or n_ex < 3:  # require ≥3 exceedances for stable miss
        return base
    s_or = classification_scores(d["ex_true"], d["ex_prob_oracle"], thr=thr)
    s_fc = classification_scores(d["ex_true"], d["ex_prob_fc"], thr=thr)
    c_or = cost_sensitive_loss(d["ex_true"], d["ex_prob_oracle"], thr=thr)
    c_fc = cost_sensitive_loss(d["ex_true"], d["ex_prob_fc"], thr=thr)
    base.update(
        {
            "oracle_miss": s_or["miss_rate"],
            "fc_miss": s_fc["miss_rate"],
            "delta_miss": s_fc["miss_rate"] - s_or["miss_rate"],
            "oracle_f1": s_or["f1"],
            "fc_f1": s_fc["f1"],
            "cost_oracle": c_or,
            "cost_fc": c_fc,
            "delta_cost": c_fc - c_or,
            "flow_nse": nse(d["q_true"].to_numpy(), d["q_pred"].to_numpy()),
            "valid_classif": True,
        }
    )
    return base


def build_scenario_table(samples: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    masks = scenario_masks()
    rows = []
    for lead, g in samples.groupby("lead"):
        avg = score_slice(g[masks["average"](g)], thr=thr)
        for name, fn in masks.items():
            sc = score_slice(g[fn(g)], thr=thr)
            sc["scenario"] = name
            sc["lead"] = int(lead)
            # robust contrasts vs average (difference / ratio on fc_miss)
            sc["delta_miss_minus_avg"] = (
                sc["delta_miss"] - avg["delta_miss"]
                if sc["valid_classif"] and avg["valid_classif"]
                else np.nan
            )
            sc["fc_miss_ratio_vs_avg"] = (
                sc["fc_miss"] / avg["fc_miss"]
                if sc["valid_classif"]
                and avg["valid_classif"]
                and avg["fc_miss"] not in (0, np.nan)
                and not np.isnan(avg["fc_miss"])
                else np.nan
            )
            sc["exceed_rate_ratio_vs_avg"] = (
                sc["exceed_rate"] / avg["exceed_rate"]
                if avg["exceed_rate"] not in (0, np.nan) and not np.isnan(avg.get("exceed_rate", np.nan))
                else np.nan
            )
            rows.append(sc)
    return pd.DataFrame(rows)


def typology_summary(table: pd.DataFrame, leads: list[int]) -> pd.DataFrame:
    """Classify each lead's dominant failure mode from agri_only vs compound_q90."""
    rows = []
    for lead in leads:
        t = table[table["lead"] == lead]
        def get(name):
            s = t[t["scenario"] == name]
            return s.iloc[0] if len(s) else None

        avg = get("average")
        agri = get("agri_only")
        comp = get("compound_q90_season")
        if avg is None or agri is None or comp is None:
            continue
        # seasonal cascade signal: agri_only delta_miss > avg and > 0
        seasonal = bool(
            agri["valid_classif"]
            and not np.isnan(agri["delta_miss"])
            and agri["delta_miss"] > 0
            and (
                np.isnan(avg["delta_miss"])
                or agri["delta_miss"] > avg["delta_miss"] + 0.01
            )
        )
        # highflow event-dense: compound has higher exceed_rate but not higher delta_miss
        event_dense = bool(
            comp["valid_classif"]
            and comp["exceed_rate"] > avg["exceed_rate"] * 1.2
            and (
                np.isnan(comp["delta_miss"])
                or np.isnan(avg["delta_miss"])
                or comp["delta_miss"] <= avg["delta_miss"] + 0.01
            )
        )
        if seasonal and event_dense:
            typ = "mixed_seasonal_cascade_and_highflow_event_dense"
        elif seasonal:
            typ = "seasonal_cascade"
        elif event_dense:
            typ = "highflow_event_dense"
        else:
            typ = "inconclusive"
        rows.append(
            {
                "lead": lead,
                "typology": typ,
                "agri_only_delta_miss": agri["delta_miss"],
                "avg_delta_miss": avg["delta_miss"],
                "compound_q90_delta_miss": comp["delta_miss"],
                "agri_only_n_exceed": agri["n_exceed"],
                "avg_n_exceed": avg["n_exceed"],
                "compound_q90_n_exceed": comp["n_exceed"],
                "compound_exceed_rate_ratio": comp["exceed_rate_ratio_vs_avg"],
                "agri_only_fc_miss": agri["fc_miss"],
                "avg_fc_miss": avg["fc_miss"],
                "compound_fc_miss": comp["fc_miss"],
            }
        )
    return pd.DataFrame(rows)


def plot_e7v2(table: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    leads = sorted(table["lead"].unique())

    # 1) counts panel
    focus = [
        "average",
        "agri_only",
        "high_flow_only_q90",
        "compound_q90_season",
        "compound_q80_season",
        "compound_highagri_q90",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    # use lead=3 for count snapshot
    lead0 = 3 if 3 in leads else leads[0]
    sub = table[table["lead"] == lead0].set_index("scenario").reindex(focus)
    axes[0].bar(range(len(focus)), sub["n"].to_numpy(dtype=float), color="#4c78a8")
    axes[0].set_xticks(range(len(focus)))
    axes[0].set_xticklabels(focus, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("n days")
    axes[0].set_title(f"Sample size (lead={lead0})")
    axes[1].bar(range(len(focus)), sub["n_exceed"].to_numpy(dtype=float), color="#e45756")
    axes[1].set_xticks(range(len(focus)))
    axes[1].set_xticklabels(focus, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("n exceed")
    axes[1].set_title(f"Exceedance counts (lead={lead0})")
    for i, s in enumerate(focus):
        axes[1].text(i, sub.loc[s, "n_exceed"], str(int(sub.loc[s, "n_exceed"])), ha="center", va="bottom", fontsize=8)
    fp = out_dir / "fig_e7v2_counts.png"
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    written.append(fp)

    # 2) main narrative: agri_only vs average delta_miss
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    for name, color, marker in [
        ("average", "#9e9e9e", "o"),
        ("agri_only", "#2a9d8f", "s"),
        ("compound_q90_season", "#e45756", "^"),
        ("compound_q80_season", "#f58518", "D"),
        ("compound_highagri_q90", "#b279a2", "v"),
    ]:
        sub = table[table["scenario"] == name].set_index("lead").reindex(leads)
        ax.plot(leads, sub["delta_miss"], marker=marker, color=color, lw=2, label=name)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Lead (days)")
    ax.set_ylabel(r"$\Delta$ miss")
    ax.set_title("E7v2 cascade: seasonal vs compound defs")
    ax.legend(frameon=False, fontsize=7)
    fp = out_dir / "fig_e7v2_delta_miss_lines.png"
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    written.append(fp)

    # 3) typology support: exceed_rate ratio vs delta_miss_minus_avg at lead 3/7
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for ax, lead in zip(axes, [3, 7] if set([3, 7]) <= set(leads) else leads[:2]):
        sub = table[table["lead"] == lead].set_index("scenario").reindex(focus)
        ax.scatter(sub["exceed_rate_ratio_vs_avg"], sub["delta_miss_minus_avg"], s=60, c="#1f4e79")
        for s in focus:
            if s not in sub.index or pd.isna(sub.loc[s, "exceed_rate_ratio_vs_avg"]):
                continue
            ax.annotate(
                s.replace("compound_", "c_").replace("high_flow_only_q90", "hf_only"),
                (
                    sub.loc[s, "exceed_rate_ratio_vs_avg"],
                    sub.loc[s, "delta_miss_minus_avg"],
                ),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(1, color="k", lw=0.6, ls="--")
        ax.set_xlabel("exceed_rate / average")
        ax.set_ylabel(r"$\Delta$miss $-$ $\Delta$miss_avg")
        ax.set_title(f"Typology plane (lead={lead})")
    fp = out_dir / "fig_e7v2_typology_plane.png"
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    written.append(fp)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/compound_e7_v2.yaml")
    args = parser.parse_args()
    ensure_dirs()
    cfg = load_config(repo_root() / args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    thr_no3 = float(cfg.get("no3_threshold", 5.0))
    df = ensure_wq()
    if "no3_mg_l" in df.columns:
        df = df.copy()
        df["exceed"] = (df["no3_mg_l"] >= thr_no3).astype(np.float32)
    val_df = val_split(df)
    lookback = int(cfg.get("lookback", 14))
    leads = list(cfg.get("leads", [1, 3, 5, 7]))
    agri_months = tuple(cfg.get("agri_months", [3, 4, 5, 6]))
    prob_thr = float(cfg.get("prob_threshold", 0.5))

    ckpt = repo_root() / cfg["nutrient_ckpt"]
    print(f"Loading nutrient head: {ckpt}", flush=True)
    model = load_nutrient_head(ckpt, cfg, device)

    print("Building FiCaR residual banks ...", flush=True)
    libs = build_lead_residual_libs(
        ensure_hydro(),
        repo_root() / cfg["hydro_ckpt_obs"],
        repo_root() / cfg["hydro_ckpt_fc"],
        leads,
        device,
        max_basins=int(cfg.get("hydro_max_basins", 60)),
        base_lead=int(cfg.get("hydro_base_lead", 3)),
        clip=float(cfg.get("residual_clip", 0.8)),
    )

    parts = []
    for lead in leads:
        print(f"Collecting samples lead={lead} ...", flush=True)
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        parts.append(
            collect_sample_table(
                model,
                val_df,
                lookback=lookback,
                lead=lead,
                device=device,
                residual_libs=libs,
                rng=rng,
            )
        )
    samples = label_flags(pd.concat(parts, ignore_index=True), agri_months)

    out_dir = repo_root() / cfg.get("output_dir", "outputs/compound_e7_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out_dir / "e7_samples.parquet", index=False)

    table = build_scenario_table(samples, thr=prob_thr)
    table.to_csv(out_dir / "e7_scenario_table.csv", index=False)

    counts = table[["lead", "scenario", "n", "n_exceed", "exceed_rate", "valid_classif"]]
    counts.to_csv(out_dir / "e7_counts.csv", index=False)

    typ = typology_summary(table, leads)
    typ.to_csv(out_dir / "e7_typology.csv", index=False)

    # key contrast table for paper
    key_sc = [
        "average",
        "agri_only",
        "compound_q90_season",
        "compound_q80_season",
        "compound_highagri_q90",
        "compound_highagri_q90_season",
        "high_flow_only_q90",
    ]
    key = table[table["scenario"].isin(key_sc)][
        [
            "lead",
            "scenario",
            "n",
            "n_exceed",
            "exceed_rate",
            "delta_miss",
            "delta_miss_minus_avg",
            "fc_miss",
            "fc_miss_ratio_vs_avg",
            "delta_cost",
            "valid_classif",
        ]
    ]
    key.to_csv(out_dir / "e7_key_contrast.csv", index=False)

    figs = plot_e7v2(table, out_dir)

    print("\n=== Counts (lead=3) ===", flush=True)
    print(
        counts[counts["lead"] == 3][["scenario", "n", "n_exceed", "exceed_rate", "valid_classif"]].to_string(
            index=False
        ),
        flush=True,
    )
    print("\n=== Key Δmiss (wide) ===", flush=True)
    print(
        table[table["scenario"].isin(key_sc)]
        .pivot(index="scenario", columns="lead", values="delta_miss")
        .reindex(key_sc),
        flush=True,
    )
    print("\n=== Typology ===", flush=True)
    print(typ.to_string(index=False), flush=True)
    print(f"\nWrote {out_dir}", flush=True)
    for f in figs:
        print(f"  {f}", flush=True)


if __name__ == "__main__":
    main()
