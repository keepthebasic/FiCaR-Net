# -*- coding: utf-8 -*-
"""Decision calibration: threshold sweep under E3 (oracle Q) vs E4 (residual Q).

H4 / Stage-5: at matched concentration head, compare management metrics
(cost, miss, FAR, F1) across probability thresholds for:
  - oracle (E3)
  - FiCaR residual transfer (E4)
  - baseline residual transfer (E4)
  - optional perturbed Q

Usage:
  python -m src.experiments.eval_decision_calib --config configs/decision_calib.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.paths import data_root, ensure_dirs, repo_root
from src.experiments.eval_cascade import (
    _perturbed_flow,
    build_lead_residual_libs,
    ensure_hydro,
    ensure_wq,
    load_nutrient_head,
    set_seed,
    val_split,
)
from src.experiments.hydro_infer import apply_residual_transfer
from src.metrics import classification_scores, cost_sensitive_loss, reliability_curve
from src.models.nutrient import NutrientRiskHead


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def collect_probs(
    model: NutrientRiskHead,
    val_df: pd.DataFrame,
    *,
    lookback: int,
    lead: int,
    device: torch.device,
    q_mode: str,
    residual_libs: dict[int, dict[str, np.ndarray]] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exceed_true, prob_oracle, prob_fc for one lead."""
    model.eval()
    y_true, p_or, p_fc = [], [], []
    for _bid, g in val_df.groupby("basin_id"):
        g = g.sort_values("date").reset_index(drop=True)
        q = g["discharge_mm"].to_numpy(np.float32)
        if q_mode == "hydro_transfer":
            assert residual_libs is not None
            q_fc = apply_residual_transfer(q, residual_libs[lead], rng)
        elif q_mode == "perturbed":
            q_fc = _perturbed_flow(q, lead, rng).astype(np.float32)
        else:
            raise ValueError(q_mode)
        ex = g["exceed"].to_numpy(np.float32)
        agri = float(g["frac_agriculture"].iloc[0])
        doy = g["date"].dt.dayofyear.to_numpy(np.float32)
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

            x0, s0 = pack(q)
            x1, s1 = pack(q_fc)
            o0 = model(x0, s0)
            o1 = model(x1, s1)
            y_true.append(float(ex[t]))
            p_or.append(float(o0["exceed_prob"]))
            p_fc.append(float(o1["exceed_prob"]))
    return (
        np.asarray(y_true, dtype=np.float32),
        np.asarray(p_or, dtype=np.float32),
        np.asarray(p_fc, dtype=np.float32),
    )


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    *,
    tag: str,
    lead: int,
    cost_fn: float,
    cost_fp: float,
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        s = classification_scores(y_true, y_prob, thr=float(thr))
        cost = cost_sensitive_loss(
            y_true, y_prob, thr=float(thr), cost_fn=cost_fn, cost_fp=cost_fp
        )
        rows.append(
            {
                "tag": tag,
                "lead": lead,
                "threshold": float(thr),
                "f1": s["f1"],
                "miss_rate": s["miss_rate"],
                "false_alarm_rate": s["false_alarm_rate"],
                "precision": s["precision"],
                "recall": s["recall"],
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "tn": s["tn"],
                "cost": cost,
                "n_exceed": int(y_true.sum()),
                "n": int(len(y_true)),
            }
        )
    return pd.DataFrame(rows)


def pick_operating_points(curve: pd.DataFrame) -> pd.DataFrame:
    """Per tag×lead: min-cost thr; and thr that hits miss≈oracle-min or 0.2."""
    rows = []
    for (tag, lead), g in curve.groupby(["tag", "lead"]):
        i_cost = g["cost"].idxmin()
        r = g.loc[i_cost].to_dict()
        r["rule"] = "min_cost"
        rows.append(r)
        # constrained: miss_rate <= 0.25, minimize FAR
        feas = g[g["miss_rate"] <= 0.25]
        if not feas.empty:
            j = feas["false_alarm_rate"].idxmin()
            r2 = g.loc[j].to_dict()
            r2["rule"] = "miss<=0.25_min_far"
            rows.append(r2)
    return pd.DataFrame(rows)


def plot_curves(curve: pd.DataFrame, out_dir: Path, focus_leads: list[int]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    colors = {
        "oracle": "#1f4e79",
        "ficar": "#c45c26",
        "baseline": "#2a9d8f",
        "perturbed": "#6b6b6b",
    }
    for lead in focus_leads:
        sub = curve[curve["lead"] == lead]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
        for tag, g in sub.groupby("tag"):
            g = g.sort_values("threshold")
            c = colors.get(tag, None)
            axes[0].plot(g["threshold"], g["cost"], "-o", ms=3, color=c, label=tag, lw=1.8)
            axes[1].plot(g["threshold"], g["miss_rate"], "-o", ms=3, color=c, label=tag, lw=1.8)
            axes[2].plot(
                g["threshold"], g["false_alarm_rate"], "-o", ms=3, color=c, label=tag, lw=1.8
            )
        axes[0].set_title(f"Cost (lead={lead})")
        axes[0].set_xlabel("Probability threshold")
        axes[0].set_ylabel("Cost-sensitive loss")
        axes[1].set_title(f"Miss rate (lead={lead})")
        axes[1].set_xlabel("Probability threshold")
        axes[1].set_ylabel("Miss rate")
        axes[2].set_title(f"False alarm rate (lead={lead})")
        axes[2].set_xlabel("Probability threshold")
        axes[2].set_ylabel("FAR")
        for ax in axes:
            ax.legend(frameon=False, fontsize=8)
        fp = out_dir / f"fig_threshold_lead{lead}.png"
        fig.savefig(fp, dpi=200)
        plt.close(fig)
        written.append(fp)

    # miss–FAR Pareto-style at one lead
    lead = focus_leads[0] if focus_leads else int(curve["lead"].iloc[0])
    sub = curve[curve["lead"] == lead]
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    for tag, g in sub.groupby("tag"):
        g = g.sort_values("miss_rate")
        ax.plot(
            g["miss_rate"],
            g["false_alarm_rate"],
            "-o",
            ms=3,
            color=colors.get(tag),
            label=tag,
            lw=1.8,
        )
    ax.set_xlabel("Miss rate")
    ax.set_ylabel("False alarm rate")
    ax.set_title(f"Decision tradeoff (lead={lead})")
    ax.legend(frameon=False, fontsize=8)
    fp = out_dir / f"fig_miss_far_lead{lead}.png"
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    written.append(fp)

    # reliability for oracle vs ficar at focus lead
    return written


def plot_reliability(
    packs: dict[str, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    lead: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    colors = {"oracle": "#1f4e79", "ficar": "#c45c26", "baseline": "#2a9d8f", "perturbed": "#6b6b6b"}
    for tag, (yt, yp) in packs.items():
        rc = reliability_curve(yt, yp, n_bins=10)
        ax.plot(
            rc["bin_centers"],
            rc["empirical"],
            "-o",
            ms=4,
            color=colors.get(tag),
            label=tag,
            lw=1.6,
        )
    ax.set_xlabel("Predicted exceedance probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability (lead={lead})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fp = out_dir / f"fig_reliability_lead{lead}.png"
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    return fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision_calib.yaml")
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
    leads = list(cfg.get("leads", [3, 7]))
    thresholds = np.asarray(cfg.get("thresholds", list(np.round(np.linspace(0.1, 0.9, 17), 3))))
    cost_fn = float(cfg.get("cost_fn", 5.0))
    cost_fp = float(cfg.get("cost_fp", 1.0))

    ckpt = repo_root() / cfg["nutrient_ckpt"]
    print(f"Loading nutrient head: {ckpt}", flush=True)
    model = load_nutrient_head(ckpt, cfg, device)

    methods = cfg.get("methods", {})
    # methods: {ficar: {flow_mode, hydro_ckpt_obs, hydro_ckpt_fc}, ...}
    hydro_df = None
    residual_cache: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for name, mcfg in methods.items():
        mode = str(mcfg.get("flow_mode", "hydro_transfer"))
        if mode == "hydro_transfer":
            if hydro_df is None:
                hydro_df = ensure_hydro()
            print(f"Building residual banks for {name} ...", flush=True)
            residual_cache[name] = build_lead_residual_libs(
                hydro_df,
                repo_root() / mcfg["hydro_ckpt_obs"],
                repo_root() / mcfg["hydro_ckpt_fc"],
                leads,
                device,
                max_basins=int(cfg.get("hydro_max_basins", 60)),
                base_lead=int(mcfg.get("hydro_base_lead", cfg.get("hydro_base_lead", 3))),
                clip=float(cfg.get("residual_clip", 0.8)),
            )

    out_dir = repo_root() / cfg.get("output_dir", "outputs/decision_calib")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    curves = []
    rel_packs: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {L: {} for L in leads}

    # oracle once per lead (same for all methods — compute with first method's loop using true Q)
    for lead in leads:
        print(f"Collecting probs lead={lead} ...", flush=True)
        # oracle from any run — use first hydro method or perturbed
        first_name = next(iter(methods))
        first = methods[first_name]
        mode0 = str(first.get("flow_mode", "hydro_transfer"))
        libs0 = residual_cache.get(first_name)
        yt, p_or, _ = collect_probs(
            model,
            val_df,
            lookback=lookback,
            lead=lead,
            device=device,
            q_mode=mode0,
            residual_libs=libs0,
            rng=np.random.default_rng(int(cfg.get("seed", 42))),
        )
        curves.append(
            sweep_thresholds(
                yt, p_or, thresholds, tag="oracle", lead=lead, cost_fn=cost_fn, cost_fp=cost_fp
            )
        )
        rel_packs[lead]["oracle"] = (yt, p_or)

        for name, mcfg in methods.items():
            mode = str(mcfg.get("flow_mode", "hydro_transfer"))
            libs = residual_cache.get(name)
            rng_m = np.random.default_rng(int(cfg.get("seed", 42)))
            yt2, _por, p_fc = collect_probs(
                model,
                val_df,
                lookback=lookback,
                lead=lead,
                device=device,
                q_mode=mode,
                residual_libs=libs,
                rng=rng_m,
            )
            assert np.allclose(yt, yt2)
            curves.append(
                sweep_thresholds(
                    yt2,
                    p_fc,
                    thresholds,
                    tag=name,
                    lead=lead,
                    cost_fn=cost_fn,
                    cost_fp=cost_fp,
                )
            )
            rel_packs[lead][name] = (yt2, p_fc)

    curve = pd.concat(curves, ignore_index=True)
    curve.to_csv(out_dir / "threshold_curves.csv", index=False)
    ops = pick_operating_points(curve)
    ops.to_csv(out_dir / "operating_points.csv", index=False)

    figs = plot_curves(curve, out_dir, focus_leads=leads)
    for lead in leads:
        figs.append(plot_reliability(rel_packs[lead], out_dir, lead))

    print("\n=== Operating points (min_cost) ===", flush=True)
    print(ops[ops["rule"] == "min_cost"][["tag", "lead", "threshold", "cost", "miss_rate", "false_alarm_rate", "f1"]], flush=True)
    print(f"\nWrote curves: {out_dir / 'threshold_curves.csv'}", flush=True)
    for f in figs:
        print(f"  {f}", flush=True)


if __name__ == "__main__":
    main()
