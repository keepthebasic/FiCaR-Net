# -*- coding: utf-8 -*-
"""Train nutrient risk head and evaluate cascade error table.

flow_mode:
  - perturbed: lead-scaled noise on Illinois Q (legacy smoke)
  - hydro_transfer: map Caravan FiCaR residuals (obs vs forecast forcing)
    onto Illinois Q — uses real hydro checkpoints (E3/E4 style)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.cascade import cascade_table, flow_error_by_quantile, save_cascade_outputs
from src.data.demo import build_demo_data
from src.data.datasets import NutrientRiskDataset
from src.data.paths import data_root, ensure_dirs, repo_root
from src.experiments.hydro_infer import (
    apply_residual_transfer,
    paired_log_residuals,
    predict_flow_series,
    residual_library_from_array,
    scale_library,
)
from src.metrics import classification_scores, cost_sensitive_loss, nse, reliability_curve
from src.models.nutrient import NutrientRiskHead, nutrient_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_wq() -> pd.DataFrame:
    fp = data_root() / "interim" / "joined_q_no3.parquet"
    if not fp.exists():
        build_demo_data()
    return pd.read_parquet(fp)


def ensure_hydro() -> pd.DataFrame:
    fp = data_root() / "interim" / "hydro_daily.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}; run build_interim --caravan first")
    return pd.read_parquet(fp)


def _perturbed_flow(q: np.ndarray, lead: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, 0.08 * lead, size=q.shape)
    return np.clip(q * (1.0 + noise), 0.05, None)


def build_lead_residual_libs(
    hydro_df: pd.DataFrame,
    ckpt_obs: Path,
    ckpt_fc: Path,
    leads: list[int],
    device: torch.device,
    max_basins: int = 60,
    base_lead: int = 3,
    clip: float = 0.8,
) -> dict[int, dict[str, np.ndarray]]:
    """Paired obs-forcing vs fc-forcing residuals; other leads scale from base_lead."""
    print(f"  obs-forcing preds from {ckpt_obs}", flush=True)
    pred_obs = predict_flow_series(
        hydro_df, ckpt_obs, device=device, forecast_lead=None, max_basins=max_basins
    )
    print(f"  fc-forcing preds from {ckpt_fc} lead={base_lead}", flush=True)
    pred_fc = predict_flow_series(
        hydro_df, ckpt_fc, device=device, forecast_lead=base_lead, max_basins=max_basins
    )
    m = pred_obs.merge(
        pred_fc[["basin_id", "date", "q_pred"]].rename(columns={"q_pred": "q_fc_hat"}),
        on=["basin_id", "date"],
        how="inner",
    )
    if m.empty:
        raise RuntimeError("No overlapping basin-date for paired residuals")
    lr = paired_log_residuals(pred_obs, pred_fc, clip=clip)
    base_lib = residual_library_from_array(lr, m["q_true"].to_numpy())
    print(
        f"  paired residual n={len(lr)} mean={lr.mean():.3f} std={lr.std():.3f} "
        f"obs_nse={nse(m['q_true'], m['q_pred']):.3f} "
        f"fc_nse={nse(m['q_true'], m['q_fc_hat']):.3f}",
        flush=True,
    )
    libs = {}
    for lead in leads:
        factor = float(lead) / float(base_lead)
        libs[lead] = scale_library(base_lib, factor)
    return libs


def _split_periods(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    tp = cfg.get("train_period")
    ep = cfg.get("test_period")
    if tp:
        t0, t1 = pd.to_datetime(tp[0]), pd.to_datetime(tp[1])
        train_df = d[(d["date"] >= t0) & (d["date"] <= t1)].copy()
    else:
        train_df = d[d["date"] <= "2020-12-31"].copy()
    if ep:
        e0, e1 = pd.to_datetime(ep[0]), pd.to_datetime(ep[1])
        val_df = d[(d["date"] >= e0) & (d["date"] <= e1)].copy()
    else:
        val_df = d[d["date"] > "2020-12-31"].copy()
    print(f"period split train={len(train_df)} val={len(val_df)}", flush=True)
    return train_df, val_df


def train_risk_head(df: pd.DataFrame, cfg: dict, device: torch.device):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if cfg.get("train_sites"):
        keep = set(cfg.get("train_sites"))
        df = df[df["basin_id"].isin(keep)].copy()
        print(f"head train sites[{len(keep)}] rows={len(df)}", flush=True)
    train_df, val_df = _split_periods(df, cfg)
    lookback = int(cfg["lookback"])
    train_ds = NutrientRiskDataset(train_df, lookback=lookback)
    val_ds = NutrientRiskDataset(val_df, lookback=lookback)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False)

    model = NutrientRiskHead(lookback=lookback, hidden=int(cfg["hidden"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        tr = []
        for x_seq, static, conc, exceed, _q in train_loader:
            x_seq, static = x_seq.to(device), static.to(device)
            conc, exceed = conc.to(device), exceed.to(device)
            out = model(x_seq, static)
            loss, stats = nutrient_loss(
                out, conc, exceed, pos_weight=float(cfg.get("pos_weight", 2.0))
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr.append(stats["total"])
        model.eval()
        conc_t, conc_p, ex_t, ex_p = [], [], [], []
        with torch.no_grad():
            for x_seq, static, conc, exceed, _q in val_loader:
                out = model(x_seq.to(device), static.to(device))
                conc_t.append(conc.numpy())
                conc_p.append(out["conc"].cpu().numpy())
                ex_t.append(exceed.numpy())
                ex_p.append(out["exceed_prob"].cpu().numpy())
        if conc_t:
            ct = np.concatenate(conc_t)
            cp = np.concatenate(conc_p)
            et = np.concatenate(ex_t)
            ep = np.concatenate(ex_p)
            scores = classification_scores(et, ep)
            metrics = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(tr)),
                "val_conc_nse": nse(ct, cp),
                "val_f1": scores["f1"],
                "val_miss": scores["miss_rate"],
                "val_cost": cost_sensitive_loss(et, ep),
            }
        else:
            metrics = {"epoch": epoch + 1, "train_loss": float(np.mean(tr))}
        history.append(metrics)
        print(metrics, flush=True)
    return model, history, val_df


def load_nutrient_head(ckpt_path: Path, cfg: dict, device: torch.device) -> NutrientRiskHead:
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    lookback = int(blob.get("cfg", cfg).get("lookback", cfg["lookback"]))
    hidden = int(blob.get("cfg", cfg).get("hidden", cfg["hidden"]))
    model = NutrientRiskHead(lookback=lookback, hidden=hidden).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def val_split(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    if cfg is not None and cfg.get("test_period"):
        ep = cfg.get("test_period")
        e0, e1 = pd.to_datetime(ep[0]), pd.to_datetime(ep[1])
        return d[(d["date"] >= e0) & (d["date"] <= e1)]
    return d[d["date"] > "2020-12-31"]


def eval_cascade(
    model: NutrientRiskHead,
    val_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    residual_libs: dict[int, dict[str, np.ndarray]] | None = None,
):
    """E3 truth-Q vs E4 forecast-Q driven exceedance (same nutrient head)."""
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    lookback = int(cfg["lookback"])
    leads = list(cfg.get("leads", [1, 3, 5, 7]))
    flow_mode = str(cfg.get("flow_mode", "perturbed"))
    model.eval()

    flow_true, flow_pred, exceed_true, exceed_prob = {}, {}, {}, {}
    oracle_rows = []

    for lead in leads:
        rows = []
        for bid, g in val_df.groupby("basin_id"):
            g = g.sort_values("date").reset_index(drop=True)
            q = g["discharge_mm"].to_numpy(np.float32)
            if flow_mode == "hydro_transfer":
                if residual_libs is None or lead not in residual_libs:
                    raise RuntimeError("hydro_transfer requires residual_libs")
                q_fc = apply_residual_transfer(q, residual_libs[lead], rng)
            else:
                q_fc = _perturbed_flow(q, lead, rng).astype(np.float32)
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
                    return torch.from_numpy(x).unsqueeze(0), torch.tensor([[agri]], dtype=torch.float32)

                with torch.no_grad():
                    x0, s0 = pack(q)
                    out0 = model(x0.to(device), s0.to(device))
                    x1, s1 = pack(q_fc)
                    out1 = model(x1.to(device), s1.to(device))
                rows.append(
                    {
                        "q_true": float(q[t]),
                        "q_pred": float(q_fc[t]),
                        "ex_true": float(ex[t]),
                        "ex_prob_oracle": float(out0["exceed_prob"]),
                        "ex_prob_fc": float(out1["exceed_prob"]),
                    }
                )
        tab = pd.DataFrame(rows)
        flow_true[lead] = tab["q_true"].to_numpy()
        flow_pred[lead] = tab["q_pred"].to_numpy()
        exceed_true[lead] = tab["ex_true"].to_numpy()
        exceed_prob[lead] = tab["ex_prob_fc"].to_numpy()
        # E3 vs E4 summary for this lead
        s_or = classification_scores(tab["ex_true"], tab["ex_prob_oracle"])
        s_fc = classification_scores(tab["ex_true"], tab["ex_prob_fc"])
        n_ex = int(tab["ex_true"].sum())
        oracle_rows.append(
            {
                "lead": lead,
                "n": len(tab),
                "n_exceed": n_ex,
                "oracle_f1": s_or["f1"],
                "oracle_miss": s_or["miss_rate"],
                "oracle_fn": s_or["fn"],
                "fc_f1": s_fc["f1"],
                "fc_miss": s_fc["miss_rate"],
                "fc_fn": s_fc["fn"],
                "delta_miss": s_fc["miss_rate"] - s_or["miss_rate"],
                "flow_nse": nse(tab["q_true"], tab["q_pred"]),
            }
        )

    e3e4 = pd.DataFrame(oracle_rows)
    table = cascade_table(leads, flow_true, flow_pred, exceed_true, exceed_prob)
    return table, e3e4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cascade_illinois.yaml")
    parser.add_argument(
        "--nutrient-ckpt",
        default="",
        help="Load existing nutrient_head.pt and skip training (fair compare)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip nutrient head training (requires --nutrient-ckpt or cfg.nutrient_ckpt)",
    )
    args = parser.parse_args()
    ensure_dirs()
    cfg = load_config(repo_root() / args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    df = ensure_wq()
    # apply threshold override into joined frame if column uses different thr
    thr = float(cfg.get("no3_threshold", 10.0))
    if "no3_mg_l" in df.columns:
        df = df.copy()
        df["exceed"] = (df["no3_mg_l"] >= thr).astype(np.float32)
    print(f"WQ basins={df['basin_id'].nunique()} rows={len(df)} thr={thr}", flush=True)
    if cfg.get("sites"):
        keep = set(cfg.get("sites"))
        df = df[df["basin_id"].isin(keep)].copy()
        print(f"site subset[{len(keep)}]: {sorted(keep)} rows={len(df)}", flush=True)

    residual_libs = None
    flow_mode = str(cfg.get("flow_mode", "perturbed"))
    if flow_mode == "hydro_transfer":
        ckpt_obs = repo_root() / cfg.get(
            "hydro_ckpt_obs", cfg.get("hydro_ckpt", "outputs/hydro_ficar_v2/model_best.pt")
        )
        ckpt_fc = repo_root() / cfg.get(
            "hydro_ckpt_fc", "outputs/hydro_ficar_fc_l3/model_best.pt"
        )
        for p in (ckpt_obs, ckpt_fc):
            if not p.exists():
                raise FileNotFoundError(p)
        hydro_df = ensure_hydro()
        leads = list(cfg.get("leads", [1, 3, 5, 7]))
        print(f"Building paired residual banks obs={ckpt_obs} fc={ckpt_fc} ...", flush=True)
        residual_libs = build_lead_residual_libs(
            hydro_df,
            ckpt_obs,
            ckpt_fc,
            leads,
            device,
            max_basins=int(cfg.get("hydro_max_basins", 60)),
            base_lead=int(cfg.get("hydro_base_lead", 3)),
            clip=float(cfg.get("residual_clip", 0.8)),
        )

    ckpt_arg = args.nutrient_ckpt or cfg.get("nutrient_ckpt", "")
    skip = bool(args.skip_train or ckpt_arg or int(cfg.get("epochs", 1)) == 0)
    history: list = []
    if skip and ckpt_arg:
        ckpt_path = repo_root() / ckpt_arg if not Path(ckpt_arg).is_absolute() else Path(ckpt_arg)
        print(f"Loading shared nutrient head: {ckpt_path}", flush=True)
        model = load_nutrient_head(ckpt_path, cfg, device)
        val_df = val_split(df, cfg)
    elif skip and not ckpt_arg:
        raise ValueError("--skip-train / epochs=0 requires --nutrient-ckpt or cfg.nutrient_ckpt")
    else:
        model, history, val_df = train_risk_head(df, cfg, device)

    table, e3e4 = eval_cascade(model, val_df, cfg, device, residual_libs=residual_libs)

    out_dir = repo_root() / cfg.get("output_dir", "outputs/cascade")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "nutrient_head.pt")
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    e3e4.to_csv(out_dir / "e3_e4_summary.csv", index=False)
    fp = save_cascade_outputs(table, out_dir)

    lead = list(cfg.get("leads", [1, 3, 5, 7]))[-1]
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    q = val_df.sort_values(["basin_id", "date"])["discharge_mm"].to_numpy()
    if flow_mode == "hydro_transfer" and residual_libs is not None:
        q_fc = apply_residual_transfer(q, residual_libs[lead], rng)
    else:
        q_fc = _perturbed_flow(q, lead, rng)
    flow_error_by_quantile(q, q_fc).to_csv(out_dir / "flow_quantile_errors.csv", index=False)

    print("\n=== E3 (oracle Q) vs E4 (forecast Q) ===", flush=True)
    print(e3e4, flush=True)
    print("\n=== Cascade table ===", flush=True)
    print(table, flush=True)
    print(f"Wrote cascade table: {fp}", flush=True)


if __name__ == "__main__":
    main()
