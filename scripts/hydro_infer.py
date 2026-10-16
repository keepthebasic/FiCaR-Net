# -*- coding: utf-8 -*-
"""Hydro checkpoint inference helpers for cascade transfer (static-aware)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.ficar import FiCaRHydro
from src.models.hydro import ProcessConstrainedHydro


def _infer_n_feat(blob, cfg) -> int:
    sd = blob.get("model", {})
    for k in ("encoder.weight_ih_l0", "encoder.weight_ih_l_0"):
        if k in sd:
            return int(sd[k].shape[1])
    return 3 + len(cfg.get("static_cols", []))


def load_hydro_ckpt(ckpt_path: Path, device: torch.device):
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    cfg = blob.get("cfg", {})
    model_name = str(blob.get("model_name", cfg.get("model", "process_lstm"))).lower()
    n_feat = _infer_n_feat(blob, cfg)
    hidden = int(cfg.get("hidden", 64))
    horizon = int(cfg.get("horizon", 7))
    use_spectral = bool(cfg.get("use_spectral", False))
    probabilistic = bool(cfg.get("probabilistic", False))
    if model_name in {"ficar", "ficar_hydro", "ficar-net"}:
        model = FiCaRHydro(
            n_features=n_feat, hidden=hidden, horizon=horizon,
            use_spectral=use_spectral, probabilistic=probabilistic,
        )
    else:
        model = ProcessConstrainedHydro(
            n_features=n_feat, hidden=hidden, horizon=horizon,
            probabilistic=probabilistic,
        )
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    return model, cfg, blob.get("norm_stats", {}), blob


def _apply_norm(df: pd.DataFrame, cols: list[str], norm_stats: dict) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns or c not in norm_stats:
            continue
        out[c] = (out[c] - norm_stats[c]["mean"]) / norm_stats[c]["std"]
    return out


@torch.no_grad()
def predict_flow_series(
    hydro_df: pd.DataFrame,
    ckpt_path: Path,
    *,
    device: torch.device,
    forecast_lead: int | None = None,
    basin_ids: list[str] | None = None,
    max_basins: int = 80,
    stride: int | None = None,
) -> pd.DataFrame:
    """Rolling next-step flow prediction."""
    model, cfg, norm_stats, blob = load_hydro_ckpt(ckpt_path, device)
    lookback = int(cfg.get("lookback", 30))
    horizon = int(cfg.get("horizon", 7))
    force = ["precip", "pet", "temp"]
    lead = forecast_lead if forecast_lead is not None else cfg.get("forecast_lead")
    if lead is not None:
        fc = f"precip_fc_l{int(lead)}"
        if fc not in hydro_df.columns:
            raise KeyError(f"Missing {fc} in hydro_daily")
        force = [fc if c == "precip" else c for c in force]

    static_cols = list(cfg.get("static_cols", []))
    if _infer_n_feat(blob, cfg) > 3 and not static_cols:
        from src.data.datasets import STATIC_COLS as _DEFAULT_STATIC

        static_cols = list(_DEFAULT_STATIC)
    use_static = len(static_cols) > 0 and _infer_n_feat(blob, cfg) > 3
    df = hydro_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    ids = basin_ids or sorted(df["basin_id"].unique().tolist())
    if max_basins and len(ids) > max_basins:
        rng = np.random.default_rng(0)
        ids = sorted(rng.choice(ids, size=max_basins, replace=False).tolist())

    step = stride if stride is not None else max(int(cfg.get("stride", 7)), 1)
    df_n = _apply_norm(df, force, norm_stats)
    smap: dict[str, np.ndarray] = {}
    if use_static:
        from src.data.paths import data_root

        afp = data_root() / "interim" / "basin_attributes.parquet"
        sdf = pd.read_parquet(afp)
        if "area_log" in static_cols and "area_log" not in sdf.columns and "area_km2" in sdf.columns:
            sdf["area_log"] = np.log1p(sdf["area_km2"].astype(float).clip(lower=0))
        snorm = blob.get("static_norm", {})
        base = sdf.set_index("basin_id")
        for bid in base.index:
            vals = []
            for c in static_cols:
                x = float(base.loc[bid, c]) if c in base.columns else 0.0
                if c in snorm:
                    x = (x - snorm[c]["mean"]) / snorm[c]["std"]
                vals.append(0.0 if not np.isfinite(x) else x)
            smap[bid] = np.array(vals, dtype=np.float32)
    rows = []
    for bid in ids:
        g = df_n.loc[df_n["basin_id"] == bid].sort_values("date")
        g0 = df.loc[df["basin_id"] == bid].sort_values("date")
        if len(g) < lookback + horizon:
            continue
        x_dyn = g[force].to_numpy(np.float32)
        if use_static:
            s = smap.get(bid, np.zeros(len(static_cols), dtype=np.float32))
            x_all = np.concatenate([x_dyn, np.tile(s[None, :], (len(g), 1))], axis=1).astype(np.float32)
        else:
            x_all = x_dyn
        q_true = g0["discharge_mm"].to_numpy(np.float32)
        dates = g0["date"].to_numpy()
        for t in range(lookback, len(g) - horizon + 1, step):
            x = torch.from_numpy(x_all[t - lookback : t]).unsqueeze(0).to(device)
            out = model(x)
            q_hat = float(out["q_total"][0, 0].cpu())
            rows.append(
                {
                    "basin_id": bid,
                    "date": dates[t],
                    "q_true": float(q_true[t]),
                    "q_pred": q_hat,
                    "forecast_lead": int(lead) if lead is not None else 0,
                }
            )
    return pd.DataFrame(rows)


def paired_log_residuals(
    pred_obs: pd.DataFrame,
    pred_fc: pd.DataFrame,
    clip: float = 0.8,
) -> np.ndarray:
    """log(q_fc / q_obs_model) on matched basin-date — forecast degradation only."""
    a = pred_obs.rename(columns={"q_pred": "q_obs_hat"})
    b = pred_fc.rename(columns={"q_pred": "q_fc_hat"})
    m = a.merge(b[["basin_id", "date", "q_fc_hat"]], on=["basin_id", "date"], how="inner")
    if m.empty:
        raise RuntimeError("No overlapping predictions for paired residuals")
    lr = np.log((m["q_fc_hat"].to_numpy() + 1e-3) / (m["q_obs_hat"].to_numpy() + 1e-3))
    return np.clip(lr, -clip, clip)


def residual_library_from_array(
    log_ratio: np.ndarray,
    q_true: np.ndarray,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.95),
) -> dict[str, np.ndarray]:
    lib = {"_all": log_ratio}
    for q in quantiles:
        thr = np.quantile(q_true, q)
        m = q_true >= thr
        if m.sum() >= 20:
            lib[f"q{q}"] = log_ratio[m]
    return lib


def scale_library(lib: dict[str, np.ndarray], factor: float) -> dict[str, np.ndarray]:
    return {k: v * factor for k, v in lib.items()}


def apply_residual_transfer(
    q_true: np.ndarray,
    lib: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.empty_like(q_true, dtype=np.float32)
    thr90 = np.quantile(q_true, 0.9)
    thr95 = np.quantile(q_true, 0.95)
    for i, q in enumerate(q_true):
        if q >= thr95 and "q0.95" in lib:
            pool = lib["q0.95"]
        elif q >= thr90 and "q0.9" in lib:
            pool = lib["q0.9"]
        else:
            pool = lib.get("q0.5", lib["_all"])
        lr = float(rng.choice(pool))
        out[i] = np.float32(max(np.exp(lr) * (q + 1e-3) - 1e-3, 0.01))
    return out
