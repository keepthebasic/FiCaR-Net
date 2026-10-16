# -*- coding: utf-8 -*-
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

from src.data.demo import build_demo_data
from src.data.datasets import BasinSequenceDataset, SeqConfig, STATIC_COLS
from src.data.paths import data_root, ensure_dirs, repo_root
from src.metrics import kge, nse, peak_relative_error, quantile_bias
from src.models.hydro import ProcessConstrainedHydro, hydro_loss
from src.models.ficar import FiCaRHydro


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_hydro_table():
    fp = data_root() / "interim" / "hydro_daily.parquet"
    if not fp.exists():
        build_demo_data()
    return pd.read_parquet(fp)


def normalize_forcings(df: pd.DataFrame, force_cols: list[str], train_ids: list[str]):
    """Z-score forcings using training basins only."""
    out = df.copy()
    stats = {}
    mask = out["basin_id"].isin(train_ids)
    for c in force_cols:
        if c not in out.columns:
            continue
        mu = float(out.loc[mask, c].mean())
        sd = float(out.loc[mask, c].std())
        sd = sd if sd > 1e-6 else 1.0
        out[c] = (out[c] - mu) / sd
        stats[c] = {"mean": mu, "std": sd}
    return out, stats


def train(cfg: dict) -> Path:
    set_seed(int(cfg.get("seed", 42)))
    ensure_dirs()
    df = ensure_hydro_table()
    basins = sorted(df["basin_id"].unique())
    n = len(basins)
    train_ids = basins[: int(0.7 * n)]
    val_ids = basins[int(0.7 * n) : int(0.85 * n)]

    seq_cfg = SeqConfig(
        lookback=int(cfg["lookback"]),
        horizon=int(cfg["horizon"]),
        forecast_lead=cfg.get("forecast_lead"),
        stride=int(cfg.get("stride", 1)),
    )
    force_cols = list(seq_cfg.force_cols)
    if cfg.get("forecast_lead") is not None:
        fc = f"precip_fc_l{cfg['forecast_lead']}"
        force_cols = [fc if c == "precip" else c for c in force_cols]

    # Keep physical precip for mass-balance; never feed z-scored precip into MB.
    if "precip" in df.columns and "precip_phys" not in df.columns:
        df = df.copy()
        df["precip_phys"] = df["precip"].astype(float)

    if bool(cfg.get("normalize_inputs", True)):
        df, norm_stats = normalize_forcings(df, force_cols, train_ids)
    else:
        norm_stats = {}
    static_cols = list(cfg.get("static_cols", STATIC_COLS))
    afp = data_root() / "interim" / "basin_attributes.parquet"
    static_df = None
    static_norm = {}
    if afp.exists() and static_cols:
        sdf = pd.read_parquet(afp)
        sdf = sdf.loc[sdf["basin_id"].isin(basins)].copy()
        if "area_log" in static_cols and "area_log" not in sdf.columns and "area_km2" in sdf.columns:
            sdf["area_log"] = np.log1p(sdf["area_km2"].astype(float).clip(lower=0))
        static_cols = [c for c in static_cols if c in sdf.columns]
        for c in static_cols:
            sdf[c] = pd.to_numeric(sdf[c], errors="coerce")
            mu = float(sdf.loc[sdf["basin_id"].isin(train_ids), c].mean())
            sd = float(sdf.loc[sdf["basin_id"].isin(train_ids), c].std())
            if not np.isfinite(mu):
                mu = 0.0
            if not np.isfinite(sd) or sd < 1e-6:
                sd = 1.0
            static_norm[c] = {"mean": mu, "std": sd}
            sdf[c] = ((sdf[c] - mu) / sd).fillna(0.0).astype(np.float32)
        static_df = sdf[["basin_id"] + static_cols]
        seq_cfg.force_cols = tuple(list(seq_cfg.force_cols) + static_cols)
        cfg["static_cols"] = static_cols
        print(f"static_cols[{len(static_cols)}]: {static_cols}")
    train_ds = BasinSequenceDataset(df, seq_cfg, train_ids, static_df, static_cols)
    val_ds = BasinSequenceDataset(df, seq_cfg, val_ids, static_df, static_cols)
    print(
        f"basins train/val={len(train_ids)}/{len(val_ids)} "
        f"samples train/val={len(train_ds)}/{len(val_ds)}"
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=str(cfg.get("device", "cpu")).startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        device = torch.device("cpu")

    model_name = str(cfg.get("model", "process_lstm")).lower()
    n_feat = len(seq_cfg.force_cols)
    probabilistic = bool(cfg.get("probabilistic", True))
    if str(cfg.get("loss_type", "nll")).lower() in {"log_mse", "logmse", "mse", "l2"}:
        # point losses do not need free sigma (avoids collapse-to-zero + inflate-sigma)
        probabilistic = bool(cfg.get("probabilistic", False))

    if model_name in {"ficar", "ficar_hydro", "ficar-net"}:
        model = FiCaRHydro(
            n_features=n_feat,
            hidden=int(cfg["hidden"]),
            horizon=int(cfg["horizon"]),
            use_spectral=bool(cfg.get("use_spectral", True)),
            probabilistic=probabilistic,
        ).to(device)
    else:
        model = ProcessConstrainedHydro(
            n_features=n_feat,
            hidden=int(cfg["hidden"]),
            horizon=int(cfg["horizon"]),
            probabilistic=probabilistic,
        ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    loss_type = str(cfg.get("loss_type", "nll"))

    history = []
    best_nse = -1e9
    best_path = None
    out_dir = repo_root() / cfg.get("output_dir", "outputs/hydro")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(cfg["epochs"])):
        model.train()
        losses = []
        for x, y, precip in train_loader:
            x, y, precip = x.to(device), y.to(device), precip.to(device)
            out = model(x)
            loss, stats = hydro_loss(
                out,
                y,
                precip,
                w_mb=float(cfg["w_mb"]),
                w_split=float(cfg["w_split"]),
                w_bias=float(cfg.get("w_bias", 0.0)),
                loss_type=loss_type,
            )
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(stats["total"])
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y, precip in val_loader:
                x = x.to(device)
                out = model(x)
                ys.append(y.numpy())
                ps.append(out["q_total"].cpu().numpy())
        if ys:
            y_all = np.concatenate(ys).ravel()
            p_all = np.concatenate(ps).ravel()
            metrics = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_nse": nse(y_all, p_all),
                "val_kge": kge(y_all, p_all),
                "val_q95_bias": quantile_bias(y_all, p_all, 0.95),
                "val_peak_rel_err": peak_relative_error(y_all, p_all),
                "val_obs_mean": float(np.mean(y_all)),
                "val_pred_mean": float(np.mean(p_all)),
            }
            if metrics["val_nse"] > best_nse:
                best_nse = metrics["val_nse"]
                best_path = out_dir / "model_best.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "model_name": model_name,
                        "norm_stats": norm_stats,
                        "static_norm": static_norm,
                        "best_nse": best_nse,
                        "epoch": epoch + 1,
                    },
                    best_path,
                )
                metrics["saved_best"] = True
        else:
            metrics = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        history.append(metrics)
        print(metrics, flush=True)

    ckpt = out_dir / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
            "model_name": model_name,
            "norm_stats": norm_stats,
            "static_norm": static_norm,
        },
        ckpt,
    )
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if best_path is not None:
        print(f"Best val_nse={best_nse:.4f} -> {best_path}", flush=True)
    return best_path or ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hydro_baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(repo_root() / args.config)
    ckpt = train(cfg)
    print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
