# -*- coding: utf-8 -*-
"""Sequence datasets for hydro and water-quality heads."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SeqConfig:
    lookback: int = 30
    horizon: int = 7
    force_cols: tuple[str, ...] = ("precip", "pet", "temp")
    target_col: str = "discharge_mm"
    forecast_lead: int | None = None  # if set, replace precip with precip_fc_l{lead}
    stride: int = 1  # subsample sliding windows to reduce autocorrelation


STATIC_COLS = ["area_log", "p_mean", "pet_mean_ERA5_LAND", "aridity_ERA5_LAND",
    "frac_snow", "high_prec_freq", "seasonality_ERA5_LAND",
    "ele_mt_sav", "slp_dg_sav", "for_pc_sse", "crp_pc_sse", "urb_pc_sse",
    "cly_pc_sav", "snd_pc_sav", "slt_pc_sav", "aet_mm_syr", "run_mm_syr"]


class BasinSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: SeqConfig, basin_ids: list[str] | None = None,
                 static_df: pd.DataFrame | None = None, static_cols: list[str] | None = None):
        self.cfg = cfg
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        ids = basin_ids or sorted(data["basin_id"].unique().tolist())
        force = list(cfg.force_cols)
        if cfg.forecast_lead is not None:
            fc = f"precip_fc_l{cfg.forecast_lead}"
            if fc not in data.columns:
                raise KeyError(f"Missing forecast column {fc}")
            force = [fc if c == "precip" else c for c in force]
        stride = max(int(cfg.stride), 1)

        static_cols = [c for c in (static_cols or []) if static_df is not None and c in static_df.columns]
        dyn_force = [c for c in force if c in data.columns]
        _sdf = static_df.set_index("basin_id")[static_cols] if static_cols else None
        static_map = {}
        if _sdf is not None:
            for bid in _sdf.index:
                static_map[bid] = np.asarray(_sdf.loc[bid], dtype=np.float32)

        for bid in ids:
            g = data.loc[data["basin_id"] == bid].sort_values("date")
            if len(g) < cfg.lookback + cfg.horizon + 1:
                continue
            x_dyn = g[dyn_force].to_numpy(dtype=np.float32)
            if bid in static_map:
                s = np.tile(static_map[bid][None, :], (len(g), 1))
                x_all = np.concatenate([x_dyn, s], axis=1).astype(np.float32)
            else:
                x_all = x_dyn
            y_all = g[cfg.target_col].to_numpy(dtype=np.float32)
            precip_all = (
                g["precip_phys"].to_numpy(dtype=np.float32)
                if "precip_phys" in g.columns
                else (
                    g["precip"].to_numpy(dtype=np.float32)
                    if "precip" in g
                    else x_all[:, 0]
                )
            )
            for t in range(cfg.lookback, len(g) - cfg.horizon + 1, stride):
                x = x_all[t - cfg.lookback : t]
                y = y_all[t : t + cfg.horizon]
                aux = precip_all[t : t + cfg.horizon]  # for mass-balance soft constraint
                self.samples.append((x, y, aux))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, aux = self.samples[idx]
        return (
            torch.from_numpy(np.array(x, copy=True)),
            torch.from_numpy(np.array(y, copy=True)),
            torch.from_numpy(np.array(aux, dtype=np.float32, copy=True)),
        )


class NutrientRiskDataset(Dataset):
    """Predict next-day NO3 and exceedance from recent Q + static agri fraction."""

    def __init__(self, df: pd.DataFrame, lookback: int = 14):
        self.samples = []
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        for bid, g in data.groupby("basin_id"):
            g = g.sort_values("date")
            need = ["discharge_mm", "no3_mg_l", "exceed", "frac_agriculture"]
            if any(c not in g.columns for c in need):
                continue
            q = g["discharge_mm"].to_numpy(np.float32)
            c = g["no3_mg_l"].to_numpy(np.float32)
            y_ex = g["exceed"].to_numpy(np.float32)
            agri = float(g["frac_agriculture"].iloc[0])
            doy = g["date"].dt.dayofyear.to_numpy(np.float32)
            for t in range(lookback, len(g)):
                x_seq = np.stack(
                    [
                        q[t - lookback : t],
                        np.sin(2 * np.pi * doy[t - lookback : t] / 365.25),
                        np.cos(2 * np.pi * doy[t - lookback : t] / 365.25),
                    ],
                    axis=-1,
                )
                static = np.array([agri], dtype=np.float32)
                self.samples.append((x_seq, static, c[t], y_ex[t], q[t]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_seq, static, conc, exceed, q = self.samples[idx]
        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(static),
            torch.tensor(conc),
            torch.tensor(exceed),
            torch.tensor(q),
        )
