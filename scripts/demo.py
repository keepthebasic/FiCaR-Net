# -*- coding: utf-8 -*-
"""Synthetic demo data that mirrors Caravan + USGS nitrate schema."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import ensure_dirs


def build_demo_data(
    n_basins: int = 24,
    n_wq_sites: int = 6,
    n_days: int = 365 * 4,
    seed: int = 42,
) -> dict[str, Path]:
    """Create demo hydro + nitrate tables under data/demo and data/interim."""
    rng = np.random.default_rng(seed)
    paths = ensure_dirs()
    demo = paths["demo"]
    interim = paths["interim"]

    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    basin_ids = [f"demo_{i:03d}" for i in range(n_basins)]
    attrs = pd.DataFrame(
        {
            "basin_id": basin_ids,
            "area_km2": rng.uniform(80, 2500, n_basins),
            "mean_precip": rng.uniform(1.5, 4.0, n_basins),
            "frac_agriculture": rng.uniform(0.05, 0.75, n_basins),
            "mean_slope": rng.uniform(0.5, 12.0, n_basins),
        }
    )
    attrs.to_parquet(demo / "basin_attributes.parquet", index=False)

    hydro_rows = []
    for _, row in attrs.iterrows():
        bid = row["basin_id"]
        seasonal = 2.5 + 1.2 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        precip = np.clip(
            row["mean_precip"] * seasonal + rng.normal(0, 1.2, n_days), 0, None
        )
        # Simple rainfall-runoff: lagged precip + baseflow
        q = 0.35 * precip + 0.25 * np.roll(precip, 1) + 0.15 * np.roll(precip, 2)
        q = q + 0.4 + 0.05 * row["frac_agriculture"]
        q = np.clip(q + rng.normal(0, 0.15, n_days), 0.05, None)
        pet = 2.0 + 1.0 * np.sin(2 * np.pi * (np.arange(n_days) - 30) / 365.25)
        # Forecast-perturbed precip: noise grows with lead proxy (stored as precip_fc_l{k})
        frame = {
            "date": dates,
            "basin_id": bid,
            "precip": precip,
            "pet": pet,
            "temp": 10 + 12 * np.sin(2 * np.pi * (np.arange(n_days) - 60) / 365.25),
            "discharge_mm": q,
        }
        for lead in (1, 3, 5, 7):
            noise = rng.normal(0, 0.25 * lead, n_days)
            frame[f"precip_fc_l{lead}"] = np.clip(precip + noise, 0, None)
        hydro_rows.append(pd.DataFrame(frame))

    hydro = pd.concat(hydro_rows, ignore_index=True)
    hydro.to_parquet(demo / "hydro_daily.parquet", index=False)
    hydro.to_parquet(interim / "hydro_daily.parquet", index=False)

    # Pair first n_wq_sites basins with nitrate (flush + seasonal agriculture signal)
    wq_basins = basin_ids[:n_wq_sites]
    wq_rows = []
    for bid in wq_basins:
        h = hydro.loc[hydro["basin_id"] == bid].sort_values("date")
        attr = attrs.loc[attrs["basin_id"] == bid].iloc[0]
        q = h["discharge_mm"].to_numpy()
        doy = h["date"].dt.dayofyear.to_numpy()
        seasonal_n = 3.0 + 2.5 * np.sin(2 * np.pi * (doy - 80) / 365.25)
        flush = 1.8 * np.log1p(q)
        c = (
            2.0
            + seasonal_n * (0.5 + attr["frac_agriculture"])
            + flush
            + rng.normal(0, 0.6, len(h))
        )
        c = np.clip(c, 0.1, None)
        # Convert mm/day proxy load: relative units for demo
        load = c * q
        wq_rows.append(
            pd.DataFrame(
                {
                    "date": h["date"].to_numpy(),
                    "basin_id": bid,
                    "site_no": f"USGS_DEMO_{bid}",
                    "no3_mg_l": c,
                    "load_proxy": load,
                    "discharge_mm": q,
                    "frac_agriculture": attr["frac_agriculture"],
                }
            )
        )

    wq = pd.concat(wq_rows, ignore_index=True)
    wq["exceed"] = (wq["no3_mg_l"] >= 10.0).astype(int)
    wq.to_parquet(demo / "wq_daily.parquet", index=False)
    wq.to_parquet(interim / "wq_daily.parquet", index=False)
    wq.to_parquet(interim / "joined_q_no3.parquet", index=False)
    attrs.to_parquet(interim / "basin_attributes.parquet", index=False)

    manifest = {
        "n_basins": n_basins,
        "n_wq_sites": n_wq_sites,
        "n_days": n_days,
        "hydro": str(interim / "hydro_daily.parquet"),
        "wq": str(interim / "joined_q_no3.parquet"),
        "attrs": str(interim / "basin_attributes.parquet"),
    }
    pd.Series(manifest).to_json(demo / "manifest.json")
    return {
        "hydro": interim / "hydro_daily.parquet",
        "wq": interim / "joined_q_no3.parquet",
        "attrs": interim / "basin_attributes.parquet",
        "manifest": demo / "manifest.json",
    }


if __name__ == "__main__":
    out = build_demo_data()
    print("Demo data written:")
    for k, v in out.items():
        print(f"  {k}: {v}")
