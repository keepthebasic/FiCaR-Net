# -*- coding: utf-8 -*-
"""USGS nutrient alignment utilities."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import data_root, ensure_dirs

# EPA MCL narrative default (mg/L as N); configurable in YAML
DEFAULT_NO3_THRESHOLD = 10.0


def usgs_csv_path() -> Path | None:
    env = os.environ.get("USGS_NUTRIENT_CSV")
    if env:
        return Path(env)
    default = data_root() / "raw" / "usgs_illinois"
    if not default.exists():
        return None
    csvs = list(default.glob("*.csv"))
    return csvs[0] if csvs else None


def normalize_nitrate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map heterogeneous USGS column names to a canonical schema."""
    colmap = {}
    lower = {c.lower(): c for c in df.columns}
    pairs = [
        ("date", ["date", "datetime", "datetimeutc", "time"]),
        (
            "site_no",
            [
                "site_no",
                "site_no.",
                "siteid",
                "station_id",
                "monitoringlocationidentifier",
                "usgs_site",
            ],
        ),
        (
            "no3_mg_l",
            [
                "nitrate",
                "no3",
                "no3_n",
                "nitrate_n",
                "nitrateplusnitrite",
                "conc_no3",
                "no3_mg_per_l",
                "no3_mg_l",
            ],
        ),
        (
            "discharge_cms",
            ["discharge", "q", "flow", "streamflow", "00060", "q_ft3_per_s", "q_cfs"],
        ),
        ("basin_id", ["basin_id", "gauge_id", "catchment_id"]),
    ]
    for canon, aliases in pairs:
        for a in aliases:
            if a in lower:
                colmap[lower[a]] = canon
                break
    out = df.rename(columns=colmap).copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def align_q_no3(
    hydro: pd.DataFrame,
    wq: pd.DataFrame,
    threshold: float = DEFAULT_NO3_THRESHOLD,
    how: str = "inner",
) -> pd.DataFrame:
    """Join discharge and nitrate on basin_id+date (or site mapping)."""
    h = hydro.copy()
    w = normalize_nitrate_columns(wq)
    h["date"] = pd.to_datetime(h["date"]).dt.normalize()
    if "basin_id" not in w.columns and "site_no" in w.columns:
        # identity map for demo / pre-mapped tables
        w["basin_id"] = w["site_no"]
    key = ["basin_id", "date"]
    for k in key:
        if k not in h.columns or k not in w.columns:
            raise KeyError(f"align_q_no3 requires '{k}' in both frames")
    merged = h.merge(w, on=key, how=how, suffixes=("", "_wq"))
    if "discharge_mm" not in merged.columns and "discharge_cms" in merged.columns:
        merged["discharge_mm"] = merged["discharge_cms"]  # placeholder without area
    if "no3_mg_l" in merged.columns:
        merged["exceed"] = (merged["no3_mg_l"] >= threshold).astype(int)
        if "discharge_mm" in merged.columns:
            merged["load_proxy"] = merged["no3_mg_l"] * merged["discharge_mm"]
    return merged


def load_or_build_joined(
    threshold: float = DEFAULT_NO3_THRESHOLD,
) -> pd.DataFrame:
    ensure_dirs()
    interim = data_root() / "interim" / "joined_q_no3.parquet"
    if interim.exists():
        return pd.read_parquet(interim)

    usgs = usgs_csv_path()
    hydro_fp = data_root() / "interim" / "hydro_daily.parquet"
    if usgs and hydro_fp.exists():
        wq = pd.read_csv(usgs)
        hydro = pd.read_parquet(hydro_fp)
        joined = align_q_no3(hydro, wq, threshold=threshold)
        joined.to_parquet(interim, index=False)
        return joined

    raise FileNotFoundError(
        "No joined_q_no3.parquet. Run `python -m src.data.demo` "
        "or place USGS CSV and interim hydro_daily.parquet."
    )


def train_val_test_split_by_time(
    df: pd.DataFrame,
    train_end: str = "2020-12-31",
    val_end: str = "2021-12-31",
) -> dict[str, pd.DataFrame]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    train = d[d["date"] <= train_end]
    val = d[(d["date"] > train_end) & (d["date"] <= val_end)]
    test = d[d["date"] > val_end]
    return {"train": train, "val": val, "test": test}
