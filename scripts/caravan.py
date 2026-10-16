# -*- coding: utf-8 -*-
"""Caravan / MultiMet loaders.

When real archives are absent, callers should use `src.data.demo.build_demo_data`.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data.paths import data_root, ensure_dirs


CARAVAN_URLS = {
    "csv_zenodo": "https://zenodo.org/records/15530021",
    "nc_zenodo": "https://doi.org/10.5281/zenodo.6522634",
    "github": "https://github.com/kratzert/Caravan",
    "multimet_nowcasts": "https://zenodo.org/records/14161235",
    "multimet_forecasts": "https://zenodo.org/records/14161281",
    "grdc_caravan": "https://doi.org/10.5281/zenodo.15349031",
    "usgs_illinois": "https://doi.org/10.5066/P13ITOD4",
    "grqa": "https://zenodo.org/records/15335450",
    "gemstat_gfqa": "https://doi.org/10.5281/zenodo.18459694",
}


def caravan_root() -> Path:
    env = os.environ.get("CARAVAN_ROOT")
    if env:
        return Path(env)
    base = data_root() / "raw" / "caravan"
    # Prefer extracted Caravan-csv layout if present
    nested = base / "Caravan-csv"
    if nested.is_dir() and (nested / "attributes").exists():
        return nested
    return base


def multimet_root() -> Path:
    env = os.environ.get("MULTIMET_ROOT")
    if env:
        return Path(env)
    return data_root() / "raw" / "multimet"


def print_download_instructions() -> None:
    ensure_dirs()
    print("=== Dataset download URLs (see docs/02_data_inventory.md) ===")
    for k, v in CARAVAN_URLS.items():
        print(f"  {k}: {v}")
    print(f"Extract Caravan into: {caravan_root()}")
    print("=== MultiMet ===")
    print("  Prefer GCP zarr subset via Caravan Colab; or Zenodo Part1/Part2.")
    print(f"Place under: {multimet_root()}")
    print("=== USGS Illinois ===")
    print(f"  Place CSV under: {data_root() / 'raw' / 'usgs_illinois'}")
    print("  or set USGS_NUTRIENT_CSV=")


def _normalize_basin_id(df: pd.DataFrame) -> pd.DataFrame:
    if "basin_id" in df.columns:
        return df
    for c in ("gauge_id", "id", "catchment_id"):
        if c in df.columns:
            return df.rename(columns={c: "basin_id"})
    return df


def load_caravan_attributes(path: Path | None = None) -> pd.DataFrame:
    """Load and concatenate all Caravan attribute tables under attributes/*.

    Caravan ships one CSV (or several) per source dataset (camels, hysets, ...).
    Earlier versions only read the first match; that yielded ~561 rows (e.g. camelsaus).
    """
    root = path or caravan_root()
    # Prefer attributes/ tree; fall back to any *attributes*.csv
    attr_dir = root / "attributes"
    if attr_dir.is_dir():
        candidates = sorted(attr_dir.rglob("*.csv")) + sorted(attr_dir.rglob("*.parquet"))
    else:
        candidates = sorted(root.rglob("*attributes*.csv")) + sorted(
            root.rglob("*attributes*.parquet")
        )
    # skip license-only or empty
    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        raise FileNotFoundError(
            f"No Caravan attributes under {root}. Run demo or download Caravan."
        )

    frames: list[pd.DataFrame] = []
    for fp in candidates:
        try:
            df = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)
        except Exception:
            continue
        df = _normalize_basin_id(df)
        if "basin_id" not in df.columns:
            continue
        src = fp.parent.name
        df = df.assign(attribute_source=src)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"Attribute files found but none readable under {root}")

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["basin_id"], keep="first")
    return out.copy()


def load_caravan_timeseries(
    basin_id: str,
    path: Path | None = None,
) -> pd.DataFrame:
    """Load one basin daily timeseries CSV/parquet."""
    root = path or caravan_root()
    # Prefer timeseries/csv
    csv_root = root / "timeseries" / "csv"
    search_roots = [csv_root] if csv_root.is_dir() else [root]
    matches: list[Path] = []
    for sr in search_roots:
        matches.extend(sr.rglob(f"*{basin_id}*.csv"))
        matches.extend(sr.rglob(f"*{basin_id}*.parquet"))
    if not matches:
        matches = list(root.rglob(f"*{basin_id}*.csv")) + list(
            root.rglob(f"*{basin_id}*.parquet")
        )
    if not matches:
        raise FileNotFoundError(f"No timeseries for {basin_id} under {root}")
    fp = matches[0]
    df = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)
    if "date" not in df.columns:
        for c in ("time", "datetime", "Date"):
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break
    df["date"] = pd.to_datetime(df["date"])
    df["basin_id"] = basin_id
    return df


def load_interim_hydro() -> pd.DataFrame:
    fp = data_root() / "interim" / "hydro_daily.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}; run: python -m src.data.demo")
    return pd.read_parquet(fp)
