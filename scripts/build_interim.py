# -*- coding: utf-8 -*-
"""Build interim tables from Caravan + USGS Illinois (real data path).

Outputs under data/interim/:
  - basin_attributes.parquet
  - hydro_daily.parquet      (precip, pet, temp, discharge_mm, precip_fc_l*)
  - wq_daily.parquet / joined_q_no3.parquet  (Illinois cascade evidence)

Usage (server)::

    export CARAVAN_ROOT=/home/ubuntu/data/cascade_water_risk/data/raw/caravan/Caravan-csv
    export USGS_NUTRIENT_CSV=.../illinois_continuous_network_wy2016-2022.csv
    python -m src.data.build_interim --caravan --illinois --max-basins 500

Until MultiMet is mounted, forecast columns precip_fc_l{1,3,5,7} are
**perturbed analysis precip** (documented placeholder for Stage-2).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.caravan import caravan_root, load_caravan_attributes
from src.data.paths import data_root, ensure_dirs
from src.data.usgs_align import DEFAULT_NO3_THRESHOLD

# Approximate drainage areas (km^2) for Illinois continuous network sites
# Refine later from USGS NWIS metadata if needed.
ILLINOIS_AREA_KM2: dict[str, float] = {
    "3339000": 2400.0,  # Embarras River
    "3346500": 3100.0,  # Little Wabash River
    "3381495": 1400.0,  # Vermilion River
    "5446500": 24700.0,  # Rock River
    "5447500": 2600.0,  # Green River
    "5586300": 69200.0,  # Illinois River at Valley City
    "5595000": 14800.0,  # Kaskaskia River
    "5599490": 5600.0,  # Big Muddy River
}

CFS_TO_CMS = 0.028316846592


def _pick_col(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    # substring soft match
    for c in columns:
        cl = c.lower()
        for name in candidates:
            if name.lower() in cl:
                return c
    return None


def _standardize_caravan_ts(df: pd.DataFrame, basin_id: str) -> pd.DataFrame | None:
    """Map Caravan basin CSV to canonical hydro columns."""
    date_c = _pick_col(list(df.columns), ["date", "time", "datetime"])
    q_c = _pick_col(list(df.columns), ["streamflow", "discharge", "q_mm", "obs_streamflow"])
    p_c = _pick_col(
        list(df.columns),
        ["total_precipitation_sum", "total_precipitation", "precipitation", "precip"],
    )
    pet_c = _pick_col(
        list(df.columns),
        [
            "potential_evaporation_sum_era5_land",
            "potential_evaporation_sum_fao_penman_monteith",
            "potential_evaporation_sum",
            "potential_evaporation",
            "pet",
        ],
    )
    t_c = _pick_col(
        list(df.columns),
        ["temperature_2m_mean", "temperature_2m", "temp_mean", "tavg", "temperature"],
    )
    if date_c is None or q_c is None or p_c is None:
        return None

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_c]).dt.normalize(),
            "basin_id": basin_id,
            "precip": pd.to_numeric(df[p_c], errors="coerce"),
            "discharge_mm": pd.to_numeric(df[q_c], errors="coerce"),
        }
    )
    if pet_c:
        out["pet"] = pd.to_numeric(df[pet_c], errors="coerce")
    else:
        out["pet"] = np.nan
    if t_c:
        out["temp"] = pd.to_numeric(df[t_c], errors="coerce")
    else:
        out["temp"] = np.nan

    # drop rows with missing target streamflow
    out = out.dropna(subset=["discharge_mm", "precip"])
    out = out[out["discharge_mm"] >= 0]
    return out


def _add_forecast_precip_placeholder(
    df: pd.DataFrame,
    leads: tuple[int, ...] = (1, 3, 5, 7),
    seed: int = 42,
) -> pd.DataFrame:
    """Stage-2 placeholder until MultiMet: lead-dependent precip noise."""
    rng = np.random.default_rng(seed + hash(str(df["basin_id"].iloc[0])) % 10_000)
    precip = df["precip"].to_numpy(dtype=float)
    for lead in leads:
        noise = rng.normal(0.0, 0.15 * lead, size=len(df))
        df[f"precip_fc_l{lead}"] = np.clip(precip * (1.0 + noise), 0.0, None)
    return df


def list_timeseries_files(root: Path, sources: list[str] | None = None) -> list[Path]:
    csv_root = root / "timeseries" / "csv"
    if not csv_root.is_dir():
        raise FileNotFoundError(f"Missing {csv_root}")
    files: list[Path] = []
    for src_dir in sorted(csv_root.iterdir()):
        if not src_dir.is_dir():
            continue
        if sources and src_dir.name.lower() not in {s.lower() for s in sources}:
            continue
        files.extend(sorted(src_dir.glob("*.csv")))
    return files


def build_caravan_hydro(
    *,
    max_basins: int = 500,
    sources: list[str] | None = None,
    start: str | None = "2010-01-01",
    end: str | None = "2022-12-31",
    min_valid_days: int = 365 * 3,
    seed: int = 42,
) -> dict[str, Path]:
    ensure_dirs()
    root = caravan_root()
    interim = data_root() / "interim"
    attrs = load_caravan_attributes(root)
    # area column soft-rename
    area_c = _pick_col(list(attrs.columns), ["area", "area_km2", "gauge_area", "area_gages2"])
    if area_c and area_c != "area_km2":
        attrs = attrs.rename(columns={area_c: "area_km2"})

    files = list_timeseries_files(root, sources=sources)
    if not files:
        raise FileNotFoundError(f"No timeseries CSV under {root}/timeseries/csv")

    rng = np.random.default_rng(seed)
    if max_basins > 0 and len(files) > max_basins:
        idx = rng.choice(len(files), size=max_basins, replace=False)
        files = [files[i] for i in sorted(idx.tolist())]

    hydro_parts: list[pd.DataFrame] = []
    kept_ids: list[str] = []
    skipped = 0
    for i, fp in enumerate(files):
        basin_id = fp.stem
        try:
            raw = pd.read_csv(fp)
        except Exception:
            skipped += 1
            continue
        ts = _standardize_caravan_ts(raw, basin_id)
        if ts is None or len(ts) < min_valid_days:
            skipped += 1
            continue
        if start:
            ts = ts[ts["date"] >= pd.Timestamp(start)]
        if end:
            ts = ts[ts["date"] <= pd.Timestamp(end)]
        if len(ts) < min_valid_days // 2:
            skipped += 1
            continue
        # fill pet/temp gaps lightly
        ts["pet"] = ts["pet"].interpolate(limit=7).fillna(ts["pet"].median())
        ts["temp"] = ts["temp"].interpolate(limit=7).fillna(ts["temp"].median())
        ts = _add_forecast_precip_placeholder(ts, seed=seed)
        hydro_parts.append(ts)
        kept_ids.append(basin_id)
        if (i + 1) % 50 == 0:
            print(f"  caravan hydro: processed {i+1}/{len(files)} files, kept {len(kept_ids)}")

    if not hydro_parts:
        raise RuntimeError("No basins passed QC; relax --min-valid-days or check paths.")

    hydro = pd.concat(hydro_parts, ignore_index=True)
    attr_sub = attrs.loc[attrs["basin_id"].isin(kept_ids)].copy()
    keep_attr = ["basin_id", "area_km2", "attribute_source"]
    preferred = [
        "p_mean", "pet_mean_ERA5_LAND", "aridity_ERA5_LAND", "frac_snow",
        "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
        "seasonality_ERA5_LAND", "moisture_index_ERA5_LAND",
        "gauge_lat", "gauge_lon",
        "ele_mt_sav", "ele_mt_smx", "slp_dg_sav",
        "for_pc_sse", "crp_pc_sse", "pst_pc_sse", "urb_pc_sse",
        "cly_pc_sav", "snd_pc_sav", "slt_pc_sav",
        "pet_mm_syr", "aet_mm_syr", "run_mm_syr",
    ]
    seen = set(keep_attr)
    extra = []
    for c in list(attr_sub.columns) + preferred:
        if c in attr_sub.columns and c not in seen:
            seen.add(c)
            extra.append(c)
    keep_attr = [c for c in keep_attr + extra if c in attr_sub.columns]
    attr_sub = attr_sub[keep_attr].drop_duplicates("basin_id")

    hydro_path = interim / "hydro_daily.parquet"
    attr_path = interim / "basin_attributes.parquet"
    hydro.to_parquet(hydro_path, index=False)
    attr_sub.to_parquet(attr_path, index=False)

    meta = {
        "n_basins": len(kept_ids),
        "n_rows": int(len(hydro)),
        "skipped_files": skipped,
        "sources": sources,
        "start": start,
        "end": end,
        "forecast_precip": "placeholder_perturbed_analysis",
        "caravan_root": str(root),
        "hydro": str(hydro_path),
        "attrs": str(attr_path),
    }
    with open(interim / "caravan_build_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Caravan interim written:", meta)
    return {"hydro": hydro_path, "attrs": attr_path, "manifest": interim / "caravan_build_manifest.json"}


def build_illinois_wq(
    *,
    threshold: float = DEFAULT_NO3_THRESHOLD,
    agri_default: float = 0.55,
) -> dict[str, Path]:
    """Build cascade WQ table from Illinois continuous nutrient release."""
    ensure_dirs()
    interim = data_root() / "interim"
    env = os.environ.get("USGS_NUTRIENT_CSV")
    if env:
        csv_path = Path(env)
    else:
        csv_path = data_root() / "raw" / "usgs_illinois" / "illinois_continuous_network_wy2016-2022.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    raw = pd.read_csv(csv_path)
    # Illinois-specific columns
    colmap = {
        "USGS_site": "site_no",
        "datetime": "date",
        "q_ft3_per_s": "q_cfs",
        "no3_mg_per_l": "no3_mg_l",
    }
    df = raw.rename(columns={k: v for k, v in colmap.items() if k in raw.columns}).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["site_no"] = df["site_no"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["basin_id"] = "usgs_" + df["site_no"]
    df["q_cfs"] = pd.to_numeric(df["q_cfs"], errors="coerce")
    df["no3_mg_l"] = pd.to_numeric(df["no3_mg_l"], errors="coerce")
    # negative q occasional in raw -> treat as missing
    df.loc[df["q_cfs"] < 0, "q_cfs"] = np.nan
    df["discharge_cms"] = df["q_cfs"] * CFS_TO_CMS
    areas = df["site_no"].map(ILLINOIS_AREA_KM2)
    df["area_km2"] = areas
    df["discharge_mm"] = np.where(
        areas.notna() & (areas > 0),
        df["discharge_cms"] * 86.4 / areas,
        df["discharge_cms"],  # fallback relative proxy
    )
    df = df.dropna(subset=["date", "no3_mg_l", "discharge_mm"])
    df["exceed"] = (df["no3_mg_l"] >= threshold).astype(int)
    df["load_proxy"] = df["no3_mg_l"] * df["discharge_mm"]
    df["frac_agriculture"] = agri_default

    keep = [
        "date",
        "basin_id",
        "site_no",
        "no3_mg_l",
        "discharge_mm",
        "discharge_cms",
        "load_proxy",
        "exceed",
        "frac_agriculture",
        "area_km2",
    ]
    out = df[keep].sort_values(["basin_id", "date"]).reset_index(drop=True)

    wq_path = interim / "wq_daily.parquet"
    joined_path = interim / "joined_q_no3.parquet"
    out.to_parquet(wq_path, index=False)
    out.to_parquet(joined_path, index=False)

    meta = {
        "n_sites": int(out["basin_id"].nunique()),
        "n_rows": int(len(out)),
        "threshold": threshold,
        "source": str(csv_path),
        "note": "Illinois Q+NO3 self-joined; Caravan hydro is separate large-sample track",
    }
    with open(interim / "illinois_build_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Illinois interim written:", meta)
    return {"wq": wq_path, "joined": joined_path, "manifest": interim / "illinois_build_manifest.json"}


def main():
    parser = argparse.ArgumentParser(description="Build real-data interim tables")
    parser.add_argument("--caravan", action="store_true", help="Build hydro from Caravan")
    parser.add_argument("--illinois", action="store_true", help="Build WQ from Illinois CSV")
    parser.add_argument("--max-basins", type=int, default=500)
    parser.add_argument(
        "--sources",
        type=str,
        default="camels,hysets",
        help="Comma-separated Caravan source folders under timeseries/csv (empty=all)",
    )
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2022-12-31")
    parser.add_argument("--min-valid-days", type=int, default=365 * 3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no3-threshold", type=float, default=DEFAULT_NO3_THRESHOLD)
    args = parser.parse_args()

    if not args.caravan and not args.illinois:
        args.caravan = True
        args.illinois = True

    sources = [s.strip() for s in args.sources.split(",") if s.strip()] or None
    if args.caravan:
        build_caravan_hydro(
            max_basins=args.max_basins,
            sources=sources,
            start=args.start,
            end=args.end,
            min_valid_days=args.min_valid_days,
            seed=args.seed,
        )
    if args.illinois:
        build_illinois_wq(threshold=args.no3_threshold)


if __name__ == "__main__":
    main()
