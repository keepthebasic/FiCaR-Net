# -*- coding: utf-8 -*-
"""Extract NWM retrospective (v3.0) hourly streamflow at validation sites -> daily series.

The NOAA NWM retrospective on AWS S3 is a Zarr store with chunk layout
``(time=672h, feature_id=30000)`` (~80.6 MB per chunk).  Reading a handful of
scattered reaches therefore costs whole chunks; this script:

  1. maps each USGS validation site to its nearest NWM reach (KNN over the
     zarr coordinate arrays; the ``gage_id`` attribute is mostly blank);
  2. estimates the download volume *before* downloading anything;
  3. enforces a ``--budget-gb`` guard (drops farthest sites until the estimate fits);
  4. streams (feature-band x time-chunk) blocks in parallel, collapses each
     block to daily mean/max and writes one tidy parquet.

Outputs (under ``data/interim/``):
  nwm_site_feature_map.csv        site -> feature_id, dist_km, lat/lon
  nwm_streamflow_sites.parquet    date, site_no, feature_id, q_cms_mean, q_cms_max

Usage::

    python -m src.data.extract_nwm_streamflow --dry-run          # mapping + budget only
    python -m src.data.extract_nwm_streamflow                    # default: rule-test sites, 2020-01..2023-01
    python -m src.data.extract_nwm_streamflow --filter all --start 2016-01-01 --budget-gb 200
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data.paths import data_root

STORE = "noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
FEATURE_CHUNK = 30_000  # from zarr metadata
TIME_CHUNK = 672        # hours


def log(msg: str) -> None:
    print(msg, flush=True)


def open_store() -> xr.Dataset:
    import s3fs
    # NOTE: do NOT pass client_kwargs={"region": ...}; s3fs forwards it to
    # aiobotocore which rejects boto-style kwargs (verified 2026-10).
    fs = s3fs.S3FileSystem(anon=True)
    return xr.open_zarr(s3fs.S3Map(STORE, s3=fs), consolidated=None)


def nearest_features(lat: np.ndarray, lon: np.ndarray,
                     site_lat: float, site_lon: float,
                     max_deg: float = 0.35) -> tuple[int, float]:
    """Nearest reach index and great-circle-ish distance (km) via widening boxes."""
    radius = 0.1
    while radius <= max_deg:
        m = (np.abs(lat - site_lat) < radius) & (np.abs(lon - site_lon) < radius)
        idx = np.where(m)[0]
        if idx.size:
            d2 = (lat[idx] - site_lat) ** 2 + ((lon[idx] - site_lon) * math.cos(math.radians(site_lat))) ** 2
            best = idx[int(np.argmin(d2))]
            d_km = math.sqrt(float(d2.min())) * 111.0
            return int(best), d_km
        radius *= 2
    # brute force (rare)
    d2 = (lat - site_lat) ** 2 + ((lon - site_lon) * math.cos(math.radians(site_lat))) ** 2
    best = int(np.argmin(d2))
    return best, math.sqrt(float(d2[best])) * 111.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites-csv", default=None,
                    help="default: data/raw/usgs_nitrate_iv/site_inventory/candidate_sites.csv")
    ap.add_argument("--filter", choices=["usable_rule", "usable_base", "all"], default="usable_rule")
    ap.add_argument("--max-sites", type=int, default=0, help="0 = no cap")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2023-01-01")
    ap.add_argument("--budget-gb", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="only map sites + print volume estimate")
    args = ap.parse_args()

    root = data_root()
    interim = root / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    sites_csv = Path(args.sites_csv) if args.sites_csv else (
        root / "raw" / "usgs_nitrate_iv" / "site_inventory" / "candidate_sites.csv")

    sites = pd.read_csv(sites_csv, dtype={"site_no": str})
    col_map = {"usable_rule": "usable_rule_test", "usable_base": "usable_base"}
    if args.filter != "all":
        col = col_map[args.filter]
        if col in sites.columns:
            sites = sites[sites[col].fillna(False).astype(bool)]
        else:
            log(f"WARNING: column '{col}' not found; using all sites")
    sites = sites.dropna(subset=["dec_lat_va", "dec_long_va"]).copy()
    if args.max_sites:
        sites = sites.head(args.max_sites)
    log(f"sites to map: {len(sites)} (filter={args.filter})")

    log("opening NWM chrtout zarr ...")
    ds = open_store()
    lat = ds["latitude"].values.astype("float64")
    lon = ds["longitude"].values.astype("float64")
    log(f"  store dims: time={ds.sizes['time']}, feature_id={ds.sizes['feature_id']}")

    # ---------- map sites -> feature index ----------
    rows = []
    for r in sites.itertuples():
        fi, dist_km = nearest_features(lat, lon, float(r.dec_lat_va), float(r.dec_long_va))
        rows.append({"site_no": r.site_no, "state": getattr(r, "state", ""),
                     "site_lat": r.dec_lat_va, "site_lon": r.dec_long_va,
                     "feature_index": fi, "feature_id": str(ds["feature_id"].values[fi]),
                     "dist_km": round(dist_km, 3)})
    fmap = pd.DataFrame(rows).sort_values("dist_km")
    log(f"mapped: median dist {fmap['dist_km'].median():.2f} km, "
        f"max {fmap['dist_km'].max():.2f} km, >10 km: {(fmap['dist_km'] > 10).sum()} sites")

    # ---------- time window -> chunk blocks ----------
    t = pd.DatetimeIndex(ds["time"].values)
    sel = np.where((t >= args.start) & (t < args.end))[0]
    if sel.size == 0:
        raise SystemExit("time window not found in store")
    tblocks = sorted({int(i // TIME_CHUNK) for i in sel})
    log(f"time window: {t[sel[0]]} .. {t[sel[-1]]} ({sel.size} hourly steps, {len(tblocks)} time chunks)")

    # ---------- budget guard: drop farthest sites until estimate fits ----------
    mb_per_block = TIME_CHUNK * FEATURE_CHUNK * 4 / 1e6
    while True:
        bands = {int(fi // FEATURE_CHUNK) for fi in fmap["feature_index"]}
        est_gb = len(bands) * len(tblocks) * mb_per_block / 1e3
        if est_gb <= args.budget_gb or len(fmap) <= 1:
            break
        dropped = fmap.iloc[-1]
        fmap = fmap.iloc[:-1]
        log(f"  budget {args.budget_gb:.0f} GB exceeded ({est_gb:.1f} GB) -> dropping {dropped.site_no} "
            f"(dist {dropped.dist_km:.1f} km)")
    log(f"final: {len(fmap)} sites, {len(bands)} feature bands x {len(tblocks)} time chunks "
        f"= {len(bands) * len(tblocks)} blocks ~ {est_gb:.1f} GB download")

    map_path = interim / "nwm_site_feature_map.csv"
    fmap_clean = fmap.drop(columns=["feature_index"])
    fmap_clean.to_csv(map_path, index=False)
    log(f"mapping saved -> {map_path}")
    if args.dry_run:
        log("dry run: nothing downloaded.")
        return

    # ---------- streaming block extraction ----------
    sv = ds["streamflow"]
    bands_needed = {int(fi // FEATURE_CHUNK): [] for fi in fmap["feature_index"]}
    for fi, fid in zip(fmap["feature_index"], fmap["feature_id"]):
        bands_needed[int(fi // FEATURE_CHUNK)].append(int(fi))

    def extract_block(band: int, tb: int) -> pd.DataFrame:
        f0 = band * FEATURE_CHUNK
        t0, t1 = tb * TIME_CHUNK, min((tb + 1) * TIME_CHUNK, ds.sizes["time"])
        block = sv.isel(time=slice(t0, t1), feature_id=slice(f0, f0 + FEATURE_CHUNK)).load().values
        times = t[t0:t1]
        out = []
        for fi in bands_needed[band]:
            col = block[:, fi - f0]
            s = pd.Series(col, index=times)
            daily = s.groupby(s.index.date)
            out.append(pd.DataFrame({
                "date": [d for d in daily.groups.keys()],
                "feature_index": fi,
                "q_cms_mean": daily.mean().values,
                "q_cms_max": daily.max().values,
            }))
        return pd.concat(out, ignore_index=True)

    tasks = [(b, tb) for b in bands_needed for tb in tblocks]
    log(f"downloading {len(tasks)} blocks with {args.workers} workers ...")
    pieces, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_block, b, tb): (b, tb) for b, tb in tasks}
        for fut in as_completed(futs):
            b, tb = futs[fut]
            try:
                pieces.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                log(f"  block band={b} tblock={tb} FAILED: {type(exc).__name__}: {exc}")
            done += 1
            if done % 25 == 0 or done == len(tasks):
                log(f"  {done}/{len(tasks)} blocks")

    if not pieces:
        raise SystemExit("no blocks extracted")
    all_df = pd.concat(pieces, ignore_index=True)
    fi2site = dict(zip(fmap["feature_index"], fmap["site_no"]))
    all_df["site_no"] = all_df["feature_index"].map(fi2site)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = (all_df.drop(columns=["feature_index"])
              .groupby(["site_no", "date"], as_index=False)
              .agg(q_cms_mean=("q_cms_mean", "mean"), q_cms_max=("q_cms_max", "max")))
    all_df = all_df.merge(fmap_clean[["site_no", "feature_id", "dist_km"]], on="site_no", how="left")
    out_path = interim / "nwm_streamflow_sites.parquet"
    all_df.sort_values(["site_no", "date"]).to_parquet(out_path, index=False)
    log(f"saved {len(all_df)} site-days ({all_df['site_no'].nunique()} sites) -> {out_path}")


if __name__ == "__main__":
    main()
