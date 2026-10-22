# -*- coding: utf-8 -*-
"""Extract WeatherBench2 IFS-HRES 6-hourly precipitation forecasts at validation sites.

Used to build the *true-forecast forcing* stream for the independent rule
validation (``Q_hat_fc`` for the new IA/IN/IL/MO sites without placeholder
perturbation).  The HRES archive (2016-01-01 .. 2023-01-10, 00/12 UTC inits,
6-hourly leads to +240 h) is a Zarr store whose chunks are whole 0.25 deg maps
(~4.15 MB each); every (init, lead) block therefore costs a full map download
no matter how small the site subset is.

This script:
  1. maps each validation site to its nearest grid cell (regular 0.25 deg grid);
  2. estimates the volume and auto-adjusts ``--day-stride`` to respect ``--budget-gb``;
  3. pulls (init x lead) blocks in parallel and collapses them to
     per-site daily precipitation forecasts for lead days 1..N.

Output (under ``data/interim/``):
  wb2_precip_fc.parquet    site_no, issue_date, lead_day, precip_mm

Usage::

    python -m src.data.extract_wb2_precip --dry-run
    python -m src.data.extract_wb2_precip --start 2020-01-01 --end 2022-12-31
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data.paths import data_root

STORES = {
    # verified chunk layout: (1, 1, 721, 1440) float32 ~= 4.15 MB per block
    "0p25": "gs://weatherbench2/datasets/hres/2016-2022-12h-6h-0p25deg-chunk-1.zarr",
}
MB_PER_BLOCK = {  # full-map float32 chunk size, MB
    "0p25": 721 * 1440 * 4 / 1e6,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites-csv", default=None,
                    help="default: data/raw/usgs_nitrate_iv/site_inventory/candidate_sites.csv")
    ap.add_argument("--store-key", choices=list(STORES), default="0p25")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--init-hour", type=int, default=0, help="UTC hour of forecast issue (0 = 00 UTC only)")
    ap.add_argument("--lead-days", type=int, default=7)
    ap.add_argument("--day-stride", type=int, default=1, help="issue every N days")
    ap.add_argument("--budget-gb", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = data_root()
    interim = root / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    sites_csv = Path(args.sites_csv) if args.sites_csv else (
        root / "raw" / "usgs_nitrate_iv" / "site_inventory" / "candidate_sites.csv")
    sites = pd.read_csv(sites_csv, dtype={"site_no": str}).dropna(
        subset=["dec_lat_va", "dec_long_va"])
    log(f"sites: {len(sites)}")

    store_url = STORES[args.store_key]
    log(f"opening {store_url} ...")
    ds = xr.open_zarr(store_url, storage_options={"token": "anon"}, consolidated=None)
    tp_name = next(v for v in ds.data_vars if "precipitation_6hr" in v)
    tp = ds[tp_name]
    lat = ds["latitude"].values.astype("float64")
    lon = ds["longitude"].values.astype("float64")
    lead_h = np.asarray(ds["prediction_timedelta"].values, dtype="float64")
    leads_used = np.where((lead_h > 0) & (lead_h <= 24 * args.lead_days))[0]
    log(f"  variable={tp_name}; dims={dict(ds.sizes)}; leads used: {len(leads_used)} of {len(lead_h)}")

    # ---------- site -> grid indices ----------
    lon_ok = lon
    site_lon = sites["dec_long_va"].to_numpy(dtype="float64")
    if float(lon.min()) >= 0:  # 0-360 longitudes
        site_lon = site_lon % 360
    lat_ix = np.abs(lat[None, :] - sites["dec_lat_va"].to_numpy(dtype="float64")[:, None]).argmin(axis=1)
    lon_ix = np.abs(lon_ok[None, :] - site_lon[:, None]).argmin(axis=1)
    res_deg = float(np.abs(lat[1] - lat[0]))
    log(f"  grid resolution ~{res_deg:.2f} deg; "
        f"sample site->cell offset: {np.abs(lat[lat_ix] - sites['dec_lat_va']).max():.2f} deg")

    # ---------- init dates ----------
    t = pd.DatetimeIndex(ds["time"].values)
    mask = ((t >= pd.Timestamp(args.start)) & (t <= pd.Timestamp(args.end))
            & (t.hour == args.init_hour))
    inits = np.where(mask)[0]

    # budget guard: bump day-stride until estimate fits
    stride = max(1, args.day_stride)
    while True:
        inits_s = inits[::stride]
        est_gb = len(inits_s) * len(leads_used) * MB_PER_BLOCK[args.store_key] / 1e3
        if est_gb <= args.budget_gb or stride >= 31:
            break
        stride += 1
    log(f"inits: {len(inits_s)} (every {stride} d) x {len(leads_used)} leads "
        f"~ {est_gb:.1f} GB download (budget {args.budget_gb:.0f} GB)")
    if args.dry_run:
        log("dry run: nothing downloaded.")
        return

    # ---------- streaming extraction ----------
    tasks = [(ti, li) for ti in inits_s for li in leads_used]

    def pull_block(ti: int, li: int) -> np.ndarray:
        field = tp.isel(time=ti, prediction_timedelta=li).load()
        return np.asarray(field.values, dtype="float32")[lat_ix, lon_ix]

    log(f"downloading {len(tasks)} blocks with {args.workers} workers ...")
    vals: dict[tuple[int, int], np.ndarray] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(pull_block, ti, li): (ti, li) for ti, li in tasks}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                vals[key] = fut.result()
            except Exception as exc:  # noqa: BLE001
                log(f"  block {key} FAILED: {type(exc).__name__}: {exc}")
            done += 1
            if done % 200 == 0 or done == len(tasks):
                log(f"  {done}/{len(tasks)} blocks")

    # ---------- collapse to daily forecasts per site ----------
    site_ids = sites["site_no"].tolist()
    day_bins = ((lead_h[leads_used] - 1) // 24 + 1).astype(int)  # lead 6h -> day 1, .. lead 168h -> day 7
    rows = []
    for ti in inits_s:
        issue = t[ti]
        day_sum = {d: np.zeros(len(site_ids)) for d in range(1, args.lead_days + 1)}
        for li, d in zip(leads_used, day_bins):
            v = vals.get((ti, int(li)))
            if v is not None:
                day_sum[int(d)] += v
        for d in range(1, args.lead_days + 1):
            rows.append(pd.DataFrame({
                "site_no": site_ids,
                "issue_date": issue.normalize(),
                "lead_day": d,
                "precip_mm": day_sum[d],
            }))
    out = pd.concat(rows, ignore_index=True)
    out_path = interim / "wb2_precip_fc.parquet"
    out.to_parquet(out_path, index=False)
    log(f"saved {len(out)} rows ({out['issue_date'].nunique()} issues x {len(site_ids)} sites) -> {out_path}")


if __name__ == "__main__":
    main()
