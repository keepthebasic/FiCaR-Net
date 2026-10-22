# -*- coding: utf-8 -*-
"""Compute cropland fraction around validation sites from the USDA CDL 2019 raster.

For each site, the fraction of agricultural pixels is computed inside circular
buffers (default radii 2/5/10 km) centred on the gauge, using windowed reads of
``2019_30m_cdls.tif`` (30 m, CONUS Albers).  Agricultural classes follow the
CDL code book: ``{1..61} U {66..80} U {204..254}`` (all crop groups; code 0 /
background excluded).

Outputs (under ``data/interim/``):
  frac_agriculture.csv        site_no, frac_ag_r2/r5/r10, valid_px_r10
  (with --merge) joined_q_no3_multistate.parquet updated in place:
  ``frac_agriculture`` <- ``frac_ag_r10`` (or --merge-col)

Requires rasterio (``pip install rasterio``).

Usage::

    python -m src.data.compute_cropland_fraction                 # compute table only
    python -m src.data.compute_cropland_fraction --merge         # + fill joined parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import data_root

CROP_CODES = set(range(1, 62)) | set(range(66, 81)) | set(range(204, 255))


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cdl", default=None, help="default: data/raw/cdl/2019_30m_cdls.tif")
    ap.add_argument("--sites-csv", default=None,
                    help="default: data/raw/usgs_nitrate_iv/site_inventory/candidate_sites.csv")
    ap.add_argument("--radii-km", default="2,5,10")
    ap.add_argument("--merge", action="store_true", help="fill frac_agriculture in joined parquet")
    ap.add_argument("--merge-col", default="frac_ag_r10")
    args = ap.parse_args()

    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.warp import transform as rio_transform
        from rasterio.windows import Window
    except ImportError:
        raise SystemExit("rasterio is required: pip install rasterio")

    root = data_root()
    cdl = Path(args.cdl) if args.cdl else root / "raw" / "cdl" / "2019_30m_cdls.tif"
    sites_csv = Path(args.sites_csv) if args.sites_csv else (
        root / "raw" / "usgs_nitrate_iv" / "site_inventory" / "candidate_sites.csv")
    radii_m = [int(float(r.strip()) * 1000) for r in args.radii_km.split(",") if r.strip()]
    crop = np.isin(np.arange(256), list(CROP_CODES))

    sites = pd.read_csv(sites_csv, dtype={"site_no": str}).dropna(
        subset=["dec_lat_va", "dec_long_va"])
    log(f"sites: {len(sites)}; radii(m): {radii_m}; cdl: {cdl}")

    rows = []
    with rasterio.open(cdl) as src:
        log(f"  raster: {src.width}x{src.height}, res={src.res}, crs={src.crs}")
        px_m = abs(src.res[0])
        xs, ys = rio_transform(CRS.from_epsg(4326), src.crs,
                               sites["dec_long_va"].tolist(), sites["dec_lat_va"].tolist())
        rmax = max(radii_m)
        half_all = int(np.ceil(rmax / px_m)) + 2
        for i, r in enumerate(sites.itertuples(index=False)):
            x, y = xs[i], ys[i]
            row, col = src.index(x, y)
            win = Window(col - half_all, row - half_all, 2 * half_all + 1, 2 * half_all + 1)
            data = src.read(1, window=win, boundless=True, fill_value=0)
            wt = src.window_transform(win)
            # window_transform is georeferenced to *local* pixel coordinates (0..width-1),
            # not global raster indices.
            cc = np.arange(win.width) + 0.5
            rr = np.arange(win.height) + 0.5
            gx = wt.c + wt.a * cc + wt.b * rr[:, None]
            gy = wt.f + wt.d * cc + wt.e * rr[:, None]
            d2 = (gx - x) ** 2 + (gy - y) ** 2
            rec = {"site_no": r.site_no}
            valid_px = None
            for rm in radii_m:
                m = (d2 <= rm ** 2) & (data > 0)
                n = int(m.sum())
                frac = float(crop[data[m]].mean()) if n else np.nan
                rec[f"frac_ag_r{rm // 1000}"] = round(frac, 4) if frac == frac else np.nan
                if rm == rmax:
                    valid_px = n
            rec["valid_px_r%d" % (rmax // 1000)] = valid_px
            rows.append(rec)
            if (i + 1) % 20 == 0 or i + 1 == len(sites):
                log(f"  {i + 1}/{len(sites)} sites")

    out = pd.DataFrame(rows)
    out_path = root / "interim" / "frac_agriculture.csv"
    out.to_csv(out_path, index=False)
    log(f"saved -> {out_path}")
    log(out.describe().round(3).to_string())

    if args.merge:
        jp = root / "interim" / "joined_q_no3_multistate.parquet"
        j = pd.read_parquet(jp)
        fill = out.set_index("site_no")[args.merge_col]
        n_before = j["frac_agriculture"].isna().sum()
        j["frac_agriculture"] = j["frac_agriculture"].fillna(j["site_no"].map(fill))
        j.to_parquet(jp, index=False)
        n_after = j["frac_agriculture"].isna().sum()
        log(f"merged into {jp}: NaN {n_before} -> {n_after}")
        missing = j.loc[j["frac_agriculture"].isna(), "site_no"].unique()
        if len(missing):
            log(f"  sites still missing frac_agriculture: {list(missing)[:10]}")


if __name__ == "__main__":
    main()
