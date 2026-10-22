# -*- coding: utf-8 -*-
"""Harvest the independent continuous-nitrate validation network (USGS pCode 99133).

For 8 Corn-Belt states (IL/IA/IN/KS/MN/MO/NE/WI) this script downloads:

  1. site inventories for 99133 and 00060 (instantaneous-value data available)
  2. expanded site metadata (drainage area, HUC)
  3. series catalogs for 99133 (begin/end dates, counts)
  4. daily values (DV): 99133 and co-located 00060, stat code 00003 (daily mean)
  5. optional WQP station cross-check (pCode=99133) to catch non-NWIS providers

and builds:

  data/raw/usgs_nitrate_iv/site_inventory/candidate_sites.csv   (selection table)
  data/raw/usgs_nitrate_iv/dv_parsed/no3_dv.parquet
  data/raw/usgs_nitrate_iv/dv_parsed/q_dv.parquet
  data/interim/joined_q_no3_multistate.parquet                  (simplified join)

Resumable: existing DV files are reused on rerun.

Usage::

    python -m src.data.wqp_harvest                     # everything
    python -m src.data.wqp_harvest --no-dv             # inventories only
    python -m src.data.wqp_harvest --states 19,18      # subset of states
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.paths import data_root

STATES = {
    17: "IL", 18: "IN", 19: "IA", 20: "KS", 27: "MN", 29: "MO", 31: "NE", 55: "WI",
}
SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
WQP_STATION_URL = "https://www.waterqualitydata.us/data/Station/search"
EXCEED_THRESHOLD = 5.0  # mg/L NO3-N (paper main threshold)
CFS_TO_CMS = 0.028316846592
SQMI_TO_KM2 = 2.589988110336


# ----------------------------------------------------------------- helpers
def _get(url: str, params: dict, tries: int = 4, timeout: int = 180) -> str | None:
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.text
            print(f"    HTTP {r.status_code}; retry {k + 1}/{tries}", flush=True)
        except requests.RequestException as exc:
            print(f"    {type(exc).__name__}; retry {k + 1}/{tries}", flush=True)
        time.sleep(2.0 * (k + 1))
    return None


def read_rdb(text: str) -> pd.DataFrame:
    """Parse a NWIS RDB text block (tolerates BOM, comments, format row)."""
    if not text:
        return pd.DataFrame()
    lines = [ln.rstrip("\r") for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        return pd.DataFrame()
    rows = [ln.split("\t") for ln in lines]
    hdr_i = next((i for i, r in enumerate(rows) if r and r[0] == "agency_cd"), None)
    if hdr_i is None:
        return pd.DataFrame()
    hdr = rows[hdr_i]
    body = [r for r in rows[hdr_i + 1:] if r and r[0] == "USGS"]
    body = [r[: len(hdr)] + [""] * (len(hdr) - len(r)) if len(r) < len(hdr) else r[: len(hdr)] for r in body]
    return pd.DataFrame(body, columns=hdr)


def fetch_site_list(state: int, parm: str, out: Path) -> pd.DataFrame:
    text = _get(SITE_URL, {
        "format": "rdb", "parameterCd": parm, "stateCd": state,
        "siteStatus": "all", "hasDataTypeCd": "iv",
    })
    if text is None:
        return pd.DataFrame()
    out.write_text(text, encoding="utf-8")
    return read_rdb(text)


def fetch_site_expanded(state: int, out: Path) -> pd.DataFrame:
    text = _get(SITE_URL, {
        "format": "rdb", "parameterCd": "99133", "stateCd": state,
        "siteStatus": "all", "hasDataTypeCd": "iv", "siteOutput": "expanded",
    })
    if text is None:
        return pd.DataFrame()
    out.write_text(text, encoding="utf-8")
    return read_rdb(text)


def fetch_catalog(site_chunk: list[str], out: Path) -> pd.DataFrame:
    text = _get(SITE_URL, {
        "format": "rdb", "sites": ",".join(site_chunk),
        "seriesCatalogOutput": "true", "parameterCd": "99133",
    })
    if text is None:
        return pd.DataFrame()
    out.write_text(text, encoding="utf-8")
    return read_rdb(text)


def fetch_dv(site: str, parm: str, start: str, end: str, out: Path) -> pd.DataFrame:
    if out.exists() and out.stat().st_size > 200:
        return read_rdb(out.read_text(encoding="utf-8", errors="replace"))
    text = _get(DV_URL, {
        "format": "rdb", "sites": site, "parameterCd": parm,
        "statCd": "00003", "startDT": start, "endDT": end,
    })
    if text is None:
        return pd.DataFrame()
    out.write_text(text, encoding="utf-8")
    time.sleep(0.15)
    return read_rdb(text)


def parse_dv(df: pd.DataFrame, site: str, parm: str, state: str) -> pd.DataFrame:
    """Parse DV output. Supports both layouts:
    - generic:  agency_cd site_no date value qualifier
    - statCd:   agency_cd site_no datetime <tsid>_<param>_00003 <tsid>_<param>_00003_cd
    """
    if df.empty:
        return pd.DataFrame()
    cols = list(df.columns)
    date_col = "date" if "date" in cols else ("datetime" if "datetime" in cols else None)
    if date_col is None:
        return pd.DataFrame()
    val_candidates = [c for c in cols
                      if c not in ("agency_cd", "site_no", date_col, "qualifier")
                      and not c.endswith("_cd")]
    if val_candidates:
        preferred = [c for c in val_candidates if f"_{parm}_00003" in c] or val_candidates
        val_col = preferred[0]
    elif "value" in cols:
        val_col = "value"
    else:
        return pd.DataFrame()
    if "qualifier" in cols:
        qual_col = "qualifier"
    else:
        qual_col = next((c for c in cols if c.endswith("_cd") and c != "agency_cd"), None)
    out = pd.DataFrame({
        "site_no": site,
        "state": state,
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "value": pd.to_numeric(df[val_col], errors="coerce"),
        "qualifier": df[qual_col].astype(str) if qual_col else "",
    })
    out = out.dropna(subset=["date", "value"]).drop_duplicates(subset=["date"]).sort_values("date")
    out["param"] = parm
    return out.reset_index(drop=True)


def count_onsets(s: pd.Series, threshold: float = EXCEED_THRESHOLD) -> int:
    """Count exceedance onsets; consecutive exceed days (gap<=3d) form one event."""
    if s.empty:
        return 0
    exc = (s >= threshold).to_numpy()
    dates = s.index.to_numpy()
    n, last_exc_day = 0, None
    for i, e in enumerate(exc):
        if not e:
            continue
        if last_exc_day is not None and (dates[i] - last_exc_day) / np.timedelta64(1, "D") <= 3:
            last_exc_day = dates[i]
            continue
        n += 1
        last_exc_day = dates[i]
    return n


def window_stats(series: pd.Series, start: str, end: str) -> dict:
    if series.empty:
        return {"n": 0, "first": None, "last": None, "onsets": 0, "exceed_days": 0}
    w = series[(series.index >= start) & (series.index <= end)]
    return {
        "n": int(w.size),
        "first": str(w.index.min().date()) if w.size else None,
        "last": str(w.index.max().date()) if w.size else None,
        "onsets": count_onsets(w),
        "exceed_days": int((w >= EXCEED_THRESHOLD).sum()),
    }


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", default=",".join(map(str, STATES)))
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--no-dv", action="store_true")
    ap.add_argument("--wqp", action="store_true", help="also fetch WQP station cross-check")
    args = ap.parse_args()

    states = [int(s) for s in args.states.split(",") if s.strip()]
    root = data_root() / "raw" / "usgs_nitrate_iv"
    inv_dir = root / "site_inventory"
    dv_raw = root / "dv_raw"
    dv_par = root / "dv_parsed"
    for d in (inv_dir, dv_raw, dv_par):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. inventories + metadata -------------------------------------
    print("== inventories ==", flush=True)
    sites_n, sites_q, expanded = {}, {}, {}
    for st in states:
        print(f"  state {st} ({STATES[st]}) ...", flush=True)
        sites_n[st] = fetch_site_list(st, "99133", inv_dir / f"sites_99133_{st}.rdb")
        sites_q[st] = fetch_site_list(st, "00060", inv_dir / f"sites_00060_{st}.rdb")
        expanded[st] = fetch_site_expanded(st, inv_dir / f"sites_99133_{st}_expanded.rdb")
        print(f"    99133: {len(sites_n[st])} sites; 00060: {len(sites_q[st])} sites", flush=True)

    inv = pd.DataFrame()
    for st in states:
        e = expanded.get(st, pd.DataFrame())
        if e.empty:
            continue
        e = e.assign(state=STATES[st], state_fips=st)
        inv = pd.concat([inv, e], ignore_index=True)
    inv = inv.drop_duplicates(subset=["site_no"], keep="first")
    inv["has_q_colocated"] = inv.apply(
        lambda r: r["site_no"] in set(sites_q.get(int(r["state_fips"]), pd.DataFrame()).get("site_no", [])),
        axis=1,
    )
    inv["drain_area_km2"] = pd.to_numeric(inv.get("drain_area_va"), errors="coerce") * SQMI_TO_KM2
    keep = ["state", "site_no", "station_nm", "site_tp_cd", "dec_lat_va", "dec_long_va",
            "drain_area_km2", "huc_cd", "has_q_colocated"]
    for c in keep:
        if c not in inv.columns:
            inv[c] = np.nan
    inv = inv[keep].rename(columns={"site_tp_cd": "site_type"})
    inv.to_csv(inv_dir / "sites_all_metadata.csv", index=False)
    print(f"  unique 99133 sites across states: {len(inv)}", flush=True)

    # ---- 2. series catalog ----------------------------------------------
    print("== series catalog ==", flush=True)
    cat = pd.DataFrame()
    site_ids = inv["site_no"].tolist()
    for i in range(0, len(site_ids), 60):
        chunk = site_ids[i: i + 60]
        c = fetch_catalog(chunk, inv_dir / f"catalog_99133_{i // 60}.rdb")
        if not c.empty:
            cat = pd.concat([cat, c], ignore_index=True)
        print(f"  catalog chunk {i // 60 + 1}: +{len(c)} series rows", flush=True)
    if not cat.empty and "parm_cd" in cat.columns:
        cat = cat[cat["parm_cd"] == "99133"].copy()
        cat["begin_date"] = pd.to_datetime(cat["begin_date"], errors="coerce")
        cat["end_date"] = pd.to_datetime(cat["end_date"], errors="coerce")
        cat["count_nu"] = pd.to_numeric(cat["count_nu"], errors="coerce")
        cat = cat.groupby("site_no").agg(
            cat_begin=("begin_date", "min"), cat_end=("end_date", "max"),
            cat_count=("count_nu", "max"), cat_n_series=("ts_id", "count"),
        ).reset_index()
        cat.to_csv(inv_dir / "catalog_99133_mean.csv", index=False)
        inv = inv.merge(cat, on="site_no", how="left")

    # ---- 3. daily values -------------------------------------------------
    no3_rows, q_rows = [], []
    if not args.no_dv:
        surface = inv[inv["site_type"] == "ST"] if "site_type" in inv.columns else inv
        print(f"== daily values: {len(surface)} surface-water sites ==", flush=True)
        for j, row in enumerate(surface.itertuples(), 1):
            site, state = row.site_no, row.state
            d = fetch_dv(site, "99133", args.start, args.end, dv_raw / f"{site}_99133_dv.rdb")
            p = parse_dv(d, site, "99133", state)
            if not p.empty:
                no3_rows.append(p)
            if bool(row.has_q_colocated):
                dq = fetch_dv(site, "00060", args.start, args.end, dv_raw / f"{site}_00060_dv.rdb")
                pq = parse_dv(dq, site, "00060", state)
                if not pq.empty:
                    q_rows.append(pq)
            if j % 10 == 0 or j == len(surface):
                print(f"  {j}/{len(surface)} sites done", flush=True)

    no3 = pd.concat(no3_rows, ignore_index=True) if no3_rows else pd.DataFrame()
    q = pd.concat(q_rows, ignore_index=True) if q_rows else pd.DataFrame()
    if not no3.empty:
        no3.to_parquet(dv_par / "no3_dv.parquet", index=False)
    if not q.empty:
        q.to_parquet(dv_par / "q_dv.parquet", index=False)
    print(f"  nitrate rows: {len(no3)}; discharge rows: {len(q)}", flush=True)

    # ---- 4. coverage / onset stats + selection table ---------------------
    print("== coverage stats ==", flush=True)
    stats = []
    for row in inv.itertuples():
        s_no3 = no3[no3["site_no"] == row.site_no].set_index("date")["value"] if not no3.empty else pd.Series(dtype=float)
        s_q = q[q["site_no"] == row.site_no].set_index("date")["value"] if not q.empty else pd.Series(dtype=float)
        base = window_stats(s_no3, "2016-01-01", "2022-12-31")
        base_q = window_stats(s_q, "2016-01-01", "2022-12-31")
        test = window_stats(s_no3, "2020-01-01", "2026-12-31")
        stats.append({
            "site_no": row.site_no,
            "n_no3_2016_2022": base["n"], "n_no3_2020_2026": test["n"],
            "no3_first": base["first"], "no3_last": base["last"],
            "n_q_2016_2022": base_q["n"],
            "exceed_days_2016_2022": base["exceed_days"],
            "onsets_2016_2019": window_stats(s_no3, "2016-01-01", "2019-12-31")["onsets"],
            "onsets_2020_2026": test["onsets"],
        })
    stats = pd.DataFrame(stats)
    inv = inv.merge(stats, on="site_no", how="left")
    inv["usable_base"] = (inv["n_no3_2016_2022"].fillna(0) >= 800) & (inv["n_q_2016_2022"].fillna(0) >= 800)
    inv["usable_rule_test"] = inv["onsets_2020_2026"].fillna(0) >= 5
    inv.sort_values(["state", "site_no"], inplace=True)
    inv.to_csv(inv_dir / "candidate_sites.csv", index=False)
    print(f"  usable_base sites: {int(inv['usable_base'].sum())}; "
          f"usable_rule_test: {int(inv['usable_rule_test'].sum())}", flush=True)
    print(inv[inv["usable_base"]][["state", "site_no", "station_nm", "n_no3_2016_2022",
                                   "n_q_2016_2022", "onsets_2020_2026"]].to_string(index=False), flush=True)

    # ---- 5. simplified joined interim table ------------------------------
    if not no3.empty and not q.empty:
        j = no3[no3["param"] == "99133"].merge(
            q[q["param"] == "00060"], on=["site_no", "state", "date"], suffixes=("", "_q"))
        area = inv.set_index("site_no")["drain_area_km2"]
        j["area_km2"] = j["site_no"].map(area)
        j["no3_mg_l"] = j["value"]
        j["discharge_cms"] = j["value_q"] * CFS_TO_CMS
        j = j[j["area_km2"].notna() & (j["area_km2"] > 0)]
        j["discharge_mm"] = j["discharge_cms"] * 86.4 / j["area_km2"]
        j["exceed"] = (j["no3_mg_l"] >= EXCEED_THRESHOLD).astype(int)
        j["basin_id"] = j["site_no"]
        j["frac_agriculture"] = np.nan  # filled by NLCD / Caravan join (next step)
        out = j[["date", "basin_id", "site_no", "state", "no3_mg_l", "discharge_cms",
                 "discharge_mm", "exceed", "frac_agriculture", "area_km2"]]
        interim = data_root() / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        out.to_parquet(interim / "joined_q_no3_multistate.parquet", index=False)
        print(f"  interim joined rows: {len(out)} "
              f"({out['site_no'].nunique()} sites) -> data/interim/joined_q_no3_multistate.parquet", flush=True)

    # ---- 6. optional WQP cross-check -------------------------------------
    if args.wqp:
        print("== WQP station cross-check ==", flush=True)
        wqp_dir = root / "wqp"
        wqp_dir.mkdir(exist_ok=True)
        params = [("pCode", "99133"), ("mimeType", "csv"), ("zip", "yes")]
        params += [("statecode", f"US:{st}") for st in states]
        try:
            r = requests.get(WQP_STATION_URL, params=params, timeout=300)
            if r.status_code == 200:
                zp = wqp_dir / "wqp_stations_99133.zip"
                zp.write_bytes(r.content)
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    z.extractall(wqp_dir)
                print(f"  WQP zip saved ({len(r.content) / 1e6:.1f} MB)", flush=True)
            else:
                print(f"  WQP HTTP {r.status_code}", flush=True)
        except requests.RequestException as exc:
            print(f"  WQP failed: {exc}", flush=True)

    manifest = {
        "states": {str(k): STATES[k] for k in states},
        "window": [args.start, args.end],
        "exceed_threshold_mg_l": EXCEED_THRESHOLD,
        "n_sites_total": int(len(inv)),
        "n_usable_base": int(inv["usable_base"].sum()) if "usable_base" in inv else None,
        "n_usable_rule_test": int(inv["usable_rule_test"].sum()) if "usable_rule_test" in inv else None,
        "files": {
            "candidate_sites": str(inv_dir / "candidate_sites.csv"),
            "no3_dv_parquet": str(dv_par / "no3_dv.parquet"),
            "q_dv_parquet": str(dv_par / "q_dv.parquet"),
        },
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (root / "harvest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
