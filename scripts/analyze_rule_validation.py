# -*- coding: utf-8 -*-
"""Partition analysis of the rule-validation audit table (duration / state / crop tercile)
plus NWM-vs-observed flow sanity diagnostics.

Reads outputs/rule_validation/rule_validation_events.csv (produced by
eval_rule_validation.py) and writes:

    partition_by_duration.csv    capture by duration bucket per (mode, variant, budget)
    partition_by_state.csv       capture by state
    partition_by_agri.csv        capture by cropland-fraction tercile
    overlap_at_weekly_budget.csv pairwise captured-event agreement (budget = 52)
    nwm_obs_corr.csv             per-site Spearman corr of NWM daily flow vs observed Q

The headline question: do adaptive variants beat fixed schedules on SHORT
(<=7 d) events and on early warning, where the rule is supposed to act?

Usage::

    python -m src.experiments.analyze_rule_validation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import data_root, repo_root

DUR_BINS = [(0, 3, "d1_3"), (4, 7, "d4_7"), (8, 30, "d8_30"), (31, 10000, "d31p")]


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    root = data_root()
    out_dir = repo_root() / "outputs" / "rule_validation"
    candidates = ["rule_validation_events.csv", "events_5mg.csv", "events_10mg.csv"]
    for name in candidates:
        p = out_dir / name
        if p.exists():
            ev = pd.read_csv(p, dtype={"site_no": str})
            log(f"reading {name}")
            break
    else:
        raise SystemExit(f"no event audit table found in {out_dir} (checked {candidates})")
    ev["onset"] = pd.to_datetime(ev["onset"])
    log(f"audit rows: {len(ev)} | modes: {sorted(ev['mode'].unique())}")

    # ---------- duration buckets ----------
    def bucket(d: int) -> str:
        for lo, hi, name in DUR_BINS:
            if lo <= d <= hi:
                return name
        return "d31p"

    ev["dur_bucket"] = ev["dur_days"].apply(bucket)

    part = (ev.groupby(["mode", "variant", "budget", "dur_bucket"])
            .agg(n=("captured", "size"), capture=("captured", "mean"),
                 pre_onset=("pre_onset", "mean"),
                 delay_med=("delay_days", "median"))
            .reset_index())
    part.to_csv(out_dir / "partition_by_duration.csv", index=False)

    # headline: weekly budget, short events
    weekly = part[(part["budget"] == 52) & (part["dur_bucket"].isin(["d1_3", "d4_7"]))]
    if not weekly.empty:
        head = (weekly.assign(short_n=weekly["n"])
                .pivot_table(index=["mode", "variant"], columns="dur_bucket",
                             values=["capture", "n"], aggfunc="first"))
        log("\n== weekly budget (52/yr): SHORT-event capture ==")
        log(head.round(3).to_string())

    # ---------- state ----------
    joined = pd.read_parquet(root / "interim" / "joined_q_no3_multistate.parquet",
                             columns=["site_no", "state"])
    smap = joined.drop_duplicates("site_no").set_index("site_no")["state"]
    ev["state"] = ev["site_no"].map(smap)
    ps = (ev.groupby(["mode", "variant", "budget", "state"])
          .agg(n=("captured", "size"), capture=("captured", "mean")).reset_index())
    ps.to_csv(out_dir / "partition_by_state.csv", index=False)

    # ---------- cropland tercile ----------
    frac_path = root / "interim" / "frac_agriculture.csv"
    if frac_path.exists():
        frac = pd.read_csv(frac_path, dtype={"site_no": str})
        if "frac_ag_r10" in frac.columns:
            f = frac.set_index("site_no")["frac_ag_r10"]
            ev["frac_ag"] = ev["site_no"].map(f)
            ev["ag_group"] = pd.qcut(ev["frac_ag"], 3, duplicates="drop").astype(str)
            pa = (ev.dropna(subset=["ag_group"])
                  .groupby(["mode", "variant", "budget", "ag_group"], observed=True)
                  .agg(n=("captured", "size"), capture=("captured", "mean")).reset_index())
            pa.to_csv(out_dir / "partition_by_agri.csv", index=False)

    # ---------- overlap between variants at weekly budget ----------
    wk = ev[ev["budget"] == 52].copy()
    wk["eid"] = wk["site_no"] + "|" + wk["onset"].astype(str)
    ov_rows = []
    for mode, g in wk.groupby("mode"):
        mats = g.pivot_table(index="eid", columns="variant", values="captured", aggfunc="max")
        vs = list(mats.columns)
        for i, a in enumerate(vs):
            for b in vs[i + 1:]:
                both = ((mats[a] == 1) & (mats[b] == 1)).sum()
                either = ((mats[a] == 1) | (mats[b] == 1)).sum()
                ov_rows.append({"mode": mode, "variant_a": a, "variant_b": b,
                                "n_events": len(mats),
                                "a_only": int(((mats[a] == 1) & (mats[b] == 0)).sum()),
                                "b_only": int(((mats[a] == 0) & (mats[b] == 1)).sum()),
                                "both": int(both),
                                "jaccard": round(both / either, 3) if either else np.nan})
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(out_dir / "overlap_at_weekly_budget.csv", index=False)
    log("\n== weekly-budget captured-event overlap (temporal) ==")
    if not ov.empty and "temporal" in set(ov["mode"]):
        log(ov[ov["mode"] == "temporal"].to_string(index=False))

    # ---------- hybrid unions: fixed schedule + adaptive trigger ----------
    hy_rows = []
    pairs = [("fixed_7d", 52, "nowcast", 52),
             ("fixed_14d", 26, "nowcast", 26),
             ("fixed_7d", 52, "season", 52)]
    for mode in ev["mode"].unique():
        g = ev[ev["mode"] == mode].copy()
        g["eid"] = g["site_no"] + "|" + g["onset"].astype(str)
        dur = g.drop_duplicates("eid").set_index("eid")["dur_days"]
        for va, ba, vb, bb in pairs:
            fa = g[(g["variant"] == va) & (g["budget"] == ba)].set_index("eid")["captured"]
            fb = g[(g["variant"] == vb) & (g["budget"] == bb)].set_index("eid")["captured"]
            if fa.empty or fb.empty:
                continue
            idx = fa.index.union(fb.index)
            fa = fa.reindex(idx, fill_value=0)
            fb = fb.reindex(idx, fill_value=0)
            d = dur.reindex(idx)
            union = (fa == 1) | (fb == 1)
            hy_rows.append({"mode": mode, "combo": f"{va}({ba}/yr)+{vb}({bb}/yr)",
                            "budget_total_per_yr": ba + bb, "n_events": len(idx),
                            "capture_a": round(float(fa.mean()), 3),
                            "capture_b": round(float(fb.mean()), 3),
                            "capture_union": round(float(union.mean()), 3),
                            "union_short_le3d": round(float(union[d <= 3].mean()), 3)
                            if (d <= 3).any() else np.nan})
    hy = pd.DataFrame(hy_rows)
    if not hy.empty:
        hy.to_csv(out_dir / "hybrid_union.csv", index=False)
        log("\n== hybrid unions (fixed + adaptive) ==")
        log(hy.to_string(index=False))

    # ---------- NWM vs observed flow correlation ----------
    nwm_path = root / "interim" / "nwm_streamflow_sites.parquet"
    if nwm_path.exists():
        nwm = pd.read_parquet(nwm_path)
        nwm["date"] = pd.to_datetime(nwm["date"])
        jo = pd.read_parquet(root / "interim" / "joined_q_no3_multistate.parquet",
                             columns=["date", "site_no", "discharge_cms"])
        jo["date"] = pd.to_datetime(jo["date"])
        m = nwm.merge(jo, on=["site_no", "date"], how="inner")
        corr = (m.groupby("site_no")
                .apply(lambda g: g["q_cms_mean"].corr(g["discharge_cms"], method="spearman"))
                .rename("spearman_q").reset_index())
        corr.to_csv(out_dir / "nwm_obs_corr.csv", index=False)
        log(f"\n== NWM vs observed daily Q (Spearman) ==")
        log(corr["spearman_q"].describe().round(3).to_string())
    else:
        log("\nNWM parquet not found locally (server-only) -> corr table skipped.")


if __name__ == "__main__":
    main()
