# -*- coding: utf-8 -*-
"""Independent validation of the frozen forecast-triggered nitrate sampling rule (v2).

Compares, on the multi-state USGS 99133 continuous-nitrate network
(IA/IN/IL/MO/KS/WI), budget-matched adaptive sampling rules against fixed
schedules, using only information available at decision time:

  variants
    nowcast   trigger if today's observed Q exceeds a frozen threshold
    nwm_fc    trigger if max NWM flow over the next 7 days exceeds the threshold
              (NWM retrospective used as the operational flow stream,
              pseudo-operational forecasts, lead 1-7 d)
    logit     trigger if a logistic model P(nitrate >= thr within 7 d) exceeds
              a frozen probability threshold
    season    trigger by climatological day-of-year exceedance probability
              (train window only; tests seasonal scheduling alone)
  baselines
    fixed_1d / fixed_7d / fixed_14d / fixed_30d  (budget = 365/52/26/12 per year)

  modes
    temporal  train window -> test window, pooled calibration
    logo      leave-one-gauge-out: thresholds/logit fit on other sites
    transfer  fit on IL sites, applied directly to IA/IN sites

Threshold calibration (v2): for every (mode, variant, budget, fold) the trigger
threshold is chosen on the TRAIN window only, maximising train-window event
capture subject to trigger frequency <= 1.3 x budget. No test-window
information enters calibration. All other parameters stay frozen via
``configs/rule_frozen.yaml``.

Outputs
    outputs/rule_validation/rule_validation_results.csv   pooled metrics
    outputs/rule_validation/rule_validation_events.csv    per-event audit table
    outputs/rule_validation/summary.json

Usage::

    python -m src.experiments.eval_rule_validation
    python -m src.experiments.eval_rule_validation --smoke      # 6 sites, 200 boot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.paths import data_root, repo_root

FIXED_GRID_DAYS = {365: 1, 52: 7, 26: 14, 12: 30}
THETA_GRID_N = 48
BUDGET_SLACK = 1.3


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ loading
def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined = pd.read_parquet(root / "interim" / "joined_q_no3_multistate.parquet",
                             columns=["date", "site_no", "state", "no3_mg_l", "discharge_cms"])
    no3 = pd.read_parquet(root / "raw" / "usgs_nitrate_iv" / "dv_parsed" / "no3_dv.parquet",
                          columns=["site_no", "date", "value"])
    nwm_path = root / "interim" / "nwm_streamflow_sites.parquet"
    nwm = pd.read_parquet(nwm_path) if nwm_path.exists() else pd.DataFrame()
    joined["date"] = pd.to_datetime(joined["date"])
    no3["date"] = pd.to_datetime(no3["date"])
    if not nwm.empty:
        nwm["date"] = pd.to_datetime(nwm["date"])
    return joined, no3, nwm


def build_events(no3: pd.DataFrame, sites: list[str], threshold: float,
                 gap_days: int) -> dict[str, pd.DataFrame]:
    """Per-site nitrate exceedance events: runs of >=threshold days joined when gaps <= gap_days."""
    events: dict[str, pd.DataFrame] = {}
    for site, g in no3[no3["site_no"].isin(sites)].groupby("site_no"):
        exc = g[g["value"] >= threshold].sort_values("date")
        if exc.empty:
            continue
        dates = exc["date"].tolist()
        runs = []
        start = prev = dates[0]
        for d in dates[1:]:
            if (d - prev).days <= gap_days:
                prev = d
            else:
                runs.append((start, prev))
                start = prev = d
        runs.append((start, prev))
        events[site] = pd.DataFrame(runs, columns=["onset", "end"])
    return events


def q50_by_site(joined: pd.DataFrame, train: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Series:
    tr = joined[(joined["date"] >= train[0]) & (joined["date"] <= train[1])]
    q50 = tr.groupby("site_no")["discharge_cms"].median()
    all_med = joined.groupby("site_no")["discharge_cms"].median()
    return q50.combine_first(all_med).replace(0, np.nan)


# ------------------------------------------------------------------ signals
def build_signals(sites: list[str], joined: pd.DataFrame, nwm: pd.DataFrame,
                  q50: pd.Series, horizon: int, no3: pd.DataFrame,
                  threshold: float,
                  train: tuple[pd.Timestamp, pd.Timestamp]
                  ) -> dict[str, dict[str, pd.Series]]:
    """variant -> site -> Series[date] of decision scores."""
    out: dict[str, dict[str, pd.Series]] = {"nowcast": {}, "nwm_fc": {}, "season": {}}
    for site in sites:
        norm = q50.get(site, np.nan)
        if not np.isfinite(norm) or norm <= 0:
            continue
        obs = joined[joined["site_no"] == site].set_index("date")["discharge_cms"].dropna()
        if not obs.empty:
            out["nowcast"][site] = obs / norm
        if not nwm.empty:
            q = nwm[nwm["site_no"] == site].set_index("date")["q_cms_mean"].dropna()
            if not q.empty:
                fwd = q.shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
                out["nwm_fc"][site] = (fwd / norm).dropna()
        n = no3[no3["site_no"] == site].set_index("date")["value"].dropna()
        if not n.empty:
            exc = (n >= threshold).astype(float)
            clin = exc.groupby(exc.index.dayofyear).mean()
            clin = clin.rolling(29, center=True, min_periods=5).mean()
            clin = clin.reindex(range(1, 367), method="nearest")
            all_days = pd.date_range(n.index.min(), n.index.max(), freq="D")
            score = clin.reindex(all_days.dayofyear)
            score.index = all_days
            score = score.fillna(score.mean())
            out["season"][site] = score
    return out


def build_logit_labels(no3: pd.DataFrame, sites: list[str], threshold: float,
                       horizon: int) -> dict[str, pd.Series]:
    """site -> Series[date] = any nitrate >= threshold within (t, t+horizon]."""
    labels = {}
    for site in sites:
        s = (no3[no3["site_no"] == site].set_index("date")["value"].dropna() >= threshold)
        if s.empty:
            continue
        fut = s.rolling(f"{horizon}D").max().shift(-horizon, freq="D")
        labels[site] = fut.astype(float)
    return labels


# ------------------------------------------------------------------ metrics
def capture_metrics(events: pd.DataFrame, trigger_days: set[pd.Timestamp],
                    window: tuple[pd.Timestamp, pd.Timestamp], horizon: int) -> dict:
    ev = events[(events["onset"] >= window[0]) & (events["onset"] <= window[1])]
    if ev.empty:
        return {"n_events": 0}
    captured, delays, pre = [], [], []
    for r in ev.itertuples():
        run = [d for d in trigger_days if r.onset <= d <= r.end]
        early = [d for d in trigger_days
                 if r.onset - pd.Timedelta(days=horizon) <= d < r.onset]
        captured.append(len(run) > 0)
        if run:
            delays.append(min((d - r.onset).days for d in run))
        pre.append(len(early) > 0)
    return {"n_events": int(len(ev)), "captured": np.array(captured, dtype=float),
            "delay": float(np.median(delays)) if delays else np.nan,
            "pre_onset": np.array(pre, dtype=float)}


def burst_rate(trigger_days: set[pd.Timestamp], exc_days: set[pd.Timestamp],
               window: tuple[pd.Timestamp, pd.Timestamp], horizon: int) -> tuple[float, float]:
    days = pd.date_range(window[0], window[1], freq="D")
    yrs = max(len(days) / 365.25, 1e-9)
    trig = [d for d in trigger_days if window[0] <= d <= window[1]]
    false = [d for d in trig
             if not any((d + pd.Timedelta(days=k)) in exc_days for k in range(horizon + 1))]
    return len(trig) / yrs, len(false) / yrs


def bootstrap_ci(flags: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if flags.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, flags.size, size=(n_boot, flags.size))
    means = flags[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def calibrate_theta(train_scores: list[pd.Series],
                    train_events: dict[str, pd.DataFrame],
                    train: tuple[pd.Timestamp, pd.Timestamp],
                    budget: int, horizon: int) -> float:
    """Choose the trigger threshold on the TRAIN window: maximise event capture
    subject to trigger frequency <= BUDGET_SLACK x budget. No test data used."""
    if not train_scores:
        return np.inf
    pooled = pd.concat(train_scores)
    years = max(len(pd.date_range(train[0], train[1], freq="D")) / 365.25, 1e-9)
    cands = np.unique(np.quantile(pooled.to_numpy(), np.linspace(0.50, 0.995, THETA_GRID_N)))
    best_theta, best_cap, best_freq = float(cands[0]), -1.0, np.inf
    for theta in cands:
        trig_sets = [set(s[s >= theta].index) for s in train_scores]
        freq = sum(len(t) for t in trig_sets) / len(train_scores) / years
        if freq > BUDGET_SLACK * budget:
            continue
        total_cap, total_ev = 0.0, 0
        for s, trig in zip((getattr(sc, "attrs", {}).get("site") for sc in train_scores), trig_sets):
            ev = train_events.get(s) if s else None
            if ev is None:
                continue
            m = capture_metrics(ev, trig, train, horizon)
            if m.get("n_events", 0):
                total_cap += float(m["captured"].sum())
                total_ev += m["n_events"]
        cap = total_cap / total_ev if total_ev else 0.0
        if cap > best_cap + 1e-9 or (abs(cap - best_cap) <= 1e-9 and freq < best_freq):
            best_theta, best_cap, best_freq = float(theta), cap, freq
    return best_theta


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override primary threshold (sensitivity run, e.g. 10.0)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    root = data_root()
    cfg_path = Path(args.config) if args.config else repo_root() / "configs" / "rule_frozen.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    thr = float(cfg["thresholds_mg_l"]["primary"])
    if args.threshold is not None:
        thr = float(args.threshold)
    gap = int(cfg["event_definition"]["onset_gap_days"])
    H = int(cfg["horizon_days"])
    budgets = [int(b) for b in cfg["budgets_per_year"]]
    split = cfg["split"]
    train = (pd.Timestamp(split["train_start"]), pd.Timestamp(split["train_end"]))
    test = (pd.Timestamp(split["test_start"]), pd.Timestamp(split["test_end"]))
    n_boot = 200 if args.smoke else int(cfg["bootstrap"]["n"])
    seed = int(cfg["bootstrap"]["seed"])

    joined, no3, nwm = load_inputs(root)
    sites = sorted(set(joined["site_no"]))
    if args.smoke:
        sites = sites[:6]
    log(f"joined sites: {len(sites)}; NWM sites: {nwm['site_no'].nunique() if not nwm.empty else 0}")

    q50 = q50_by_site(joined, train)
    events = build_events(no3, sites, thr, gap)
    signals = build_signals(sites, joined, nwm, q50, H, no3, thr, train)
    labels = build_logit_labels(no3, sites, thr, H)
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        LogisticRegression = None
        log("WARNING: scikit-learn missing -> logit variant skipped")

    exc_days_all = {s: set(g["date"]) for s, g in
                    no3[(no3["site_no"].isin(sites)) & (no3["value"] >= thr)].groupby("site_no")}
    for v, d in signals.items():
        log(f"signal coverage [{v}]: {len(d)} sites")

    rows: list[dict] = []
    audit: list[dict] = []

    def add_audit(site, variant, budget, mode, trig):
        ev = events.get(site)
        if ev is None:
            return None
        m = capture_metrics(ev, trig, test, H)
        if not m.get("n_events", 0):
            return None
        evw = ev[(ev["onset"] >= test[0]) & (ev["onset"] <= test[1])]
        for (r, cap, pre) in zip(evw.itertuples(), m["captured"], m["pre_onset"]):
            run = [d for d in trig if r.onset <= d <= r.end]
            audit.append({"mode": mode, "variant": variant, "budget": budget,
                          "site_no": site, "onset": r.onset,
                          "dur_days": (r.end - r.onset).days + 1,
                          "captured": bool(cap), "pre_onset": bool(pre),
                          "delay_days": (min(run) - r.onset).days if run else np.nan})
        return m

    def feat_frame(site: str) -> pd.DataFrame | None:
        sn = signals["nwm_fc"].get(site)
        sw = signals["nowcast"].get(site)
        if sn is None:
            return None
        df = pd.DataFrame({"s_nwm_max7": sn, "s_now": sw.reindex(sn.index) if sw is not None else np.nan})
        df["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        lb = labels.get(site)
        df["y"] = lb.reindex(df.index) if lb is not None else np.nan
        return df

    feats = ["s_nwm_max7", "s_now", "doy_sin", "doy_cos"]

    def fold_train_scores(variant: str, fold_sites: list[str]):
        """Return (train_scores list with attrs['site'], scorer(site)->full Series, events used)."""
        ev_pool = {s: events[s] for s in fold_sites if s in events}
        if variant == "logit":
            frames = []
            for s in fold_sites:
                f = feat_frame(s)
                if f is not None:
                    frames.append(f)
            if not frames:
                return None, None, ev_pool
            trn = pd.concat(frames).dropna(subset=["s_nwm_max7", "y"])
            if len(trn) < 100 or trn["y"].nunique() < 2:
                return None, None, ev_pool
            model = LogisticRegression(max_iter=500)
            model.fit(trn[feats].fillna(-1), trn["y"])

            def scorer(site: str) -> pd.Series | None:
                f = feat_frame(site)
                if f is None:
                    return None
                return pd.Series(model.predict_proba(f[feats].fillna(-1))[:, 1], index=f.index)

            train_scores = []
            for s in fold_sites:
                sc = scorer(s)
                if sc is None:
                    continue
                sc = sc[(sc.index >= train[0]) & (sc.index <= train[1])]
                sc.attrs["site"] = s
                train_scores.append(sc)
            return train_scores, scorer, ev_pool

        def scorer(site: str) -> pd.Series | None:
            return signals[variant].get(site)

        train_scores = []
        for s in fold_sites:
            ser = signals[variant].get(s)
            if ser is None:
                continue
            sub = ser[(ser.index >= train[0]) & (ser.index <= train[1])]
            sub = sub.copy()
            sub.attrs["site"] = s
            train_scores.append(sub)
        return train_scores, scorer, ev_pool

    for mode in cfg["modes"]:
        if mode == "transfer":
            src = set(joined.loc[joined["state"].isin(cfg["transfer"]["source_states"]), "site_no"])
            tgt = set(joined.loc[joined["state"].isin(cfg["transfer"]["target_states"]), "site_no"])
        else:
            src = tgt = set(sites)

        # ---- fixed-schedule baselines
        for budget in budgets:
            k = FIXED_GRID_DAYS[budget]
            flags = []
            days = set(pd.date_range(test[0], test[1], freq="D")[::k])
            for site in tgt:
                m = add_audit(site, f"fixed_{k}d", budget, mode, days)
                if m:
                    flags.extend(m["captured"].tolist())
            lo, hi = bootstrap_ci(np.asarray(flags), n_boot, seed)
            rows.append({"mode": mode, "variant": f"fixed_{k}d", "budget": budget,
                         "n_sites": len(tgt), "n_events": len(flags),
                         "capture_rate": float(np.mean(flags)) if flags else np.nan,
                         "capture_ci_lo": lo, "capture_ci_hi": hi,
                         "median_delay_days": np.nan, "pre_onset_coverage": np.nan,
                         "trig_per_year": budget, "false_per_year": np.nan})

        # ---- adaptive variants (fold-based: logo per site, others single fold)
        for variant in ["nowcast", "nwm_fc", "season", "logit"]:
            if variant == "logit" and LogisticRegression is None:
                continue
            if variant == "logit":
                pool_sites = [s for s in sorted(src) if s in signals["nwm_fc"] and s in labels] \
                    if mode == "transfer" else \
                    [s for s in sorted(tgt) if s in signals["nwm_fc"] and s in labels]
            else:
                pool_sites = [s for s in sorted(src) if s in signals[variant]] \
                    if mode == "transfer" else [s for s in sorted(tgt) if s in signals[variant]]
            if len(pool_sites) < 2:
                log(f"  [{mode}/{variant}] skipped (too few sites)")
                continue

            if mode == "transfer":
                if variant == "logit":
                    eval_sites_all = [s for s in sorted(tgt)
                                      if s in signals["nwm_fc"] and s in labels]
                else:
                    eval_sites_all = [s for s in sorted(tgt) if s in signals[variant]]
                if not eval_sites_all:
                    log(f"  [{mode}/{variant}] skipped (no evaluation sites)")
                    continue
            else:
                eval_sites_all = pool_sites

            if mode == "logo":
                folds = [(s, [t for t in pool_sites if t != s]) for s in pool_sites]
            else:
                folds = [(None, pool_sites)]

            acc = {b: {"flags": [], "delays": [], "pre": [], "trig": [], "false": []}
                   for b in budgets}
            for fold_site, fold_sites in folds:
                train_scores, scorer, _ = fold_train_scores(variant, fold_sites)
                if not train_scores:
                    continue
                eval_sites = [fold_site] if fold_site else eval_sites_all
                for budget in budgets:
                    theta = calibrate_theta(train_scores, events, train, budget, H)
                    for site in eval_sites:
                        score = scorer(site)
                        if score is None:
                            continue
                        trig = set(score[score >= theta].index)
                        m = add_audit(site, variant, budget, mode, trig)
                        if m:
                            acc[budget]["flags"].extend(m["captured"].tolist())
                            if np.isfinite(m["delay"]):
                                acc[budget]["delays"].append(m["delay"])
                            acc[budget]["pre"].extend(m["pre_onset"].tolist())
                        tp, fp = burst_rate(trig, exc_days_all.get(site, set()), test, H)
                        acc[budget]["trig"].append(tp)
                        acc[budget]["false"].append(fp)
            for budget in budgets:
                a = acc[budget]
                lo, hi = bootstrap_ci(np.asarray(a["flags"]), n_boot, seed)
                rows.append({"mode": mode, "variant": variant, "budget": budget,
                             "n_sites": len(pool_sites), "n_events": len(a["flags"]),
                             "capture_rate": float(np.mean(a["flags"])) if a["flags"] else np.nan,
                             "capture_ci_lo": lo, "capture_ci_hi": hi,
                             "median_delay_days": float(np.median(a["delays"])) if a["delays"] else np.nan,
                             "pre_onset_coverage": float(np.mean(a["pre"])) if a["pre"] else np.nan,
                             "trig_per_year": float(np.mean(a["trig"])) if a["trig"] else np.nan,
                             "false_per_year": float(np.mean(a["false"])) if a["false"] else np.nan})
                log(f"  [{mode}/{variant}] budget {budget}/yr: "
                    f"capture={rows[-1]['capture_rate']:.3f} (events={len(a['flags'])})")

    out_dir = repo_root() / "outputs" / "rule_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows).sort_values(["mode", "variant", "budget"]).reset_index(drop=True)
    res.to_csv(out_dir / "rule_validation_results.csv", index=False)
    pd.DataFrame(audit).to_csv(out_dir / "rule_validation_events.csv", index=False)
    summary = {"config": str(cfg_path), "threshold_mg_l": thr, "horizon_days": H,
               "split": split, "n_sites_total": len(sites), "smoke": args.smoke,
               "calibration": f"max train capture, freq <= {BUDGET_SLACK}x budget"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    log(f"\nresults -> {out_dir / 'rule_validation_results.csv'}")
    log(res.to_string(index=False))


if __name__ == "__main__":
    main()
