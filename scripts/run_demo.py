# -*- coding: utf-8 -*-
"""End-to-end demo: synthetic data -> hydro train -> cascade eval."""
from __future__ import annotations

import json
from pathlib import Path

from src.data.caravan import print_download_instructions
from src.data.demo import build_demo_data
from src.data.paths import ensure_dirs, repo_root
from src.experiments.train_hydro import load_config, train as train_hydro
import sys
import yaml
from src.experiments import eval_cascade as ec


def run():
    ensure_dirs()
    print("=== Download instructions (for real data) ===")
    print_download_instructions()
    print("\n=== Building demo data ===")
    paths = build_demo_data()
    print(paths)

    print("\n=== Train hydro baseline ===")
    cfg = load_config(repo_root() / "configs" / "hydro_baseline.yaml")
    cfg["epochs"] = 3
    cfg["output_dir"] = "outputs/demo_hydro"
    ckpt = train_hydro(cfg)
    print("Hydro ckpt:", ckpt)

    print("\n=== Train hydro with forecast lead=3 ===")
    cfg_fc = load_config(repo_root() / "configs" / "hydro_forecast.yaml")
    cfg_fc["epochs"] = 3
    cfg_fc["output_dir"] = "outputs/demo_hydro_fc"
    ckpt_fc = train_hydro(cfg_fc)
    print("Hydro FC ckpt:", ckpt_fc)

    print("\n=== Train FiCaR hydro ===")
    cfg_f = load_config(repo_root() / "configs" / "hydro_ficar.yaml")
    cfg_f["epochs"] = 3
    cfg_f["output_dir"] = "outputs/demo_hydro_ficar"
    ckpt_f = train_hydro(cfg_f)
    print("FiCaR hydro ckpt:", ckpt_f)

    print("\n=== Cascade nutrient risk eval ===")
    with open(repo_root() / "configs" / "cascade_illinois.yaml", encoding="utf-8") as f:
        ccfg = yaml.safe_load(f)
    ccfg["epochs"] = 3
    ccfg["output_dir"] = "outputs/demo_cascade"
    tmp = repo_root() / "configs" / "_demo_cascade.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(ccfg, f)
    sys.argv = ["eval_cascade", "--config", str(tmp.relative_to(repo_root()))]
    ec.main()

    summary = {
        "hydro_ckpt": str(ckpt),
        "hydro_fc_ckpt": str(ckpt_fc),
        "ficar_hydro_ckpt": str(ckpt_f),
        "cascade_dir": str(repo_root() / "outputs" / "demo_cascade"),
        "status": "ok",
    }
    out = repo_root() / "outputs" / "demo_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nDemo complete:", out)
    return summary


if __name__ == "__main__":
    run()
