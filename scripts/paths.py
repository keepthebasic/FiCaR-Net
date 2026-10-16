# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    # src/data/paths.py -> parents[0]=data, [1]=src, [2]=cascade_water_risk
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    env = os.environ.get("CASCADE_DATA_ROOT")
    if env:
        return Path(env)
    return repo_root() / "data"


def ensure_dirs() -> dict[str, Path]:
    root = data_root()
    paths = {
        "root": root,
        "raw": root / "raw",
        "interim": root / "interim",
        "processed": root / "processed",
        "demo": root / "demo",
        "caravan": root / "raw" / "caravan",
        "multimet": root / "raw" / "multimet",
        "usgs": root / "raw" / "usgs_illinois",
        "outputs": repo_root() / "outputs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
