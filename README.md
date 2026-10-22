# FiCaR-Net

Forecast-informed cascading risk network for riverine nitrate exceedance, with a
frozen-rule independent validation on an eight-state continuous-nitrate network.

## Contents

- `models/` — network definitions (`ficar.py`, `hydro.py`, `nutrient.py`)
- `scripts/` — data preparation, training, evaluation, figure generation, and the
  rule-validation pipeline:
  - `wqp_harvest.py` — harvest USGS continuous nitrate (parameter 99133) stations from the Water Quality Portal
  - `extract_nwm_streamflow.py` — extract NWM v3.0 retrospective daily flows at the nearest reach
  - `extract_wb2_precip.py` — extract IFS-HRES precipitation forecasts from WeatherBench 2
  - `compute_cropland_fraction.py` — cropland fractions from the USDA CDL (2/5/10-km buffers)
  - `eval_rule_validation.py` — frozen, budget-matched rule validation (temporal / leave-one-gauge-out / transfer; 5 and 10 mg/L)
  - `analyze_rule_validation.py` — duration/state/cropland partitions, hybrid unions, NWM diagnostics
  - `make_rule_figures.py` — submission-style figures (capture-budget curves, short-event bars, delay distributions)
- `configs/` — locked hyperparameters for the reported experiments,
  including the frozen rule configuration `rule_frozen.yaml`

## Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## Data

Raw forcing and water-quality archives are **not** redistributed here. Public sources:

| Item | Detail |
|------|--------|
| Caravan | https://doi.org/10.5281/zenodo.15530021 |
| Caravan-MultiMet nowcasts/forecasts | https://doi.org/10.5281/zenodo.14161235 |
| Illinois nutrients | https://doi.org/10.5066/P13ITOD4 |
| ERA5-Land | https://cds.climate.copernicus.eu/ |
| Continuous nitrate (8 states, parameter 99133) | https://www.waterqualitydata.us/ |
| NWM v3.0 retrospective flows | s3://noaa-nwm-retrospective-3-0-pds (AWS Open Data) |
| IFS-HRES precipitation forecasts | gs://weatherbench2 (Google Cloud) |
| USDA Cropland Data Layer 2019 | https://nassgeodata.gmu.edu/CropScape/ |

Build processed windows with `scripts/build_interim.py`, then train with `scripts/train_hydro.py`.

## Reproduce

Manuscript core cascade experiments:

```bash
python scripts/train_hydro.py --config configs/hydro_ficar_full.yaml
python scripts/eval_cascade.py --config configs/cascade_shared_head.yaml
python scripts/eval_compound_e7.py --config configs/compound_e7_v2.yaml
python scripts/eval_decision_calib.py --config configs/decision_calib.yaml
python scripts/make_figs.py
```

Independent rule validation (eight-state network; frozen rule in `configs/rule_frozen.yaml`):

```bash
# 1. harvest continuous nitrate stations (8 states) and build paired flow–nitrate records
python scripts/wqp_harvest.py --states IA,IL,IN,KS,MO,MN,NE,WI
# 2. extract operational flows and true-forecast precipitation, and attach CDL cropland fractions
python scripts/extract_nwm_streamflow.py
python scripts/extract_wb2_precip.py
python scripts/compute_cropland_fraction.py
# 3. frozen rule validation at 5 mg/L (primary) and 10 mg/L (sensitivity)
python scripts/eval_rule_validation.py                     # 5 mg/L
python scripts/eval_rule_validation.py --threshold 10      # 10 mg/L
# 4. partitions, hybrid unions and NWM diagnostics
python scripts/analyze_rule_validation.py
# 5. submission-style figures
python scripts/make_rule_figures.py
```

Rule thresholds are calibrated on the 2020-01–2021-06 training window only
(maximal training capture subject to a trigger frequency of at most 1.3x budget);
the 2021-07–2022-12 test window never enters calibration.
