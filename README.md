# FiCaR-Net

Forecast-informed cascading risk network for riverine nitrate exceedance.

## Contents

- `models/` — network definitions (`ficar.py`, `hydro.py`, `nutrient.py`)
- `scripts/` — data preparation, training, evaluation, and figure generation
- `configs/` — locked hyperparameters for the reported experiments

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

Build processed windows with `scripts/build_interim.py`, then train with `scripts/train_hydro.py`.

## Reproduce

```bash
python scripts/train_hydro.py --config configs/hydro_ficar_full.yaml
python scripts/eval_cascade.py --config configs/cascade_shared_head.yaml
python scripts/eval_compound_e7.py --config configs/compound_e7_v2.yaml
python scripts/eval_decision_calib.py --config configs/decision_calib.yaml
python scripts/make_figs.py
```
