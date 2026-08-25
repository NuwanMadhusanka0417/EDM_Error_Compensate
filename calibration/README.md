# EDM day-to-day drift calibration

Two-stage calibration for a **frozen** Element Description Method (EDM) force estimator
when sEMG envelope scale drifts across recording days.

## Data layout (matches `mycode/train_compensation.py`)

Uses CSV files under `../Data/`:

| Split | Session folder |
|-------|----------------|
| Day-0 (EDM fit) | `3rd_Base` |
| Train | `1st_Comp`, `2nd_Comp`, `3rd_Base` |
| Validation | `4th_Valid` |
| Test | `5th_Comp` |

Columns: `Time`, `Cali_LPF_PASF_sEMG` (envelope), `Force`.

Each CSV is split temporally into pseudo-trials: **rest → reference → calibration → task**.
Only `task` segments are used for evaluation; calibrators fit on the other segments.

## Install

```powershell
cd calibration
pip install -r requirements.txt
```

## Run on real data

```powershell
python scripts/00_make_manifest.py
python scripts/02_identify_day0.py
python scripts/03_run_arms.py
```

Outputs:
- `data/processed/edm_params.json` — frozen day-0 EDM
- `results/metrics/arm_rmse.csv` — RMSE per arm on val/test
- `results/metrics/decomposition.csv` — drift loss decomposition

## Run on synthetic data (no real CSVs required)

```powershell
python synth/generate.py
set CALIBRATION_CONFIG=config.synth.yaml
python scripts/00_make_manifest.py
python scripts/02_identify_day0.py
python scripts/03_run_arms.py
pytest
```

## Arms

| ID | Name | Stage 1 (input) | Stage 2 (output) |
|----|------|-----------------|------------------|
| A | uncalibrated | no | no |
| B | stage1_only | yes | no |
| C | stage2_only | no | yes |
| D | cascade | yes | yes |
| E | oracle_affine | no | yes (analysis only) |

## Interpreting the decomposition

- `E_floor` — RMSE of frozen EDM on day-0 (best achievable, no drift)
- `E_A` — RMSE uncalibrated on drifted day
- `L_total = E_A - E_floor` — loss caused by drift
- `R_arm = (E_A - E_arm) / L_total` — fraction of drift loss removed
- `nonaffine` — RMSE after best affine fit; Stage 2 alone cannot reduce the non-affine part

## Tests

```powershell
pytest
```
