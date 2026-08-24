


## Run training

```powershell
cd mycode

# Default: saves PNG plots only
python train_compensation.py --stride 10 --batch 32

# Also save CSV files, model, config, validation outputs
python train_compensation.py --stride 10 --batch 32 --save-all
```

## Default outputs (`Estimation/`)

| File | Contents |
|------|----------|
| `training_loss.png` | Train / val / test loss vs epoch (best + early-stop marked) |
| `test/Data1.png` … | Real vs predicted force line plot with metrics at bottom |

## With `--save-all`

Also saves: `training_history.csv`, `run_summary.json`, `model.pt`, `run_config.json`, validation CSVs, and test CSVs.

## Low RAM

```powershell
python train_compensation.py --stride 20 --batch 16
```
