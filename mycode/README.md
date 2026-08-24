# Compensation model training (`train_compensation.py`)

Trains a Conv + BiLSTM + Transformer compensation model (Pattern 3: sEMG + base-model output) with early stopping on validation data.

**Train:** `1st_Comp` + `2nd_Comp` + `3rd_Base`  
**Validation:** `4th_Valid` (early stopping)  
**Test:** `5th_Comp`

---

## Run training

```powershell
cd mycode

# Default settings (stride=5, batch=64) — saves PNG plots only
python train_compensation.py

# Recommended for ~4 GB RAM
python train_compensation.py --stride 10 --batch 32

# Also save CSV files, model weights, config, validation outputs
python train_compensation.py --stride 10 --batch 32 --save-all
```

---

## Command-line parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--stride` | `5` | Step between sliding-window start positions. Higher = fewer training windows, **less RAM**, faster training, slightly less temporal coverage. `1` uses every sample (most data, highest RAM). |
| `--batch` | `64` | Mini-batch size for training and evaluation. **Lower = less RAM** and less GPU memory; may be slightly noisier. |
| `--epochs` | `100` | Maximum number of training epochs. Training usually stops earlier via early stopping (patience = 15 on validation loss). |
| `--save-all` | off | Also save CSV outputs, `model.pt`, `run_config.json`, validation predictions, and test CSVs. Without this flag, only PNG plots are saved. |
| `--debug-anomaly` | off | Enable PyTorch autograd anomaly detection. Use when debugging NaN/Inf errors; prints a traceback for the first bad operation. Slower — not for normal runs. |

### `--stride` in more detail

Each sample is a window of **100 timesteps** (`SEQ = 100`). Stride controls how far the window moves each step:

| Stride | Effect | Typical RAM |
|--------|--------|-------------|
| `1` | Every timestep → most windows (~950k train) | ~2.5–3 GB+ |
| `5` | Default; every 5th start | ~0.7–1 GB |
| `10` | Good for ~4 GB machines | ~0.5 GB |
| `20` | Good for ~2–3 GB machines | ~0.3 GB |

### `--batch` in more detail

| Batch | When to use |
|-------|-------------|
| `64` | Default; fine if you have enough RAM |
| `32` | Safer on 4–8 GB RAM |
| `16` | Low-RAM machines or very long sequences |

---

## Outputs

### Always saved (`Estimation/`)

| File | Contents |
|------|----------|
| `training_loss.png` | Train / val / test loss vs epoch; best epoch and early-stop epoch marked |
| `test/Data1.png` … | Real vs predicted force line plot; RMSE, MAE, R², MSE (final + base) at bottom |

Console also prints **best epoch** and whether **early stopping** triggered.

### With `--save-all`

| File / folder | Contents |
|---------------|----------|
| `training_history.csv` | Per-epoch losses and RMSE values |
| `run_summary.json` | Best epoch, early-stop epoch, final metrics |
| `model.pt` | Best model weights |
| `run_config.json` | Scalers, base-model params, run settings |
| `base_model_parameters.csv` | Frozen TANH base-model parameters |
| `validation/` | Full validation prediction CSVs |
| `test_csv/` | Test force/prediction CSVs |

---

## Examples

```powershell
# Full training, low RAM
python train_compensation.py --stride 20 --batch 16

# Cap epochs for a quick test
python train_compensation.py --stride 10 --batch 32 --epochs 20

# Debug NaN during training
python train_compensation.py --stride 10 --batch 32 --debug-anomaly

# Full export for downstream use
python train_compensation.py --stride 5 --batch 64 --save-all
```

---

## Fixed settings (edit in script if needed)

These are not CLI flags; change them in `train_compensation.py` if required:

| Setting | Value | Meaning |
|---------|-------|---------|
| `PATTERN` | `3` | Input = scaled sEMG + base-model output |
| `SEQ` | `100` | Sliding-window length (timesteps) |
| `EARLY_STOP_PATIENCE` | `15` | Stop if val loss does not improve for 15 epochs |
| `LR` | `3e-4` | Learning rate after warmup |
| `MAX_EPOCHS` | `100` | Upper epoch limit (overridable with `--epochs`) |
