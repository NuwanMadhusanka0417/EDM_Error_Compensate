"""Within-day (same-session) compensation experiment.

Companion to train_compensation.py, which does the cross-day (Leave-One-Day-Out)
experiment. The two scripts share every component - base model, calibration,
normalisation, windowing, architectures, training loop - and differ only in how
recordings are split:

    train_compensation.py : train on some DAYS, test on a different DAY
    train_within_day.py   : train on some FILES of a day, test on other FILES
                            of the SAME day

Running both answers the question the cross-day results raise on their own:
is the compensator failing because transfer across days is hard, or because
there is nothing left in the sEMG to compensate at all? If the models beat the
base model here but not across days, the problem is transfer. If they fail here
too, the input simply does not carry the residual.

Splits are at file level, never inside a recording, so no window straddles the
train/test boundary.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

import train_compensation as tc
from models import DISPLAY_NAMES, MODEL_NAMES

OUT = Path(__file__).resolve().parent / "Estimation" / "within_day"


def split_files(paths, fold, n_folds, val_files=1):
    """File-level train/val/test split inside one session.

    ``fold`` rotates which block of files is held out for testing, so across
    ``n_folds`` runs every recording is tested at least once.
    """
    n = len(paths)
    if n < 3:
        raise ValueError(
            f"Need at least 3 recordings to split a day; got {n}."
        )

    n_test = max(1, n // n_folds if n_folds > 1 else max(1, round(0.2 * n)))
    n_val = max(1, min(val_files, n - n_test - 1))

    shift = (fold * n_test) % n
    order = list(paths[shift:]) + list(paths[:shift])

    test = order[:n_test]
    val = order[n_test:n_test + n_val]
    train = order[n_test + n_val:]

    if not train:
        raise ValueError(
            f"Split left no training files (n={n}, test={n_test}, val={n_val})."
        )
    return train, val, test


def run_day(session, fold, args):
    """Train and evaluate every selected model inside a single session."""
    paths = tc.session_files(session)
    train_paths, val_paths, test_paths = split_files(
        paths, fold, args.day_folds, args.val_files
    )

    print(f"\n{'=' * 70}")
    print(f"WITHIN-DAY  session={session}  fold={fold + 1}/{args.day_folds}")
    print(f"  train : {', '.join(p.stem for p in train_paths)}")
    print(f"  val   : {', '.join(p.stem for p in val_paths)}")
    print(f"  test  : {', '.join(p.stem for p in test_paths)}")
    print(f"{'=' * 70}", flush=True)

    use_lpf = not args.no_lpf

    # Base model fitted on this day's TRAINING files only - never on the files
    # it will be evaluated against.
    theta, base_result = tc.fit_base_model(train_paths, use_lpf=use_lpf)
    if use_lpf:
        print(
            f"  base: gain={theta[0]:.4f} slope={theta[1]:.4f} "
            f"fc={theta[2]:.2f} Hz (cost={base_result.cost:.4f})",
            flush=True,
        )
    else:
        print(
            f"  base: gain={theta[0]:.4f} slope={theta[1]:.4f} "
            f"(cost={base_result.cost:.4f})",
            flush=True,
        )

    train_records = tc.build_records(train_paths, theta, args.calib_seconds, use_lpf)
    val_records = tc.build_records(val_paths, theta, args.calib_seconds, use_lpf)
    test_records = tc.build_records(test_paths, theta, args.calib_seconds, use_lpf)

    x_scaler = tc.fit_feature_scaler(train_records)

    X_train, Y_train = tc.records_to_xy(train_records, x_scaler, args.stride, "train")
    X_val, Y_val = tc.records_to_xy(val_records, x_scaler, args.stride, "val")

    train_ds = TensorDataset(
        tc.tensor_from_numpy(X_train, "X_train"),
        tc.tensor_from_numpy(Y_train, "Y_train"),
    )
    val_ds = TensorDataset(
        tc.tensor_from_numpy(X_val, "X_val"),
        tc.tensor_from_numpy(Y_val, "Y_val"),
    )
    n_features = X_train.shape[2]

    day_dir = OUT / session / f"fold{fold + 1}"
    day_dir.mkdir(parents=True, exist_ok=True)

    base_ref = {}
    model_rows = {}
    csv_rows = []

    for model_name in args.models:
        model, best_state, history, summary = tc.train_one_model(
            model_name, train_ds, val_ds, n_features, args
        )

        tc.save_training_loss_plot(
            history, summary,
            day_dir / f"training_loss_{model_name}.png",
            title_suffix=(
                f"  ({DISPLAY_NAMES.get(model_name, model_name)}, "
                f"within {session})"
            ),
        )

        test_rows = tc.predict_records(model, test_records, x_scaler, args.stride)

        model_rows[model_name] = {}
        for row in test_rows:
            stem = row["path"].stem
            model_rows[model_name][stem] = row
            base_ref.setdefault(
                stem,
                {
                    "t": row["t"],
                    "y": row["y"],
                    "y_base": row["y_base"],
                    "session": f"{session} (within-day)",
                },
            )
            csv_rows.append(
                {
                    "Experiment": "within_day",
                    "Session": session,
                    "Fold": fold + 1,
                    "Model": model_name,
                    "Model_Name": DISPLAY_NAMES.get(model_name, model_name),
                    "Data": stem,
                    "Session_Gain": row["session_gain"],
                    "N_Params": summary["n_params"],
                    "Best_Epoch": summary["best_epoch"],
                    **{f"Base_{k}": v for k, v in row["base_metrics"].items()},
                    **{f"Final_{k}": v for k, v in row["final_metrics"].items()},
                }
            )

        base_rmse = float(np.mean([r["base_metrics"]["RMSE"] for r in test_rows]))
        final_rmse = float(np.mean([r["final_metrics"]["RMSE"] for r in test_rows]))
        final_r2 = float(np.mean([r["final_metrics"]["R2"] for r in test_rows]))
        improved = sum(
            1 for r in test_rows
            if r["final_metrics"]["RMSE"] < r["base_metrics"]["RMSE"]
        )
        print(
            f"    -> test RMSE={final_rmse:.4f} (base {base_rmse:.4f}) "
            f"R2={final_r2:.4f} | improved {improved}/{len(test_rows)}",
            flush=True,
        )

        if args.save_all:
            tc.save_checkpoint(best_state, day_dir / f"model_{model_name}.pt")
            tc.save_prediction_csvs(test_rows, day_dir / f"test_csv_{model_name}")

    tc.save_comparison_plots(base_ref, model_rows, day_dir / "test")
    tc.save_rmse_bar_plot(csv_rows, day_dir, f"{session} (within-day)")

    if args.save_all:
        (day_dir / "split.json").write_text(
            json.dumps(
                {
                    "session": session,
                    "fold": fold + 1,
                    "train": [p.stem for p in train_paths],
                    "val": [p.stem for p in val_paths],
                    "test": [p.stem for p in test_paths],
                    "base_theta": theta.tolist(),
                },
                indent=2,
            )
        )

    return csv_rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Within-day (same-session) compensation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available models:\n  "
            + "\n  ".join(
                f"{name:26s} {DISPLAY_NAMES.get(name, name)}"
                for name in MODEL_NAMES
            )
            + "\n\nExample:\n"
            "  python train_within_day.py --models all --stride 20 "
            "--epochs 100 --calib-seconds 10"
        ),
    )
    p.add_argument(
        "--models", nargs="+", default=["all"], metavar="MODEL",
        help=f"Models to compare. Use 'all'. Choices: {' '.join(MODEL_NAMES)}",
    )
    p.add_argument(
        "--sessions", nargs="+", default=None, metavar="SESSION",
        help=f"Days to run. Default: all. Choices: {' '.join(tc.SESSIONS)}",
    )
    p.add_argument(
        "--day-folds", type=int, default=1,
        help=(
            "Rotating within-day splits per session. 1 uses a single "
            "60/20/20 file split; higher values rotate the test block so every "
            "recording is tested (costs proportionally more time)."
        ),
    )
    p.add_argument(
        "--val-files", type=int, default=1,
        help="Number of files held out for validation inside each day.",
    )
    p.add_argument("--stride", type=int, default=tc.WINDOW_STRIDE)
    p.add_argument("--batch", type=int, default=tc.BATCH)
    p.add_argument("--epochs", type=int, default=tc.MAX_EPOCHS)
    p.add_argument(
        "--calib-seconds", type=float, default=0.0,
        help=(
            "Per-recording gain calibration window, excluded from all metrics. "
            "Use the same value as the cross-day run so the two are comparable."
        ),
    )
    p.add_argument("--no-lpf", action="store_true",
                   help="Ablation: static TANH base model with no dynamics.")
    outputs = p.add_mutually_exclusive_group()
    outputs.add_argument("--save-all", action="store_true",
                         help="Also save checkpoints, prediction CSVs and splits.")
    outputs.add_argument("--images-only", action="store_true",
                         help="Save only PNGs plus the final comparison CSV.")
    p.add_argument("--debug-anomaly", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.stride = max(1, args.stride)
    args.day_folds = max(1, args.day_folds)
    tc.BATCH = max(1, args.batch)
    args.models = tc.resolve_models(args.models)

    if args.debug_anomaly:
        torch.autograd.set_detect_anomaly(True)

    sessions = args.sessions or tc.SESSIONS
    unknown = [s for s in sessions if s not in tc.SESSIONS]
    if unknown:
        print(
            f"ERROR: unknown session(s): {', '.join(unknown)}\n"
            f"Choices: {', '.join(tc.SESSIONS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)

    print("DEVICE =", tc.DEVICE)
    print("Experiment : WITHIN-DAY (train and test on the same session)")
    print(f"Sessions   : {', '.join(sessions)}")
    print(f"Models     : {', '.join(args.models)}")
    print(
        f"stride={args.stride} batch={tc.BATCH} epochs={args.epochs} "
        f"day_folds={args.day_folds} calib_seconds={args.calib_seconds}"
    )

    all_rows = []
    for session in sessions:
        for fold in range(args.day_folds):
            all_rows.extend(run_day(session, fold, args))

    frame = pd.DataFrame(all_rows)
    combined = OUT / "within_day_comparison.csv"
    frame.to_csv(combined, index=False)

    print(f"\n{'=' * 70}")
    print("WITHIN-DAY RESULTS  (mean RMSE per model, per session)")
    print(f"{'=' * 70}")
    pivot = frame.pivot_table(
        index="Model_Name", columns="Session",
        values="Final_RMSE", aggfunc="mean",
    )
    pivot.loc["EDM base model"] = frame.groupby("Session")["Base_RMSE"].mean()
    print(pivot.round(4).to_string())

    print(f"\n{'=' * 70}")
    print("OVERALL")
    print(f"{'=' * 70}")
    overall = frame.groupby("Model_Name").agg(
        RMSE_mean=("Final_RMSE", "mean"),
        RMSE_std=("Final_RMSE", "std"),
        R2_mean=("Final_R2", "mean"),
        MAE_mean=("Final_MAE", "mean"),
        Params=("N_Params", "first"),
    ).sort_values("RMSE_mean")
    print(overall.round(4).to_string())
    print(
        f"\nEDM base model     RMSE={frame['Base_RMSE'].mean():.4f}  "
        f"R2={frame['Base_R2'].mean():.4f}"
    )

    beat = frame[frame["Final_RMSE"] < frame["Base_RMSE"]]
    print(f"\nFiles where a model beat the base model: {len(beat)}/{len(frame)}")

    # The comparison that decides the story.
    best = overall.index[0]
    best_rmse = overall.iloc[0]["RMSE_mean"]
    base_rmse = frame["Base_RMSE"].mean()
    print(f"\n{'=' * 70}")
    if best_rmse < base_rmse:
        print(
            f"WITHIN-DAY COMPENSATION WORKS: {best} reaches RMSE "
            f"{best_rmse:.4f} vs base {base_rmse:.4f} "
            f"({100 * (base_rmse - best_rmse) / base_rmse:.1f}% better).\n"
            "If the cross-day run does not show this, the limitation is "
            "transfer between sessions, not the sEMG signal itself."
        )
    else:
        print(
            f"WITHIN-DAY COMPENSATION DOES NOT HELP: best model {best} reaches "
            f"RMSE {best_rmse:.4f} vs base {base_rmse:.4f}.\n"
            "Training and testing on the same day removes transfer entirely, so "
            "this points at the input: after a calibrated TANH->LPF base model, "
            "the residual is not recoverable from this sEMG channel."
        )
    print(f"{'=' * 70}")

    if not args.images_only:
        overall.to_csv(OUT / "within_day_summary.csv")

    print(f"\nCombined results CSV : {combined.resolve()}")
    print(f"Plots and per-day outputs: {OUT.resolve()}")


if __name__ == "__main__":
    main()
