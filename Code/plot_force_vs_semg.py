from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("../Data")
OUT = DATA.parent / "Analysis" / "Force_vs_RectifiedsEMG"
XCOL, YCOL = "Cali_Rectified_PASF_sEMG", "Force"  # Cali_Rectified_PASF_sEMG, Cali_LPF_PASF_sEMG

for folder in sorted(p for p in DATA.iterdir() if p.is_dir()):
    files = sorted(folder.glob("Data*.csv"))
    if not files:
        continue

    n = len(files)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle(folder.name, fontsize=14)

    for ax, path in zip(axes.flat, files):
        df = pd.read_csv(path, usecols=[XCOL, YCOL])
        x = pd.to_numeric(df[XCOL], errors="coerce").to_numpy()
        y = pd.to_numeric(df[YCOL], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]

        ax.scatter(x[::5], y[::5], s=1, alpha=0.25, c="steelblue", rasterized=True)

        if len(x) > 1 and np.ptp(x) > 0:
            grad, intercept = np.polyfit(x, y, 1)
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, grad * xx + intercept, "r-", lw=1.5)
            ax.set_title(f"{path.stem}  gradient = {grad:.4f}", fontsize=9)
        else:
            ax.set_title(path.stem, fontsize=9)

        ax.set_xlabel(XCOL, fontsize=8)
        ax.set_ylabel(YCOL, fontsize=8)

    for ax in axes.flat[n:]:
        ax.axis("off")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{folder.name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / folder.name}.png")

print(f"\nDone. Images in:\n{OUT.resolve()}")
