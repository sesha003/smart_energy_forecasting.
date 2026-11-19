"""Step 4 - Compare all models on validation (choose) and test (confirm once).

Usage:
    python -m src.compare        # writes models/results_val.csv and models/results_test.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .build_dataset import load_dataset
from .config import MODELS_DIR
from .data_prep import WindowDataset, prepare, unscale_targets
from .evaluate import (baseline_forecast, clip_solar, fit_baseline_band, metrics, raw_target_array,
                       results_table, true_values)
from .forecast import MODEL_LABELS
from .patchtst import load_patchtst_arrays
from .train import load_model, predict


def all_forecasts(split: str):
    """Returns (days, y_true (N,24,2), {model: q (N,24,2,3)}) for 'val' or 'test'."""
    df = load_dataset()
    full, values, scaler, w = prepare(df, fit=False)
    raw = raw_target_array(full)
    starts = w[split].starts
    y = true_values(raw, starts)
    preds = {}
    for lag, key in ((168, "base_week"), (24, "base_day")):
        preds[key] = baseline_forecast(raw, starts, lag, fit_baseline_band(raw, w["train"].starts, lag))
    for name in ("lstm", "cnn"):
        if (MODELS_DIR / f"{name}.pt").exists():
            q = predict(load_model(name), WindowDataset(values, starts), device="cpu")
            preds[name] = clip_solar(unscale_targets(q, scaler))
    if split == "test":
        q = load_patchtst_arrays(w[split].days())
        if q is not None and not np.isnan(q).all():
            ok = ~np.isnan(q).any(axis=(1, 2, 3))
            preds["patchtst"] = q
            if not ok.all():  # evaluate everyone on the same days
                y = y[ok]
                preds = {k: v[ok] for k, v in preds.items()}
                return w[split].days()[ok], y, preds
    return w[split].days(), y, preds


def compare(split: str) -> pd.DataFrame:
    _, y, preds = all_forecasts(split)
    table = results_table({MODEL_LABELS[k]: metrics(y, q) for k, q in preds.items()})
    table.to_csv(MODELS_DIR / f"results_{split}.csv")
    return table


def best_model(split: str = "val") -> str:
    """Lowest average (scaled) pinball over price and solar on the validation set."""
    t = pd.read_csv(MODELS_DIR / f"results_{split}.csv")
    t["norm"] = t.groupby("target")["Pinball"].transform(lambda s: s / s.min())
    return t.groupby("model")["norm"].mean().idxmin()


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    for split in ("val", "test"):
        print(f"\n=== {split} ===")
        print(compare(split))
    print("\nbest on validation:", best_model("val"))
