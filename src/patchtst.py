"""Step 4 - PatchTST (Transformer) from the neuralforecast library.

PatchTST cuts the 168-hour history into 24-hour patches; attention learns which past days matter.
It is univariate (channel-independent): ONE model is trained on two series, `price` and `solar`.

Training data (same time split as the LSTM / 1D-CNN):
    inputs/targets : hourly `price` (ct/kWh) and `solar` (MWh) from data/processed/dataset_hourly.csv
                     (source: SMARD wholesale price DE/LU + actual photovoltaic generation)
    train          : 2019-01-01 -> 2024-12-31   (the model learns from this)
    validation     : 2025-01-01 -> 2025-12-31   (early stopping only - no weight updates)
    test           : 2026-01-01 -> last complete day (rolling day-ahead forecasts, never trained on)
    not used       : wind, load, residual load, calendar features (PatchTST here sees only the
                     history of the series it forecasts)

Hardware: Apple GPU (MPS) when available, else CUDA, else CPU.

Outputs:
    models/patchtst/                 saved model -> used by the Streamlit app for live forecasts
    models/patchtst/training_info.json  data, periods, device, steps
    models/patchtst_forecasts.csv    test-period forecasts (for the comparison table)

Usage:
    python -m src.patchtst --max-steps 2000
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .build_dataset import load_dataset
from .config import HORIZON, INPUT_HOURS, MODELS_DIR, TARGETS, TRAIN_END, VAL_END

FORECAST_CSV = MODELS_DIR / "patchtst_forecasts.csv"
MODEL_DIR = MODELS_DIR / "patchtst"
INFO_JSON = MODEL_DIR / "training_info.json"
COLS = {"PatchTST-lo-80": "lo", "PatchTST-median": "mid", "PatchTST-hi-80": "hi"}


def accelerator() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"       # Apple GPU
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def to_long(df: pd.DataFrame, end_on_full_day: bool = True) -> pd.DataFrame:
    """Wide hourly table -> neuralforecast long format (unique_id, ds, y), no gaps."""
    full = df[TARGETS].reindex(pd.date_range(df.index.min(), df.index.max(), freq="1h"))
    full = full.interpolate(limit_direction="both")
    if end_on_full_day:  # end at 23:00 so the test cutoffs fall on midnight
        full = full.loc[:full.index[full.index.hour == 23].max()]
    long = full.reset_index(names="ds").melt(id_vars="ds", var_name="unique_id", value_name="y")
    return long[["unique_id", "ds", "y"]]


def _tidy(q: pd.DataFrame) -> pd.DataFrame:
    """Sort quantiles (no crossing) and clip solar at 0."""
    lo, mid, hi = np.sort(q[["lo", "mid", "hi"]].values, axis=1).T
    q = q.assign(lo=lo, mid=mid, hi=hi)
    solar = q.target == "solar"
    q.loc[solar, ["lo", "mid", "hi"]] = q.loc[solar, ["lo", "mid", "hi"]].clip(lower=0)
    return q


def run_patchtst(df: pd.DataFrame, max_steps: int = 2000, verbose: bool = True) -> pd.DataFrame:
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import PatchTST

    long = to_long(df)
    end = long.ds.max()
    test_hours = int((end - pd.Timestamp(VAL_END)) / pd.Timedelta(hours=1))
    val_hours = int((pd.Timestamp(VAL_END) - pd.Timestamp(TRAIN_END)) / pd.Timedelta(hours=1))
    acc = accelerator()
    if verbose:
        print(f"PatchTST on {acc.upper()} | train {long.ds.min():%Y-%m-%d} -> {TRAIN_END[:10]} | "
              f"val {val_hours // 24} days (2025) | test {test_hours // 24} days -> {end:%Y-%m-%d}")

    model = PatchTST(
        h=HORIZON, input_size=INPUT_HOURS, patch_len=24, stride=24,
        loss=MQLoss(level=[80]),            # 10 % / 50 % / 90 % quantiles
        scaler_type="standard", learning_rate=1e-3, batch_size=64,
        max_steps=max_steps, val_check_steps=100, early_stop_patience_steps=5,
        random_seed=42, enable_progress_bar=verbose,
        accelerator=acc, devices=1,
    )
    nf = NeuralForecast(models=[model], freq="h")
    t0 = time.time()
    # fits on train (val for early stopping), then forecasts every test day without refitting
    cv = nf.cross_validation(df=long, val_size=val_hours, test_size=test_hours,
                             step_size=HORIZON, n_windows=None, refit=False)
    seconds = time.time() - t0
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    out = cv.rename(columns={"ds": "timestamp", "unique_id": "target", **COLS})
    out = out[out["cutoff"].dt.hour == 23]          # one forecast per test day (made at 23:00)
    out = _tidy(out[["timestamp", "target", "lo", "mid", "hi", "y"]].sort_values(["target", "timestamp"]))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(FORECAST_CSV, index=False)
    nf.save(path=str(MODEL_DIR), save_dataset=False, overwrite=True)
    info = {
        "model": "PatchTST (neuralforecast)",
        "device": acc,
        "series": TARGETS,
        "data_file": "data/processed/dataset_hourly.csv (SMARD: wholesale price DE/LU, actual photovoltaics)",
        "train": f"{long.ds.min():%Y-%m-%d} to {TRAIN_END[:10]}",
        "validation_early_stopping": f"{(pd.Timestamp(TRAIN_END) + pd.Timedelta(hours=1)):%Y-%m-%d} to {VAL_END[:10]}",
        "test": f"{(pd.Timestamp(VAL_END) + pd.Timedelta(hours=1)):%Y-%m-%d} to {end:%Y-%m-%d}",
        "input_hours": INPUT_HOURS, "horizon_hours": HORIZON, "patch_len": 24,
        "quantiles": [0.1, 0.5, 0.9], "max_steps": max_steps,
        "training_seconds": round(seconds, 1),
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    INFO_JSON.write_text(json.dumps(info, indent=1))
    if verbose:
        print(f"saved model -> {MODEL_DIR}  |  {len(out)} test rows -> {FORECAST_CSV}  |  {seconds:.0f}s on {acc}")
    return out


# ------------------------------------------------------------ live use ------
def load_patchtst():
    """Load the saved model (or None if it was not trained yet)."""
    if not any(MODEL_DIR.glob("configuration.*")):
        return None
    from neuralforecast import NeuralForecast
    return NeuralForecast.load(path=str(MODEL_DIR), allow_pickle=True)


def training_info() -> dict:
    return json.loads(INFO_JSON.read_text()) if INFO_JSON.exists() else {}


def forecast_patchtst(nf, history: pd.DataFrame) -> np.ndarray:
    """history: hourly price/solar ending at 23:00 the day before -> (24, 2, 3) forecast."""
    long = history[TARGETS].iloc[-INPUT_HOURS:].reset_index(names="ds").melt(
        id_vars="ds", var_name="unique_id", value_name="y")
    pred = nf.predict(df=long[["unique_id", "ds", "y"]]).reset_index()
    pred = _tidy(pred.rename(columns={"unique_id": "target", **COLS}))
    out = np.empty((HORIZON, len(TARGETS), 3))
    for t, name in enumerate(TARGETS):
        out[:, t, :] = pred[pred.target == name].sort_values("ds")[["lo", "mid", "hi"]].values
    return out


def load_patchtst_arrays(days: pd.DatetimeIndex) -> np.ndarray | None:
    """Saved test-period forecasts as (N_days, 24, 2, 3), or None."""
    if not FORECAST_CSV.exists():
        return None
    fc = pd.read_csv(FORECAST_CSV, parse_dates=["timestamp"])
    out = np.full((len(days), HORIZON, len(TARGETS), 3), np.nan)
    for t, name in enumerate(TARGETS):
        s = fc[fc.target == name].set_index("timestamp")[["lo", "mid", "hi"]]
        for d, day in enumerate(days):
            hrs = pd.date_range(day, periods=HORIZON, freq="1h")
            if hrs.isin(s.index).all():
                out[d, :, t, :] = s.loc[hrs].values
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=2000)
    a = p.parse_args()
    run_patchtst(load_dataset(), a.max_steps)
