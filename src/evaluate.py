"""Step 4 - Baselines and evaluation metrics.

Metrics per target (price in ct/kWh, solar in MWh):
    MAE / RMSE   error of the middle forecast
    Pinball      quality of the whole low/mid/high forecast
    Coverage     share of real values inside the low-high band (goal ~80 %)
    Band width   average distance between low and high
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZON, QUANTILES, TARGETS
from .data_prep import TARGET_IDX


# ------------------------------------------------------------ baselines -----
def baseline_point(raw_targets: np.ndarray, starts: np.ndarray, lag: int) -> np.ndarray:
    """Copy the same hours from `lag` hours earlier.  lag=24 -> yesterday, lag=168 -> last week."""
    return np.stack([raw_targets[i - lag:i - lag + HORIZON] for i in starts])   # (N, 24, 2)


def true_values(raw_targets: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.stack([raw_targets[i:i + HORIZON] for i in starts])               # (N, 24, 2)


def fit_baseline_band(raw_targets: np.ndarray, train_starts: np.ndarray, lag: int) -> np.ndarray:
    """Residual quantiles (per hour of day and target) on training days -> turns the
    point baseline into a low/mid/high forecast.  Returns offsets (24, 2, 3)."""
    starts = train_starts[(train_starts % 24) == (train_starts[0] % 24)]  # one window per day
    resid = true_values(raw_targets, starts) - baseline_point(raw_targets, starts, lag)
    return np.nanquantile(resid, QUANTILES, axis=0).transpose(1, 2, 0)


def baseline_forecast(raw_targets, starts, lag, offsets) -> np.ndarray:
    point = baseline_point(raw_targets, starts, lag)
    q = point[..., None] + offsets[None]
    q[..., 1] = point                       # middle = the plain baseline
    return np.sort(q, axis=-1)


# --------------------------------------------------------------- metrics ----
def pinball(y: np.ndarray, q: np.ndarray) -> float:
    qs = np.array(QUANTILES)
    err = y[..., None] - q
    return float(np.mean(np.maximum(qs * err, (qs - 1) * err)))


def metrics(y: np.ndarray, q: np.ndarray) -> dict:
    """y (N, 24, 2) and q (N, 24, 2, 3) in real units -> dict of metrics per target."""
    out = {}
    for t, name in enumerate(TARGETS):
        yt, qt = y[..., t], q[..., t, :]
        mid = qt[..., 1]
        out[name] = {
            "MAE": float(np.mean(np.abs(yt - mid))),
            "RMSE": float(np.sqrt(np.mean((yt - mid) ** 2))),
            "Pinball": pinball(yt, qt),
            "Coverage": float(np.mean((yt >= qt[..., 0]) & (yt <= qt[..., 2]))),
            "BandWidth": float(np.mean(qt[..., 2] - qt[..., 0])),
        }
    return out


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    """{model: metrics(...)} -> tidy table with one row per model and target."""
    rows = [{"model": m, "target": t, **v} for m, res in results.items() for t, v in res.items()]
    df = pd.DataFrame(rows).set_index(["target", "model"]).sort_index()
    df["Coverage"] = (df["Coverage"] * 100).round(1)
    return df.round(3)


def clip_solar(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    q[..., TARGETS.index("solar"), :] = np.clip(q[..., TARGETS.index("solar"), :], 0, None)
    return q


def raw_target_array(full: pd.DataFrame) -> np.ndarray:
    return full[TARGETS].values.astype(np.float64)


assert TARGET_IDX == [0, 1]
