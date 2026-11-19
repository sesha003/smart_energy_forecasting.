"""Step 3 - Prepare the data for training.

* split by time: train 2019-2024, validation 2025, test 2026
* StandardScaler fitted on the training rows only
* sliding windows: 168 h x 12 features  ->  next 24 h of price and solar
  - training: every hour is a possible start (~50,000 examples)
  - validation / test: only windows starting at midnight ("forecast tomorrow")
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .config import FEATURES, HORIZON, INPUT_HOURS, MODELS_DIR, TARGETS, TRAIN_END, VAL_END

TARGET_IDX = [FEATURES.index(t) for t in TARGETS]
SCALER_PATH = MODELS_DIR / "scaler.joblib"


def to_full_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Re-insert dropped hours as NaN rows so that row position == hour offset."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    return df.reindex(full)


def split_periods(df: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """(first target hour, last target hour) for each part."""
    t_end, v_end = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
    return {
        "train": (df.index.min(), t_end),
        "val": (t_end + pd.Timedelta(hours=1), v_end),
        "test": (v_end + pd.Timedelta(hours=1), df.index.max()),
    }


def fit_scaler(df: pd.DataFrame, save: bool = True) -> StandardScaler:
    scaler = StandardScaler().fit(df.loc[:TRAIN_END, FEATURES].dropna().values)
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    return scaler


def load_scaler() -> StandardScaler:
    return joblib.load(SCALER_PATH)


def scale(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df[FEATURES].values).astype(np.float32)


def unscale_targets(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse-scale an array whose axis with size 2 at position `-1` (or -2 for quantiles)
    holds [price, solar]. Works for (..., 2) and (..., 2, 3)."""
    mean = scaler.mean_[TARGET_IDX]
    std = scaler.scale_[TARGET_IDX]
    if y.shape[-1] == len(TARGETS):
        return y * std + mean
    return y * std[:, None] + mean[:, None]


def window_starts(full: pd.DataFrame, values: np.ndarray, first: pd.Timestamp, last: pd.Timestamp,
                  midnight_only: bool) -> np.ndarray:
    """Row positions i where target = rows [i, i+24) lies inside [first, last]
    and the 168-hour input plus target contain no missing values."""
    idx = full.index
    ok_row = ~np.isnan(values).any(axis=1)
    # rolling "all rows ok" over INPUT_HOURS + HORIZON rows
    bad = (~ok_row).astype(np.int32)
    csum = np.concatenate([[0], np.cumsum(bad)])
    span = INPUT_HOURS + HORIZON
    starts = np.arange(INPUT_HOURS, len(idx) - HORIZON + 1)
    clean = (csum[starts + HORIZON] - csum[starts - INPUT_HOURS]) == 0
    in_period = (idx[starts] >= first) & (idx[starts + HORIZON - 1] <= last)
    keep = clean & in_period
    if midnight_only:
        keep &= idx[starts].hour == 0
    assert span > 0
    return starts[keep]


@dataclass
class Windows:
    starts: np.ndarray   # row positions of the first forecast hour
    index: pd.DatetimeIndex  # full hourly index (to map starts -> timestamps)

    def days(self) -> pd.DatetimeIndex:
        return self.index[self.starts]


class WindowDataset(Dataset):
    """Builds (input, target) pairs on the fly to save memory."""

    def __init__(self, values: np.ndarray, starts: np.ndarray):
        self.x = torch.from_numpy(np.nan_to_num(values))
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, k: int):
        i = int(self.starts[k])
        x = self.x[i - INPUT_HOURS:i]                 # (168, 12)
        y = self.x[i:i + HORIZON][:, TARGET_IDX]      # (24, 2)
        return x, y


def prepare(df: pd.DataFrame, fit: bool = True):
    """Everything step 3 produces: full frame, scaled values, scaler and windows per split."""
    full = to_full_hourly(df)
    scaler = fit_scaler(full) if fit else load_scaler()
    values = scale(full, scaler)
    periods = split_periods(full)
    windows = {
        name: Windows(window_starts(full, values, a, b, midnight_only=(name != "train")), full.index)
        for name, (a, b) in periods.items()
    }
    return full, values, scaler, windows
