"""Step 2 - Turn the raw SMARD files into one clean hourly table: dataset_hourly.csv

* timestamp = start of the hour, German local time (Europe/Berlin)
* clock change: October duplicate hour dropped, March missing hour filled
* quarter-hourly values (prices since Oct 2025) averaged to hourly
* price converted from EUR/MWh to ct/kWh (divide by 10)
* gaps up to 3 hours interpolated, still-incomplete rows dropped
* calendar features: hour, weekday, is_weekend, is_holiday, hour/day-of-year sine-cosine

Usage:
    python -m src.build_dataset
"""
from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from .config import DATASET_CSV, MUST_HAVE, RAW_DIR, SMARD_SERIES, START_DATE, TIMEZONE


# ---------------------------------------------------------------- reading ---
def read_api_csv(path: Path) -> pd.Series:
    """Read a file written by download_data.py (timestamp_utc_ms, value)."""
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.timestamp_utc_ms, unit="ms", utc=True).dt.tz_convert(TIMEZONE)
    return pd.Series(df.value.values, index=ts, name=path.stem)


def read_smard_download_csv(path: Path) -> pd.DataFrame:
    """Read a CSV from the SMARD download center (manual download).

    SMARD uses ';' as separator, German numbers (1.234,56) and '-' for missing.
    Returns a DataFrame indexed by the local start-of-interval timestamp.
    """
    df = pd.read_csv(path, sep=";", decimal=",", thousands=".", na_values=["-"])
    date_col = df.columns[0]  # "Datum von" / "Start date"
    ts = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.drop(columns=[c for c in df.columns[:2]]).set_index(ts)
    df.index.name = "timestamp"
    return df.apply(pd.to_numeric, errors="coerce")


# ------------------------------------------------------------- cleaning -----
def to_local_hourly(s: pd.Series) -> pd.Series:
    """tz-aware series -> hourly local-time series with the clock change fixed."""
    s = s.sort_index()
    # quarter-hourly -> hourly mean (no-op for data that is already hourly)
    s = s.resample("1h").mean()
    local = s.copy()
    local.index = local.index.tz_localize(None)       # drop tz: local wall-clock time
    local = local[~local.index.duplicated(keep="first")]  # October: hour appears twice
    full = pd.date_range(local.index.min(), local.index.max(), freq="1h")
    return local.reindex(full)                          # March: missing hour -> NaN (filled later)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    de_holidays = holidays.Germany(years=range(idx.year.min(), idx.year.max() + 2))
    df["hour"] = idx.hour
    df["weekday"] = idx.day_of_week
    df["is_weekend"] = (idx.day_of_week >= 5).astype(int)
    df["is_holiday"] = np.array([d in de_holidays for d in idx.date]).astype(int)
    doy = idx.day_of_year - 1
    days_in_year = np.where(idx.is_leap_year, 366, 365)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / days_in_year)
    df["doy_cos"] = np.cos(2 * np.pi * doy / days_in_year)
    return df


def build_dataset(raw_dir: Path = RAW_DIR, out: Path = DATASET_CSV, verbose: bool = True) -> pd.DataFrame:
    series = {}
    for name in SMARD_SERIES:
        path = raw_dir / f"{name}.csv"
        if path.exists():
            series[name] = to_local_hourly(read_api_csv(path))
        elif name in MUST_HAVE:
            raise FileNotFoundError(f"{path} missing - run `python -m src.download_data` first")

    df = pd.DataFrame(series)
    df.index.name = "timestamp"
    df = df[df.index >= START_DATE]

    df["price"] = df["price"] / 10.0  # EUR/MWh -> ct/kWh

    # fill small gaps (up to 3 hours), then drop rows where a must-have column is still empty
    df = df.interpolate(limit=3, limit_area="inside")
    before = len(df)
    df = df.dropna(subset=MUST_HAVE)
    df = add_calendar_features(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out)
    if verbose:
        print(f"dataset: {len(df)} rows ({before - len(df)} incomplete dropped), "
              f"{df.index.min()} -> {df.index.max()}  saved to {out}")
    return df


def load_dataset(path: Path = DATASET_CSV) -> pd.DataFrame:
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)


if __name__ == "__main__":
    build_dataset()
