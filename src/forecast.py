"""Forecast one day (24 hours) with any trained model - used by the Streamlit app."""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
import torch

from .build_dataset import load_dataset
from .config import HORIZON, INPUT_HOURS, MODELS_DIR, TARGETS
from .data_prep import load_scaler, prepare, scale, to_full_hourly, unscale_targets
from .evaluate import baseline_forecast, clip_solar, fit_baseline_band, raw_target_array
from .patchtst import MODEL_DIR as PATCHTST_DIR
from .patchtst import forecast_patchtst, load_patchtst, load_patchtst_arrays

MODEL_LABELS = {
    "cnn": "1D-CNN",
    "lstm": "LSTM",
    "patchtst": "PatchTST",
    "base_week": "Baseline: same hour last week",
    "base_day": "Baseline: same hour yesterday",
}

# What each model learned from (all use the SMARD data in data/processed/dataset_hourly.csv)
_SPLIT = "Train 2019–2024 · validation (early stopping) 2025 · test 2026"
MODEL_TRAINING_DATA = {
    "cnn": f"12 inputs: price, solar, wind on/offshore, load, residual load + 6 calendar features. {_SPLIT}. Apple GPU (MPS).",
    "lstm": f"12 inputs: price, solar, wind on/offshore, load, residual load + 6 calendar features. {_SPLIT}. Apple GPU (MPS).",
    "patchtst": f"Only the history of price and solar (univariate, one shared model). {_SPLIT}. Apple GPU (MPS).",
    "base_week": "No training: copies the same hour 7 days earlier. Low/high band from its errors in 2019–2024.",
    "base_day": "No training: copies the same hour 1 day earlier. Low/high band from its errors in 2019–2024.",
}


class ForecastService:
    def __init__(self):
        self.df = load_dataset()
        self.full = to_full_hourly(self.df)
        # extend by one day so "tomorrow" (after the last row) can be forecast
        nxt = pd.date_range(self.full.index.max() + pd.Timedelta(hours=1), periods=2 * HORIZON, freq="1h")
        self.full = pd.concat([self.full, pd.DataFrame(index=nxt, columns=self.full.columns, dtype=float)])
        self.scaler = load_scaler()
        self.values = scale(self.full, self.scaler)
        self.raw = raw_target_array(self.full)
        self.actuals = self._load_actuals()
        self._torch_models = {}

    def _load_actuals(self) -> pd.DataFrame:
        """Real price/solar straight from the raw files: the day-ahead auction already fixes
        today's (and after ~13:00 tomorrow's) prices, even if other columns are not published yet."""
        from .build_dataset import read_api_csv, to_local_hourly
        from .config import RAW_DIR
        cols = {}
        for name in TARGETS:
            path = RAW_DIR / f"{name}.csv"
            if path.exists():
                cols[name] = to_local_hourly(read_api_csv(path)) / (10.0 if name == "price" else 1.0)
        return pd.DataFrame(cols) if cols else self.full[TARGETS]

    # ---------------------------------------------------------------- info --
    def available_models(self) -> list[str]:
        names = [n for n in ("cnn", "lstm") if (MODELS_DIR / f"{n}.pt").exists()]
        if any(PATCHTST_DIR.glob("configuration.*")) or (MODELS_DIR / "patchtst_forecasts.csv").exists():
            names.append("patchtst")
        return names + ["base_week", "base_day"]

    def forecastable_days(self) -> pd.DatetimeIndex:
        """Midnights whose previous 168 hours are complete."""
        ok = ~np.isnan(self.values).any(axis=1)
        csum = np.concatenate([[0], np.cumsum(~ok)])
        idx = self.full.index
        pos = np.where((idx.hour == 0) & (np.arange(len(idx)) >= INPUT_HOURS))[0]
        good = pos[(csum[pos] - csum[pos - INPUT_HOURS]) == 0]
        return idx[good]

    def latest_day(self) -> pd.Timestamp:
        return self.forecastable_days().max()

    @cached_property
    def _baseline_offsets(self):
        _, _, _, w = prepare(self.df, fit=False)
        return {lag: fit_baseline_band(raw_target_array(to_full_hourly(self.df)), w["train"].starts, lag)
                for lag in (24, 168)}

    # ------------------------------------------------------------- forecast --
    def _torch_model(self, name):
        if name not in self._torch_models:
            from .train import load_model
            self._torch_models[name] = load_model(name)
        return self._torch_models[name]

    def forecast_array(self, day: pd.Timestamp, model: str) -> np.ndarray:
        """(24, 2, 3) forecast in real units."""
        i = self.full.index.get_loc(pd.Timestamp(day))
        if model in ("lstm", "cnn"):
            x = torch.from_numpy(self.values[i - INPUT_HOURS:i][None])
            with torch.no_grad():
                q = self._torch_model(model)(x).numpy()
            q = unscale_targets(q, self.scaler)[0]
        elif model == "patchtst":
            if "patchtst" not in self._torch_models:
                self._torch_models["patchtst"] = load_patchtst()
            nf = self._torch_models["patchtst"]
            if nf is not None:   # live forecast from the saved model - works for any day, incl. tomorrow
                q = forecast_patchtst(nf, self.full[TARGETS].iloc[:i])
            else:                # fallback: stored test-period forecasts
                q = load_patchtst_arrays(pd.DatetimeIndex([day]))[0]
                if np.isnan(q).all():
                    raise ValueError("PatchTST model not saved - run `python -m src.patchtst` first.")
        else:
            lag = 24 if model == "base_day" else 168
            q = baseline_forecast(self.raw, np.array([i]), lag, self._baseline_offsets[lag])[0]
        return clip_solar(q[None])[0]

    def forecast_day(self, day, model: str) -> pd.DataFrame:
        day = pd.Timestamp(day).normalize()
        q = self.forecast_array(day, model)
        hours = pd.date_range(day, periods=HORIZON, freq="1h")
        out = pd.DataFrame({"timestamp": hours})
        for t, name in enumerate(TARGETS):
            out[f"{name}_lo"], out[f"{name}_mid"], out[f"{name}_hi"] = q[:, t, 0], q[:, t, 1], q[:, t, 2]
            out[f"{name}_actual"] = self.actuals[name].reindex(hours).values
        return out
