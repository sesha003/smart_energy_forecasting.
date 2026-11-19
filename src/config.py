"""Central settings for the Smart Energy Forecasting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DATASET_CSV = PROCESSED_DIR / "dataset_hourly.csv"

TIMEZONE = "Europe/Berlin"
START_DATE = "2019-01-01"

# SMARD chart-data series: short name -> (filter id, region, description)
# Source: Bundesnetzagentur | SMARD.de  (licence CC BY 4.0)
SMARD_SERIES = {
    # must-have
    "price":         (4169, "DE-LU", "Market > Wholesale price DE/LU (EUR/MWh)"),
    "solar":         (4068, "DE",    "Actual generation > Photovoltaics (MWh)"),
    "wind_onshore":  (4067, "DE",    "Actual generation > Wind onshore (MWh)"),
    "wind_offshore": (1225, "DE",    "Actual generation > Wind offshore (MWh)"),
    "load":          (410,  "DE",    "Actual consumption > Total grid load (MWh)"),
    "residual_load": (4359, "DE",    "Actual consumption > Residual load (MWh)"),
    # recommended (day-ahead forecasts: extra inputs + benchmark)
    "solar_fc":         (125,  "DE", "Forecasted generation day-ahead > Photovoltaics (MWh)"),
    "wind_onshore_fc":  (123,  "DE", "Forecasted generation day-ahead > Wind onshore (MWh)"),
    "wind_offshore_fc": (3791, "DE", "Forecasted generation day-ahead > Wind offshore (MWh)"),
    "load_fc":          (411,  "DE", "Forecasted consumption > Total grid load (MWh)"),
}
MUST_HAVE = ["price", "solar", "wind_onshore", "wind_offshore", "load", "residual_load"]

# Model inputs (12 columns) and targets
FEATURES = [
    "price", "solar", "wind_onshore", "wind_offshore", "load", "residual_load",
    "is_weekend", "is_holiday", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
TARGETS = ["price", "solar"]
QUANTILES = [0.1, 0.5, 0.9]

INPUT_HOURS = 168   # 7 days of history
HORIZON = 24        # forecast the next 24 hours

# Time-based split (never random)
TRAIN_END = "2024-12-31 23:00"
VAL_END = "2025-12-31 23:00"

# Training settings from the project guide
LR = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 5
SEED = 42
