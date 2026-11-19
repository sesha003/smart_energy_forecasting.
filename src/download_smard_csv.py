"""Download the ORIGINAL SMARD files - exactly as the website's download center gives them.

https://www.smard.de/home/downloadcenter/download-marktdaten/
Settings: resolution Hour, format CSV, 01.01.2019 - today, one file per category and year.

The files are NOT cleaned: ';' separator, German numbers (1.234,56), '-' for missing values,
German column names, clock-change hours as published.  Use them to see what the raw data looks like
(build_dataset.read_smard_download_csv() can read them).

Also saves one example week of the raw JSON that the SMARD chart API returns
(this is what src/download_data.py downloads before writing data/raw/<name>.csv).

Output:
    data/raw/smard_download_center/<category>/<category>_<year>.csv
    data/raw/smard_api_json/<series>_example_week.json

Usage:
    python -m src.download_smard_csv
    python -m src.download_smard_csv --categories wholesale_prices --start-year 2024
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import requests

from .config import RAW_DIR, SMARD_SERIES

URL = "https://www.smard.de/nip-download-manager/nip/download/market-data"
OUT_DIR = RAW_DIR / "smard_download_center"
JSON_DIR = RAW_DIR / "smard_api_json"

# download-center categories -> (region, module ids = the columns of the file)
CATEGORIES = {
    "wholesale_prices": ("DE-LU", [
        8004169,  # Deutschland/Luxemburg
        8005078,  # average of neighbours DE/LU
        8000251,  # DE/AT/LU (until Sep 2018)
        8004996, 8000252, 8000253, 8000254, 8000255, 8000256, 8004997,  # BE, DK1, DK2, FR, IT-N, NL, NO2
        8004170, 8000257, 8000258, 8000259, 8000260, 8000261, 8000262,  # AT, PL, SE4, CH, SI, CZ, HU
    ]),
    "actual_generation": ("DE", [
        1004066, 1001226, 1001225, 1004067, 1004068, 1001228,  # biomass, hydro, wind off, wind on, PV, other RE
        1001224, 1001223, 1004069, 1004071, 1004070, 1001227,  # nuclear, lignite, hard coal, gas, pumped st., other conv.
    ]),
    "forecasted_generation_day_ahead": ("DE", [2000122, 2005097, 2003791, 2000123, 2000125, 2000715]),
    "actual_consumption": ("DE", [5000410, 5004387, 5004359]),         # grid load, pumped storage, residual load
    "forecasted_consumption": ("DE", [6000411, 6004362]),             # grid load, residual load
}


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def download_category_year(category: str, year: int, resolution: str = "hour") -> str:
    region, ids = CATEGORIES[category]
    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin")
    end = min(pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Berlin"),
              pd.Timestamp.now(tz="Europe/Berlin").normalize() + pd.Timedelta(days=2))
    body = {"request_form": [{
        "format": "CSV", "moduleIds": ids, "region": region, "type": "discrete", "language": "de",
        "timestamp_from": _ms(start), "timestamp_to": _ms(end),                     # end is exclusive
        "resolution": resolution,
    }]}
    r = requests.post(URL, json=body, timeout=180)
    r.raise_for_status()
    return r.content.decode("utf-8-sig")


def download_all_csv(categories=None, start_year: int = 2019) -> None:
    this_year = pd.Timestamp.now().year
    for cat in categories or CATEGORIES:
        folder = OUT_DIR / cat
        folder.mkdir(parents=True, exist_ok=True)
        for year in range(start_year, this_year + 1):
            text = download_category_year(cat, year)
            path = folder / f"{cat}_{year}.csv"
            path.write_text(text, encoding="utf-8")
            n = len(text.splitlines()) - 1
            print(f"{cat:34s} {year}  {n:5d} rows  {len(text) / 1e6:5.2f} MB  -> {path.relative_to(RAW_DIR.parent)}")


def save_api_json_examples() -> None:
    """One week of the raw chart-API JSON per series (the format download_data.py receives)."""
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    base = "https://www.smard.de/app/chart_data"
    for name, (fid, region, desc) in SMARD_SERIES.items():
        index = requests.get(f"{base}/{fid}/{region}/index_hour.json", timeout=30).json()["timestamps"]
        ts = index[-2]  # last complete week
        data = requests.get(f"{base}/{fid}/{region}/{fid}_{region}_hour_{ts}.json", timeout=30).json()
        data["_note"] = f"{desc}. series = [timestamp in ms since 1970 (UTC), value]"
        (JSON_DIR / f"{name}_example_week.json").write_text(json.dumps(data, indent=1))
    print(f"saved {len(SMARD_SERIES)} example JSON files -> {JSON_DIR.relative_to(RAW_DIR.parent)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="*", choices=list(CATEGORIES), help="default: all")
    p.add_argument("--start-year", type=int, default=2019)
    a = p.parse_args()
    download_all_csv(a.categories, a.start_year)
    save_api_json_examples()
