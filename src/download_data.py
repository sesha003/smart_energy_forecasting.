"""Step 1 - Download hourly market data from SMARD (Bundesnetzagentur).

Uses the public SMARD chart-data API (the same data as the download center at
https://www.smard.de/home/downloadcenter/download-marktdaten/). No account needed.
Licence CC BY 4.0 - credit: "Bundesnetzagentur | SMARD.de".

Each series is saved to data/raw/<name>.csv with columns:
    timestamp_utc_ms, value

Usage:
    python -m src.download_data                 # all series
    python -m src.download_data --only price solar
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .config import RAW_DIR, SMARD_SERIES, START_DATE

BASE = "https://www.smard.de/app/chart_data"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "smart-energy-forecasting-student-project"


def _get_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
    return {}


def download_series(name: str, start: str = START_DATE, resolution: str = "hour") -> pd.DataFrame:
    """Download one SMARD series from `start` until today."""
    filter_id, region, _ = SMARD_SERIES[name]
    index = _get_json(f"{BASE}/{filter_id}/{region}/index_{resolution}.json")["timestamps"]
    start_ms = pd.Timestamp(start, tz="Europe/Berlin").value // 10**6
    # each chunk covers one week; keep the chunk that contains `start` and all later ones
    chunks = [t for t in index if t >= start_ms - 7 * 24 * 3600 * 1000]

    def fetch(ts: int):
        url = f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{ts}.json"
        return _get_json(url).get("series", [])

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [row for part in pool.map(fetch, chunks) for row in part]

    df = pd.DataFrame(rows, columns=["timestamp_utc_ms", "value"])
    df = df.dropna().drop_duplicates("timestamp_utc_ms").sort_values("timestamp_utc_ms")
    return df[df.timestamp_utc_ms >= start_ms].reset_index(drop=True)


def download_all(names: list[str] | None = None, start: str = START_DATE, verbose: bool = True) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in names or list(SMARD_SERIES):
        df = download_series(name, start)
        path = RAW_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        if verbose:
            last = pd.to_datetime(df.timestamp_utc_ms.iloc[-1], unit="ms", utc=True).tz_convert("Europe/Berlin")
            try:
                print(f"{name:18s} {len(df):7d} rows  last={last:%Y-%m-%d %H:%M}  -> {path.name}")
            except OSError:
                pass  # stdout not writable (e.g. inside Streamlit)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", help="series names to download (default: all)")
    p.add_argument("--start", default=START_DATE)
    a = p.parse_args()
    download_all(a.only, a.start)
