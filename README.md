# ⚡ Smart Energy Forecasting with an AI Planning Assistant

Forecast tomorrow's hourly **German electricity price** and **solar generation** (low / middle / high),
compare **baselines vs LSTM vs 1D-CNN vs PatchTST**, and let an **AI planning assistant** pick the
cheapest hours to charge an electric car or run a heat pump - all in a **Streamlit** app.

Data: Bundesnetzagentur | SMARD.de (CC BY 4.0), Germany/Luxembourg market area, hourly, 2019 → today.

## Folder structure

```
smart_energy_forecasting/
├── app.py                         # Streamlit app  (streamlit run app.py)
├── requirements.txt
├── README.md
├── .streamlit/
│   └── secrets.toml.example       # copy to secrets.toml and add ANTHROPIC_API_KEY (optional)
├── notebooks/
│   └── smart_energy_forecasting.ipynb   # full walkthrough: steps 1-6
├── src/
│   ├── config.py                  # paths, SMARD series ids, features, split dates, training settings
│   ├── download_data.py           # Step 1  download SMARD data          -> data/raw/*.csv
│   ├── download_smard_csv.py      # original download-center files (raw, German format) - for inspection
│   ├── build_dataset.py           # Step 2  clean + merge + calendar     -> data/processed/dataset_hourly.csv
│   ├── data_prep.py               # Step 3  time split, scaler, 168h -> 24h windows
│   ├── models.py                  # Step 4  pinball loss, LSTM, dilated 1D-CNN
│   ├── train.py                   # Step 4  training with early stopping  -> models/*.pt
│   ├── patchtst.py                # Step 4  PatchTST (neuralforecast)     -> models/patchtst/ + patchtst_forecasts.csv
│   ├── evaluate.py                # Step 4  baselines + MAE/RMSE/pinball/coverage/band width
│   ├── compare.py                 # Step 4  results tables               -> models/results_{val,test}.csv
│   ├── forecast.py                # forecast any day with any model (used by the app)
│   ├── planner.py                 # Step 5  optimizer: cheapest hours + cost
│   └── assistant.py               # Step 5  LLM explanation (falls back to plain summary)
├── data/
│   ├── raw/                       # downloaded SMARD series (one tidy CSV per series)
│   │   ├── smard_download_center/ # ORIGINAL website files: 5 categories x 2019-2026, ';' + German numbers
│   │   └── smard_api_json/        # one example week of the raw API JSON per series
│   └── processed/                 # dataset_hourly.csv
└── models/                        # scaler, trained weights, histories, result tables
```

## Setup (Python 3.10 – 3.12)

```bash
cd smart_energy_forecasting
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`neuralforecast` (PatchTST) is the heaviest package. If it gives you trouble, remove it from
`requirements.txt` - everything else works without it. On Google Colab only
`pip install holidays neuralforecast` is needed.

## Run the pipeline

```bash
python -m src.download_data     # Step 1 (~30 s, all series 2019 -> today)
python -m src.download_smard_csv   # optional: original download-center CSVs to inspect (~3 min)
python -m src.build_dataset     # Step 2 -> data/processed/dataset_hourly.csv
python -m src.train             # Step 4: LSTM + 1D-CNN (50 epochs max, early stopping)
python -m src.patchtst          # Step 4: PatchTST on the Apple GPU (~1.5 min), saved for live use
python -m src.compare           # Step 4: comparison tables on validation + test
streamlit run app.py            # Step 6: http://localhost:8501
```

Or open `notebooks/smart_energy_forecasting.ipynb` and run all cells (switches at the top control
downloading, training and PatchTST).

## The app

* **Sidebar** - Demo mode (any 2026 test day vs reality) or Real mode (download newest data, forecast the next day); model choice.
* **Forecast** - interactive price and solar charts with the 10–90 % band; dotted line = actual values.
* **Planning assistant** - car / heat pump, energy, power, allowed hours, normal / safe mode → cheapest hours, cost, saving, explanation.
* **Model comparison** - validation and test tables.

### AI explanation (optional)

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your key
```
The optimizer always does the maths; the LLM (Claude, `claude-opus-5-5`) only explains the plan.
Without a key the app shows a plain text summary. Never commit `secrets.toml`.

### Deploy for free
Push to GitHub → [Streamlit Community Cloud](https://streamlit.io/cloud) → New app → `app.py`;
add `ANTHROPIC_API_KEY` under *Settings → Secrets*. Commit `data/processed/dataset_hourly.csv`,
`data/raw/*.csv` and `models/` so the app starts without retraining.

## Method notes

**Training data and hardware** (all models: SMARD data in `data/processed/dataset_hourly.csv`)

| Model | Inputs | Train | Early stopping | Test | Hardware |
|---|---|---|---|---|---|
| LSTM, 1D-CNN | 12 columns: price, solar, wind on/offshore, load, residual load, weekend, holiday, hour & day-of-year sin/cos | 2019–2024 | 2025 | 2026 | Apple GPU (MPS) |
| PatchTST | price and solar history only (univariate, one shared model) | 2019–2024 | 2025 | 2026 | Apple GPU (MPS) |
| Baselines | no training (copy yesterday / last week; band from 2019–2024 errors) | – | – | 2026 | – |

PatchTST is saved to `models/patchtst/` (details in `training_info.json`), so the app forecasts live with it
for any day, including Real mode. The scripts pick the Apple GPU automatically (CUDA on Colab, else CPU).


* **Split by time:** train 2019–2024, validation 2025, test 2026 (scaler fitted on train only).
* **Windows:** 168 h × 12 inputs → 24 h × (price, solar) × (10 %, 50 %, 90 %) = 72 outputs.
* **Choosing the model:** lowest validation pinball loss. PatchTST uses 2025 for its own early stopping,
  so it only has test results - compare it on the test table.
* **Cost** is the wholesale price only; taxes and grid fees come on top.

## Limitations
* Day-ahead prices are fixed in an auction around noon - a real forecast must be made before that.
* At prediction time only weather forecasts exist; measured weather would overstate solar accuracy.
* Negative prices and the 2022 energy crisis cause larger errors.
* Since Oct 2025 prices are quarter-hourly; this project uses hourly averages.
* SMARD publishes generation/consumption a few hours late, so Real mode forecasts the first day after the last complete day.
