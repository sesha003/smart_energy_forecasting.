# ⚡ Smart Energy Forecasting: Deep Learning Project Q&A & Technical Deep-Dive

---

## Table of Contents
1. [Project Overview & Objectives](#1-project-overview--objectives)
2. [The Dataset & Feature Engineering](#2-the-dataset--feature-engineering)
3. [Deep Learning Architectures (LSTM, 1D-CNN, PatchTST)](#3-deep-learning-architectures)
4. [Deep Dive: How PatchTST Works](#4-deep-dive-how-patchtst-works)
5. [Loss Functions: What is Pinball Loss & Why Not MSE/MAE?](#5-loss-functions-pinball-loss-vs-msemae)
6. [Why Deep Learning Instead of Classical ML (Random Forest, ARIMA, XGBoost)?](#6-why-deep-learning-instead-of-classical-ml)
7. [Evaluation Metrics & Benchmark Comparison](#7-evaluation-metrics--benchmark-comparison)
8. [Dashboard & Planning Assistant (How the Streamlit App Works)](#8-dashboard--planning-assistant)

---

## 1. Project Overview & Objectives

### What does this project do?
This project forecasts German electricity spot market prices (**ct/kWh**) and national solar energy generation (**MWh**) for the next **24 hours (day-ahead horizon)** using deep learning time-series models.

### Why is this important?
With the rise of dynamic electricity tariffs and renewable integration:
* Electricity prices fluctuate every hour (ranging from negative prices during surplus solar/wind hours to spikes during peak demand).
* Accurate probabilistic forecasts allow smart consumers, electric vehicle (EV) owners, and heat pump operators to schedule energy consumption during the cheapest and greenest hours.

### The 2 Prediction Targets
* **Wholesale Electricity Price** (`price`, ct/kWh): Direct economic signal for charging/heating schedules.
* **Solar Photovoltaic Generation** (`solar`, MWh): Major driver of the German energy supply curve.

---

## 2. The Dataset & Feature Engineering

### Data Source
* **Bundesnetzagentur | SMARD.de** (German Federal Network Agency, open data licensed under CC BY 4.0).
* Hourly data spanning from **January 1, 2019 to 2026** (>50,000 consecutive hourly timestamps).

### 12 Model Features (168-Hour Lookback)
Every prediction uses a history of **168 hours (7 days)** of 12 input features:

| # | Feature | Category | Description |
|---|---|---|---|
| 1 | `price` | Market | Hourly wholesale electricity price |
| 2 | `solar` | Renewable | Photovoltaic generation in Germany |
| 3 | `wind_onshore` | Renewable | Onshore wind generation |
| 4 | `wind_offshore` | Renewable | Offshore wind generation |
| 5 | `load` | Demand | Total power consumption across the grid |
| 6 | `residual_load` | Demand | Grid load minus solar and wind (power fossil fuels must provide) |
| 7 | `is_weekend` | Calendar | Binary flag (0 = weekday, 1 = weekend) |
| 8 | `is_holiday` | Calendar | Binary flag for German public holidays |
| 9 | `hour_sin` | Cyclical | $\sin(2\pi \cdot \text{hour} / 24)$ |
| 10 | `hour_cos` | Cyclical | $\cos(2\pi \cdot \text{hour} / 24)$ |
| 11 | `doy_sin` | Cyclical | $\sin(2\pi \cdot \text{day of year} / 365)$ |
| 12 | `doy_cos` | Cyclical | $\cos(2\pi \cdot \text{day of year} / 365)$ |

> **Why Cyclical Sine/Cosine?**
> A numerical hour `23` (11 PM) and `0` (midnight) are numerically far apart (distance = 23), but temporally adjacent (distance = 1 hour). Projecting time onto a unit circle using $\sin$ and $\cos$ preserves continuous temporal topology.

### Sequential Time Split (Strict Temporal Integrity)
* **Training Set:** 2019-01-01 to 2024-12-31 (~52,417 sliding windows)
* **Validation Set:** 2025-01-01 to 2025-12-31 (365 daily windows, used for model selection & early stopping)
* **Test Set:** 2026-01-01 to Present (321 daily windows, held-out final benchmark)

---

## 3. Deep Learning Architectures

All models map an input of shape `(batch, 168 hours, 12 features)` to an output of shape `(batch, 24 hours, 2 targets, 3 quantiles) = 144 outputs`.

### A. LSTM (Long Short-Term Memory)
* **Mechanism:** Recurrent neural network that consumes the 168 hours step-by-step. Hidden gates update internal memory states.
* **Architecture:** 2 LSTM layers (hidden size 128, dropout 0.1) followed by an MLP head.
* **Limitation:** In long sequences (168 steps), information from 7 days ago suffers from gradient decay and memory saturation.

### B. 1D-CNN (Dilated Causal Convolutions)
* **Mechanism:** 1D convolutions sliding across the time axis.
  * **Causal:** Zero-padding ensures the network only looks backward in time (no future leakage).
  * **Dilated:** Dilation factors ($2^0, 2^1, 2^2, 2^3, 2^4, 2^5 = 1, 2, 4, 8, 16, 32$) expand the receptive field exponentially to **190 hours** without increasing parameter count.
* **Advantages:** Computes representations across the entire week in parallel on GPU; captures multi-scale temporal patterns (hourly noise vs. daily trends).

### C. PatchTST (Patch Time-Series Transformer)
* **Mechanism:** State-of-the-art Transformer tailored for time-series forecasting. Treats non-overlapping 24-hour time chunks as "patches" (tokens).

---

## 4. Deep Dive: How PatchTST Works

### 1. Patching (Tokens for Time-Series)
Standard Transformers operating at the single-hour level suffer from high computational complexity ($O(L^2)$) and noise sensitivity. PatchTST groups hours into patches:
* **Lookback:** 168 hours
* **Patch Length:** 24 hours (1 full day)
* **Stride:** 24 hours (non-overlapping)
* **Result:** The 168-hour history is represented as **7 patch tokens** (Day -7 through Day -1).

### 2. Multi-Head Self-Attention
Self-attention computes dynamic weights answering: *"Which past day is most relevant to tomorrow's 24 hours?"*
* If tomorrow is Monday, attention assigns higher weights to Day -7 (last Monday).
* If weather conditions (cloud cover/wind) abruptly shifted yesterday, attention focuses on Day -1.

### 3. Native Quantile Estimation
Trained with Multi-Quantile Loss (`MQLoss([80])`) to directly predict the 10th, 50th, and 90th percentiles for every hour.

---

## 5. Loss Functions: Pinball Loss vs. MSE/MAE

### The Limitation of Standard Loss Functions (MSE / MAE)
1. **Symmetric Penalty:**
   * If actual price = 10 ct/kWh:
     * Predicting 15 ct (+5 error) yields $|10 - 15| = 5$.
     * Predicting 5 ct (-5 error) yields $|10 - 5| = 5$.
   * Because both errors are penalized equally, the model converges to the single conditional mean or median.
2. **Point Forecasts Lack Risk Information:**
   * **Calm Day:** Prediction = 10 ct/kWh. Reality will fall between 9 ct and 11 ct (low risk).
   * **Stormy Day:** Prediction = 10 ct/kWh. Reality could swing between -5 ct (wind surge) and 35 ct (grid shortage).
   * A single point prediction looks identical for both days, hiding critical operational risk.

### How Pinball Loss (Quantile Loss) Solves This
Pinball loss applies an asymmetric penalty dictated by quantile $q \in (0, 1)$:

$$\text{Loss}(q, y, \hat{y}) = \max\Big(q \cdot (y - \hat{y}), (q - 1) \cdot (y - \hat{y})\Big)$$

* **$q = 0.90$ (High Estimate / Ceiling):** If prediction is too low, penalty is $0.90 \times \text{error}$; if too high, penalty is only $0.10 \times \text{error}$. This pushes the prediction up so **90% of actual values stay below it**.
* **$q = 0.50$ (Mid Estimate / Median):** Symmetric penalty (standard MAE).
* **$q = 0.10$ (Low Estimate / Floor):** If prediction is too high, penalty is $0.90 \times \text{error}$. This pushes the prediction down so **only 10% of actual values fall below it**.

```
                         Hour 14:00 (Calm Day)        Hour 14:00 (Volatile Day)
High (90% quantile)  -->        10.8 ct/kWh                   28.5 ct/kWh
Mid  (50% quantile)  -->        10.0 ct/kWh                   10.0 ct/kWh
Low  (10% quantile)  -->         9.2 ct/kWh                    2.1 ct/kWh
                         [Narrow Band: Confident]      [Wide Band: High Risk]
```

### Quantile Crossing Prevention
In `src/models.py`, monotonic consistency is strictly guaranteed by sorting outputs along the quantile dimension:
```python
return torch.sort(out, dim=-1).values  # Guarantees low <= mid <= high
```

---

## 6. Why Deep Learning Instead of Classical ML?

### Myth: "This is a small amount of data"
* The dataset contains **>52,000 consecutive hours** (6 years).
* Sliding window generation yields **52,417 training sequences**.
* With 168 time steps and 12 features per window:
  $$52,417 \times 168 \times 12 \approx \mathbf{105{,}672{,}000 \text{ data points}}$$
  This volume is well-suited for deep neural networks.

### Classical ML Bottlenecks on this Task:
1. **Direct Multi-Horizon Forecasting (24 Hours):**
   * Classical models (ARIMA, standard XGBoost) predict one step ($t+1$).
   * To predict 24 hours, they must feed predicted values back as inputs iteratively. Prediction errors compound exponentially by step 15–24.
   * Deep Learning outputs all 24 hours simultaneously in a single forward pass.
2. **Simultaneous Quantile Outputs (144 numbers):**
   * Predicting 2 targets $\times$ 24 hours $\times$ 3 quantiles = **144 targets**.
   * Classical ML requires training 144 independent regressors, which cannot share representations and frequently suffer from quantile crossing ($q_{10} > q_{90}$).
3. **Loss of Temporal Structure in Tabularization:**
   * Flattening 168 hours $\times$ 12 features into tabular form yields **2,016 separate columns**.
   * Tree-based models treat column `price_lag_167` and `price_lag_168` as independent features, losing temporal continuity, periodicity, and time-equivariance.

---

## 7. Evaluation Metrics & Benchmark Comparison

### What the 5 Metrics Mean

1. **MAE (Mean Absolute Error):** Average magnitude of error in real units (lower is better).
2. **RMSE (Root Mean Squared Error):** Heavily penalizes large outlier errors (lower is better).
3. **Pinball Loss:** Statistical score evaluating the accuracy of the entire 3-quantile distribution (lower is better).
4. **Coverage (%):** Percentage of real values that fall between the Low (10%) and High (90%) bounds. **Ideal target is 80.0%**.
5. **BandWidth:** Average distance between High and Low bounds. Measures prediction uncertainty.

---

### Test Benchmark Results (Year 2026 Held-Out Data)

#### ⚡ Electricity Price (ct/kWh)

| Model | MAE ↓ | RMSE ↓ | Pinball ↓ | Coverage | BandWidth |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **PatchTST** | **2.588** | **4.007** | **0.844** | **80.4%** | 7.926 |
| 🥈 **1D-CNN** | 2.656 | 4.021 | 0.872 | 73.3% | **6.875** |
| 🥉 **LSTM** | 3.091 | 4.452 | 1.015 | 67.8% | 7.888 |
| Baseline (Yesterday) | 3.327 | 5.158 | 1.192 | 72.8% | 8.185 |
| Baseline (Last Week) | 4.461 | 6.789 | 1.599 | 71.0% | 10.601 |

#### ☀️ Solar Generation (MWh)

| Model | MAE ↓ | RMSE ↓ | Pinball ↓ | Coverage | BandWidth |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **PatchTST** | **2,222.8** | **4,214.5** | **722.8** | **81.5%** | 6,826.3 |
| 🥈 Baseline (Yesterday) | 2,254.2 | 4,644.0 | 803.9 | 57.6% | 4,619.8 |
| 🥉 **1D-CNN** | 2,461.4 | 4,517.5 | 855.6 | 64.6% | **4,614.7** |
| **LSTM** | 2,482.2 | 4,525.2 | 859.6 | 63.1% | 4,907.5 |
| Baseline (Last Week) | 3,462.1 | 6,642.4 | 1,191.7 | 53.5% | 6,919.8 |

---

### What These Numbers Suggest:

1. **PatchTST is the Clear Overall Winner:**
   * Achieves the lowest MAE on both Price (2.59 ct/kWh) and Solar (2,223 MWh).
   * **Flawlessly calibrated uncertainty:** Coverage is **80.4%** on Price and **81.5%** on Solar (almost exactly on the 80% theoretical target).
2. **1D-CNN Outperforms LSTM:**
   * CNN achieves 2.66 ct/kWh MAE (14% better than LSTM's 3.09 ct/kWh).
   * Demonstrates that dilated receptive fields capture long multi-day patterns better than recurrent recurrence on this dataset.
3. **The Solar Persistence Phenomenon:**
   * For solar generation, "Same hour yesterday" achieves a strong point MAE (2,254 MWh) due to weather autocorrelation (consecutive sunny/overcast days).
   * However, Baseline-Yesterday has poor coverage (**57.6%** vs. PatchTST's **81.5%**), failing to capture uncertainty when weather fronts move in.
4. **All Deep Learning Models Beat Historical Baselines on Price:**
   * Deep models reduce price error by **20% to 42%** compared to naive persistence, proving they learn supply/demand dynamics rather than memorizing historical values.

---

## 8. Dashboard & Planning Assistant

The Streamlit dashboard (`app.py`) serves two functions:

### 1. Interactive 24-Hour Forecasting Tab
* Visualizes the median prediction line alongside the shaded 10%–90% uncertainty ribbon.
* Plots actual realized market prices and solar generation in real time for accuracy verification.

### 2. Device Scheduling Optimizer (Planning Assistant)
Solves a constrained knapsack problem for electrical devices (EVs, heat pumps):
* **Inputs:** Energy required ($E$, kWh), Device max power ($P$, kW), Permitted operating window ($t_{\text{start}} \to t_{\text{end}}$).
* **Hours required:** $N = \lceil E / P \rceil$.
* **Optimization Modes:**
  * **Normal Mode:** Minimizes cost using $q_{50}$ (expected median price).
  * **Safe Mode:** Minimizes cost using $q_{90}$ (upper bound price), hedging against price-spike risks.
* **Economic Output:** Calculates total charging cost, average tariff rate, and net monetary savings compared to naive immediate charging.
