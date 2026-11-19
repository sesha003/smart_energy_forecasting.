"""Smart Energy Forecasting - Streamlit app.

Run:  streamlit run app.py      (opens http://localhost:8501)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import DATASET_CSV, MODELS_DIR, VAL_END
from src.evaluate import metrics
from src.forecast import MODEL_LABELS, ForecastService

# Fallback or imported training descriptions
try:
    from src.forecast import MODEL_TRAINING_DATA
except ImportError:
    _SPLIT = "Train 2019–2024 · validation (early stopping) 2025 · test 2026"
    MODEL_TRAINING_DATA = {
        "cnn": f"12 inputs: price, solar, wind on/offshore, load, residual load + 6 calendar features. {_SPLIT}. Apple GPU (MPS).",
        "lstm": f"12 inputs: price, solar, wind on/offshore, load, residual load + 6 calendar features. {_SPLIT}. Apple GPU (MPS).",
        "patchtst": f"Only the history of price and solar (univariate, one shared model). {_SPLIT}. Apple GPU (MPS).",
        "base_week": "No training: copies the same hour 7 days earlier. Low/high band from its errors in 2019–2024.",
        "base_day": "No training: copies the same hour 1 day earlier. Low/high band from its errors in 2019–2024.",
    }

st.set_page_config(page_title="Smart Energy Forecasting", page_icon="⚡", layout="wide")


# ------------------------------------------------------------------ setup ---
@st.cache_resource(show_spinner="Loading data and models ...")
def get_service() -> ForecastService:
    return ForecastService()


def api_key() -> str | None:
    try:
        return st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets.toml
        return os.environ.get("ANTHROPIC_API_KEY")


if not DATASET_CSV.exists() or not (MODELS_DIR / "scaler.joblib").exists():
    st.error("No dataset / trained models yet. Run:\n\n"
             "`python -m src.download_data`  →  `python -m src.build_dataset`  →  `python -m src.train`")
    st.stop()

svc = get_service()

# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("⚡ Smart Energy")
mode = st.sidebar.radio("Mode", ["Demo (test period)", "Real (tomorrow)"],
                        help="Demo: pick a past 2026 day and compare forecast vs reality. "
                             "Real: download the newest SMARD data and forecast the next day.")
models = svc.available_models()
model = st.sidebar.selectbox("Model", models, format_func=lambda m: MODEL_LABELS[m])
if model in MODEL_TRAINING_DATA:
    st.sidebar.caption(f"**Trained on:** {MODEL_TRAINING_DATA[model]}")

days = svc.forecastable_days()
if mode.startswith("Demo"):
    test_days = days[(days > pd.Timestamp(VAL_END)) & (days + pd.Timedelta(hours=23) <= svc.df.index.max())]
    day = st.sidebar.date_input("Day to forecast", value=test_days[-1].date(),
                                min_value=test_days[0].date(), max_value=test_days[-1].date())
else:
    if st.sidebar.button("⬇️ Download newest SMARD data", width="stretch"):
        from src.build_dataset import build_dataset
        from src.download_data import download_all
        import requests as _requests
        with st.spinner("Downloading from SMARD and rebuilding the dataset ..."):
            try:
                download_all(verbose=False)
                build_dataset(verbose=False)
                get_service.clear()
                st.rerun()
            except _requests.exceptions.ConnectionError as _ce:
                st.sidebar.error(
                    "❌ **No internet access.**\n\n"
                    "Could not reach `www.smard.de`. Please check your network "
                    "connection and try again.\n\n"
                    "In the meantime, use **Demo (test period)** mode which works "
                    "fully offline with the pre-downloaded dataset."
                )
    day = svc.latest_day().date()
    st.sidebar.info(f"Data until **{svc.df.index.max():%d.%m.%Y %H:%M}**.\n\n"
                    f"Forecasting **{day:%A %d.%m.%Y}** (first day after the last complete day).")

st.sidebar.caption("Data: Bundesnetzagentur | SMARD.de (CC BY 4.0)")

try:
    fc = svc.forecast_day(day, model)
except Exception as exc:
    st.error(f"Cannot forecast {day} with {MODEL_LABELS[model]}: {exc}")
    st.stop()

has_actual = fc["price_actual"].notna().all() and fc["solar_actual"].notna().all()


# ---------------------------------------------------------------- helpers ---
def band_chart(df: pd.DataFrame, target: str, unit: str, color: str, highlight=None) -> go.Figure:
    x = df["timestamp"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df[f"{target}_hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=df[f"{target}_lo"], fill="tonexty", line=dict(width=0),
                             fillcolor=color.replace("rgb", "rgba").replace(")", ",0.18)"),
                             name="Low–high (10–90 %)"))
    fig.add_trace(go.Scatter(x=x, y=df[f"{target}_mid"], line=dict(color=color, width=3), name="Forecast (middle)"))
    if df[f"{target}_actual"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=df[f"{target}_actual"], line=dict(color="gray", dash="dot", width=2),
                                 name="Actual"))
    if highlight is not None and len(highlight):
        fig.add_trace(go.Scatter(x=highlight["timestamp"], y=highlight[f"{target}_mid"], mode="markers",
                                 marker=dict(size=13, color="#2ca02c", symbol="star"), name="Chosen hours"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=unit,
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    fig.update_xaxes(tickformat="%H:%M")
    return fig


# ------------------------------------------------------------------- page ---
st.title("Smart Energy Forecasting")
st.caption(f"{MODEL_LABELS[model]} · forecast for **{pd.Timestamp(day):%A, %d %B %Y}** · "
           "German/Luxembourg day-ahead market")

tab_fc, tab_plan, tab_cmp = st.tabs(["📈 Forecast", "🤖 Planning assistant", "📊 Model comparison"])

with tab_fc:
    c1, c2, c3, c4 = st.columns(4)
    cheapest = fc.loc[fc["price_mid"].idxmin()]
    priciest = fc.loc[fc["price_mid"].idxmax()]
    c1.metric("Average price (forecast)", f"{fc['price_mid'].mean():.1f} ct/kWh")
    c2.metric("Cheapest hour", f"{cheapest.timestamp:%H:00}", f"{cheapest.price_mid:.1f} ct/kWh", delta_color="off")
    c3.metric("Most expensive hour", f"{priciest.timestamp:%H:00}", f"{priciest.price_mid:.1f} ct/kWh", delta_color="off")
    c4.metric("Solar energy (forecast)", f"{fc['solar_mid'].sum() / 1000:.0f} GWh")

    left, right = st.columns(2)
    left.subheader("Electricity price")
    left.plotly_chart(band_chart(fc, "price", "ct/kWh", "rgb(31,119,180)"), width="stretch")
    right.subheader("Solar generation")
    right.plotly_chart(band_chart(fc, "solar", "MWh", "rgb(255,127,14)"), width="stretch")

    if has_actual:
        y = fc[["price_actual", "solar_actual"]].values[None]
        q = np.stack([fc[[f"{t}_lo", f"{t}_mid", f"{t}_hi"]].values for t in ("price", "solar")], axis=1)[None]
        m = metrics(y, q)
        st.markdown(
            f"**How good was this forecast?**  Price MAE **{m['price']['MAE']:.2f} ct/kWh**, "
            f"{m['price']['Coverage'] * 100:.0f} % of hours inside the band · "
            f"Solar MAE **{m['solar']['MAE']:,.0f} MWh**, {m['solar']['Coverage'] * 100:.0f} % inside the band."
        )
    with st.expander("Forecast table"):
        st.dataframe(fc.set_index("timestamp").round(2), width="stretch")

with tab_plan:
    from src.assistant import explain_plan
    from src.planner import make_plan

    st.subheader("When should I run my device?")
    a, b, c = st.columns(3)
    task = a.selectbox("Task", ["Electric car", "Heat pump"])
    defaults = {"Electric car": (20.0, 11.0, 0, 7), "Heat pump": (12.0, 3.0, 0, 24)}[task]
    energy = a.number_input("Energy needed (kWh)", 0.5, 200.0, defaults[0], 0.5, key=f"e_{task}")
    power = b.number_input("Device power (kW)", 0.5, 50.0, defaults[1], 0.5, key=f"p_{task}")
    start_h, end_h = b.select_slider("Allowed hours (start → end)", options=list(range(25)),
                                     value=(defaults[2], defaults[3]), key=f"w_{task}")
    plan_mode = c.radio("Mode", ["normal", "safe"], horizontal=True,
                        help="normal: uses the middle forecast · safe: uses the high forecast (avoids risky hours)")
    c.caption(f"Hours needed: **{int(np.ceil(energy / power - 1e-9))}** = ⌈{energy:g} ÷ {power:g}⌉")

    if st.button("Find the cheapest hours", type="primary"):
        plan = make_plan(fc, task.lower(), energy, power, start_h, end_h % 24 if end_h < 24 else 0, plan_mode)
        k1, k2, k3 = st.columns(3)
        k1.metric("Estimated cost", f"{plan.est_cost_eur:.2f} €",
                  f"range {plan.cost_low_eur:.2f}–{plan.cost_high_eur:.2f} €", delta_color="off")
        k2.metric("Avg. price of chosen hours", f"{plan.avg_price_ct:.1f} ct/kWh",
                  f"{plan.avg_price_ct - plan.window_avg_price_ct:+.1f} vs window average", delta_color="inverse")
        k3.metric("Saving vs. starting right away", f"{plan.savings_eur:.2f} €")
        st.plotly_chart(band_chart(fc, "price", "ct/kWh", "rgb(31,119,180)", highlight=plan.chosen),
                        width="stretch")
        st.dataframe(plan.chosen[["timestamp", "energy_kwh", "price_lo", "price_mid", "price_hi"]]
                     .rename(columns={"energy_kwh": "kWh", "price_lo": "low", "price_mid": "middle",
                                      "price_hi": "high"}).set_index("timestamp").round(2),
                     width="stretch")
        with st.spinner("Assistant is writing the explanation ..."):
            text, source = explain_plan(plan, api_key())
        st.markdown("#### 🤖 Assistant")
        st.info(text)
        if source == "summary":
            st.caption("Simple summary (add ANTHROPIC_API_KEY to .streamlit/secrets.toml for the AI explanation).")
    st.caption("Cost is based on the wholesale price only. Households pay taxes and grid fees on top; "
               "only dynamic tariffs pass on the hourly price changes.")

with tab_cmp:
    st.subheader("Model comparison")
    for split, label in (("val", "Validation 2025 (used to choose the model)"),
                         ("test", "Test 2026 (final check, used once)")):
        path = MODELS_DIR / f"results_{split}.csv"
        if path.exists():
            st.markdown(f"**{label}**")
            st.dataframe(pd.read_csv(path).set_index(["target", "model"]), width="stretch")
    if not (MODELS_DIR / "results_val.csv").exists():
        st.info("Run `python -m src.compare` (or the notebook) to create the comparison tables.")
    st.caption("MAE/RMSE: error of the middle forecast · Pinball: quality of the whole range · "
               "Coverage: % of real values inside the low–high band (goal ≈ 80 %) · BandWidth: low–high distance.")
