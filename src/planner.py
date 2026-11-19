"""Step 5a - The optimizer: pick the cheapest hours from the forecast.

The maths is done here in plain Python - the LLM only explains the result.
Cost uses the wholesale price only (no taxes / grid fees).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Plan:
    task: str
    energy_kwh: float
    power_kw: float
    mode: str                       # "normal" (middle) or "safe" (high)
    hours_needed: int
    chosen: pd.DataFrame            # rows of the forecast that were selected
    est_cost_eur: float
    cost_low_eur: float
    cost_high_eur: float
    avg_price_ct: float
    window_avg_price_ct: float      # average (middle) price over all allowed hours
    savings_eur: float              # vs. charging in the first hours of the window
    feasible: bool = True
    notes: list[str] = field(default_factory=list)


def allowed_mask(hours: pd.Series, start_hour: int, end_hour: int) -> pd.Series:
    """Hours in [start, end). If start > end the window wraps (e.g. 22 -> 7)."""
    if start_hour == end_hour:
        return pd.Series(True, index=hours.index)
    if start_hour < end_hour:
        return (hours >= start_hour) & (hours < end_hour)
    return (hours >= start_hour) | (hours < end_hour)


def _energy_per_slot(n: int, energy_kwh: float, power_kw: float) -> list[float]:
    """Full power in every chosen hour except the last one, which gets the remainder."""
    full = [power_kw] * (n - 1)
    return full + [energy_kwh - power_kw * (n - 1)]


def make_plan(forecast: pd.DataFrame, task: str, energy_kwh: float, power_kw: float,
              start_hour: int = 0, end_hour: int = 7, mode: str = "normal") -> Plan:
    """forecast: one row per hour with columns timestamp, price_lo, price_mid, price_hi (ct/kWh)."""
    notes = []
    hours_needed = math.ceil(energy_kwh / power_kw - 1e-9)
    fc = forecast.copy()
    fc["hour"] = pd.to_datetime(fc["timestamp"]).dt.hour
    window = fc[allowed_mask(fc["hour"], start_hour, end_hour)]
    feasible = hours_needed <= len(window)
    if not feasible:
        notes.append(f"Only {len(window)} hours allowed but {hours_needed} needed - using all allowed hours.")
        hours_needed = len(window)

    key = "price_hi" if mode == "safe" else "price_mid"
    chosen = window.nsmallest(hours_needed, key).sort_values("timestamp")

    # put the partial (last) slot into the most expensive chosen hour -> cheaper total
    kwh = _energy_per_slot(hours_needed, min(energy_kwh, power_kw * hours_needed), power_kw)
    order = chosen[key].rank(method="first").astype(int).values - 1
    chosen = chosen.assign(energy_kwh=[sorted(kwh, reverse=True)[o] for o in order])

    def cost(col):  # ct/kWh * kWh / 100 = EUR
        return float((chosen[col] * chosen["energy_kwh"]).sum() / 100)

    naive = window.sort_values("timestamp").head(hours_needed)
    naive_cost = float((naive["price_mid"].values * sorted(kwh, reverse=True)).sum() / 100)
    est = cost("price_mid")
    if (window["price_lo"] < 0).any():
        notes.append("Some hours may have negative prices (you could be paid to use power).")

    return Plan(
        task=task, energy_kwh=energy_kwh, power_kw=power_kw, mode=mode, hours_needed=hours_needed,
        chosen=chosen, est_cost_eur=est, cost_low_eur=cost("price_lo"), cost_high_eur=cost("price_hi"),
        avg_price_ct=float(chosen["price_mid"].mean()) if len(chosen) else float("nan"),
        window_avg_price_ct=float(window["price_mid"].mean()) if len(window) else float("nan"),
        savings_eur=naive_cost - est, feasible=feasible, notes=notes,
    )


def plan_summary(plan: Plan) -> str:
    """Plain text summary used when no LLM API key is available."""
    hours = ", ".join(f"{t:%H:00}" for t in pd.to_datetime(plan.chosen["timestamp"]))
    band = plan.chosen["price_hi"] - plan.chosen["price_lo"]
    certainty = "fairly certain" if band.mean() < 4 else "quite uncertain"
    text = (
        f"Run the {plan.task} at {hours} ({plan.hours_needed} h at {plan.power_kw:g} kW for "
        f"{plan.energy_kwh:g} kWh). Expected wholesale cost about {plan.est_cost_eur:.2f} EUR "
        f"(range {plan.cost_low_eur:.2f}-{plan.cost_high_eur:.2f} EUR), average "
        f"{plan.avg_price_ct:.1f} ct/kWh vs {plan.window_avg_price_ct:.1f} ct/kWh over the allowed window. "
        f"That saves about {plan.savings_eur:.2f} EUR compared with starting right away. "
        f"The forecast is {certainty} for these hours (average band {band.mean():.1f} ct/kWh)."
    )
    return " ".join([text, *plan.notes])
