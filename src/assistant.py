"""Step 5b - The LLM explains the plan (it never calculates it).

Needs an Anthropic API key in ANTHROPIC_API_KEY or in Streamlit secrets.
Without a key - or if the call fails - the plain text summary from planner.py is used.
"""
from __future__ import annotations

import os

import pandas as pd

from .planner import Plan, plan_summary

MODEL = "claude-opus-5-5"

SYSTEM = (
    "You are an energy planning assistant for a household in Germany with a dynamic electricity tariff. "
    "An optimizer has already chosen the cheapest hours. Explain its plan to the user in 3-4 short, "
    "friendly sentences of plain language. Use only the numbers given - do not recalculate anything. "
    "Say how certain the forecast is (based on the low-high band) and mention that the cost is the "
    "wholesale price only, without taxes and grid fees."
)


def _plan_facts(plan: Plan) -> str:
    rows = "\n".join(
        f"- {pd.Timestamp(r.timestamp):%H:00}: {r.energy_kwh:.1f} kWh, price low/mid/high "
        f"{r.price_lo:.1f}/{r.price_mid:.1f}/{r.price_hi:.1f} ct/kWh"
        for r in plan.chosen.itertuples()
    )
    return (
        f"Task: {plan.task}, energy {plan.energy_kwh} kWh, device power {plan.power_kw} kW, "
        f"mode: {plan.mode} ({'uses the high forecast to be safe' if plan.mode == 'safe' else 'uses the middle forecast'}).\n"
        f"Chosen hours:\n{rows}\n"
        f"Estimated cost: {plan.est_cost_eur:.2f} EUR (range {plan.cost_low_eur:.2f}-{plan.cost_high_eur:.2f} EUR).\n"
        f"Average price of chosen hours: {plan.avg_price_ct:.1f} ct/kWh; average over all allowed hours: "
        f"{plan.window_avg_price_ct:.1f} ct/kWh. Saving vs starting immediately: {plan.savings_eur:.2f} EUR.\n"
        f"Notes: {' '.join(plan.notes) or 'none'}"
    )


def explain_plan(plan: Plan, api_key: str | None = None) -> tuple[str, str]:
    """Returns (text, source) where source is 'llm' or 'summary'."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return plan_summary(plan), "summary"
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": _plan_facts(plan)}],
        )
        if response.stop_reason == "refusal":
            return plan_summary(plan), "summary"
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return (text, "llm") if text else (plan_summary(plan), "summary")
    except Exception as exc:  # no network, bad key, rate limit ... -> fall back, app keeps working
        return plan_summary(plan) + f"\n\n(LLM unavailable: {type(exc).__name__})", "summary"
