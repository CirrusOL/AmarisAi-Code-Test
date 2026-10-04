"""
LLM-written nutritional insights via the Groq API.

Design: pandas does all the arithmetic. This module distills the (filtered)
drink and food data into a compact "facts" dictionary - top items, averages,
ratios - and asks the model only to *summarize and explain* those facts. The
model never sees the raw tables, so it can't invent numbers from rows it
wasn't given, and requests stay small.

Setup:
    pip install groq
    PowerShell:  $env:GROQ_API_KEY = "your_key"      (this session only)
    bash/zsh:    export GROQ_API_KEY="your_key"
Optional: set GROQ_MODEL to override the default model.

If the SDK, key or network is unavailable, `offline_summary()` produces a
plain, deterministic summary from the same facts, so the feature degrades
gracefully instead of crashing.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

# llama-3.3-70b-versatile was retired by Groq on 2026-08-16; this is the replacement
# Groq recommends. Override with the GROQ_MODEL environment variable if needed.
DEFAULT_MODEL = "openai/gpt-oss-120b"
MIN_MACRO_G_FOR_DENSITY = 10   # ignore near-zero-macro items in the density proxy

SYSTEM_PROMPT = """You are a nutrition data analyst writing a brief, plain-English \
summary of Starbucks menu nutrition data for a general reader.

Rules:
- Use ONLY the figures in the JSON the user provides. Never use outside knowledge \
about specific menu items, and never invent, estimate or round-trip numbers; quote \
values exactly as given.
- The data has NO sugar and NO caffeine columns. If asked about sugar, say it is \
not available and use carbohydrates (carbs_g) as the closest proxy, stating clearly \
that carbs include sugar but are not the same thing.
- All values are per menu item. Serving size/weight is not in the data, so true \
calories-per-gram "calorie density" cannot be computed. "kcal_per_macro_g" is only \
a proxy (calories per gram of fat+carbs+protein; higher means fattier). Say so if \
you use it.
- Items without nutrition data are excluded; mention the counts if relevant.
- Drinks and food are different categories; do not imply a drink and a food item \
are equivalent portions.
- Be concise (about 150-250 words). Use short headed sections or bullets. End with \
one sentence of takeaways. Do not give medical advice."""


class InsightsError(RuntimeError):
    """Raised when the LLM summary can't be produced (caller may fall back)."""


# --------------------------------------------------------------------------- #
# Facts (all numbers computed here, not by the model)
# --------------------------------------------------------------------------- #
def _num(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    v = float(v)
    return int(v) if v.is_integer() else round(v, 2)


def _top(df: pd.DataFrame, col: str, n: int, ascending: bool = False) -> list[dict]:
    """Top-n items by `col` with a few context columns."""
    if col not in df.columns:
        return []
    ranked = df.dropna(subset=[col]).sort_values(
        [col, "item"], ascending=[ascending, True]).head(n)
    keep = [c for c in ("item", col, "calories", "fat_g", "carbs_g", "protein_g")
            if c in ranked.columns]
    keep = list(dict.fromkeys(keep))
    return [{k: (_num(v) if k != "item" else str(v)) for k, v in row.items()}
            for row in ranked[keep].to_dict("records")]


def dataset_facts(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Summarize one dataset into LLM-ready facts."""
    valid = df.dropna(subset=["calories"]).copy()
    facts: dict[str, Any] = {"items_total": int(len(df)),
                             "items_with_nutrition_data": int(len(valid))}
    if valid.empty:
        return facts

    macro_g = valid[["fat_g", "carbs_g", "protein_g"]].sum(axis=1)
    valid["kcal_per_macro_g"] = (valid["calories"] / macro_g).where(
        macro_g >= MIN_MACRO_G_FOR_DENSITY)

    cols = [c for c in ("calories", "fat_g", "carbs_g", "net_carbs_g",
                        "fiber_g", "protein_g", "sodium_mg") if c in valid.columns]
    protein_total = valid["protein_g"].sum()
    facts.update({
        "averages_per_item": {c: _num(valid[c].mean()) for c in cols},
        "total_calories_all_items": _num(valid["calories"].sum()),
        "fat_to_protein_ratio_total": (_num(valid["fat_g"].sum() / protein_total)
                                       if protein_total > 0 else None),
        "top_by_calories": _top(valid, "calories", top_n),
        "top_by_carbs_g": _top(valid, "carbs_g", top_n),
        "top_by_fat_g": _top(valid, "fat_g", top_n),
        "top_by_protein_g": _top(valid, "protein_g", top_n),
        "top_by_kcal_per_macro_g_proxy": _top(valid, "kcal_per_macro_g", top_n),
        "lowest_calorie_items": _top(valid, "calories", 3, ascending=True),
    })
    return facts


def build_facts(drinks: pd.DataFrame, foods: pd.DataFrame, top_n: int = 5,
                filter_note: str | None = None) -> dict:
    """Facts for both datasets plus a cross-dataset comparison."""
    facts = {
        "units": "calories in kcal, fat/carbs/fiber/protein in grams, sodium in mg; "
                 "all per menu item",
        "active_filters": filter_note or "none (full menu)",
        "drinks": dataset_facts(drinks, top_n),
        "food": dataset_facts(foods, top_n),
    }
    d = facts["drinks"].get("averages_per_item")
    f = facts["food"].get("averages_per_item")
    if d and f and d.get("calories"):
        facts["comparison"] = {
            "avg_calories_food_vs_drink_ratio": _num(f["calories"] / d["calories"]),
            "avg_calories_drink": d["calories"],
            "avg_calories_food": f["calories"],
        }
    return facts


# --------------------------------------------------------------------------- #
# Prompting + API call
# --------------------------------------------------------------------------- #
def build_messages(facts: dict, question: str | None = None) -> list[dict]:
    task = (question.strip() if question else
            "Summarize the key nutritional insights: which drinks and food items "
            "are highest in carbs (the sugar proxy), fat and calories, which are "
            "most calorie-dense, and how drinks and food compare overall.")
    user = (f"Task: {task}\n\nData (JSON):\n"
            f"{json.dumps(facts, ensure_ascii=False)}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def get_client():
    """Create a Groq client or raise InsightsError with a helpful message."""
    try:
        from groq import Groq
    except ImportError as err:
        raise InsightsError("The 'groq' package is not installed. "
                            "Run: pip install groq") from err
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise InsightsError(
            "GROQ_API_KEY is not set.\n"
            "  PowerShell:  $env:GROQ_API_KEY = \"your_key\"\n"
            "  bash/zsh:    export GROQ_API_KEY=\"your_key\"")
    return Groq(api_key=key)


def generate_insights(drinks: pd.DataFrame, foods: pd.DataFrame, *,
                      question: str | None = None, top_n: int = 5,
                      filter_note: str | None = None, model: str | None = None,
                      client=None, return_facts: bool = False):
    """Ask Groq for a summary of the data. Raises InsightsError on failure."""
    facts = build_facts(drinks, foods, top_n, filter_note)
    if not facts["drinks"].get("items_with_nutrition_data") and \
       not facts["food"].get("items_with_nutrition_data"):
        raise InsightsError("No items with nutrition data to summarize "
                            "(filters may be too strict).")

    client = client or get_client()
    model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    try:
        kwargs = {}
        if "gpt-oss" in model:
            # Reasoning model: keep thinking short so it doesn't eat the token budget.
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        response = client.chat.completions.create(
            messages=build_messages(facts, question),
            model=model,
            temperature=0.2,     # low: we want faithful summaries, not creativity
            max_tokens=1500,     # headroom for reasoning tokens + a ~250-word answer
            **kwargs,
        )
    except Exception as err:     # translate SDK errors into friendly messages
        raise InsightsError(_explain_api_error(err, model)) from err

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise InsightsError("The model returned an empty response.")
    return (text, facts) if return_facts else text


def _explain_api_error(err: Exception, model: str) -> str:
    try:
        import groq
    except ImportError:
        return f"API error: {err}"
    if isinstance(err, groq.AuthenticationError):
        return "Groq rejected the API key (401). Check GROQ_API_KEY, or create a new key."
    if isinstance(err, groq.RateLimitError):
        return "Groq rate limit reached (429). Wait a moment and try again."
    if isinstance(err, groq.NotFoundError):
        return (f"Model '{model}' was not found (it may have been retired). Set the "
                f"GROQ_MODEL environment variable, or --model on the command line, to a "
                f"model available to your account.")
    if isinstance(err, groq.BadRequestError) and "decommission" in str(err).lower():
        return (f"Model '{model}' has been decommissioned by Groq. Set the GROQ_MODEL "
                f"environment variable to a current model.")
    if isinstance(err, groq.APIConnectionError):
        return "Could not reach the Groq API. Check your internet connection."
    if isinstance(err, groq.APIStatusError):
        return f"Groq API error ({err.status_code}): {err.message}"
    return f"Unexpected error calling Groq: {err}"


# --------------------------------------------------------------------------- #
# Offline fallback (no LLM)
# --------------------------------------------------------------------------- #
def _fmt_items(rows: list[dict], col: str, unit: str) -> str:
    return "; ".join(f"{r['item']} ({r[col]:g}{unit})" for r in rows) or "n/a"


def offline_summary(facts: dict) -> str:
    """Plain deterministic summary built from the same facts (no API needed)."""
    lines = ["Offline summary (no LLM; generated directly from the data)"]
    if facts.get("active_filters") and not facts["active_filters"].startswith("none"):
        lines.append(f"Filters: {facts['active_filters']}")
    for key, label in (("drinks", "Drinks"), ("food", "Food")):
        f = facts[key]
        if not f.get("items_with_nutrition_data"):
            lines.append(f"\n{label}: no items with nutrition data.")
            continue
        avg = f["averages_per_item"]
        lines.append(f"\n{label} ({f['items_with_nutrition_data']} items with data, "
                     f"avg {avg['calories']:g} kcal)")
        lines.append(f"  Highest calories: {_fmt_items(f['top_by_calories'], 'calories', ' kcal')}")
        lines.append(f"  Highest carbs (sugar proxy): "
                     f"{_fmt_items(f['top_by_carbs_g'], 'carbs_g', ' g')}")
        lines.append(f"  Highest fat: {_fmt_items(f['top_by_fat_g'], 'fat_g', ' g')}")
        lines.append(f"  Highest kcal per macro gram (density proxy): "
                     f"{_fmt_items(f['top_by_kcal_per_macro_g_proxy'], 'kcal_per_macro_g', '')}")
    if "comparison" in facts:
        c = facts["comparison"]
        lines.append(f"\nOn average a food item has {c['avg_calories_food_vs_drink_ratio']:g}x "
                     f"the calories of a drink ({c['avg_calories_food']:g} vs "
                     f"{c['avg_calories_drink']:g} kcal).")
    return "\n".join(lines)
