"""
Starbucks nutrition analysis.

Imports the loading/cleaning functions from starbucks_loader.py and then:
  1. prints descriptive statistics for the drink and food datasets,
  2. compares key metrics between the two,
  3. saves visualizations (PNG) comparing calories, carbs and fats.

Usage (standalone, unfiltered):
    python starbucks_analysis.py drinks.csv food.csv
For filtering plus analysis in one tool, use main.py.

Note: the source data has no sugar column. Carbohydrates (and "net carbs" =
carbs - fiber) are used as the closest available stand-in.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Functions imported from the previous script.
from starbucks_loader import load_starbucks_data

NUTRIENTS = ["calories", "fat_g", "carbs_g", "fiber_g", "protein_g"]
LABELS = {
    "calories": "Calories",
    "fat_g": "Fat (g)",
    "carbs_g": "Carbs (g)",
    "fiber_g": "Fiber (g)",
    "protein_g": "Protein (g)",
    "sodium_mg": "Sodium (mg)",
    "net_carbs_g": "Net carbs (g)",
    "fat_to_protein": "Fat-to-protein ratio",
}
DRINK_COLOR, FOOD_COLOR = "#00704A", "#C8873A"   # green / caramel


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add net carbs and a per-item fat-to-protein ratio."""
    df = df.copy()
    df["net_carbs_g"] = (df["carbs_g"] - df["fiber_g"]).clip(lower=0)
    # Ratio is undefined when protein is 0 -> NaN instead of inf.
    df["fat_to_protein"] = df["fat_g"] / df["protein_g"].where(df["protein_g"] > 0)
    return df


def with_nutrition(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that actually have nutrition data (many drinks are all-NaN)."""
    return df.dropna(subset=["calories"])


# --------------------------------------------------------------------------- #
# 1. Descriptive statistics
# --------------------------------------------------------------------------- #
def describe_dataset(df: pd.DataFrame, label: str) -> dict:
    """Print and return headline statistics for one dataset."""
    valid = with_nutrition(df)
    cols = [c for c in NUTRIENTS + ["sodium_mg"] if c in valid.columns]

    print(f"\n{'=' * 64}\n{label.upper()}  "
          f"({len(df)} items, {len(valid)} with nutrition data, "
          f"{len(df) - len(valid)} without)\n{'=' * 64}")

    summary = valid[cols].agg(["count", "mean", "median", "min", "max", "sum"]).T
    summary.index = [LABELS[c] for c in summary.index]
    print(summary.round(1).to_string())

    total_cal = valid["calories"].sum()
    avg_carbs = valid["carbs_g"].mean()
    avg_net = valid["net_carbs_g"].mean()
    protein_total = valid["protein_g"].sum()
    agg_ratio = valid["fat_g"].sum() / protein_total if protein_total > 0 else np.nan
    ratios = valid["fat_to_protein"].dropna()
    excluded = len(valid) - len(ratios)

    print(f"\nTotal calories (sum of all items):  {total_cal:,.0f}")
    print(f"Average carbs per item:             {avg_carbs:.1f} g "
          f"(net of fiber: {avg_net:.1f} g)")
    if np.isnan(agg_ratio):
        print("Fat-to-protein ratio: n/a (no protein in these items)")
    else:
        print(f"Fat-to-protein ratio (total fat / total protein): {agg_ratio:.2f}")
        print(f"Fat-to-protein ratio per item: median {ratios.median():.2f}, "
              f"mean {ratios.mean():.2f}  "
              f"({excluded} items excluded: 0 g protein)")

    hi = valid.loc[valid["calories"].idxmax()]
    lo = valid.loc[valid["calories"].idxmin()]
    print(f"Highest-calorie item: {hi['item']} ({hi['calories']:.0f} kcal)")
    print(f"Lowest-calorie item:  {lo['item']} ({lo['calories']:.0f} kcal)")

    return {"summary": summary, "total_calories": total_cal,
            "aggregate_fat_to_protein": agg_ratio}


# --------------------------------------------------------------------------- #
# 2. Comparison
# --------------------------------------------------------------------------- #
def compare_datasets(drinks: pd.DataFrame, foods: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side averages for metrics both datasets share."""
    d, f = with_nutrition(drinks), with_nutrition(foods)
    metrics = ["calories", "fat_g", "carbs_g", "net_carbs_g", "fiber_g", "protein_g"]

    table = pd.DataFrame({
        "Drinks (avg)": d[metrics].mean(),
        "Food (avg)": f[metrics].mean(),
        "Drinks (median)": d[metrics].median(),
        "Food (median)": f[metrics].median(),
    })
    table["Food / Drink (avg)"] = table["Food (avg)"] / table["Drinks (avg)"]
    def _agg(x):
        return x["fat_g"].sum() / x["protein_g"].sum() if x["protein_g"].sum() > 0 else np.nan

    table.loc["fat_to_protein"] = [
        _agg(d),
        _agg(f),
        d["fat_to_protein"].median(), f["fat_to_protein"].median(), np.nan,
    ]
    table.loc["fat_to_protein", "Food / Drink (avg)"] = (
        table.loc["fat_to_protein", "Food (avg)"]
        / table.loc["fat_to_protein", "Drinks (avg)"])
    table.index = [LABELS[i] for i in table.index]

    print(f"\n{'=' * 64}\nDRINKS vs FOOD\n{'=' * 64}")
    print(table.round(2).to_string())
    print("\n(Fat-to-protein row: 'avg' columns = total fat / total protein.)")

    cal_d, cal_f = table.loc["Calories", ["Drinks (avg)", "Food (avg)"]]
    print(f"\n- The average food item has {cal_f / cal_d:.1f}x the calories of the "
          f"average drink ({cal_f:.0f} vs {cal_d:.0f} kcal).")
    fp = table.loc[LABELS["fat_to_protein"]]
    print(f"- Drinks carry {fp['Drinks (avg)']:.2f} g fat per g protein; "
          f"food carries {fp['Food (avg)']:.2f}.")
    cf = d["calories"].corr(d["fat_g"]), f["calories"].corr(f["fat_g"])
    print(f"- Calories correlate with fat at r={cf[0]:.2f} (drinks) and "
          f"r={cf[1]:.2f} (food).")
    n_d, n_f = len(d), len(f)
    print(f"- Sample sizes: {n_d} drinks vs {n_f} food items (drinks' serving "
          f"size is not given in the data).")
    return table


# --------------------------------------------------------------------------- #
# 3. Visualizations
# --------------------------------------------------------------------------- #
def plot_average_comparison(drinks, foods, path: Path):
    """Grouped bars: average nutrients, drinks vs food (calories on own axis)."""
    d, f = with_nutrition(drinks), with_nutrition(foods)
    grams = ["fat_g", "carbs_g", "net_carbs_g", "fiber_g", "protein_g"]

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [1, 3.2]})
    # Calories panel
    vals = [d["calories"].mean(), f["calories"].mean()]
    bars = ax1.bar(["Drinks", "Food"], vals, color=[DRINK_COLOR, FOOD_COLOR])
    ax1.bar_label(bars, fmt="%.0f", padding=3)
    ax1.set_title("Average calories per item")
    ax1.set_ylabel("kcal")
    ax1.set_ylim(0, max(vals) * 1.15)

    # Grams panel
    x, w = np.arange(len(grams)), 0.38
    b1 = ax2.bar(x - w / 2, d[grams].mean(), w, label=f"Drinks (n={len(d)})",
                 color=DRINK_COLOR)
    b2 = ax2.bar(x + w / 2, f[grams].mean(), w, label=f"Food (n={len(f)})",
                 color=FOOD_COLOR)
    ax2.bar_label(b1, fmt="%.1f", padding=2, fontsize=9)
    ax2.bar_label(b2, fmt="%.1f", padding=2, fontsize=9)
    ax2.set_xticks(x, [LABELS[g] for g in grams])
    ax2.set_title("Average macronutrients per item")
    ax2.set_ylabel("grams")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Starbucks drinks vs food: average nutrition", fontsize=14)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_top_items(drinks, foods, path: Path, top: int = 10):
    """Horizontal bars of the highest-calorie items, colored by fat share."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, df, name, color in ((axes[0], drinks, "drinks", DRINK_COLOR),
                                (axes[1], foods, "food items", FOOD_COLOR)):
        top_df = with_nutrition(df).nlargest(top, "calories").iloc[::-1]
        names = top_df["item"].map(lambda n: n if len(n) <= 36 else n[:35].rstrip() + "…")
        bars = ax.barh(names, top_df["calories"], color=color)
        ax.bar_label(bars, labels=[f"{c:.0f} kcal  |  {fat:.0f} g fat"
                                   for c, fat in zip(top_df["calories"],
                                                     top_df["fat_g"])],
                     padding=3, fontsize=8)
        ax.set_title(f"Top {top} {name} by calories")
        ax.set_xlim(0, top_df["calories"].max() * 1.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_macro_split(drinks, foods, path: Path):
    """Pie charts: where the calories come from (Atwater factors 9/4/4)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#D9534F", "#F0AD4E", "#5BC0DE"]
    for ax, df, name in ((axes[0], drinks, "Drinks"), (axes[1], foods, "Food")):
        v = with_nutrition(df)
        parts = pd.Series({
            "Fat": v["fat_g"].sum() * 9,
            "Carbs": v["carbs_g"].sum() * 4,
            "Protein": v["protein_g"].sum() * 4,
        })
        ax.pie(parts, labels=parts.index, colors=colors, autopct="%1.0f%%",
               startangle=90, counterclock=False,
               wedgeprops={"edgecolor": "white"})
        ax.set_title(f"{name}: estimated calorie sources")
    fig.suptitle("Share of macronutrient calories (fat 9, carbs 4, protein 4 kcal/g)",
                 fontsize=11)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def make_visualizations(drinks, foods, outdir: Path, top: int, show: bool,
                        note: str | None = None):
    """Save the three charts. `note` (e.g. active filters) is stamped on each."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / "avg_comparison.png",
             outdir / "top_calorie_items.png",
             outdir / "macro_split.png"]
    figs = [plot_average_comparison(drinks, foods, paths[0]),
            plot_top_items(drinks, foods, paths[1], top),
            plot_macro_split(drinks, foods, paths[2])]
    if note:
        for fig, path in zip(figs, paths):
            fig.subplots_adjust(bottom=0.12)
            fig.text(0.5, 0.012, note, ha="center", fontsize=8, color="#555555")
            fig.savefig(path, dpi=150)
    print("\nSaved charts:\n  " + "\n  ".join(str(p) for p in paths))
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return paths


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Starbucks nutrition data.")
    parser.add_argument("drinks_csv", nargs="?",
                        default="starbucks-menu-nutrition-drinks.csv")
    parser.add_argument("food_csv", nargs="?",
                        default="starbucks-menu-nutrition-food.csv")
    parser.add_argument("--outdir", default="charts", help="Where to save PNG charts")
    parser.add_argument("--top", type=int, default=10, help="Items in top-N chart")
    parser.add_argument("--show", action="store_true", help="Open chart windows")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show loader cleaning details")
    args = parser.parse_args(argv)

    if not args.show:
        matplotlib.use("Agg")   # headless-safe

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")
    try:
        drinks, foods = load_starbucks_data(args.drinks_csv, args.food_csv)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    drinks, foods = add_derived_columns(drinks), add_derived_columns(foods)

    describe_dataset(drinks, "Drinks")
    describe_dataset(foods, "Food")
    compare_datasets(drinks, foods)
    make_visualizations(drinks, foods, Path(args.outdir), args.top, args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
