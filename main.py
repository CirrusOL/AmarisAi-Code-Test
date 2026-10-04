"""
Starbucks nutrition tool: load -> filter -> analyze -> visualize.

Ties together:
  starbucks_loader.py    robust CSV loading and cleaning
  starbucks_analysis.py  descriptive stats, drink-vs-food comparison, charts

and adds filtering, so every statistic and chart can be restricted to the
items you care about.

Examples
--------
Full report on everything:
    python main.py drinks.csv food.csv

Food under 500 calories, and drinks with more than 10 g protein:
    python main.py drinks.csv food.csv --food-filter "calories<500" \\
                                       --drink-filter "protein_g>10"

A filter applied to both datasets, sorted and exported:
    python main.py drinks.csv food.csv --filter "fat_g<=5" --sort calories --export low_fat.csv

Name search (case- and accent-insensitive), list only:
    python main.py drinks.csv food.csv --name latte --no-stats --no-charts

AI-written summary of the (filtered) data via the Groq API:
    python main.py drinks.csv food.csv --insights
    python main.py drinks.csv food.csv --food-filter "calories<500" --insights --no-charts
    python main.py drinks.csv food.csv --insights --question "Which drinks are highest in carbs?"
Requires `pip install groq` and the GROQ_API_KEY environment variable. Without
them, a plain offline summary is printed instead.

Filter syntax: COLUMN OPERATOR NUMBER, with operators < <= > >= == !=
(quote it so your shell doesn't treat < and > as redirects). Repeat a flag to
combine criteria; all must match. Columns can be written with or without the
unit suffix ("protein" = "protein_g"). Derived columns net_carbs_g and
fat_to_protein can be filtered on too.

Note: the data has no sugar or caffeine columns. Filtering on a column that
doesn't exist gives an error listing what is available; if you load a dataset
that has one (e.g. caffeine_mg), the same filter syntax works unchanged.
"""

from __future__ import annotations

import argparse
import logging
import operator
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

# ---- functions from the previous scripts ---------------------------------- #
from starbucks_loader import load_starbucks_data
from starbucks_insights import (
    DEFAULT_MODEL,
    InsightsError,
    build_facts,
    generate_insights,
    offline_summary,
)
from starbucks_analysis import (
    add_derived_columns,
    compare_datasets,
    describe_dataset,
    make_visualizations,
    with_nutrition,
)

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
       "==": operator.eq, "=": operator.eq, "!=": operator.ne}
CRITERION_RE = re.compile(r"^\s*(\w+)\s*(<=|>=|==|!=|<|>|=)\s*(-?\d+(?:\.\d+)?)\s*$")
LIST_COLUMNS = ["item", "calories", "fat_g", "carbs_g", "fiber_g", "protein_g",
                "sodium_mg", "caffeine_mg"]


# --------------------------------------------------------------------------- #
# Locating input files
# --------------------------------------------------------------------------- #
DEFAULT_DRINKS = "starbucks-menu-nutrition-drinks.csv"
DEFAULT_FOOD = "starbucks-menu-nutrition-food.csv"


def locate_csv(path: str) -> Path:
    """Return the first existing match for `path`.

    Relative paths are tried against the current directory, then the folder
    containing this script, then a `data/` folder next to it. If nothing
    exists, the original path is returned so the loader reports the error.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    script_dir = Path(__file__).resolve().parent
    for base in (Path.cwd(), script_dir, script_dir / "data"):
        if (base / candidate).is_file():
            return base / candidate
    return candidate


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Criterion:
    column: str
    op: str
    value: float

    def __str__(self) -> str:
        return f"{self.column}{self.op}{self.value:g}"


def parse_criterion(text: str) -> Criterion:
    """'calories<500' -> Criterion('calories', '<', 500.0)."""
    match = CRITERION_RE.match(text)
    if not match:
        raise ValueError(
            f"Invalid filter {text!r}. Use COLUMN OPERATOR NUMBER, e.g. "
            f"\"calories<500\" (operators: < <= > >= == !=)")
    column, op, value = match.groups()
    return Criterion(column.lower(), op, float(value))


def resolve_column(name: str, df: pd.DataFrame, numeric_only: bool = False) -> str | None:
    """Match 'protein' to 'protein_g', 'sodium' to 'sodium_mg', etc."""
    for candidate in (name, f"{name}_g", f"{name}_mg"):
        if candidate in df.columns and (
                not numeric_only or pd.api.types.is_numeric_dtype(df[candidate])):
            return candidate
    return None


def filterable_columns(df: pd.DataFrame) -> list[str]:
    """Columns a numeric filter can be applied to (text columns are excluded)."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def apply_numeric_filters(df: pd.DataFrame, criteria: list[Criterion],
                          label: str) -> pd.DataFrame | None:
    """Keep rows matching ALL criteria. Rows with a missing value never match.

    Returns None if the dataset lacks a filtered column (so it can't satisfy
    the criterion), after printing a warning.
    """
    mask = pd.Series(True, index=df.index)
    for crit in criteria:
        col = resolve_column(crit.column, df, numeric_only=True)
        if col is None:
            print(f"Warning: {label} has no '{crit.column}' column, so it is "
                  f"excluded from results for filter {crit}.", file=sys.stderr)
            return None
        mask &= OPS[crit.op](df[col], crit.value).fillna(False)
    return df[mask]


def _normalize(text: str) -> str:
    """Lowercase and strip accents so 'latte' matches 'Caffè Latte'."""
    decomposed = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def apply_name_filters(df: pd.DataFrame, contains: list[str],
                       excludes: list[str]) -> pd.DataFrame:
    """Keep items whose name contains every `contains` term and no `excludes`."""
    names = df["item"].map(_normalize)
    mask = pd.Series(True, index=df.index)
    for term in contains:
        mask &= names.str.contains(_normalize(term), regex=False)
    for term in excludes:
        mask &= ~names.str.contains(_normalize(term), regex=False)
    return df[mask]


def validate_criteria(criteria: list[Criterion], frames: dict[str, pd.DataFrame],
                      parser: argparse.ArgumentParser) -> None:
    """Fail early if a column exists in none of the datasets it applies to."""
    for crit, targets in criteria:
        if not any(resolve_column(crit.column, frames[t], numeric_only=True)
                   for t in targets):
            available = sorted({c for t in targets for c in filterable_columns(frames[t])})
            parser.error(f"Numeric column '{crit.column}' (filter {crit}) does not exist "
                         f"in the {' / '.join(targets)} data. "
                         f"Available columns: {', '.join(available)}")


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def show_items(df: pd.DataFrame, label: str, sort: str | None, desc: bool,
               limit: int | None) -> None:
    """Print a compact table of matching items."""
    print(f"\n--- {label}: {len(df)} matching item(s) ---")
    if df.empty:
        return
    if sort:
        col = resolve_column(sort.lower(), df)
        if col is None:
            print(f"(cannot sort {label} by '{sort}': column not present)")
        else:
            df = df.sort_values(col, ascending=not desc, na_position="last")
    cols = [c for c in LIST_COLUMNS if c in df.columns]
    table = df[cols].copy()
    table["item"] = table["item"].map(lambda n: n if len(n) <= 46 else n[:45] + "…")
    shown = table if limit is None else table.head(limit)
    print(shown.to_string(index=False, na_rep="n/a", float_format=lambda v: f"{v:g}"))
    if limit is not None and len(table) > limit:
        print(f"... {len(table) - limit} more (use --limit 0 to show all)")


def describe_filters(args, drink_c, food_c) -> str:
    """Human-readable summary of active filters, stamped on charts."""
    parts = []
    if args.name:
        parts.append("name contains " + " & ".join(args.name))
    if args.exclude_name:
        parts.append("name excludes " + ", ".join(args.exclude_name))
    if drink_c:
        parts.append("drinks: " + ", ".join(map(str, drink_c)))
    if food_c:
        parts.append("food: " + ", ".join(map(str, food_c)))
    return "Filters - " + "; ".join(parts) if parts else ""


def run_insights(args, drinks_f: pd.DataFrame, food_f: pd.DataFrame, note: str) -> None:
    """Print an LLM summary; fall back to an offline summary on any failure."""
    print(f"\n{'=' * 64}\nAI INSIGHTS (Groq)\n{'=' * 64}")
    try:
        text, facts = generate_insights(
            drinks_f, food_f, question=args.question, top_n=args.insights_top,
            filter_note=note.removeprefix("Filters - ") or None,
            model=args.model, return_facts=True)
        print(text)
    except InsightsError as err:
        print(f"Could not get an AI summary: {err}\n", file=sys.stderr)
        facts = build_facts(drinks_f, food_f, args.insights_top,
                            note.removeprefix("Filters - ") or None)
        print(offline_summary(facts))
    if args.show_facts:
        import json
        print("\n--- facts sent to the model ---")
        print(json.dumps(facts, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load, filter, analyze and visualize Starbucks nutrition data.",
        epilog="Filter syntax: \"COLUMN<OP>NUMBER\" e.g. \"calories<500\". "
               "Repeat flags to combine (all must match).")
    p.add_argument("drinks_csv", nargs="?", default=DEFAULT_DRINKS,
                   help=f"drinks CSV (default: {DEFAULT_DRINKS})")
    p.add_argument("food_csv", nargs="?", default=DEFAULT_FOOD,
                   help=f"food CSV (default: {DEFAULT_FOOD})")

    g = p.add_argument_group("filtering")
    g.add_argument("--filter", "-f", action="append", default=[], metavar="EXPR",
                   help="numeric filter applied to BOTH drinks and food")
    g.add_argument("--drink-filter", action="append", default=[], metavar="EXPR",
                   help="numeric filter for drinks only")
    g.add_argument("--food-filter", action="append", default=[], metavar="EXPR",
                   help="numeric filter for food only")
    g.add_argument("--name", action="append", default=[], metavar="TEXT",
                   help="keep items whose name contains TEXT (repeatable)")
    g.add_argument("--exclude-name", action="append", default=[], metavar="TEXT",
                   help="drop items whose name contains TEXT (repeatable)")

    g = p.add_argument_group("output")
    g.add_argument("--list", action="store_true",
                   help="list matching items (automatic when any filter is set)")
    g.add_argument("--sort", metavar="COLUMN", help="sort the item list by COLUMN")
    g.add_argument("--desc", action="store_true", help="sort descending")
    g.add_argument("--limit", type=int, default=20,
                   help="max rows to list per dataset (0 = all; default 20)")
    g.add_argument("--export", metavar="CSV",
                   help="write filtered items to CSV (adds _drinks / _food suffix)")
    g.add_argument("--no-stats", action="store_true", help="skip statistics/comparison")
    g.add_argument("--no-charts", action="store_true", help="skip charts")
    g.add_argument("--outdir", default="charts", help="where to save charts")
    g.add_argument("--top", type=int, default=10, help="items in the top-N chart")
    g.add_argument("--show", action="store_true", help="open chart windows")
    g.add_argument("-v", "--verbose", action="store_true",
                   help="show loader cleaning details")

    g = p.add_argument_group("AI insights (Groq)")
    g.add_argument("--insights", action="store_true",
                   help="write an LLM summary of the (filtered) data")
    g.add_argument("--question", metavar="TEXT",
                   help="ask a specific question instead of the default summary "
                        "(implies --insights)")
    g.add_argument("--model", metavar="NAME",
                   help=f"Groq model (default: $GROQ_MODEL or {DEFAULT_MODEL})")
    g.add_argument("--insights-top", type=int, default=5, metavar="N",
                   help="items per ranking given to the model (default 5)")
    g.add_argument("--show-facts", action="store_true",
                   help="print the exact facts sent to the model, to verify its claims")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Windows consoles/redirects may use a legacy codec; item names contain
    # characters like ™ ® è, so never crash on printing them.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    if not args.show:
        matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    # Parse filters up front so bad syntax fails before any work is done.
    try:
        common = [parse_criterion(t) for t in args.filter]
        drink_c = common + [parse_criterion(t) for t in args.drink_filter]
        food_c = common + [parse_criterion(t) for t in args.food_filter]
    except ValueError as err:
        parser.error(str(err))

    # Load + clean (loader), then add derived columns (analysis).
    drinks_path, food_path = locate_csv(args.drinks_csv), locate_csv(args.food_csv)
    try:
        drinks, foods = load_starbucks_data(drinks_path, food_path)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        if isinstance(err, FileNotFoundError):
            print(f"  Looked in: {Path.cwd()} (current folder), "
                  f"{Path(__file__).resolve().parent} (script folder)\n"
                  f"  Pass the file paths explicitly:\n"
                  f"    python main.py path/to/drinks.csv path/to/food.csv",
                  file=sys.stderr)
        return 1
    frames = {"drinks": add_derived_columns(drinks), "food": add_derived_columns(foods)}

    validate_criteria([(c, ["drinks", "food"]) for c in common]
                      + [(c, ["drinks"]) for c in drink_c if c not in common]
                      + [(c, ["food"]) for c in food_c if c not in common],
                      frames, parser)

    # Apply filters.
    criteria = {"drinks": drink_c, "food": food_c}
    filtered: dict[str, pd.DataFrame] = {}
    print("Dataset sizes (before -> after filtering):")
    for key, df in frames.items():
        out = apply_name_filters(df, args.name, args.exclude_name)
        out = apply_numeric_filters(out, criteria[key], key)
        out = out if out is not None else df.iloc[0:0]
        filtered[key] = out
        print(f"  {key:<7}{len(df):>4} -> {len(out):>4}")

    drinks_f, food_f = filtered["drinks"], filtered["food"]
    note = describe_filters(args, drink_c, food_c)
    any_filter = bool(note)
    if note:
        print(f"\n{note}")

    # Item listing.
    if args.list or any_filter:
        limit = None if args.limit == 0 else args.limit
        show_items(drinks_f, "Drinks", args.sort, args.desc, limit)
        show_items(food_f, "Food", args.sort, args.desc, limit)

    # Export.
    if args.export:
        base = Path(args.export)
        for key, df in filtered.items():
            target = base.with_name(f"{base.stem}_{key}{base.suffix or '.csv'}")
            df.to_csv(target, index=False)
            print(f"Exported {len(df)} {key} rows -> {target}")

    # Statistics and comparison on the filtered data.
    d_ok, f_ok = not with_nutrition(drinks_f).empty, not with_nutrition(food_f).empty
    if not args.no_stats:
        for df, label, ok in ((drinks_f, "Drinks", d_ok), (food_f, "Food", f_ok)):
            if ok:
                describe_dataset(df, label)
            else:
                print(f"\n{label}: no items with nutrition data after filtering; "
                      f"statistics skipped.")
        if d_ok and f_ok:
            compare_datasets(drinks_f, food_f)
        else:
            print("\nDrink-vs-food comparison skipped (one side has no data).")

    # AI insights on the filtered data.
    if args.insights or args.question:
        run_insights(args, drinks_f, food_f, note)

    # Charts on the filtered data.
    if not args.no_charts:
        if d_ok and f_ok:
            try:
                make_visualizations(drinks_f, food_f, Path(args.outdir), args.top,
                                    args.show, note=note or None)
            except ValueError as err:
                print(f"Charts skipped: {err}", file=sys.stderr)
        else:
            print("\nCharts skipped (both datasets need at least one item with data).")
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:               # only exit non-zero on failure, so IDEs and
        sys.exit(exit_code)     # notebooks don't show SystemExit on success
