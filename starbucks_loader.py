"""
Starbucks nutrition data loader.

Loads the Starbucks drink and food nutrition CSVs into pandas DataFrames and
copes with common CSV problems: unknown encodings, odd delimiters, placeholder
values for missing data ("-", "N/A", ...), units embedded in numbers ("12g",
"1,200 mg"), messy headers, stray whitespace, duplicates and malformed rows.

Usage:
    python starbucks_loader.py drinks.csv food.csv
    # or, from other code:
    from starbucks_loader import load_starbucks_data
    drinks, foods = load_starbucks_data("drinks.csv", "food.csv")
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger("starbucks_loader")

# Strings that should be treated as "no value".
NA_VALUES = ["", " ", "-", "--", "—", "–", "n/a", "N/A", "NA", "na", "NaN", "nan",
             "null", "NULL", "None", "none", "?", "unknown", "Unknown"]

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]  # latin-1 never fails

# Canonical column names. Keys are regexes matched against the cleaned header.
COLUMN_ALIASES = {
    r"^(unnamed_0|beverage|drink|food|item|product|menu_item|name)$": "item",
    r"^beverage_category$": "category",
    r"^beverage_prep$": "prep",
    r"^calories?$": "calories",
    r"^(total_)?fat(_g)?$": "fat_g",
    r"^total_fat_g$": "fat_g",
    r"^saturated_fat_g$": "saturated_fat_g",
    r"^trans_fat_g$": "trans_fat_g",
    r"^carb(s|ohydrates?)?(_g)?$": "carbs_g",
    r"^total_carbohydrates_g$": "carbs_g",
    r"^(dietary_)?fib(er|re)(_g)?$": "fiber_g",
    r"^sugars?(_g)?$": "sugar_g",
    r"^protein(_g)?$": "protein_g",
    r"^sodium(_mg)?$": "sodium_mg",
    r"^cholesterol(_mg)?$": "cholesterol_mg",
    r"^caffeine(_mg)?$": "caffeine_mg",
}


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _candidate_encodings(path: Path) -> list[str]:
    """Pick encodings to try, putting any BOM / UTF-16 evidence first.

    Single-byte codecs like cp1252 "succeed" on UTF-16 bytes and produce
    garbage, so UTF-16 has to be detected up front rather than by failure.
    """
    head = path.read_bytes()[:4096]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return ["utf-16"] + ENCODINGS                # BOM present
    if head.startswith(b"\xef\xbb\xbf"):
        return ENCODINGS                             # utf-8-sig is first
    if b"\x00" in head:                              # BOM-less UTF-16 heuristic
        even_nuls = head[0::2].count(0)
        odd_nuls = head[1::2].count(0)
        return ["utf-16-le" if odd_nuls > even_nuls else "utf-16-be"] + ENCODINGS
    return ENCODINGS


def _sniff_delimiter(path: Path, encoding: str) -> str:
    """Guess the delimiter from the first few lines; default to comma."""
    with open(path, "r", encoding=encoding, newline="") as fh:
        sample = fh.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    """Read a CSV, trying several encodings and skipping malformed rows."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty: {path}")

    last_err: Exception | None = None
    for enc in _candidate_encodings(path):
        try:
            sep = _sniff_delimiter(path, enc)
            df = pd.read_csv(
                path,
                sep=sep,
                encoding=enc,
                na_values=NA_VALUES,
                keep_default_na=True,
                skipinitialspace=True,
                skip_blank_lines=True,
                on_bad_lines="warn",   # skip rows with the wrong number of fields
                engine="python",
            )
            log.info("Read %s (encoding=%s, delimiter=%r): %d rows, %d columns",
                     path.name, enc, sep, *df.shape)
            return df
        except UnicodeDecodeError as err:
            last_err = err
            continue
        except pd.errors.EmptyDataError as err:
            raise ValueError(f"No data found in {path}") from err
    raise ValueError(f"Could not decode {path}: {last_err}")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _clean_header(name: object) -> str:
    """'Carb. (g)' -> 'carb_g', 'Unnamed: 0' -> 'unnamed_0'."""
    s = str(name).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "unnamed"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers and map common variants to canonical names."""
    cleaned = [_clean_header(c) for c in df.columns]
    renamed = []
    for col in cleaned:
        new = col
        for pattern, canonical in COLUMN_ALIASES.items():
            if re.match(pattern, col):
                new = canonical
                break
        renamed.append(new)

    # Avoid duplicate column names after renaming (keep first, suffix others).
    seen: dict[str, int] = {}
    unique = []
    for col in renamed:
        seen[col] = seen.get(col, 0) + 1
        unique.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
    df = df.copy()
    df.columns = unique

    # If no column is called "item", assume the first text column is the name.
    if "item" not in df.columns:
        first_text = next((c for c in df.columns
                           if not pd.api.types.is_numeric_dtype(df[c])), None)
        if first_text:
            df = df.rename(columns={first_text: "item"})
    return df


def _to_number(value: object) -> float:
    """Parse '1,200 mg', '12g', '45%' or '  7 ' into a float; else NaN."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d[\d,]*\.?\d*|-?\.\d+", str(value))
    if not match:
        return float("nan")
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return float("nan")


def clean_dataframe(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Standardize headers, text, numeric types, blanks and duplicates."""
    df = standardize_columns(df)
    start_rows = len(df)

    # Text columns: trim, collapse internal whitespace, blank -> NaN.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (df[col].astype("string")
                          .str.replace(r"\s+", " ", regex=True)
                          .str.strip()
                          .replace("", pd.NA))

    # Numeric columns: any non-text column, plus text columns that are mostly numeric.
    text_cols = {"item", "category", "prep"}
    for col in df.columns:
        if col in text_cols:
            continue
        converted = df[col].map(_to_number)
        non_null_before = df[col].notna().sum()
        # Only convert if most existing values actually looked numeric.
        if non_null_before == 0 or converted.notna().sum() >= 0.8 * non_null_before:
            df[col] = converted
        else:
            log.warning("[%s] column %r left as text (not mostly numeric)", label, col)

    # Drop rows with no item name or with no data at all.
    if "item" in df.columns:
        df = df.dropna(subset=["item"])
    df = df.dropna(how="all")

    # Negative nutrition values are invalid -> treat as missing.
    for col in df.select_dtypes(include="number").columns:
        bad = df[col] < 0
        if bad.any():
            log.warning("[%s] %d negative values in %r set to NaN", label, bad.sum(), col)
            df.loc[bad, col] = float("nan")

    # Remove exact duplicate rows only. Rows sharing a name but with different
    # values are kept (they may be genuine variants) and flagged for review.
    before = len(df)
    df = df.drop_duplicates()
    if before - len(df):
        log.info("[%s] removed %d exact duplicate rows", label, before - len(df))
    if "item" in df.columns:
        conflicts = df.loc[df.duplicated("item", keep=False), "item"].unique()
        for name in conflicts:
            log.warning("[%s] %r appears more than once with different values", label, name)

    df = df.reset_index(drop=True)
    log.info("[%s] %d -> %d rows after cleaning", label, start_rows, len(df))
    return df


def report_missing(df: pd.DataFrame, label: str) -> None:
    """Log how much data is missing in each column."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        log.info("[%s] no missing values", label)
        return
    for col, n in missing.items():
        log.info("[%s] %-18s missing %4d / %d (%.1f%%)",
                 label, col, n, len(df), 100 * n / len(df))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_nutrition_csv(path: str | Path, label: str) -> pd.DataFrame:
    """Load and clean a single nutrition CSV."""
    df = clean_dataframe(read_csv_robust(path), label)
    report_missing(df, label)
    return df


REQUIRED_COLUMNS = ["item", "calories", "fat_g", "carbs_g", "fiber_g", "protein_g"]


def validate_columns(df: pd.DataFrame, label: str) -> None:
    """Raise a readable ValueError if the file lacks columns the analysis needs."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The {label} file is missing required column(s): {', '.join(missing)}. "
            f"Expected a nutrition file with item name, calories, fat, carbs, fiber "
            f"and protein.")
    if df.empty:
        raise ValueError(f"The {label} file has no usable rows.")


def load_starbucks_data(drinks_path: str | Path,
                        food_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (drinks_df, food_df), both cleaned."""
    drinks = load_nutrition_csv(drinks_path, "drinks")
    foods = load_nutrition_csv(food_path, "food")
    validate_columns(drinks, "drinks")
    validate_columns(foods, "food")
    return drinks, foods


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load Starbucks nutrition CSVs.")
    parser.add_argument("drinks_csv", help="Path to the drink nutrition CSV")
    parser.add_argument("food_csv", help="Path to the food nutrition CSV")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show cleaning details")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")
    try:
        drinks, foods = load_starbucks_data(args.drinks_csv, args.food_csv)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    pd.set_option("display.width", 120, "display.max_columns", 20)
    for name, df in (("DRINKS", drinks), ("FOOD", foods)):
        print(f"\n=== {name}: {df.shape[0]} rows x {df.shape[1]} columns ===")
        print(df.head())
        print("\nColumn types:")
        print(df.dtypes.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
