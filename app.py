"""
Starbucks Nutrition Explorer - Flask web interface.

Upload the drinks and food CSVs, filter them, and view statistics, a
drink-vs-food comparison, charts and an optional Groq-written summary, all in
the browser. It reuses the same code as the command-line tool:

    starbucks_loader.py    CSV loading/cleaning
    starbucks_analysis.py  statistics, comparison, charts
    starbucks_insights.py  Groq summaries
    main.py                filter parsing (same syntax as the CLI)

Run:
    pip install -r requirements.txt
    python app.py                    # then open http://127.0.0.1:5000
Optional environment variables:
    GROQ_API_KEY       enables the AI summary (otherwise an offline summary is shown)
    FLASK_SECRET_KEY   keeps sessions valid across restarts (random if unset)

This is a local/single-user tool: uploads are kept in a temp folder and removed after an hour
without use (checked on each page load), and the AI button spends your Groq key. Don't expose it to the internet as is.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")   # headless rendering; must happen before pyplot is used

import base64
import contextlib
import io
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
from flask import (Flask, Response, abort, flash, jsonify, redirect,
                   render_template, request, session, url_for)

from starbucks_analysis import (LABELS, add_derived_columns, compare_datasets,
                                describe_dataset, plot_average_comparison,
                                plot_macro_split, plot_top_items, with_nutrition)
from starbucks_insights import (InsightsError, build_facts, generate_insights,
                                offline_summary)
from starbucks_loader import load_starbucks_data
from main import (LIST_COLUMNS, OPS, apply_name_filters, describe_filters,
                  filterable_columns, parse_criterion, resolve_column)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "starbucks_web_uploads"
UPLOAD_TTL_SECONDS = 3600
REQUIRED_COLUMNS = ["item", "calories", "fat_g", "carbs_g", "fiber_g", "protein_g"]
TOKEN_RE = re.compile(r"[0-9a-f]{32}")
MAX_ROWS = 500

BASE_DIR = Path(__file__).resolve().parent
# Normally index.html lives in templates/. If it sits next to app.py, use that.
_flat = (BASE_DIR / "index.html").is_file() and not (BASE_DIR / "templates" / "index.html").is_file()
app = Flask(__name__, template_folder=str(BASE_DIR) if _flat else "templates")
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_SAMESITE="Lax",
)

# pyplot and stdout redirection are process-global, so analysis is serialized.
LOCK = threading.RLock()
_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# Security: CSRF protection for every POST
# --------------------------------------------------------------------------- #
def csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def check_csrf():
    if request.method == "POST":
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        expected = session.get("csrf", "")
        if not expected or not secrets.compare_digest(sent.encode(), expected.encode()):
            abort(400, "Your session expired. Reload the page and try again.")


@app.errorhandler(400)
def bad_request(err):
    if request.path == url_for("insights"):
        return jsonify(error=err.description), 400
    flash(err.description, "error")
    return redirect(url_for("index"))


@app.errorhandler(413)
def too_large(_err):
    flash(f"Upload too large. The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
          f"for both files together.", "error")
    return redirect(url_for("index"))


# --------------------------------------------------------------------------- #
# Stored uploads + loading (reuses starbucks_loader)
# --------------------------------------------------------------------------- #
class _LogCapture(logging.Handler):
    """Collects the loader's cleaning notes so they can be shown to the user."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.records: list[tuple[str, str]] = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def cleanup_old_uploads() -> None:
    if not UPLOAD_ROOT.exists():
        return
    cutoff = time.time() - UPLOAD_TTL_SECONDS
    for d in UPLOAD_ROOT.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def load_dataset(token: str) -> dict:
    """Load + clean the stored CSVs for this token (cached by file mtime)."""
    folder = UPLOAD_ROOT / token
    drinks_path, food_path = folder / "drinks.csv", folder / "food.csv"
    key = (token, drinks_path.stat().st_mtime_ns, food_path.stat().st_mtime_ns)
    with LOCK:
        if key in _CACHE:
            return _CACHE[key]
        log = logging.getLogger("starbucks_loader")
        handler, old_level = _LogCapture(), log.level
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            drinks, foods = load_starbucks_data(drinks_path, food_path)
        finally:
            log.removeHandler(handler)
            log.setLevel(old_level)

        for name, df in (("drinks", drinks), ("food", foods)):
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"The {name} file is missing required column(s): "
                    f"{', '.join(missing)}. Expected a nutrition file with item "
                    f"name, calories, fat, carbs, fiber and protein.")
            if df.empty:
                raise ValueError(f"The {name} file has no usable rows.")

        result = {"frames": {"drinks": add_derived_columns(drinks),
                             "food": add_derived_columns(foods)},
                  "notes": handler.records}
        if len(_CACHE) >= 8:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = result
        return result


def current_data() -> dict | None:
    token = session.get("token")
    if not token or not TOKEN_RE.fullmatch(token):
        return None
    folder = UPLOAD_ROOT / token
    if not (folder / "drinks.csv").is_file() or not (folder / "food.csv").is_file():
        session.pop("token", None)
        return None
    with contextlib.suppress(OSError):
        os.utime(folder)    # mark as recently used so active sessions aren't cleaned up
    try:
        return load_dataset(token)
    except Exception:   # stored files no longer load (e.g. cleaned up) -> start over
        app.logger.exception("Could not reload stored upload")
        session.pop("token", None)
        return None


# --------------------------------------------------------------------------- #
# Filtering (parsing and matching come from main.py, same syntax as the CLI)
# --------------------------------------------------------------------------- #
def split_terms(text: str | None) -> list[str]:
    return [t.strip() for t in re.split(r"[,\n;]+", text or "") if t.strip()]


def filter_frames(frames: dict, values) -> tuple[dict, list[str], list[str], str]:
    """Apply the form's filters. Returns (filtered, messages, errors, note)."""
    messages: list[str] = []
    try:
        common = [parse_criterion(t) for t in split_terms(values.get("all_filters"))]
        drink_only = [parse_criterion(t) for t in split_terms(values.get("drink_filters"))]
        food_only = [parse_criterion(t) for t in split_terms(values.get("food_filters"))]
    except ValueError as err:
        return frames, [], [str(err)], ""

    errors = []
    checks = ([(c, ["drinks", "food"]) for c in common]
              + [(c, ["drinks"]) for c in drink_only]
              + [(c, ["food"]) for c in food_only])
    for crit, targets in checks:
        if not any(resolve_column(crit.column, frames[t], numeric_only=True)
                   for t in targets):
            available = sorted({c for t in targets for c in filterable_columns(frames[t])})
            errors.append(f"There is no numeric '{crit.column}' column in the "
                          f"{' / '.join(targets)} data. Columns you can filter on: "
                          f"{', '.join(available)}.")
    if errors:
        return frames, [], errors, ""

    contains = split_terms(values.get("name"))
    excludes = split_terms(values.get("exclude_name"))
    criteria = {"drinks": common + drink_only, "food": common + food_only}
    filtered = {}
    for key, df in frames.items():
        out = apply_name_filters(df, contains, excludes)
        mask = pd.Series(True, index=out.index)
        lacking = None
        for crit in criteria[key]:
            col = resolve_column(crit.column, out, numeric_only=True)
            if col is None:
                lacking = crit
                break
            mask &= OPS[crit.op](out[col], crit.value).fillna(False)
        if lacking:
            messages.append(f"The {key} data has no '{lacking.column}' column, so no "
                            f"{key} items can match '{lacking}'.")
            out = out.iloc[0:0]
        else:
            out = out[mask]
        filtered[key] = out

    note = describe_filters(SimpleNamespace(name=contains, exclude_name=excludes),
                            criteria["drinks"], criteria["food"])
    return filtered, messages, [], note


# --------------------------------------------------------------------------- #
# Building the page (reuses starbucks_analysis)
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:g}"


def item_table(df: pd.DataFrame, sort: str | None, desc: bool, limit: int,
               max_cal: float) -> dict:
    """Rows for the item table; calories carry a bar scaled to `max_cal`."""
    if sort:
        col = resolve_column(sort.lower(), df)
        if col:
            df = df.sort_values(col, ascending=not desc, na_position="last")
    cols = [c for c in LIST_COLUMNS if c in df.columns]
    others = [c for c in cols if c not in ("item", "calories")]
    rows = []
    for rec in df[cols].head(limit).to_dict("records"):
        cal = rec["calories"]
        has = not pd.isna(cal)
        rows.append({"item": rec["item"], "cal": _fmt(cal),
                     "pct": round(100 * cal / max_cal, 1) if has and max_cal else 0,
                     "cells": [_fmt(rec[c]) for c in others]})
    return {"headers": [LABELS.get(c, c) for c in others], "rows": rows,
            "total": len(df), "shown": len(rows)}


def dataset_stats(df: pd.DataFrame, label: str) -> dict:
    valid = with_nutrition(df)
    if valid.empty:
        return {"label": label, "empty": True, "n_total": len(df)}
    with contextlib.redirect_stdout(io.StringIO()):     # reuse the CLI report code
        info = describe_dataset(df, label)
    protein = valid["protein_g"].sum()
    hi = valid.loc[valid["calories"].idxmax()]
    return {
        "label": label, "empty": False, "n_total": len(df), "n_valid": len(valid),
        "headline": [
            ("Average calories", f"{valid['calories'].mean():.0f} kcal"),
            ("Total calories, all items", f"{valid['calories'].sum():,.0f}"),
            ("Average carbs", f"{valid['carbs_g'].mean():.1f} g"),
            ("Fat-to-protein ratio",
             f"{valid['fat_g'].sum() / protein:.2f}" if protein > 0 else "n/a"),
            ("Highest-calorie item", f"{hi['item']} ({hi['calories']:.0f} kcal)"),
        ],
        "table": info["summary"].round(1).to_html(classes="grid", border=0, na_rep="–"),
    }


def render_chart(fn, drinks, foods, note: str, **kwargs) -> str | None:
    """Draw one of the analysis charts to a base64 PNG (None if it can't be drawn)."""
    fig = None
    try:
        buf = io.BytesIO()
        fig = fn(drinks, foods, None, **kwargs)     # None: draw only, don't save
        if note:
            fig.subplots_adjust(bottom=0.12)
            fig.text(0.5, 0.012, note, ha="center", fontsize=8, color="#555555")
        fig.savefig(buf, format="png", dpi=150)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ValueError:
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def build_view(data: dict, values) -> dict:
    frames = data["frames"]
    filtered, messages, errors, note = filter_frames(frames, values)
    view = {"errors": errors, "messages": messages, "note": note}
    if errors:
        return view

    applied = "applied" in values
    show_charts = ("charts" in values) if applied else True
    try:
        limit = max(1, min(MAX_ROWS, int(values.get("limit", 25))))
    except ValueError:
        limit = 25
    sort, desc = values.get("sort") or None, "desc" in values

    max_cal = max(with_nutrition(frames[k])["calories"].max() for k in frames)
    drinks_f, food_f = filtered["drinks"], filtered["food"]
    view["sizes"] = {k: (len(frames[k]), len(filtered[k])) for k in frames}
    view["tables"] = {k: item_table(filtered[k], sort, desc, limit, max_cal)
                      for k in filtered}

    d_ok, f_ok = (not with_nutrition(drinks_f).empty), (not with_nutrition(food_f).empty)
    with LOCK:
        view["stats"] = [dataset_stats(drinks_f, "Drinks"), dataset_stats(food_f, "Food")]
        if d_ok and f_ok:
            with contextlib.redirect_stdout(io.StringIO()):
                table = compare_datasets(drinks_f, food_f)
            ratio = table.loc["Calories", "Food / Drink (avg)"]
            view["comparison"] = {
                "html": table.round(2).to_html(classes="grid", border=0, na_rep="–"),
                "sentence": (f"The average food item has {ratio:.1f}x the calories of "
                             f"the average drink "
                             f"({table.loc['Calories', 'Food (avg)']:.0f} vs "
                             f"{table.loc['Calories', 'Drinks (avg)']:.0f} kcal)."),
            }
        if show_charts and d_ok and f_ok:
            charts = [("Average nutrition", render_chart(plot_average_comparison, drinks_f, food_f, note)),
                      ("Highest-calorie items", render_chart(plot_top_items, drinks_f, food_f, note, top=10)),
                      ("Where calories come from", render_chart(plot_macro_split, drinks_f, food_f, note))]
            view["charts"] = [(t, img) for t, img in charts if img]
    view["show_charts"] = show_charts
    view["charts_possible"] = d_ok and f_ok
    return view


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    cleanup_old_uploads()
    data = current_data()
    ctx = {"loaded": data is not None, "values": request.args,
           "query": request.query_string.decode()}
    if data:
        frames = data["frames"]
        cols = sorted({c for df in frames.values() for c in filterable_columns(df)})
        ctx.update(columns=cols,
                   totals={k: len(df) for k, df in frames.items()},
                   sort_options=[c for c in LIST_COLUMNS if c in cols],
                   notes=data["notes"], view=build_view(data, request.args),
                   labels=LABELS)
    return render_template("index.html", **ctx)


@app.post("/upload")
def upload():
    files = {}
    for kind in ("drinks", "food"):
        f = request.files.get(kind)
        if not f or not f.filename:
            flash(f"Choose a {kind} CSV file.", "error")
            return redirect(url_for("index"))
        if not f.filename.lower().endswith(".csv"):
            flash(f"The {kind} file must be a .csv file (got '{Path(f.filename).name}').", "error")
            return redirect(url_for("index"))
        files[kind] = f

    cleanup_old_uploads()
    token = secrets.token_hex(16)
    folder = UPLOAD_ROOT / token
    folder.mkdir(parents=True, exist_ok=True)
    for kind, f in files.items():          # saved under fixed names, never user-supplied ones
        f.save(folder / f"{kind}.csv")

    try:
        load_dataset(token)
    except Exception as err:
        shutil.rmtree(folder, ignore_errors=True)
        message = str(err) if isinstance(err, (ValueError, FileNotFoundError)) else \
            "That file could not be read as a CSV. Check that it is a valid CSV export."
        if not isinstance(err, (ValueError, FileNotFoundError)):
            app.logger.exception("Upload failed to load")
        flash(message, "error")
        return redirect(url_for("index"))

    old = session.get("token")
    if old and TOKEN_RE.fullmatch(old):
        shutil.rmtree(UPLOAD_ROOT / old, ignore_errors=True)
    session["token"] = token
    flash("Files loaded. Showing everything; add filters to narrow it down.", "info")
    return redirect(url_for("index"))


@app.post("/reset")
def reset():
    old = session.pop("token", None)
    if old and TOKEN_RE.fullmatch(old):
        shutil.rmtree(UPLOAD_ROOT / old, ignore_errors=True)
    return redirect(url_for("index"))


@app.get("/download/<which>.csv")
def download(which: str):
    if which not in ("drinks", "food"):
        abort(404)
    data = current_data()
    if not data:
        abort(400, "Upload your CSV files first.")
    filtered, _messages, errors, _note = filter_frames(data["frames"], request.args)
    if errors:
        abort(400, " ".join(errors))
    return Response(filtered[which].to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition":
                             f"attachment; filename=starbucks_{which}_filtered.csv"})


@app.post("/insights")
def insights():
    data = current_data()
    if not data:
        return jsonify(error="Upload your CSV files first."), 400
    now = time.time()
    if now - session.get("last_ai", 0) < 3:
        return jsonify(error="Please wait a few seconds between summaries."), 429
    session["last_ai"] = now

    filtered, _messages, errors, note = filter_frames(data["frames"], request.form)
    if errors:
        return jsonify(error=" ".join(errors)), 400
    question = (request.form.get("question") or "").strip()[:300] or None
    filter_note = note.removeprefix("Filters - ") or None
    drinks_f, food_f = filtered["drinks"], filtered["food"]
    try:
        text, facts = generate_insights(drinks_f, food_f, question=question,
                                        filter_note=filter_note, return_facts=True)
        source, warning = "groq", None
    except InsightsError as err:
        facts = build_facts(drinks_f, food_f, 5, filter_note)
        text, source, warning = offline_summary(facts), "offline", str(err)
    return jsonify(text=text, source=source, warning=warning, facts=facts)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
