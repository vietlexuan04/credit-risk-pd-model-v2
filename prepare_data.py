"""
STEP 1 - Data preparation: raw file -> modelling datasets + scoring database + data quality report.

Input (you place these by hand)
-------------------------------
    data/cs-training.csv   REQUIRED. Kaggle "Give Me Some Credit" training file (has the target).
    data/cs-test.csv       OPTIONAL. Kaggle test file (target is empty). Only loaded into the
                           scoring database so the app has "new customers" to score.

What it does
------------
1. Reads the raw training file and renames the columns to short snake_case names.
2. Splits it ONCE into development (70%) and hold-out (30%): stratified on the target,
   duplicate-aware, seed 42 (identical to notebook 02).
3. Applies the 6 deterministic cleaning rules R1-R6 (identical to notebook 03) to each set
   and checks the result with validate_clean().
4. Builds a small SQLite database that plays the role of the bank's Oracle source:
   RAW (uncleaned) hold-out rows with their target + RAW test rows without target.
   The Streamlit app queries it by customer ID and cleans/scores on the fly.
5. Writes an Excel data quality report.

Output: data/processed/
-----------------------
    raw_dev.csv, raw_holdout.csv     raw rows after the split (used by notebooks 03+)
    dev.csv, holdout.csv             CLEANED rows -> input of train.py
    scoring.db                       SQLite, table `customers` (customer_id = HO_<id> or TS_<id>)
    data_quality_report.xlsx         profile, split, cleaning log, missing values, bad rate by decile

Usage (from anywhere):  python prepare_data.py
"""
import hashlib
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_processing import (HOLDOUT_FRAC, LATE_COLS, RENAME_MAP, SEED, SPECIAL_CODES,  # noqa: E402
                             clean, load_raw, split_dev_holdout, validate_clean)
from report_excel import ExcelReport  # noqa: E402

DATA_DIR = ROOT / "data"
TRAIN_FILE = DATA_DIR / "cs-training.csv"
TEST_FILE = DATA_DIR / "cs-test.csv"
OUT_DIR = DATA_DIR / "processed"
FEATURES = list(RENAME_MAP.values())[1:]          # the 10 predictors (first entry is the target)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ profiling helpers
def raw_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATURES:
        x = df[c]
        rows.append({
            "variable": c, "n_missing": int(x.isna().sum()), "pct_missing": x.isna().mean(),
            "n_zero": int((x == 0).sum()), "min": x.min(), "p01": x.quantile(0.01), "median": x.median(),
            "p99": x.quantile(0.99), "max": x.max(),
            "n_codes_96_98": int(x.isin(SPECIAL_CODES).sum()) if c in LATE_COLS else 0,
        })
    return pd.DataFrame(rows)


def split_summary(dev, hold) -> pd.DataFrame:
    return pd.DataFrame([
        {"set": n, "rows": len(d), "bad": int(d["target"].sum()), "bad_rate": d["target"].mean(),
         "share_of_rows": len(d) / (len(dev) + len(hold))}
        for n, d in (("development", dev), ("hold-out", hold))
    ])


def missing_after_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATURES:
        m = df[c].isna()
        rows.append({"variable": c, "n_missing": int(m.sum()), "pct_missing": m.mean(),
                     "bad_rate_missing": df.loc[m, "target"].mean() if m.any() else np.nan,
                     "bad_rate_present": df.loc[~m, "target"].mean()})
    return pd.DataFrame(rows)


def bad_rate_by_decile(df: pd.DataFrame) -> pd.DataFrame:
    """Development bad rate by decile of each cleaned variable (+ a missing bucket)."""
    out = []
    for c in FEATURES:
        x = df[c]
        q = pd.qcut(x, 10, duplicates="drop")
        g = df.groupby(q, observed=True)["target"].agg(rows="size", bad_rate="mean").reset_index()
        g.columns = ["bucket", "rows", "bad_rate"]
        g["bucket"] = g["bucket"].astype(str)
        if x.isna().any():
            g.loc[len(g)] = ["Missing", int(x.isna().sum()), df.loc[x.isna(), "target"].mean()]
        g.insert(0, "variable", c)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def plot_bad_rate_by_decile(t: pd.DataFrame, overall: float, path: Path) -> str:
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, (v, g) in zip(axes.ravel(), t.groupby("variable", sort=False)):
        colors = ["#7f7f7f" if b == "Missing" else "#4f81bd" for b in g["bucket"]]
        ax.bar(range(len(g)), g["bad_rate"] * 100, color=colors)
        ax.axhline(overall * 100, color="k", ls="--", lw=0.8)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([b if b == "Missing" else str(i + 1) for i, b in enumerate(g["bucket"])], fontsize=7)
        ax.set_title(v, fontsize=10)
        ax.set_ylabel("bad rate %")
    fig.suptitle("Development bad rate by decile of each cleaned variable (grey = missing, dashed = average)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


# ------------------------------------------------------------------ scoring database
def build_scoring_db(raw_hold: pd.DataFrame, raw_test: pd.DataFrame | None, path: Path) -> pd.DataFrame:
    """RAW rows only: cleaning happens at scoring time, exactly like a production pipeline."""
    parts = []
    h = raw_hold.reset_index()
    h.insert(0, "customer_id", "HO_" + h["id"].astype(str))
    h.insert(1, "source", "holdout")
    parts.append(h)
    if raw_test is not None:
        t = raw_test.reset_index()
        t.insert(0, "customer_id", "TS_" + t["id"].astype(str))
        t.insert(1, "source", "test")
        t["target"] = np.nan
        parts.append(t)
    db = pd.concat(parts, ignore_index=True).drop(columns="id")[["customer_id", "source", "target"] + FEATURES]
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as con:
        db.to_sql("customers", con, index=False)
        con.execute("CREATE UNIQUE INDEX idx_customer_id ON customers(customer_id)")
    return db.groupby("source").agg(rows=("customer_id", "size"), with_target=("target", "count")).reset_index()


# ------------------------------------------------------------------ main
def main():
    if not TRAIN_FILE.exists():
        sys.exit(f"Missing {TRAIN_FILE}. Download 'Give Me Some Credit' from Kaggle and put cs-training.csv in data/.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {TRAIN_FILE.relative_to(ROOT)}")
    raw = load_raw(TRAIN_FILE)

    # split once, on RAW data
    raw_dev, raw_hold = split_dev_holdout(raw)
    raw_dev.to_csv(OUT_DIR / "raw_dev.csv")
    raw_hold.to_csv(OUT_DIR / "raw_holdout.csv")

    # clean (rules are fixed in advance, so applying them to the hold-out set does not leak information)
    dev, log_dev = clean(raw_dev)
    hold, log_hold = clean(raw_hold)
    validate_clean(dev)
    validate_clean(hold)
    dev.to_csv(OUT_DIR / "dev.csv")
    hold.to_csv(OUT_DIR / "holdout.csv")
    print(f"Development {len(dev):,} rows (bad {dev.target.mean():.2%}) | hold-out {len(hold):,} rows "
          f"(bad {hold.target.mean():.2%})")

    raw_test = None
    if TEST_FILE.exists():
        raw_test = load_raw(TEST_FILE)
        print(f"Test file found: {len(raw_test):,} rows (no target), loaded into the scoring database")
    else:
        print("No data/cs-test.csv: the scoring database will contain hold-out customers only")
    db_summary = build_scoring_db(raw_hold, raw_test, OUT_DIR / "scoring.db")

    # ---------------- Excel report
    dup = raw.duplicated()
    by_decile = bad_rate_by_decile(dev)
    fig = plot_bad_rate_by_decile(by_decile, dev["target"].mean(), OUT_DIR / "figures" / "bad_rate_by_decile.png")

    r = ExcelReport()
    r.add_text("Summary", "Data quality report - Give Me Some Credit", style="title")
    r.add_kv("Summary", {
        "created": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(TRAIN_FILE.relative_to(ROOT)), "input_md5": md5(TRAIN_FILE),
        "raw_rows": len(raw), "raw_bad_rate": f"{raw['target'].mean():.4%}",
        "target_definition": "SeriousDlqin2yrs = 1: 90+ days past due within 2 years",
        "split": f"{1 - HOLDOUT_FRAC:.0%} development / {HOLDOUT_FRAC:.0%} hold-out, stratified, duplicate-aware, seed {SEED}",
        "test_file": str(TEST_FILE.relative_to(ROOT)) if raw_test is not None else "not provided",
    }, title="Input")
    r.add_table("Summary", split_summary(dev, hold), title="Split (after cleaning)")
    r.add_table("Summary", db_summary, title="Scoring database data/processed/scoring.db (raw rows)")
    r.add_kv("Summary", {p: "written" for p in ["raw_dev.csv", "raw_holdout.csv", "dev.csv (cleaned)",
                                                 "holdout.csv (cleaned)", "scoring.db"]}, title="Files in data/processed/")

    r.add_text("Raw_Profile", "Raw training file, before any cleaning", style="title")
    r.add_table("Raw_Profile", raw_profile(raw))
    r.add_kv("Raw_Profile", {"fully duplicated rows": int(dup.sum()),
                             "bad rate of duplicated rows": f"{raw.loc[dup, 'target'].mean():.2%}",
                             "bad rate of other rows": f"{raw.loc[~dup, 'target'].mean():.2%}"},
             title="Duplicates (kept, rule R6; the split keeps every copy on the same side)")

    r.add_text("Cleaning_Log", "Cleaning rules R1-R6 (fixed in advance, same rules on both sets)", style="title")
    r.add_table("Cleaning_Log", log_dev, title="Development")
    r.add_table("Cleaning_Log", log_hold, title="Hold-out")

    r.add_text("Missing_After", "Missing values after cleaning (development)", style="title")
    r.add_text("Missing_After", "No imputation: missing values get their own WoE bin (scorecard) or are routed "
                                "natively by the tree models.")
    r.add_table("Missing_After", missing_after_cleaning(dev))

    r.add_text("Bad_Rate_Profile", "Bad rate by decile of each cleaned variable (development)", style="title")
    r.add_image("Bad_Rate_Profile", fig)
    r.add_table("Bad_Rate_Profile", by_decile)

    if raw_test is not None:
        r.add_text("Test_File", "cs-test.csv (no target): raw profile", style="title")
        r.add_table("Test_File", raw_profile(raw_test))

    xlsx = r.save(OUT_DIR / "data_quality_report.xlsx")
    print(f"Done. Outputs in {OUT_DIR.relative_to(ROOT)}/  (report: {Path(xlsx).name})")
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
