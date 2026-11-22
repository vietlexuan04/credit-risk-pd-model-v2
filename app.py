"""
STEP 3 - Streamlit scoring app.

    python -m streamlit run app.py

Reads
-----
    runs/champion.json          which run and model are "in production" (written by `python train.py --promote`).
                                If it does not exist, the latest run is used.
    runs/<run_id>/models/       scorecard specs, XGBoost, LightGBM, reference.json
    data/processed/scoring.db   SQLite with RAW customer records (written by prepare_data.py)

Pages
-----
    1. Score a customer     look up a customer ID in the database (or type the values) -> the app
                            cleans the raw record with the same rules as training -> PD, score,
                            grade, decision, and the reasons (scorecard points / SHAP).
    2. Batch scoring        upload a CSV (or take a whole table from the database) -> score every row;
                            if the file has the target column, the app also VALIDATES the model on it
                            (AUC, Gini, KS, calibration, rank ordering) and checks stability (PSI).
    3. Model report         the run's metrics, decision and figures, plus its Excel report.

Optional login: set the environment variable APP_PASSWORD before starting the app.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from challengers import FEATURES, load_tree_model  # noqa: E402
from champion import load_champion  # noqa: E402
from data_processing import RENAME_MAP, clean  # noqa: E402
from evaluation import (assign_grade, calibration_table, discrimination_metrics,  # noqa: E402
                        probability_scores, rank_ordering_table)

RUNS_DIR = ROOT / "runs"
CHAMPION_FILE = RUNS_DIR / "champion.json"
DB_FILE = ROOT / "data" / "processed" / "scoring.db"
MODEL_NAMES = ["scorecard", "xgboost", "lightgbm"]
LABELS = {
    "revolving_util": "Revolving utilization (balance / limit)",
    "age": "Age",
    "n_30_59_late": "Times 30-59 days late (2 years)",
    "n_60_89_late": "Times 60-89 days late (2 years)",
    "n_90_late": "Times 90+ days late",
    "debt_ratio": "Debt ratio (monthly debt / income)",
    "monthly_income": "Monthly income",
    "n_open_lines": "Open credit lines and loans",
    "n_real_estate": "Real estate loans",
    "n_dependents": "Dependents",
}

st.set_page_config(page_title="PD scoring app", layout="wide")


# =============================================================================== loading
def list_runs():
    return sorted([p.name for p in RUNS_DIR.glob("*") if (p / "metrics.json").exists()], reverse=True)


@st.cache_resource(show_spinner="Loading models ...")
def load_run(run_id: str) -> dict:
    d = RUNS_DIR / run_id
    metrics = json.loads((d / "metrics.json").read_text())
    ref = json.loads((d / "models" / "reference.json").read_text())
    sc = load_champion(d / "models" / "scorecard")
    models = {
        "scorecard": sc,
        "xgboost": load_tree_model("xgboost", d / "models" / "xgboost_model.json", sc.factor, sc.offset),
        "lightgbm": load_tree_model("lightgbm", d / "models" / "lightgbm_model.txt", sc.factor, sc.offset),
    }
    for m in ref["models"].values():
        m["grade_edges"] = np.array(m["grade_edges"])
        m["master_pd"] = {int(k): v for k, v in m["master_pd"].items()}
    return {"dir": d, "metrics": metrics, "ref": ref, "models": models}


@st.cache_resource(show_spinner=False)
def tree_explainer(run_id: str, name: str):
    import warnings

    import shap
    m = load_run(run_id)["models"][name].model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return shap.TreeExplainer(m.booster_ if name == "lightgbm" else m)


def query(sql: str, params=()) -> pd.DataFrame:
    with sqlite3.connect(DB_FILE) as con:
        return pd.read_sql_query(sql, con, params=params)


# =============================================================================== scoring
def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Accept Kaggle column names or the short names used in this project."""
    df = df.rename(columns=RENAME_MAP)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "SeriousDlqin2yrs" in df.columns:
        df = df.rename(columns={"SeriousDlqin2yrs": "target"})
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def score_all(run: dict, clean_df: pd.DataFrame) -> pd.DataFrame:
    """PD, score, grade and master-scale PD of every model for every row."""
    out = pd.DataFrame(index=clean_df.index)
    for n in MODEL_NAMES:
        m, r = run["models"][n], run["ref"]["models"][n]
        s = np.asarray(m.score(clean_df), dtype=float)
        g = assign_grade(s, r["grade_edges"]).to_numpy()
        out[f"{n}_pd"] = np.asarray(m.predict_pd(clean_df))
        out[f"{n}_score"] = np.round(s, 0)
        out[f"{n}_grade"] = g
        out[f"{n}_master_pd"] = [r["master_pd"][int(x)] for x in g]
    return out


def psi_vs_dev(scores, ref_model: dict) -> float:
    edges = ref_model["grade_edges"]
    e = np.clip(np.array(ref_model["dev_bucket_share"]), 1e-6, None)
    a = np.bincount(np.searchsorted(edges, np.asarray(scores, dtype=float), side="right"), minlength=len(e)) / len(scores)
    a = np.clip(a, 1e-6, None)
    return float(((a - e) * np.log(a / e)).sum())


def explain_row(run_id: str, run: dict, model: str, row: pd.DataFrame) -> pd.DataFrame:
    """Contribution of each variable to the log-odds of bad (positive = riskier)."""
    if model == "scorecard":
        sc = run["models"]["scorecard"]
        mean_woe = run["ref"]["scorecard_mean_woe_dev"]
        w = sc.woe(row).iloc[0]
        pts = sc.points_breakdown(row).set_index("variable")["points"]
        t = pd.DataFrame({"value": row[sc.features].iloc[0],
                          "contribution": [sc.coefs[v] * (w[v] - mean_woe[v]) for v in sc.features],
                          "points": pts})
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sv = tree_explainer(run_id, model).shap_values(row[FEATURES])
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)[0]
        t = pd.DataFrame({"value": row[FEATURES].iloc[0], "contribution": sv})
    t.index.name = "variable"
    return t.sort_values("contribution", key=np.abs, ascending=False)


def plot_contributions(t: pd.DataFrame, title: str):
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(t) + 1))
    colors = ["#c0504d" if v > 0 else "#4f81bd" for v in t["contribution"]]
    ax.barh([f"{i} = {v:.4g}" if pd.notna(v) else f"{i} = missing" for i, v in zip(t.index, t["value"])],
            t["contribution"], color=colors)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("contribution to log-odds of bad (red = riskier, blue = safer)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================== login + sidebar
pw = os.environ.get("APP_PASSWORD")
if pw and not st.session_state.get("auth"):
    st.title("PD scoring app")
    with st.form("login"):
        user = st.text_input("User")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            if p == pw:
                st.session_state["auth"], st.session_state["user"] = True, user
                st.rerun()
            st.error("Wrong password")
    st.stop()

runs = list_runs()
if not runs:
    st.error("No training run found in runs/. Run `python prepare_data.py` then `python train.py`.")
    st.stop()

champion = json.loads(CHAMPION_FILE.read_text()) if CHAMPION_FILE.exists() else None
with st.sidebar:
    st.header("Model in use")
    if champion and champion["run_id"] in runs:
        st.success(f"Promoted: run **{champion['run_id']}**, model **{champion['model']}**")
        default_run, default_model = champion["run_id"], champion["model"]
    else:
        st.warning("No promoted model (runs/champion.json). Using the latest run.")
        default_run, default_model = runs[0], None
    run_id = st.selectbox("Run", runs, index=runs.index(default_run))
    run = load_run(run_id)
    if run_id != default_run or default_model is None:
        default_model = run["metrics"]["recommended_model"]
    model = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(default_model))
    if champion and (run_id, model) != (champion["run_id"], champion["model"]):
        st.info("You are looking at a model that is not the promoted one.")

    st.header("Decision rule")
    ref_m = run["ref"]["models"][model]
    target = st.select_slider("Target approval rate", options=["0.50", "0.60", "0.70", "0.80", "0.90"], value="0.80")
    cutoff = st.number_input("Cut-off score (approve if score >= cut-off)",
                             value=float(round(ref_m["cutoff_by_approval"][target])), step=1.0)
    st.caption("The default cut-off is the development-score quantile for the chosen approval rate.")
    st.divider()
    st.caption(f"Run decision: {run['metrics']['decision']}")
    if not DB_FILE.exists():
        st.error("data/processed/scoring.db missing: run `python prepare_data.py`.")

st.title("PD scoring app - Give Me Some Credit")
tab1, tab2, tab3 = st.tabs(["Score a customer", "Batch scoring & validation", "Model report"])

# =============================================================================== 1. single customer
with tab1:
    mode = st.radio("Input", ["Customer ID (database)", "Manual input"], horizontal=True)
    raw = None
    if mode == "Customer ID (database)":
        c1, c2, c3 = st.columns([3, 1, 1])
        if c2.button("Random hold-out customer"):
            st.session_state["cid"] = query("SELECT customer_id FROM customers WHERE source='holdout' "
                                            "ORDER BY RANDOM() LIMIT 1").iloc[0, 0]
        if c3.button("Random test customer"):
            st.session_state["cid"] = query("SELECT customer_id FROM customers WHERE source='test' "
                                            "ORDER BY RANDOM() LIMIT 1").iloc[0, 0]
        cid = c1.text_input("Customer ID (HO_<id> = hold-out, TS_<id> = Kaggle test file)",
                            value=st.session_state.get("cid", "HO_1"))
        if cid:
            rec = query("SELECT * FROM customers WHERE customer_id = ?", (cid.strip(),))
            if rec.empty:
                st.error(f"{cid} not found in the database")
            else:
                raw = rec.set_index("customer_id")
                st.caption("Raw record queried from the database (before cleaning)")
                st.dataframe(raw, width="stretch")
    else:
        with st.form("manual"):
            cols = st.columns(2)
            v = {}
            v["revolving_util"] = cols[0].number_input(LABELS["revolving_util"], 0.0, value=0.30, step=0.05, format="%.4f")
            v["age"] = cols[1].number_input(LABELS["age"], 0, 120, value=40)
            v["n_30_59_late"] = cols[0].number_input(LABELS["n_30_59_late"], 0, 98, value=0)
            v["n_60_89_late"] = cols[1].number_input(LABELS["n_60_89_late"], 0, 98, value=0)
            v["n_90_late"] = cols[0].number_input(LABELS["n_90_late"], 0, 98, value=0)
            v["debt_ratio"] = cols[1].number_input(LABELS["debt_ratio"], 0.0, value=0.35, step=0.05, format="%.4f")
            inc = cols[0].number_input(LABELS["monthly_income"], 0.0, value=5000.0, step=500.0)
            inc_missing = cols[0].checkbox("Income not reported")
            v["n_open_lines"] = cols[1].number_input(LABELS["n_open_lines"], 0, 100, value=8)
            v["n_real_estate"] = cols[0].number_input(LABELS["n_real_estate"], 0, 60, value=1)
            dep = cols[1].number_input(LABELS["n_dependents"], 0, 20, value=0)
            dep_missing = cols[1].checkbox("Dependents not reported")
            submitted = st.form_submit_button("Score")
        v["monthly_income"] = np.nan if inc_missing else inc
        v["n_dependents"] = np.nan if dep_missing else dep
        raw = pd.DataFrame([v], index=pd.Index(["manual"], name="customer_id"))
        raw.insert(0, "target", np.nan)

    if raw is not None:
        clean_df, log = clean(to_model_frame(raw.drop(columns=["source"], errors="ignore")))
        if clean_df.empty:
            st.error("Record rejected by cleaning rule R1 (age < 18): it cannot be scored.")
        else:
            changed = log[(log["rows_affected"] > 0)]["rule"].tolist()
            if changed:
                st.info("Cleaning rules applied to this record: " + "; ".join(changed))
            res = score_all(run, clean_df).iloc[0]
            s, p, g = res[f"{model}_score"], res[f"{model}_pd"], int(res[f"{model}_grade"])
            approve = s >= cutoff
            k = st.columns(5)
            k[0].metric("Score", f"{s:.0f}")
            k[1].metric("PD (model)", f"{p:.2%}")
            k[2].metric("Grade (1 = safest)", g)
            k[3].metric("PD (master scale)", f"{res[f'{model}_master_pd']:.2%}")
            k[4].metric("Decision", "APPROVE" if approve else "DECLINE")
            k[4].caption(f"approve if score >= {cutoff:.0f}")
            tgt = raw["target"].iloc[0]
            if pd.notna(tgt):
                st.caption(f"Actual outcome in the data: {'BAD (90+ days past due within 2 years)' if tgt == 1 else 'GOOD'}")

            left, right = st.columns([3, 2])
            with left:
                t = explain_row(run_id, run, model, clean_df)
                st.pyplot(plot_contributions(t, f"Why this result? ({model})"))
            with right:
                st.markdown("**All models**")
                st.dataframe(pd.DataFrame({n: {"score": f"{res[f'{n}_score']:.0f}", "PD": f"{res[f'{n}_pd']:.2%}",
                                               "grade": int(res[f"{n}_grade"])} for n in MODEL_NAMES}).T,
                             width="stretch")
                st.markdown("**Inputs after cleaning**")
                st.dataframe(clean_df[FEATURES].T.rename(columns=lambda c: "value"), width="stretch")
            if model == "scorecard":
                st.markdown("**Scorecard points per variable** (total = score)")
                st.dataframe(t[["value", "points"]].sort_values("points"), width="stretch")
            st.caption("Contributions are in log-odds of bad relative to the average development customer. "
                       "Tree models: SHAP values. Scorecard: coefficient x (WoE - mean WoE), its exact SHAP value.")

# =============================================================================== 2. batch
with tab2:
    src = st.radio("Data", ["Upload a CSV", "Database: hold-out customers", "Database: test customers"],
                   horizontal=True)
    df = None
    if src == "Upload a CSV":
        st.caption("Same columns as cs-training.csv (Kaggle names or the short names). "
                   "If the file has SeriousDlqin2yrs / target, the model is also validated on it.")
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            first = df.columns[0]
            if first.startswith("Unnamed") or first in ("id", "customer_id"):
                df = df.set_index(first)
                df.index.name = "id"
    else:
        source = "holdout" if "hold-out" in src else "test"
        df = query("SELECT * FROM customers WHERE source = ?", (source,)).set_index("customer_id").drop(columns="source")
        if source == "holdout":
            st.caption("Scoring the hold-out set from the database reproduces the numbers of the training report.")

    if df is not None:
        try:
            df = to_model_frame(df)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        clean_df, log = clean(df)
        n_drop = len(df) - len(clean_df)
        st.write(f"**{len(df):,}** rows read, **{len(clean_df):,}** scored"
                 + (f" ({n_drop} rejected by rule R1, age < 18)" if n_drop else ""))
        with st.expander("Cleaning log"):
            st.dataframe(log, width="stretch")
        scores = score_all(run, clean_df)
        st.dataframe(pd.concat([clean_df[FEATURES], scores], axis=1).head(200), width="stretch")
        st.download_button("Download all scores (CSV)", pd.concat([clean_df, scores], axis=1).to_csv().encode(),
                           file_name=f"scores_{run_id}.csv", mime="text/csv")

        st.subheader("Stability vs development (PSI of the score)")
        psi_t = pd.DataFrame({"psi": {n: psi_vs_dev(scores[f"{n}_score"], run["ref"]["models"][n]) for n in MODEL_NAMES}})
        psi_t["reading"] = np.where(psi_t["psi"] < 0.1, "stable", np.where(psi_t["psi"] < 0.25, "monitor", "shift"))
        st.dataframe(psi_t.style.format({"psi": "{:.4f}"}), width="stretch")

        has_target = "target" in clean_df.columns and clean_df["target"].notna().all() \
            and clean_df["target"].nunique() == 2
        if not has_target:
            st.info("No complete target column: scores only, no validation.")
        else:
            y = clean_df["target"].astype(int).to_numpy()
            st.subheader("Validation on this file")
            rows = {}
            for n in MODEL_NAMES:
                p = scores[f"{n}_pd"].to_numpy()
                ok = scores[f"{n}_score"].to_numpy() >= (cutoff if n == model else
                                                          run["ref"]["models"][n]["cutoff_by_approval"][target])
                rows[n] = {**discrimination_metrics(y, p), **probability_scores(y, p),
                           "approval_rate": ok.mean(), "bad_rate_approved": y[ok].mean() if ok.any() else np.nan}
            mt = pd.DataFrame(rows).T
            st.dataframe(mt.style.format("{:.4f}"), width="stretch")
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                for n in MODEL_NAMES:
                    fpr, tpr, _ = roc_curve(y, scores[f"{n}_pd"])
                    ax.plot(fpr, tpr, label=f"{n} (AUC {mt.loc[n, 'auc']:.4f})")
                ax.plot([0, 1], [0, 1], "k--", lw=0.7)
                ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.legend()
                ax.set_title("ROC")
                fig.tight_layout()
                st.pyplot(fig)
            with c2:
                edges = run["ref"]["models"][model]["grade_edges"]
                ro = rank_ordering_table(y, scores[f"{model}_score"], scores[f"{model}_pd"], edges)
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                ax.bar(ro["grade"], ro["bad_rate"] * 100, color="#c0504d", label="observed bad rate")
                ax.plot(ro["grade"], ro["avg_pd"] * 100, "ko-", label="mean predicted PD")
                ax.set_yscale("log"); ax.set_xlabel("grade (1 = safest)"); ax.set_ylabel("% (log)")
                ax.set_title(f"Rank ordering and calibration - {model}"); ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
            st.markdown(f"**Rank ordering - {model}** (grades fixed on the development score)")
            st.dataframe(ro.style.format({"bad_rate": "{:.2%}", "avg_pd": "{:.2%}", "pct_rows": "{:.2%}",
                                          "min_score": "{:.0f}", "max_score": "{:.0f}"}), width="stretch")
            cal = calibration_table(y, scores[f"{model}_pd"], scores[f"{model}_grade"])
            st.markdown(f"**Calibration by grade - {model}**")
            st.dataframe(cal.style.format({"avg_pd": "{:.2%}", "bad_rate": "{:.2%}", "ratio_obs_pred": "{:.3f}",
                                           "binom_p_value": "{:.4f}"}), width="stretch")

# =============================================================================== 3. report
with tab3:
    m = run["metrics"]
    st.subheader(f"Run {run_id}")
    st.write(m["decision"])
    h = pd.DataFrame(m["holdout"]).T
    st.dataframe(h.style.format({c: "{:.4f}" for c in h.columns if c != "rank_ordering_ok"}),
                 width="stretch")
    st.markdown("**Challenger minus champion (paired bootstrap, hold-out)**")
    st.dataframe(pd.DataFrame(m["vs_champion"]).set_index("challenger").style.format("{:.4f}"),
                 width="stretch")
    st.markdown("**Decision rule checks**")
    st.dataframe(pd.DataFrame(m["recommendation"]).set_index("challenger"), width="stretch")

    xlsx = next(run["dir"].glob("model_report_*.xlsx"), None)
    if xlsx:
        st.download_button("Download the Excel model report", xlsx.read_bytes(), file_name=xlsx.name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    figs = run["dir"] / "figures"
    order = ["metrics_bar", "roc_holdout", "ks_holdout", "calibration_by_grade", "rank_ordering",
             "approval_vs_badrate", "shap_importance", "shap_beeswarm_xgboost", "shap_beeswarm_lightgbm",
             "shap_dependence_xgboost", "shap_dependence_lightgbm", "tuning_cv_auc"]
    for name in order:
        f = figs / f"{name}.png"
        if f.exists():
            st.image(str(f), caption=name.replace("_", " "))
    with st.expander("Run information"):
        st.json(m["run"])
