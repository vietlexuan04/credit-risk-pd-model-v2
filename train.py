"""
STEP 2 - Training run: WoE logistic scorecard + XGBoost + LightGBM, compared on the hold-out set.

Input
-----
    data/processed/dev.csv, data/processed/holdout.csv   (cleaned; created by prepare_data.py)

What it does
------------
1. Trains the scorecard from scratch (WoE binning -> backward selection -> logistic
   regression -> PDO scaling), exactly the Phase 1 method (notebooks 04-06).
2. Tunes XGBoost and LightGBM with 5-fold CV on the development set only, then refits them.
3. Evaluates all three models on the hold-out set with the same metrics.
4. Applies a decision rule fixed in advance (src/comparison.py, DECISION_RULE).
5. Writes everything to a NEW folder; earlier runs are never overwritten:

    runs/<run_id>/
    ├── models/
    │   ├── scorecard/          binning_spec.json, model_spec.json, scorecard_spec.json, points, master scale
    │   ├── xgboost_model.json, lightgbm_model.txt, challenger_params.json
    │   └── reference.json      grade edges, master-scale PD, cut-offs, development distributions (used by the app)
    ├── figures/                ROC, KS, calibration, rank ordering, approval, SHAP ...
    ├── tables/                 every table as CSV
    ├── metrics.json            headline numbers, parameters, recommendation
    └── model_report_<run_id>.xlsx

A run does NOT change the model used by the app. After reading the report you promote it:

    python train.py --promote <run_id>                    # model = the recommendation of that run
    python train.py --promote <run_id> --model lightgbm   # or choose the model yourself

which writes runs/champion.json. The app reads that file (or the latest run if it does not exist).

Usage
-----
    python train.py                 # full run, 20 tuning trials per challenger (about 20 min on 2 cores)
    python train.py --n-iter 5      # quick run
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import comparison as cmp  # noqa: E402
from challengers import FEATURES, TreeModel, best_params, fit_final, tune  # noqa: E402
from champion import ChampionScorecard  # noqa: E402
from data_processing import SEED, validate_clean  # noqa: E402
from evaluation import assign_grade  # noqa: E402
from feature_engineering import WoEBinner, iv_summary  # noqa: E402
from modeling import backward_select, coef_table  # noqa: E402
from report_excel import ExcelReport  # noqa: E402
from scorecard import Scorecard, scaling_params  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
PHASE1_SCORECARD_DIR = ROOT / "outputs" / "scorecard"   # saved by notebooks 04-07 (only used for a check)
SCORECARD_DIR = PHASE1_SCORECARD_DIR                    # kept for notebook 08 (check_refit default)
RUNS_DIR = ROOT / "runs"
CHAMPION_FILE = RUNS_DIR / "champion.json"

N_BOOT = 500        # bootstrap resamples for confidence intervals
SHAP_SAMPLE = 5000  # hold-out rows used for SHAP (enough for stable importances, fast)
APPROVAL_DEFAULT = 0.80   # default cut-off of the app = development-score quantile for 80% approval


# =====================================================================================
# 1. Data
# =====================================================================================
def load_data():
    f_dev, f_hold = PROCESSED / "dev.csv", PROCESSED / "holdout.csv"
    if not (f_dev.exists() and f_hold.exists()):
        sys.exit("data/processed/dev.csv or holdout.csv not found. Run `python prepare_data.py` first.")
    dev = pd.read_csv(f_dev, index_col="id")
    hold = pd.read_csv(f_hold, index_col="id")
    validate_clean(dev)
    validate_clean(hold)
    return dev, hold


# =====================================================================================
# 2. Scorecard training (notebooks 04-06 in one function)
# =====================================================================================
# Binning settings of notebook 04: variable -> (business trend, minimum share of rows per bin)
BINNING_CONFIG = {
    "revolving_util": ("increasing_risk", 0.02),   # higher utilization -> more stress
    "age":            ("decreasing_risk", 0.05),   # older -> historically more stable
    "n_30_59_late":   ("increasing_risk", 0.01),   # more past-due events -> riskier
    "n_60_89_late":   ("increasing_risk", 0.01),
    "n_90_late":      ("increasing_risk", 0.01),
    "debt_ratio":     ("increasing_risk", 0.05),   # heavier debt burden -> riskier
    "monthly_income": ("decreasing_risk", 0.05),   # more income -> more capacity to repay
    "n_open_lines":   ("decreasing_risk", 0.01),   # thin file -> riskier
    "n_real_estate":  ("free",            0.01),   # U-shape is plausible, no constraint
    "n_dependents":   ("increasing_risk", 0.01),   # more dependents -> more burden
}
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20


def train_scorecard(dev: pd.DataFrame) -> dict:
    """WoE binning -> variable selection -> logistic regression -> PDO scaling.

    Input: the CLEANED development set. Returns the specs (same format as outputs/scorecard/)
    plus the fitted objects and tables used in the report.
    """
    y = dev["target"]

    # Step 1 (notebook 04): coarse classing with business-driven monotonic WoE.
    #   Bins are learned on the development set only; NaN gets its own bin (no imputation).
    binners = {v: WoEBinner(v, trend=t, min_share=s).fit(dev[v], y) for v, (t, s) in BINNING_CONFIG.items()}

    # Step 2 (notebook 05): replace every variable by its WoE, then backward elimination
    #   in this order: wrong coefficient sign -> VIF > 5 -> p-value > 0.05.
    #   Plain statsmodels Logit: no class weights, no penalty, so the output is a real PD.
    X = pd.DataFrame({f"woe_{v}": b.transform(dev[v]) for v, b in binners.items()})
    kept, final, log = backward_select(X, y, alpha=0.05, max_vif=5.0, expected_sign=-1)
    model_spec = {
        "intercept": float(final.params["const"]),
        "coefficients": {c.replace("woe_", ""): float(final.params[c]) for c in kept},
        "dropped": log.to_dict(orient="records"),
        "sign_convention": "target=1 is bad; WoE=ln(%good/%bad); coefficients are negative",
    }

    # Step 3 (notebook 06): points. Factor = PDO / ln 2, Offset = 600 - Factor * ln 50,
    #   points of a bin = -Factor * coef * WoE + (Offset - Factor * intercept) / n_variables.
    factor, offset = scaling_params(BASE_SCORE, BASE_ODDS, PDO)
    card = Scorecard(binners, model_spec["intercept"], model_spec["coefficients"], factor, offset)
    scorecard_spec = {
        "base_score": BASE_SCORE, "base_odds_good_to_bad": BASE_ODDS, "pdo": PDO,
        "factor": factor, "offset": offset, "intercept": model_spec["intercept"],
        "n_variables": len(card.coefs), "base_points_per_variable": card.base_points,
        "round_points": True, "coefficients": model_spec["coefficients"],
    }
    bin_tables = pd.concat([binners[v].table_ for v in model_spec["coefficients"]], ignore_index=True)
    return {
        "binning_spec": {v: b.to_dict() for v, b in binners.items()},
        "model_spec": model_spec, "scorecard_spec": scorecard_spec,
        # objects / tables for the report
        "binners": binners, "iv": iv_summary(binners), "selection_log": log,
        "coef_table": coef_table(final).reset_index(names="term"),
        "points": card.points_table(bin_tables),
    }


def check_refit(refit: dict, scorecard_dir=PHASE1_SCORECARD_DIR, atol: float = 1e-8) -> pd.DataFrame | None:
    """Coefficient-by-coefficient comparison with the Phase 1 scorecard (None if it is not there)."""
    f = Path(scorecard_dir) / "model_spec.json"
    if not f.exists():
        return None
    saved = json.loads(f.read_text())
    rows = [{"term": "intercept", "saved": saved["intercept"], "refit": refit["model_spec"]["intercept"]}]
    rows += [{"term": v, "saved": c, "refit": refit["model_spec"]["coefficients"].get(v, np.nan)}
             for v, c in saved["coefficients"].items()]
    t = pd.DataFrame(rows)
    t["abs_diff"] = (t["saved"] - t["refit"]).abs()
    t["match"] = t["abs_diff"] < atol
    return t


# =====================================================================================
# 3. Challengers: XGBoost and LightGBM (tuned by CV on development only)
# =====================================================================================
def train_challengers(dev: pd.DataFrame, factor: float, offset: float, n_iter: int = 20, verbose=True):
    X, y = dev[FEATURES], dev["target"]
    models, cv_results, params = {}, {}, {}
    for kind in ("xgboost", "lightgbm"):
        cv = tune(kind, X, y, n_iter=n_iter, seed=SEED, verbose=verbose)
        p, n_trees = best_params(cv, kind)
        models[kind] = TreeModel(kind, fit_final(kind, p, n_trees, X, y, seed=SEED), factor, offset)
        cv_results[kind], params[kind] = cv, {**p, "n_estimators": n_trees}
    return models, cv_results, params


# =====================================================================================
# 4. Evaluation on the hold-out set (all models, same metrics)
# =====================================================================================
def evaluate(models: dict, dev: pd.DataFrame, hold: pd.DataFrame, fig_dir, cv_results=None,
             n_boot: int = N_BOOT, shap_n: int = SHAP_SAMPLE) -> dict:
    fig_dir = Path(fig_dir)
    y_dev, y_hold = dev["target"].to_numpy(), hold["target"].to_numpy()
    preds = cmp.collect_predictions(models, dev, hold)

    res = {"preds": preds}
    res["discrimination"] = cmp.discrimination_table(preds, y_dev, y_hold, n_boot=n_boot)
    res["vs_champion"] = cmp.challenger_vs_champion(preds, y_hold, n_boot=n_boot)
    res["probability"] = cmp.probability_table(preds, y_hold)
    res["rank_ordering"], res["calibration"] = cmp.grade_tables(preds, y_hold)
    res["rank_ordering_ok"] = cmp.rank_ordering_ok(res["rank_ordering"])
    res["approval"] = cmp.approval_table(preds, y_hold)
    res["psi"] = cmp.score_psi_table(preds)
    res["csi"] = cmp.csi_table(dev, hold, FEATURES)
    res["recommendation"] = cmp.recommend(res["discrimination"], res["vs_champion"], res["probability"],
                                          res["rank_ordering_ok"])
    res["decision"] = cmp.decision_text(res["recommendation"], res["discrimination"])
    winners = res["recommendation"][res["recommendation"]["replaces_champion"]]
    res["recommended_model"] = (winners.sort_values("gini_gain", ascending=False).iloc[0]["challenger"]
                                if len(winners) else "scorecard")

    # SHAP on a fixed hold-out sample (log-odds units for all three models)
    sample = hold.sample(min(shap_n, len(hold)), random_state=SEED)
    shap_by_model = {"scorecard": (models["scorecard"].contributions(sample, dev).to_numpy(),
                                   models["scorecard"].features)}
    for n in ("xgboost", "lightgbm"):
        shap_by_model[n] = (cmp.tree_shap(models[n].model, sample[FEATURES]), FEATURES)
    res["shap_sample"], res["shap_values"] = sample, shap_by_model
    res["shap_importance"] = cmp.shap_importance(shap_by_model)

    # Figures
    figs = {
        "metrics": cmp.plot_metric_bars(res["discrimination"], res["probability"], fig_dir / "metrics_bar.png"),
        "roc": cmp.plot_roc(preds, y_hold, fig_dir / "roc_holdout.png"),
        "ks": cmp.plot_ks(preds, y_hold, fig_dir / "ks_holdout.png"),
        "calibration": cmp.plot_calibration(res["calibration"], fig_dir / "calibration_by_grade.png"),
        "rank_ordering": cmp.plot_rank_ordering(res["rank_ordering"], fig_dir / "rank_ordering.png"),
        "approval": cmp.plot_approval(res["approval"], y_hold, fig_dir / "approval_vs_badrate.png"),
        "shap_importance": cmp.plot_shap_importance(res["shap_importance"], fig_dir / "shap_importance.png"),
    }
    for n in ("xgboost", "lightgbm"):
        sv = shap_by_model[n][0]
        figs[f"shap_{n}"] = cmp.plot_shap_beeswarm(sv, sample[FEATURES], f"SHAP summary - {n} (hold-out sample)",
                                                   fig_dir / f"shap_beeswarm_{n}.png")
        top3 = list(pd.Series(np.abs(sv).mean(0), index=FEATURES).nlargest(3).index)
        figs[f"shap_dep_{n}"] = cmp.plot_shap_dependence(sv, sample[FEATURES], top3,
                                                         f"SHAP dependence - {n}, top 3 variables",
                                                         fig_dir / f"shap_dependence_{n}.png")
    if cv_results:
        figs["tuning"] = cmp.plot_tuning(cv_results, fig_dir / "tuning_cv_auc.png")
    res["figures"] = figs
    return res


def reference_info(models: dict, preds: dict, dev: pd.DataFrame) -> dict:
    """What the app needs to turn a PD into a grade, a decision and a stability check,
    using only development-set information."""
    y_dev = dev["target"].to_numpy()
    ref = {"approval_default": APPROVAL_DEFAULT, "models": {}}
    for n, p in preds.items():
        edges = p["edges"]
        g_dev = assign_grade(p["s_dev"], edges).to_numpy()
        n_buckets = len(edges) + 1
        dist = np.bincount(np.searchsorted(edges, p["s_dev"], side="right"), minlength=n_buckets) / len(p["s_dev"])
        ref["models"][n] = {
            "grade_edges": edges.tolist(),
            "master_pd": {int(g): float(y_dev[g_dev == g].mean()) for g in sorted(np.unique(g_dev))},
            "dev_bucket_share": dist.tolist(),          # for PSI: buckets ordered from lowest score up
            "cutoff_by_approval": {f"{a:.2f}": float(np.quantile(p["s_dev"], 1 - a))
                                   for a in (0.5, 0.6, 0.7, 0.8, 0.9)},
            "dev_score_min": float(np.min(p["s_dev"])), "dev_score_max": float(np.max(p["s_dev"])),
        }
    sc = models["scorecard"]
    ref["scorecard_mean_woe_dev"] = {v: float(w) for v, w in sc.woe(dev).mean().items()}
    ref["features"] = FEATURES
    return ref


# =====================================================================================
# 5. Outputs: models, CSV tables, metrics.json, Excel report
# =====================================================================================
def _file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_info(run_id, dev, hold, n_iter, seconds):
    import lightgbm, shap, sklearn, statsmodels, xgboost
    return {
        "run_id": run_id, "created": datetime.now().isoformat(timespec="seconds"),
        "input_dev": "data/processed/dev.csv", "input_dev_md5": _file_md5(PROCESSED / "dev.csv"),
        "input_holdout": "data/processed/holdout.csv", "input_holdout_md5": _file_md5(PROCESSED / "holdout.csv"),
        "dev_rows": len(dev), "dev_bad_rate": float(dev["target"].mean()),
        "holdout_rows": len(hold), "holdout_bad_rate": float(hold["target"].mean()),
        "seed": SEED, "tuning_trials_per_model": n_iter, "cv_folds": 5, "bootstrap_resamples": N_BOOT,
        "shap_sample_rows": SHAP_SAMPLE, "runtime_seconds": round(seconds, 1),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
        "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "shap": shap.__version__,
    }


def write_outputs(run_dir: Path, info: dict, sc: dict, refit_check, models: dict, params: dict,
                  cv_results: dict, res: dict, reference: dict) -> Path:
    mdir, tables = run_dir / "models", run_dir / "tables"
    (mdir / "scorecard").mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)

    # models
    for k in ("binning_spec", "model_spec", "scorecard_spec"):
        (mdir / "scorecard" / f"{k}.json").write_text(json.dumps(sc[k], indent=2))
    sc["points"].to_csv(mdir / "scorecard" / "scorecard_points.csv", index=False)
    master = pd.DataFrame({"grade": list(reference["models"]["scorecard"]["master_pd"]),
                           "pd_master_scale": list(reference["models"]["scorecard"]["master_pd"].values())})
    master.to_csv(mdir / "scorecard" / "grade_master_scale.csv", index=False)
    models["xgboost"].save(mdir / "xgboost_model.json")
    models["lightgbm"].save(mdir / "lightgbm_model.txt")
    (mdir / "challenger_params.json").write_text(json.dumps(params, indent=2))
    (mdir / "reference.json").write_text(json.dumps(reference, indent=2))

    # tables
    named = {k: res[k] for k in ("discrimination", "vs_champion", "probability", "rank_ordering", "calibration",
                                 "approval", "psi", "csi", "shap_importance", "recommendation")}
    named.update({"scorecard_iv": sc["iv"], "scorecard_selection_log": sc["selection_log"],
                  "scorecard_coefficients": sc["coef_table"], "scorecard_points": sc["points"]})
    for k, df in named.items():
        df.to_csv(tables / f"{k}.csv", index=False)
    for k, cv in cv_results.items():
        cv.to_csv(tables / f"tuning_{k}.csv", index=False)
    if refit_check is not None:
        refit_check.to_csv(tables / "scorecard_vs_phase1.csv", index=False)

    # metrics.json
    disc = res["discrimination"].set_index("model")
    metrics = {
        "run": info,
        "holdout": {m: {"auc": disc.loc[m, "hold_auc"], "gini": disc.loc[m, "hold_gini"], "ks": disc.loc[m, "hold_ks"],
                        **res["probability"].set_index("model").loc[m, ["brier", "log_loss", "mean_pd"]].to_dict(),
                        "score_psi": float(res["psi"].set_index("model").loc[m, "score_psi_dev_vs_hold"]),
                        "rank_ordering_ok": res["rank_ordering_ok"][m]}
                    for m in disc.index},
        "vs_champion": res["vs_champion"].to_dict(orient="records"),
        "decision_rule": cmp.DECISION_RULE,
        "recommendation": res["recommendation"].to_dict(orient="records"),
        "decision": res["decision"],
        "recommended_model": res["recommended_model"],
        "challenger_params": params,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    xlsx = run_dir / f"model_report_{info['run_id']}.xlsx"
    build_excel(xlsx, info, sc, refit_check, params, cv_results, res)
    return xlsx


def build_excel(path, info, sc, refit_check, params, cv_results, res):
    f = res["figures"]
    r = ExcelReport()

    # Summary
    r.add_text("Summary", f"PD model report - run {info['run_id']}", style="title")
    r.add_text("Summary", f"Decision: {res['decision']}", style="subtitle")
    r.add_text("Summary", "Champion = WoE logistic scorecard (trained in this run). "
                          "Challengers = XGBoost, LightGBM on the same split and the same cleaned inputs.")
    r.add_text("Summary", "")
    headline = res["discrimination"][["model", "hold_auc", "hold_auc_lo", "hold_auc_hi", "hold_gini", "hold_ks"]].merge(
        res["probability"][["model", "brier", "log_loss", "mean_pd", "observed_bad_rate", "calib_slope"]], on="model"
    ).merge(res["psi"][["model", "score_psi_dev_vs_hold"]], on="model")
    headline["rank_ordering_ok"] = headline["model"].map(res["rank_ordering_ok"])
    r.add_table("Summary", headline, title="Hold-out headline metrics")
    r.add_table("Summary", res["vs_champion"], title="Challenger minus champion (paired bootstrap, hold-out)")
    r.add_kv("Summary", cmp.DECISION_RULE, title="Decision rule (fixed before evaluation)")
    r.add_table("Summary", res["recommendation"], title="Decision rule checks")
    r.add_image("Summary", f["metrics"], title="Metrics overview")

    # Scorecard build
    r.add_text("Scorecard", "Scorecard development (WoE + logistic regression + PDO scaling)", style="title")
    r.add_table("Scorecard", sc["iv"], title="Information Value of the 10 candidates (development)")
    r.add_table("Scorecard", sc["selection_log"], title="Backward selection log (sign -> VIF > 5 -> p-value > 0.05)")
    r.add_table("Scorecard", sc["coef_table"], title="Final logistic regression (statsmodels)")
    r.add_kv("Scorecard", {k: sc["scorecard_spec"][k] for k in ("base_score", "base_odds_good_to_bad", "pdo",
                                                                "factor", "offset", "base_points_per_variable")},
             title="Scaling: 600 points at odds 50:1, 20 points to double the odds")
    r.add_table("Scorecard", sc["points"], title="Points per bin")
    if refit_check is not None:
        r.add_table("Scorecard", refit_check, title="Check: same coefficients as the Phase 1 scorecard (outputs/scorecard)")

    # Discrimination
    r.add_text("Discrimination", "Discrimination: AUC, Gini, KS", style="title")
    r.add_table("Discrimination", res["discrimination"], title="Development vs hold-out (95% bootstrap CI for hold-out AUC)")
    r.add_image("Discrimination", f["roc"], title="ROC curve")
    r.add_image("Discrimination", f["ks"], title="KS")

    # Calibration
    r.add_text("Calibration", "Calibration: predicted PD vs observed bad rate", style="title")
    r.add_table("Calibration", res["probability"], title="Overall (hold-out). Slope 1 and intercept 0 = perfect")
    r.add_table("Calibration", res["calibration"], title="By grade (grades fixed on development score deciles)")
    r.add_image("Calibration", f["calibration"])

    # Rank ordering
    r.add_text("Rank_Ordering", "Rank ordering by grade (hold-out)", style="title")
    r.add_kv("Rank_Ordering", res["rank_ordering_ok"], title="Bad rate strictly increasing from grade 1 to 10")
    r.add_table("Rank_Ordering", res["rank_ordering"])
    r.add_image("Rank_Ordering", f["rank_ordering"])

    # Cut-off
    r.add_text("Cutoff", "Business view: bad rate at the same approval rate", style="title")
    r.add_text("Cutoff", "Cut-off = development-score quantile for the target approval rate; applied to hold-out.")
    r.add_table("Cutoff", res["approval"])
    r.add_image("Cutoff", f["approval"])

    # Stability
    r.add_text("Stability", "Stability: PSI of scores and CSI of inputs (development vs hold-out)", style="title")
    r.add_text("Stability", "Random split, so this mainly checks the split; it says nothing about stability over time.")
    r.add_table("Stability", res["psi"], title="Score PSI (<0.1 stable, 0.1-0.25 monitor, >0.25 shift)")
    r.add_table("Stability", res["csi"], title="CSI per input variable (development deciles + missing bucket)")

    # SHAP
    r.add_text("SHAP", "Explainability: SHAP (log-odds of bad)", style="title")
    r.add_text("SHAP", f"Hold-out sample of {len(res['shap_sample']):,} rows. Scorecard: coef x (WoE - mean WoE), "
                       "which is the exact SHAP value of a linear model. monthly_income is not in the scorecard.")
    r.add_table("SHAP", res["shap_importance"], title="Mean |SHAP| per variable")
    r.add_image("SHAP", f["shap_importance"])
    for n in ("xgboost", "lightgbm"):
        r.add_image("SHAP", f[f"shap_{n}"], title=f"Beeswarm - {n}")
        r.add_image("SHAP", f[f"shap_dep_{n}"], title=f"Dependence - {n}")

    # Tuning / parameters
    r.add_text("Tuning", "Hyperparameter search (5-fold CV on development only)", style="title")
    r.add_kv("Tuning", {f"{m}.{k}": v for m, p in params.items() for k, v in p.items()}, title="Selected parameters")
    for n, cv in cv_results.items():
        r.add_table("Tuning", cv, title=f"All trials - {n} (best first)")
    if "tuning" in f:
        r.add_image("Tuning", f["tuning"])

    # Run info
    r.add_text("Run_Info", "Run information", style="title")
    r.add_kv("Run_Info", info)
    return r.save(path)


# =====================================================================================
# 6. Promotion: choose which run / model the app uses
# =====================================================================================
def promote(run_id: str, model: str | None):
    run_dir = RUNS_DIR / run_id
    if not (run_dir / "metrics.json").exists():
        sys.exit(f"{run_dir} is not a finished run (metrics.json missing)")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    model = model or metrics["recommended_model"]
    if model not in metrics["holdout"]:
        sys.exit(f"unknown model {model}; choose from {list(metrics['holdout'])}")
    rec = {"run_id": run_id, "model": model, "promoted_at": datetime.now().isoformat(timespec="seconds"),
           "recommended_by_run": metrics["recommended_model"],
           "holdout_auc": metrics["holdout"][model]["auc"], "holdout_gini": metrics["holdout"][model]["gini"]}
    CHAMPION_FILE.write_text(json.dumps(rec, indent=2))
    print(f"Promoted run {run_id}, model {model} -> {CHAMPION_FILE.relative_to(ROOT)}")
    if model != metrics["recommended_model"]:
        print(f"Note: the run recommended {metrics['recommended_model']}, you chose {model}.")


# =====================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-iter", type=int, default=20, help="random-search trials per challenger (default 20)")
    ap.add_argument("--run-id", default=None, help="folder name under runs/ (default: timestamp)")
    ap.add_argument("--promote", metavar="RUN_ID", help="do not train; make RUN_ID the one used by the app")
    ap.add_argument("--model", choices=["scorecard", "xgboost", "lightgbm"], help="model to promote (with --promote)")
    a = ap.parse_args(argv)

    if a.promote:
        return promote(a.promote, a.model)

    t0 = time.time()
    run_id = a.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists; runs are never overwritten")
    print(f"Run {run_id}")

    dev, hold = load_data()
    print(f"Data: development {len(dev):,} rows (bad {dev.target.mean():.2%}), hold-out {len(hold):,} rows")

    print("Training scorecard ...")
    sc = train_scorecard(dev)
    champion = ChampionScorecard(sc["binners"], sc["model_spec"], sc["scorecard_spec"])
    refit_check = check_refit(sc)
    if refit_check is not None:
        print("  same coefficients as the Phase 1 scorecard:", bool(refit_check["match"].all()))

    print("Training challengers ...")
    challengers, cv_results, params = train_challengers(dev, champion.factor, champion.offset, n_iter=a.n_iter)
    models = {"scorecard": champion, **challengers}

    print("Evaluating on hold-out ...")
    res = evaluate(models, dev, hold, run_dir / "figures", cv_results=cv_results)
    reference = reference_info(models, res["preds"], dev)
    info = run_info(run_id, dev, hold, a.n_iter, time.time() - t0)
    xlsx = write_outputs(run_dir, info, sc, refit_check, models, params, cv_results, res, reference)

    print(res["discrimination"][["model", "hold_auc", "hold_gini", "hold_ks"]].round(4).to_string(index=False))
    print("Decision:", res["decision"])
    print("Report:", xlsx.relative_to(ROOT))
    print(f"To use this run in the app:  python train.py --promote {run_id}")


if __name__ == "__main__":
    main()
