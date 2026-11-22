"""
Champion vs challenger comparison on the hold-out set: metrics, tables and figures.

Every model object must expose .predict_pd(df) (PD of bad) and .score(df)
(points on the 600 / 50:1 / PDO 20 scale, higher = safer). The scorecard gives
rounded points; the tree models give the same scale computed from their PD.

Grades, cut-offs and PSI bins are always fixed on the DEVELOPMENT score and then
applied to the hold-out set, exactly as in notebook 07.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from evaluation import (assign_grade, bootstrap_ci, calibration_table, discrimination_metrics,  # noqa: E402
                        grade_edges, hosmer_lemeshow, probability_scores, psi, rank_ordering_table)

COLORS = {"scorecard": "#4f81bd", "xgboost": "#c0504d", "lightgbm": "#9bbb59"}
N_GRADES = 10
APPROVAL_TARGETS = [0.5, 0.6, 0.7, 0.8, 0.9]

# Decision rule, fixed BEFORE looking at hold-out results.
# A challenger replaces the champion only if ALL hold:
DECISION_RULE = {
    "min_gini_gain": 0.03,          # at least +3 Gini points on hold-out
    "diff_ci_excludes_zero": True,  # paired bootstrap 95% CI of the AUC gain is above 0
    "brier_not_worse": True,        # Brier score on hold-out <= champion's
    "rank_ordering_holds": True,    # bad rate strictly increases over the 10 grades
}


# --------------------------------------------------------------- predictions
def collect_predictions(models: dict, dev: pd.DataFrame, hold: pd.DataFrame) -> dict:
    out = {}
    for name, m in models.items():
        out[name] = {
            "pd_dev": np.asarray(m.predict_pd(dev)), "pd_hold": np.asarray(m.predict_pd(hold)),
            "s_dev": np.asarray(m.score(dev), dtype=float), "s_hold": np.asarray(m.score(hold), dtype=float),
        }
        out[name]["edges"] = grade_edges(out[name]["s_dev"], N_GRADES)
    return out


# --------------------------------------------------------------- discrimination
def paired_auc_diff(y, p_base, p_new, n_boot: int = 500, seed: int = 42) -> dict:
    """Paired bootstrap of AUC(new) - AUC(base): both models are scored on the SAME resample."""
    y, p_base, p_new = map(np.asarray, (y, p_base, p_new))
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        d[i] = roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_base[idx])
    point = roc_auc_score(y, p_new) - roc_auc_score(y, p_base)
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"auc_diff": float(point), "auc_diff_lo": float(lo), "auc_diff_hi": float(hi),
            "gini_diff": float(2 * point), "share_boot_positive": float((d > 0).mean())}


def discrimination_table(preds: dict, y_dev, y_hold, n_boot: int = 500) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        d, h = discrimination_metrics(y_dev, p["pd_dev"]), discrimination_metrics(y_hold, p["pd_hold"])
        ci = bootstrap_ci(y_hold, p["pd_hold"], n_boot=n_boot, seed=42)
        rows.append({"model": name, "dev_auc": d["auc"], "dev_gini": d["gini"], "dev_ks": d["ks"],
                     "hold_auc": h["auc"], "hold_auc_lo": ci["auc_lo"], "hold_auc_hi": ci["auc_hi"],
                     "hold_gini": h["gini"], "hold_ks": h["ks"], "gini_dev_minus_hold": d["gini"] - h["gini"]})
    return pd.DataFrame(rows)


def challenger_vs_champion(preds: dict, y_hold, champion: str = "scorecard", n_boot: int = 500) -> pd.DataFrame:
    base = preds[champion]["pd_hold"]
    rows = [{"challenger": n, **paired_auc_diff(y_hold, base, p["pd_hold"], n_boot=n_boot)}
            for n, p in preds.items() if n != champion]
    return pd.DataFrame(rows)


# --------------------------------------------------------------- calibration
def probability_table(preds: dict, y_hold) -> pd.DataFrame:
    lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    rows = []
    for name, p in preds.items():
        m = sm.Logit(np.asarray(y_hold), sm.add_constant(lg(p["pd_hold"]))).fit(disp=0)
        grades = assign_grade(p["s_hold"], p["edges"])
        hl = hosmer_lemeshow(y_hold, p["pd_hold"], grades, dof_reduction=0)
        rows.append({"model": name, **probability_scores(y_hold, p["pd_hold"]),
                     "calib_intercept": float(m.params[0]), "calib_slope": float(m.params[1]),
                     "hl_statistic": hl["hl_statistic"], "hl_p_value": hl["p_value"]})
    return pd.DataFrame(rows)


def grade_tables(preds: dict, y_hold) -> tuple:
    """(rank ordering, calibration by grade) on hold-out, long format with a `model` column."""
    ro, cal = [], []
    for name, p in preds.items():
        t = rank_ordering_table(y_hold, p["s_hold"], p["pd_hold"], p["edges"])
        t.insert(0, "model", name)
        ro.append(t)
        c = calibration_table(y_hold, p["pd_hold"], assign_grade(p["s_hold"], p["edges"]))
        c.insert(0, "model", name)
        cal.append(c)
    return pd.concat(ro, ignore_index=True), pd.concat(cal, ignore_index=True)


def rank_ordering_ok(ro: pd.DataFrame) -> dict:
    return {m: bool(np.all(np.diff(g.sort_values("grade")["bad_rate"]) > 0)) for m, g in ro.groupby("model")}


# --------------------------------------------------------------- business view
def approval_table(preds: dict, y_hold) -> pd.DataFrame:
    """Bad rate among approved at the same target approval rate. The cut-off is the development-score
    quantile for that rate, so it is fixed before the hold-out is scored."""
    y = np.asarray(y_hold)
    rows = []
    for name, p in preds.items():
        for a in APPROVAL_TARGETS:
            c = float(np.quantile(p["s_dev"], 1 - a))
            ok = p["s_hold"] >= c
            rows.append({"model": name, "target_approval": a, "cutoff_score": c,
                         "hold_approval": ok.mean(), "bad_rate_approved": y[ok].mean(),
                         "bads_rejected_share": y[~ok].sum() / y.sum()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- stability
def score_psi_table(preds: dict) -> pd.DataFrame:
    rows = [{"model": n, "score_psi_dev_vs_hold": psi(p["s_dev"], p["s_hold"], p["edges"])} for n, p in preds.items()]
    t = pd.DataFrame(rows)
    t["reading"] = np.where(t["score_psi_dev_vs_hold"] < 0.1, "stable",
                            np.where(t["score_psi_dev_vs_hold"] < 0.25, "monitor", "shift"))
    return t


def csi_table(dev: pd.DataFrame, hold: pd.DataFrame, features) -> pd.DataFrame:
    """Characteristic stability of each raw input: development deciles + a NaN bucket."""
    rows = []
    for v in features:
        x_d, x_h = dev[v].astype(float), hold[v].astype(float)
        edges = np.unique(np.quantile(x_d.dropna(), np.linspace(0, 1, 11)[1:-1]))
        k = len(edges) + 2  # numeric buckets + NaN

        def dist(x):
            idx = np.where(x.isna(), k - 1, np.searchsorted(edges, x.fillna(0).to_numpy(), side="right"))
            return np.clip(np.bincount(idx, minlength=k) / len(x), 1e-6, None)

        e, a = dist(x_d), dist(x_h)
        rows.append({"variable": v, "csi": float(((a - e) * np.log(a / e)).sum()), "n_buckets": k})
    t = pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)
    t["reading"] = np.where(t["csi"] < 0.1, "stable", "check")
    return t


# --------------------------------------------------------------- SHAP
def tree_shap(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values (log-odds of bad) of a fitted XGBoost / LightGBM sklearn model."""
    import warnings

    import shap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # LightGBM output-format notice
        sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):          # older LightGBM output: one array per class
        sv = sv[1]
    return np.asarray(sv)


def shap_importance(shap_by_model: dict) -> pd.DataFrame:
    """Mean |SHAP| per variable and model (log-odds units), plus the rank within each model."""
    t = pd.DataFrame({m: pd.Series(np.abs(v).mean(axis=0), index=cols) for m, (v, cols) in shap_by_model.items()})
    t = t.sort_values(t.columns[-1], ascending=False)
    ranks = t.rank(ascending=False).astype("Int64").add_suffix("_rank")
    return pd.concat([t, ranks], axis=1).reset_index(names="variable")


# --------------------------------------------------------------- decision
def recommend(disc: pd.DataFrame, diff: pd.DataFrame, prob: pd.DataFrame, ro_ok: dict,
              champion: str = "scorecard") -> pd.DataFrame:
    """Apply DECISION_RULE to every challenger. Returns one row per challenger with each check."""
    r = DECISION_RULE
    champ_brier = float(prob.set_index("model").loc[champion, "brier"])
    rows = []
    for _, d in diff.iterrows():
        n = d["challenger"]
        checks = {
            "gini_gain": float(d["gini_diff"]),
            "check_gini_gain": d["gini_diff"] >= r["min_gini_gain"],
            "check_ci_above_zero": (d["auc_diff_lo"] > 0) if r["diff_ci_excludes_zero"] else True,
            "check_brier_not_worse": (float(prob.set_index("model").loc[n, "brier"]) <= champ_brier)
                                     if r["brier_not_worse"] else True,
            "check_rank_ordering": ro_ok[n] if r["rank_ordering_holds"] else True,
        }
        checks["replaces_champion"] = all(v for k, v in checks.items() if k.startswith("check_"))
        rows.append({"challenger": n, **checks})
    out = pd.DataFrame(rows)
    return out


def decision_text(rec: pd.DataFrame, disc: pd.DataFrame, champion: str = "scorecard") -> str:
    winners = rec[rec["replaces_champion"]]
    if len(winners):
        best = winners.sort_values("gini_gain", ascending=False).iloc[0]
        return (f"Recommend {best['challenger']} as the new champion: +{best['gini_gain'] * 100:.1f} Gini points "
                f"and every check of the decision rule passes.")
    best = rec.sort_values("gini_gain", ascending=False).iloc[0]
    failed = [k.replace("check_", "") for k in rec.columns if k.startswith("check_") and not best[k]]
    return (f"Keep the {champion} as champion. Best challenger ({best['challenger']}) gains "
            f"{best['gini_gain'] * 100:+.1f} Gini points but fails: {', '.join(failed)}. "
            f"Use it as a benchmark / challenger.")


# =============================================================== figures
def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_roc(preds, y_hold, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n, p in preds.items():
        fpr, tpr, _ = roc_curve(y_hold, p["pd_hold"])
        ax.plot(fpr, tpr, color=COLORS.get(n), label=f"{n} (AUC {roc_auc_score(y_hold, p['pd_hold']):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("false positive rate (good flagged as bad)"); ax.set_ylabel("true positive rate (bad flagged)")
    ax.set_title("ROC curve, hold-out"); ax.legend(loc="lower right")
    return _save(fig, path)


def plot_ks(preds, y_hold, path):
    y = np.asarray(y_hold)
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4), sharey=True)
    for ax, (n, p) in zip(np.atleast_1d(axes), preds.items()):
        s = p["s_hold"]
        grid = np.quantile(s, np.linspace(0, 1, 401))
        cb = np.array([(s[y == 1] <= g).mean() for g in grid])
        cg = np.array([(s[y == 0] <= g).mean() for g in grid])
        k = int(np.argmax(cb - cg))
        ax.plot(grid, cb, color="#c0504d", label="bad"); ax.plot(grid, cg, color="#4f81bd", label="good")
        ax.vlines(grid[k], cg[k], cb[k], color="k", ls="--")
        ax.set_title(f"{n}: KS {cb[k] - cg[k]:.3f} at {grid[k]:.0f}"); ax.set_xlabel("score")
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel("cumulative share")
    return _save(fig, path)


def plot_metric_bars(disc, prob, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    m = disc.set_index("model")[["hold_auc", "hold_gini", "hold_ks"]]
    m.columns = ["AUC", "Gini", "KS"]
    x, w = np.arange(len(m.columns)), 0.8 / len(m)
    for i, (n, r) in enumerate(m.iterrows()):
        ax[0].bar(x + i * w, r.values, w, color=COLORS.get(n), label=n)
        for xi, v in zip(x + i * w, r.values):
            ax[0].text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax[0].set_xticks(x + w * (len(m) - 1) / 2); ax[0].set_xticklabels(m.columns)
    ax[0].set_ylim(0.4, 1); ax[0].set_title("Discrimination, hold-out"); ax[0].legend()
    b = prob.set_index("model")["brier"]
    ax[1].bar(b.index, b.values, color=[COLORS.get(n) for n in b.index])
    for i, v in enumerate(b.values):
        ax[1].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
    ax[1].set_ylim(b.min() * 0.97, b.max() * 1.01); ax[1].set_title("Brier score, hold-out (lower = better)")
    return _save(fig, path)


def plot_calibration(cal, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    lo, hi = cal[["avg_pd", "bad_rate"]].min().min() * 0.8, cal[["avg_pd", "bad_rate"]].max().max() * 1.2
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=0.7, label="perfect")
    for n, g in cal.groupby("model", sort=False):
        ax[0].plot(g["avg_pd"], g["bad_rate"], marker="o", color=COLORS.get(n), label=n)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("predicted PD (grade mean)"); ax[0].set_ylabel("observed bad rate")
    ax[0].set_title("Calibration by grade, hold-out (log-log)"); ax[0].legend()
    w = 0.8 / cal["model"].nunique()
    for i, (n, g) in enumerate(cal.groupby("model", sort=False)):
        ax[1].bar(g["grade"] + (i - 1) * w, g["ratio_obs_pred"], w, color=COLORS.get(n), label=n)
    ax[1].axhline(1, color="k", lw=0.8); ax[1].set_xlabel("grade (1 = safest)")
    ax[1].set_ylabel("observed / predicted"); ax[1].set_title("Observed / predicted bad rate"); ax[1].legend()
    return _save(fig, path)


def plot_rank_ordering(ro, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / ro["model"].nunique()
    for i, (n, g) in enumerate(ro.groupby("model", sort=False)):
        ax.bar(g["grade"] + (i - 1) * w, g["bad_rate"] * 100, w, color=COLORS.get(n), label=n)
    ax.set_yscale("log"); ax.set_xticks(range(1, N_GRADES + 1))
    ax.set_xlabel("grade (1 = safest)"); ax.set_ylabel("bad rate % (log)")
    ax.set_title("Rank ordering by grade, hold-out"); ax.legend()
    return _save(fig, path)


def plot_approval(appr, y_hold, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n, g in appr.groupby("model", sort=False):
        ax.plot(g["hold_approval"] * 100, g["bad_rate_approved"] * 100, marker="o", color=COLORS.get(n), label=n)
    ax.axhline(np.mean(y_hold) * 100, color="k", ls="--", lw=0.8, label="no model")
    ax.set_xlabel("approval rate %"); ax.set_ylabel("bad rate among approved %")
    ax.set_title("Approval rate vs bad rate, hold-out"); ax.legend()
    return _save(fig, path)


def plot_shap_beeswarm(shap_values, X, title, path, max_display=10):
    import shap
    shap.summary_plot(shap_values, X, show=False, max_display=max_display, plot_size=(8, 5))
    fig = plt.gcf()
    fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def plot_shap_importance(imp, path):
    models = [c for c in imp.columns if c != "variable" and not c.endswith("_rank")]
    t = imp.set_index("variable")[models].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    h = 0.8 / len(models)
    y = np.arange(len(t))
    for i, m in enumerate(models):
        ax.barh(y + i * h, t[m], h, color=COLORS.get(m), label=m)
    ax.set_yticks(y + h * (len(models) - 1) / 2); ax.set_yticklabels(t.index)
    ax.set_xlabel("mean |SHAP| (log-odds of bad)"); ax.set_title("Variable importance, hold-out sample")
    ax.legend()
    return _save(fig, path)


def plot_shap_dependence(shap_values, X, features, title, path):
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    for ax, f in zip(np.atleast_1d(axes), features):
        j = list(X.columns).index(f)
        x = X[f].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        hi = np.nanquantile(x, 0.99)
        sel = ok & (x <= hi)
        ax.scatter(x[sel], shap_values[sel, j], s=3, alpha=0.3, color="#c0504d")
        if (~ok).any():
            ax.scatter(np.full((~ok).sum(), np.nanmin(x) - 0.05 * (hi - np.nanmin(x))), shap_values[~ok, j],
                       s=3, alpha=0.3, color="#7f7f7f", label="missing")
            ax.legend()
        ax.axhline(0, color="k", lw=0.6); ax.set_xlabel(f"{f} (up to 99th pct)"); ax.set_ylabel("SHAP")
    fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def plot_tuning(cv_by_model: dict, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for n, cv in cv_by_model.items():
        t = cv.sort_values("trial")
        ax.errorbar(t["trial"], t["cv_auc_mean"], yerr=t["cv_auc_std"], fmt="o", ms=4, capsize=2,
                    color=COLORS.get(n), label=n, alpha=0.8)
    ax.set_xlabel("trial"); ax.set_ylabel("5-fold CV AUC (development)"); ax.set_title("Random search results")
    ax.legend()
    return _save(fig, path)
