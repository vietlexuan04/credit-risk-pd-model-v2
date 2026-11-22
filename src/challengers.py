"""
Challenger models: XGBoost and LightGBM on the same data as the scorecard.

Rules (so the comparison with the scorecard is fair)
----------------------------------------------------
* Same development / hold-out split and the same deterministic clean() as Phase 1.
* Inputs are the 10 original columns after cleaning. No WoE, no derived features,
  no imputation: both libraries route NaN to a learned side of each split.
* No class weights (scale_pos_weight / is_unbalance) and no resampling, so the
  output stays a real PD that can be compared with observed bad rates.
* Tuning uses stratified k-fold CV on the DEVELOPMENT set only. Early stopping
  inside each fold uses that fold's own validation part (still development data).
  The hold-out set is never seen before the final evaluation.
* Final model: refitted on the full development set with the chosen parameters and
  n_estimators = mean best iteration across the CV folds (so no hold-out early stopping).
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

FEATURES = ["revolving_util", "age", "n_30_59_late", "debt_ratio", "monthly_income",
            "n_open_lines", "n_90_late", "n_real_estate", "n_60_89_late", "n_dependents"]

MAX_TREES = 3000
EARLY_STOP = 100


# ------------------------------------------------------------ search spaces
def _sample_xgb(rng) -> dict:
    return {
        "max_depth": int(rng.integers(2, 7)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
        "min_child_weight": float(np.exp(rng.uniform(np.log(1), np.log(200)))),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(20)))),
        "gamma": float(rng.uniform(0, 2)),
    }


def _sample_lgbm(rng) -> dict:
    return {
        "num_leaves": int(rng.integers(7, 64)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
        "min_child_samples": int(np.exp(rng.uniform(np.log(20), np.log(500)))),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(20)))),
        "min_split_gain": float(rng.uniform(0, 0.5)),
    }


SAMPLERS = {"xgboost": _sample_xgb, "lightgbm": _sample_lgbm}


# ------------------------------------------------------------ model factory
def make_model(kind: str, params: dict, n_estimators: int = MAX_TREES, early_stopping: bool = True, seed: int = 42):
    if kind == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=n_estimators, tree_method="hist", objective="binary:logistic",
            eval_metric="logloss", early_stopping_rounds=EARLY_STOP if early_stopping else None,
            random_state=seed, n_jobs=-1, **params)
    if kind == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=n_estimators, objective="binary", subsample_freq=1,
            random_state=seed, n_jobs=-1, verbose=-1, **params)
    raise ValueError(kind)


def _fit_fold(kind, params, X_tr, y_tr, X_va, y_va, seed):
    m = make_model(kind, params, seed=seed)
    if kind == "xgboost":
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best = int(m.best_iteration) + 1
    else:
        m.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        best = int(m.best_iteration_)
    p = m.predict_proba(X_va)[:, 1]
    return best, roc_auc_score(y_va, p), log_loss(y_va, p)


def tune(kind: str, X: pd.DataFrame, y: pd.Series, n_iter: int = 20, n_splits: int = 5, seed: int = 42,
         verbose: bool = True) -> pd.DataFrame:
    """Random search with stratified k-fold CV on the development set.

    Early stopping uses log loss (a proper scoring rule, protects the PD); trials are ranked by
    mean CV AUC. Returns one row per trial, best first.
    """
    rng = np.random.default_rng(seed)
    folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y))
    rows = []
    for t in range(n_iter):
        params = SAMPLERS[kind](rng)
        t0 = time.time()
        res = [_fit_fold(kind, params, X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], seed) for tr, va in folds]
        best, aucs, lls = zip(*res)
        rows.append({"trial": t, **params, "cv_auc_mean": np.mean(aucs), "cv_auc_std": np.std(aucs),
                     "cv_logloss_mean": np.mean(lls), "best_iter_mean": int(np.mean(best)),
                     "seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"[{kind}] trial {t + 1:>2}/{n_iter}  CV AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  "
                  f"trees {int(np.mean(best)):>4}  ({rows[-1]['seconds']}s)")
    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)


def best_params(cv_results: pd.DataFrame, kind: str) -> tuple:
    """(params, n_estimators) of the best trial."""
    template = SAMPLERS[kind](np.random.default_rng(0))   # only used for the keys and their types
    top = cv_results.iloc[0]
    params = {k: (int(top[k]) if isinstance(v, int) else float(top[k])) for k, v in template.items()}
    return params, int(top["best_iter_mean"])


def fit_final(kind: str, params: dict, n_estimators: int, X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """Refit on the whole development set with a fixed number of trees."""
    m = make_model(kind, params, n_estimators=n_estimators, early_stopping=False, seed=seed)
    m.fit(X, y)
    return m


class TreeModel:
    """Common interface: .predict_pd(df) and .score(df) on the scorecard's 600/50/20 scale."""

    def __init__(self, name: str, model, factor: float, offset: float, features=FEATURES):
        self.name, self.model, self.factor, self.offset, self.features = name, model, factor, offset, list(features)

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])[:, 1]

    def score(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(pd_to_score(self.predict_pd(df), self.factor, self.offset), index=df.index)

    def save(self, path):
        if self.name == "xgboost":
            self.model.save_model(str(path))
        else:
            self.model.booster_.save_model(str(path))


def pd_to_score(p, factor: float, offset: float):
    """Inverse of scorecard.score_to_pd: Score = Offset + Factor * ln(odds good:bad). Not rounded."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return offset + factor * np.log((1 - p) / p)


def load_tree_model(name: str, path, factor: float, offset: float) -> TreeModel:
    """Reload a saved challenger (used by the Streamlit app)."""
    if name == "xgboost":
        m = xgb.XGBClassifier()
        m.load_model(str(path))
        return TreeModel(name, m, factor, offset)

    booster = lgb.Booster(model_file=str(path))

    class _Wrap:  # minimal predict_proba wrapper around a native Booster
        def __init__(self, b):
            self.booster_ = b

        def predict_proba(self, X):
            p = self.booster_.predict(X)
            return np.column_stack([1 - p, p])

    return TreeModel(name, _Wrap(booster), factor, offset)
