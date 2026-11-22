import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import comparison as cmp  # noqa: E402
from challengers import best_params, fit_final, pd_to_score, tune, TreeModel  # noqa: E402
from report_excel import ExcelReport  # noqa: E402
from scorecard import scaling_params, score_to_pd  # noqa: E402


def test_pd_to_score_is_the_inverse_of_score_to_pd():
    factor, offset = scaling_params(600, 50, 20)
    s = np.array([450.0, 560.0, 600.0, 640.0])
    assert np.allclose(pd_to_score(score_to_pd(s, factor, offset), factor, offset), s)
    assert np.isclose(pd_to_score(1 / 51, factor, offset), 600)


def test_paired_auc_diff_is_zero_for_identical_models_and_positive_for_a_better_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    good = y + rng.normal(0, 0.8, 3000)
    noisy = y + rng.normal(0, 2.0, 3000)
    same = cmp.paired_auc_diff(y, good, good, n_boot=50)
    assert same["auc_diff"] == 0 and same["auc_diff_lo"] == 0 and same["auc_diff_hi"] == 0
    better = cmp.paired_auc_diff(y, noisy, good, n_boot=100)
    assert better["auc_diff"] > 0 and better["auc_diff_lo"] > 0


def test_recommend_keeps_champion_when_gain_is_small():
    disc = pd.DataFrame({"model": ["scorecard", "lgbm"]})
    diff = pd.DataFrame([{"challenger": "lgbm", "gini_diff": 0.01, "auc_diff_lo": 0.001}])
    prob = pd.DataFrame({"model": ["scorecard", "lgbm"], "brier": [0.05, 0.049]})
    rec = cmp.recommend(disc, diff, prob, {"scorecard": True, "lgbm": True})
    assert not rec["replaces_champion"].iloc[0]
    assert "Keep the scorecard" in cmp.decision_text(rec, disc)


def test_recommend_promotes_a_clearly_better_challenger():
    disc = pd.DataFrame({"model": ["scorecard", "lgbm"]})
    diff = pd.DataFrame([{"challenger": "lgbm", "gini_diff": 0.05, "auc_diff_lo": 0.01}])
    prob = pd.DataFrame({"model": ["scorecard", "lgbm"], "brier": [0.05, 0.049]})
    rec = cmp.recommend(disc, diff, prob, {"scorecard": True, "lgbm": True})
    assert rec["replaces_champion"].iloc[0]


def test_csi_is_zero_for_identical_samples_and_counts_missing():
    df = pd.DataFrame({"x": np.r_[np.arange(100, dtype=float), [np.nan] * 10]})
    t = cmp.csi_table(df, df, ["x"])
    assert np.isclose(t["csi"].iloc[0], 0)
    shifted = df.copy()
    shifted.loc[:50, "x"] = np.nan
    assert cmp.csi_table(df, shifted, ["x"])["csi"].iloc[0] > 0.1


@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
def test_challenger_tuning_and_fit_on_toy_data(kind, tmp_path):
    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    X.loc[rng.random(n) < 0.1, "b"] = np.nan          # NaN must be accepted without imputation
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(-2 + 1.5 * X["a"])))).astype(int))
    cv = tune(kind, X, y, n_iter=2, n_splits=3, verbose=False)
    assert cv["cv_auc_mean"].iloc[0] >= cv["cv_auc_mean"].iloc[-1] and cv["cv_auc_mean"].iloc[0] > 0.7
    p, n_trees = best_params(cv, kind)
    m = TreeModel(kind, fit_final(kind, p, n_trees, X, y), *scaling_params(), features=["a", "b"])
    pd_ = m.predict_pd(X)
    assert ((pd_ > 0) & (pd_ < 1)).all()
    assert abs(pd_.mean() - y.mean()) < 0.02            # no class weights: mean PD close to bad rate


def test_excel_report_writes_tables_and_images(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    img = tmp_path / "f.png"
    fig.savefig(img)
    r = ExcelReport()
    r.add_text("Summary", "Title", style="title")
    r.add_table("Summary", pd.DataFrame({"model": ["a"], "hold_auc": [0.86], "bad_rate": [0.07]}), title="t")
    r.add_image("Summary", img)
    out = r.save(tmp_path / "r.xlsx")
    from openpyxl import load_workbook
    ws = load_workbook(out)["Summary"]
    assert ws["A1"].value == "Title" and len(ws._images) == 1
