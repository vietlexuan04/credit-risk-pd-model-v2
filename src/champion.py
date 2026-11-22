"""
Champion model: the Phase 1 WoE logistic-regression scorecard.

load_champion() rebuilds the scorecard from the specs saved in Phase 1
(binning_spec.json, model_spec.json, scorecard_spec.json). This is the exact model
validated in notebook 07, so its numbers match the README and the reports.

The training code itself (notebooks 04-06 in one function) is train_scorecard()
in train.py; it is only run with `python train.py --refit-scorecard`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import WoEBinner
from scorecard import Scorecard, score_to_pd


class ChampionScorecard:
    """Thin wrapper so the scorecard has the same interface as the challengers."""

    name = "scorecard"

    def __init__(self, binners: dict, model_spec: dict, scorecard_spec: dict):
        self.factor, self.offset = scorecard_spec["factor"], scorecard_spec["offset"]
        self.intercept = model_spec["intercept"]
        self.coefs = model_spec["coefficients"]
        self.binners = {v: binners[v] for v in self.coefs}
        self.card = Scorecard(binners, self.intercept, self.coefs, self.factor, self.offset, round_points=True)
        self.features = list(self.coefs)

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Rounded scorecard points (the score a credit officer would see)."""
        return self.card.score(df)

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        """PD implied by the rounded score (same definition as notebook 07)."""
        return np.asarray(score_to_pd(self.score(df), self.factor, self.offset))

    def woe(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({v: b.transform(df[v]) for v, b in self.binners.items()}, index=df.index)

    def contributions(self, df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        """Log-odds contribution of each variable, centred on the reference (development) mean.

        For a linear model with independent inputs this IS the SHAP value:
        phi_i = coef_i * (WoE_i - mean WoE_i). Units: log-odds of bad, same as tree SHAP.
        """
        w, w_ref = self.woe(df), self.woe(reference)
        return pd.DataFrame({v: self.coefs[v] * (w[v] - w_ref[v].mean()) for v in self.coefs}, index=df.index)

    def points_breakdown(self, row: pd.DataFrame) -> pd.DataFrame:
        """Points of each variable for ONE applicant (row = one-row DataFrame)."""
        out = []
        for v, b in self.binners.items():
            idx = int(b.bin_index(row[v]).iloc[0])
            out.append({"variable": v, "value": row[v].iloc[0], "bin_index": idx, "points": self.card.points[v][idx]})
        return pd.DataFrame(out)


def load_champion(scorecard_dir) -> ChampionScorecard:
    d = Path(scorecard_dir)
    binners = {v: WoEBinner.from_dict(s) for v, s in json.loads((d / "binning_spec.json").read_text()).items()}
    model_spec = json.loads((d / "model_spec.json").read_text())
    scorecard_spec = json.loads((d / "scorecard_spec.json").read_text())
    return ChampionScorecard(binners, model_spec, scorecard_spec)
