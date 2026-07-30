"""Tests for the walk-forward model gate: it should FIND a planted signal and
REJECT pure noise (so a passing verdict means something)."""

import numpy as np
import pandas as pd

from sterling.research import model


def _synth(n_dates=90, n_tickers=40, signal_strength=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_dates).date
    rows = []
    for d in dates:
        for t in range(n_tickers):
            f1, f2, f3 = rng.normal(), rng.normal(), rng.normal()
            noise = rng.normal(0, 0.05)
            label = signal_strength * f1 + noise
            rows.append({"date": d, "ticker": f"T{t}", "f1": f1, "f2": f2, "f3": f3, "label": label})
    return pd.DataFrame(rows)


class TestWalkForwardGate:
    def test_finds_planted_signal(self):
        df = _synth(signal_strength=0.5, seed=1)
        rep = model.walk_forward(df, n_folds=3, horizon=2, features=["f1", "f2", "f3"])
        assert rep["pooled_ic"] > 0.1
        assert rep["fold_ic_positive_frac"] >= 0.66

    def test_rejects_pure_noise(self):
        df = _synth(signal_strength=0.0, seed=2)
        rep = model.walk_forward(df, n_folds=3, horizon=2, features=["f1", "f2", "f3"])
        assert abs(rep["pooled_ic"]) < 0.1
        assert rep["ship_recommendation"] is False

    def test_usable_features_drops_all_nan(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan], "c": [5.0, 5.0, 5.0]})
        used = model._usable_features(train, ["a", "b", "c"])
        assert used == ["a"]  # b all-NaN, c constant → both dropped

    def test_usable_features_ignores_missing_columns(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        assert model._usable_features(train, ["a", "not_there"]) == ["a"]
