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


class TestFitResilient:
    def test_noop_when_fit_succeeds(self):
        # a healthy fit must not drop any feature (scores stay identical to a plain fit)
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
        y = X["a"] + rng.normal(0, 0.1, 500)
        m, cols = model.fit_resilient(model._make_model(), X, y)
        assert cols == ["a", "b"]

    def test_recovers_by_dropping_binning_crash_feature(self, monkeypatch):
        # simulate sklearn's binning ValueError on the first fit, success once the
        # near-constant feature is dropped
        X = pd.DataFrame({"good": [float(i) for i in range(20)],
                          "bad": [1.0] * 19 + [2.0]})   # fewest distinct
        y = pd.Series([float(i) for i in range(20)])
        calls = {"n": 0}
        real_make = model._make_model()

        class FakeModel:
            def fit(self, Xf, yf):
                calls["n"] += 1
                if "bad" in Xf.columns:
                    raise ValueError("window shape cannot be larger than input array shape")
                self.cols_ = list(Xf.columns)
                return self

        m, cols = model.fit_resilient(FakeModel(), X, y)
        assert cols == ["good"]           # the near-constant feature was dropped
        assert calls["n"] == 2            # failed once, succeeded on retry

    def test_reraises_unrelated_valueerror(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([1.0, 2.0, 3.0])

        class Boom:
            def fit(self, Xf, yf):
                raise ValueError("something else entirely")

        import pytest
        with pytest.raises(ValueError, match="something else"):
            model.fit_resilient(Boom(), X, y)
