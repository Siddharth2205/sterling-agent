"""Tests for sterling/analyst.py — determinism, scoring logic, macro overlay."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from sterling import analyst


def _make_hist(n: int = 250, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV history."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    base = 100.0
    if trend == "up":
        closes = base + np.cumsum(rng.normal(0.3, 1.0, n))
    elif trend == "down":
        closes = base + np.cumsum(rng.normal(-0.3, 1.0, n))
    else:
        closes = base + np.cumsum(rng.normal(0.0, 1.0, n))

    closes = np.clip(closes, 5.0, None)
    df = pd.DataFrame({
        "Open":   closes * rng.uniform(0.99, 1.01, n),
        "High":   closes * rng.uniform(1.00, 1.02, n),
        "Low":    closes * rng.uniform(0.98, 1.00, n),
        "Close":  closes,
        "Volume": rng.integers(200_000, 2_000_000, n).astype(float),
    }, index=dates)
    return df


class TestScoreTechnical:
    def test_returns_in_range(self):
        hist = _make_hist(250)
        score = analyst.score_technical(hist)
        assert 0.0 <= score <= 100.0

    def test_deterministic(self):
        hist = _make_hist(250, seed=99)
        s1 = analyst.score_technical(hist)
        s2 = analyst.score_technical(hist)
        assert s1 == s2

    def test_output_is_always_in_valid_range(self):
        # Run on 10 different random histories — score must always be [0, 100].
        for seed in range(10):
            for trend in ("up", "flat", "down"):
                hist = _make_hist(300, trend=trend, seed=seed)
                score = analyst.score_technical(hist)
                assert 0.0 <= score <= 100.0, f"Out of range: {score} (trend={trend}, seed={seed})"

    def test_insufficient_data_returns_50(self):
        thin = _make_hist(30)
        assert analyst.score_technical(thin) == 50.0

    def test_none_returns_50(self):
        assert analyst.score_technical(None) == 50.0


class TestScoreFundamental:
    def test_low_pe_scores_high(self):
        score = analyst.score_fundamental({"pe": 10, "fcf_yield_pct": 6, "roe": 0.18})
        assert score > 65

    def test_negative_pe_scores_low(self):
        score = analyst.score_fundamental({"pe": -5})
        assert score < 30

    def test_empty_dict_returns_50(self):
        assert analyst.score_fundamental({}) == 50.0

    def test_high_fcf_yield_bullish(self):
        s_high = analyst.score_fundamental({"fcf_yield_pct": 10})
        s_low = analyst.score_fundamental({"fcf_yield_pct": 1})
        assert s_high > s_low

    def test_negative_fcf_bearish(self):
        score = analyst.score_fundamental({"fcf_yield_pct": -3})
        assert score < 40

    def test_high_debt_scores_low(self):
        low_debt = analyst.score_fundamental({"debt_to_equity": 20})
        high_debt = analyst.score_fundamental({"debt_to_equity": 200})
        assert low_debt > high_debt


class TestScoreMacro:
    def test_low_vix_bullish(self):
        s_low = analyst.score_macro({"vix": 12, "tsx_change_pct": 0.5})
        s_high = analyst.score_macro({"vix": 30, "tsx_change_pct": -2.5})
        assert s_low > s_high

    def test_tsx_crash_bearish(self):
        score = analyst.score_macro({"vix": 15, "tsx_change_pct": -3.0})
        assert score < 45

    def test_empty_macro_returns_50(self):
        assert analyst.score_macro({}) == 50.0


class TestConfidenceToRec:
    @pytest.mark.parametrize("score,expected", [
        (80, "BUY"),
        (75, "BUY"),
        (70, "ACCUMULATE"),
        (60, "ACCUMULATE"),
        (55, "HOLD"),
        (40, "HOLD"),
        (35, "TRIM"),
        (25, "TRIM"),
        (20, "SELL"),
        (0,  "SELL"),
    ])
    def test_mapping(self, score, expected):
        assert analyst._confidence_to_rec(score) == expected


class TestMacroOverlay:
    def test_risk_off_downgrades_by_15(self):
        macro = {"risk_off": True, "oil_crash": False}
        original = 75.0
        adjusted = analyst.apply_macro_overlay(original, macro, "SHOP.TO", False)
        assert adjusted == pytest.approx(60.0)

    def test_oil_crash_downgrades_energy(self):
        macro = {"risk_off": False, "oil_crash": True}
        adjusted = analyst.apply_macro_overlay(75.0, macro, "ENB.TO", is_energy=True)
        assert adjusted == pytest.approx(60.0)

    def test_oil_crash_no_effect_on_non_energy(self):
        macro = {"risk_off": False, "oil_crash": True}
        adjusted = analyst.apply_macro_overlay(75.0, macro, "SHOP.TO", is_energy=False)
        assert adjusted == pytest.approx(75.0)

    def test_clamps_to_zero(self):
        macro = {"risk_off": True}
        adjusted = analyst.apply_macro_overlay(10.0, macro)
        assert adjusted >= 0.0

    def test_no_overlay_when_risk_on(self):
        macro = {"risk_off": False, "oil_crash": False}
        adjusted = analyst.apply_macro_overlay(72.0, macro)
        assert adjusted == pytest.approx(72.0)


class TestPositionSizingConstants:
    def test_min_price(self):
        assert analyst.MIN_PRICE == 5.0

    def test_min_volume(self):
        assert analyst.MIN_AVG_VOLUME == 100_000

    def test_max_risk(self):
        assert analyst.MAX_RISK_CAD == 20.0

    def test_max_positions(self):
        assert analyst.MAX_POSITIONS == 4


class TestCheckTradeable:
    def test_passes_valid(self):
        ok, reason = analyst._check_tradeable("SHOP.TO", 100.0, 500_000)
        assert ok is True
        assert reason == ""

    def test_fails_low_price(self):
        ok, reason = analyst._check_tradeable("SHOP.TO", 4.99, 500_000)
        assert ok is False
        assert "Price" in reason

    def test_fails_low_volume(self):
        ok, reason = analyst._check_tradeable("SHOP.TO", 100.0, 50_000)
        assert ok is False
        assert "volume" in reason.lower()


class TestStopAndTarget:
    def test_rr_positive(self):
        hist = _make_hist(250)
        stop, target, rr = analyst._compute_stop_and_target(100.0, 70.0, hist)
        assert target > 100.0
        assert stop < 100.0
        assert rr > 0

    def test_target_above_price(self):
        hist = _make_hist(250)
        _, target, _ = analyst._compute_stop_and_target(50.0, 60.0, hist)
        assert target > 50.0


class TestWeights:
    def test_weights_sum_to_one(self):
        total = sum(analyst.WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_all_axes_present(self):
        expected = {"technical", "fundamental", "sentiment", "macro", "insider"}
        assert set(analyst.WEIGHTS.keys()) == expected
