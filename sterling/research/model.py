"""Gradient-boosted forward-return model with time-based walk-forward validation.

This module is the ship/no-ship gate. It does NOT tune until something looks good; it
trains on the past, predicts the future it has never seen, and reports whether the
predictions actually correlate with realized excess returns out-of-sample.

Guards against the classic ways a model lies to you:
  - Time-ordered folds (never train on the future).
  - An embargo gap between train and test equal to the label horizon, so a training
    row's forward-return window cannot overlap the test period (label leakage).
  - All metrics are computed on test folds only, then pooled.

Ship criteria (stated before seeing results):
  - Pooled out-of-sample IC clearly > 0 and positive in a majority of folds.
  - Top-quantile realized excess return > 0 with t-stat > 2.
  - Roughly monotonic quantile buckets.
If these fail, the honest output is "no edge" and nothing gets integrated.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from sterling.research.features import ALL_FEATURES

logger = logging.getLogger(__name__)


def _usable_features(train: pd.DataFrame, features: list) -> list:
    """Features with >=2 distinct non-NaN values in this training set.

    Early folds have all-NaN fundamental columns (statements don't reach that far back);
    HistGradientBoosting's binner errors on a constant/all-NaN feature. Dropping them is
    also the honest thing — the model can't use fundamentals in periods that had none.
    """
    return [f for f in features if f in train.columns and train[f].nunique(dropna=True) >= 2]


def _make_model(params: dict | None = None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    # Modest capacity + regularization: limited depth, L2, early-ish stopping. The point
    # is a robust signal, not a maximally-fit training curve. `params` lets the autonomous
    # search vary depth/learning-rate/leaf-size etc.
    base = dict(max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
                min_samples_leaf=200, early_stopping=True, validation_fraction=0.15,
                random_state=42)
    if params:
        base.update(params)
    return HistGradientBoostingRegressor(**base)


def _ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on ranks; no scipy needed)."""
    if len(pred) < 10:
        return float("nan")
    p = pd.Series(pred).rank()
    a = pd.Series(actual).rank()
    return float(p.corr(a))


def _bucket_stats(pred: np.ndarray, label: np.ndarray, n: int = 5) -> list:
    df = pd.DataFrame({"pred": pred, "label": label})
    try:
        df["q"] = pd.qcut(df["pred"].rank(method="first"), n, labels=range(1, n + 1))
    except ValueError:
        return []
    g = df.groupby("q", observed=True)["label"]
    return [{"bucket": int(q), "mean_excess_pct": round(float(m) * 100, 3), "count": int(c)}
            for q, m, c in zip(g.mean().index, g.mean(), g.count())]


def _top_bucket_by_date(test: pd.DataFrame, top_q: float = 0.8) -> dict:
    """Per-date top-quantile basket realized excess return → mean + t-stat."""
    means = []
    for _, day in test.groupby("date"):
        if len(day) < 5:
            continue
        thr = day["pred"].quantile(top_q)
        top = day[day["pred"] >= thr]
        if not top.empty:
            means.append(float(top["label"].mean()))
    if len(means) < 2:
        return {}
    arr = np.array(means)
    t = arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))) if arr.std(ddof=1) > 0 else 0.0
    return {
        "dates": len(arr),
        "mean_excess_pct": round(float(arr.mean()) * 100, 3),
        "hit_rate_pct": round(float((arr > 0).mean()) * 100, 1),
        "t_stat": round(float(t), 2),
    }


def walk_forward(
    df: pd.DataFrame,
    n_folds: int = 5,
    horizon: int = 63,
    features: Optional[list] = None,
    model_params: dict | None = None,
) -> dict:
    """Expanding-window walk-forward. Returns a JSON-friendly report."""
    features = features or ALL_FEATURES
    data = df.dropna(subset=["label"]).sort_values("date").reset_index(drop=True)
    dates = np.array(sorted(data["date"].unique()))
    if len(dates) < (n_folds + 1) * 5:
        return {"error": "not enough distinct dates for walk-forward"}

    # Chronological fold boundaries; first block is the initial training seed.
    blocks = np.array_split(dates, n_folds + 1)
    embargo = pd.Timedelta(days=int(horizon * 1.6))  # calendar gap ≈ trading horizon

    fold_reports = []
    pooled_pred, pooled_label, pooled_frames = [], [], []

    for k in range(1, n_folds + 1):
        test_dates = set(blocks[k])
        test_start = min(blocks[k])
        train = data[data["date"] < (test_start - embargo)]
        test = data[data["date"].isin(test_dates)].copy()
        if len(train) < 500 or len(test) < 100:
            continue

        used = _usable_features(train, features)
        model = _make_model(model_params)
        model.fit(train[used], train["label"])
        test["pred"] = model.predict(test[used])

        ic = _ic(test["pred"].to_numpy(), test["label"].to_numpy())
        fold_reports.append({
            "fold": k,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_start": str(test_start),
            "test_end": str(max(blocks[k])),
            "features_used": len(used),
            "ic": round(ic, 4),
            "buckets": _bucket_stats(test["pred"].to_numpy(), test["label"].to_numpy()),
            "top_bucket": _top_bucket_by_date(test),
        })
        pooled_pred.append(test["pred"].to_numpy())
        pooled_label.append(test["label"].to_numpy())
        pooled_frames.append(test[["date", "ticker", "pred", "label"]])

    if not fold_reports:
        return {"error": "no valid folds"}

    allp = np.concatenate(pooled_pred)
    alll = np.concatenate(pooled_label)
    pooled_test = pd.concat(pooled_frames, ignore_index=True)
    ics = [f["ic"] for f in fold_reports if f["ic"] == f["ic"]]
    if not ics:
        return {"error": "no folds produced a finite IC"}
    pooled_ic = _ic(allp, alll)
    buckets = _bucket_stats(allp, alll)
    top = _top_bucket_by_date(pooled_test)

    ship = bool(
        pooled_ic > 0.02
        and sum(1 for x in ics if x > 0) >= (len(ics) / 2)
        and top.get("t_stat", 0) > 2.0
        and top.get("mean_excess_pct", 0) > 0
    )

    return {
        "n_folds": len(fold_reports),
        "features": len(features),
        "pooled_ic": round(pooled_ic, 4),
        "fold_ics": ics,
        "fold_ic_positive_frac": round(sum(1 for x in ics if x > 0) / len(ics), 2),
        "pooled_buckets": buckets,
        "pooled_top_bucket_vs_benchmark": top,
        "ship_recommendation": ship,
        "folds": fold_reports,
    }


def generate_oos_predictions(
    df: pd.DataFrame,
    n_folds: int = 5,
    horizon: int = 63,
    features: Optional[list] = None,
    model_params: dict | None = None,
) -> pd.DataFrame:
    """Re-run the walk-forward and return the pooled out-of-sample prediction panel
    with realized returns attached: [date, ticker, market, pred, fwd_return, bench_fwd,
    label]. Used by the portfolio simulator so P&L is computed only on genuinely
    unseen predictions."""
    features = features or ALL_FEATURES
    keep = ["date", "ticker", "market", "fwd_return", "bench_fwd", "label"]
    data = df.dropna(subset=["label"]).sort_values("date").reset_index(drop=True)
    dates = np.array(sorted(data["date"].unique()))
    blocks = np.array_split(dates, n_folds + 1)
    embargo = pd.Timedelta(days=int(horizon * 1.6))

    frames = []
    for k in range(1, n_folds + 1):
        test_start = min(blocks[k])
        train = data[data["date"] < (test_start - embargo)]
        test = data[data["date"].isin(set(blocks[k]))].copy()
        if len(train) < 500 or len(test) < 100:
            continue
        used = _usable_features(train, features)
        m = _make_model(model_params)
        m.fit(train[used], train["label"])
        test["pred"] = m.predict(test[used])
        frames.append(test[keep + ["pred"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=keep + ["pred"])


def train_final_model(df: pd.DataFrame, features: Optional[list] = None):
    """Fit on all labelled data (for integration only AFTER walk-forward passes)."""
    features = features or ALL_FEATURES
    data = df.dropna(subset=["label"])
    used = _usable_features(data, features)
    model = _make_model()
    model.fit(data[used], data["label"])
    return model, used
