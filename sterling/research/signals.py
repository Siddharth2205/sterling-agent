"""Multiple signals + a blend — the 'breadth' lever.

One signal, however well-built, gives a modest market-neutral Sharpe. The professional
edge comes from combining several *uncorrelated* signals: their Sharpes add roughly in
quadrature (the fundamental law of active management), so a handful of mediocre-but-
independent signals blend into a strong one.

Signals are combined by averaging their cross-sectional (per-date) z-scores, which puts
every signal on the same scale before blending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    """Cross-sectional z-score of `col` within each date (mean 0, std 1 per day)."""
    g = df.groupby("date")[col]
    sd = g.transform("std")
    return (df[col] - g.transform("mean")) / sd.where(sd > 0, np.nan)


def combine(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Equal-weight blend of several signals via their per-date z-scores."""
    zs = [zscore_by_date(df, c) for c in cols]
    return sum(zs) / len(zs)
