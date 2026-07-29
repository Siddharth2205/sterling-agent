"""Point-in-time feature engineering for the research model.

Every feature for (ticker, date t) is computable from information available at t:
price history up to t, fundamentals filed by t, and macro readings at t. No look-ahead.

Design: compute rolling indicators over each ticker's full series once (vectorized),
then sample rows every `step` trading days to cut autocorrelation between observations.
Fundamentals (which don't reach the full 10y from free data) are joined as-of each date
and left as NaN when unavailable — HistGradientBoosting handles missing values natively.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

# Feature columns produced (kept explicit so the model layer can rely on the schema).
PRICE_FEATURES = [
    "ret_21", "ret_63", "ret_126", "ret_252", "mom_252_21", "rev_5",
    "vol_21", "vol_63", "rsi_14", "macd_hist_norm",
    "sma50_ratio", "sma200_ratio", "bb_pctb", "vol_ratio", "high_252_prox",
]
FUND_FEATURES = ["pe", "revenue_growth", "debt_to_equity", "roe", "fcf_yield_pct"]
MACRO_FEATURES = ["vix", "mkt_ret_63", "mkt_vol_21"]
ALL_FEATURES = PRICE_FEATURES + FUND_FEATURES + MACRO_FEATURES


def _price_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized rolling features over one ticker's full OHLCV history."""
    close, vol = df["Close"], df["Volume"]
    out = pd.DataFrame(index=df.index)

    out["ret_21"] = close / close.shift(21) - 1
    out["ret_63"] = close / close.shift(63) - 1
    out["ret_126"] = close / close.shift(126) - 1
    out["ret_252"] = close / close.shift(252) - 1
    out["mom_252_21"] = close.shift(21) / close.shift(252) - 1   # 12m momentum, skip last month
    out["rev_5"] = close / close.shift(5) - 1                    # short-term reversal
    ret_1d = close.pct_change()
    out["vol_21"] = ret_1d.rolling(21).std()
    out["vol_63"] = ret_1d.rolling(63).std()

    try:
        out["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        out["rsi_14"] = np.nan
    try:
        macd = ta.trend.MACD(close)
        hist = macd.macd() - macd.macd_signal()
        out["macd_hist_norm"] = hist / close      # scale-free
    except Exception:
        out["macd_hist_norm"] = np.nan

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out["sma50_ratio"] = close / sma50 - 1
    out["sma200_ratio"] = close / sma200 - 1

    try:
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        rng = bb.bollinger_hband() - bb.bollinger_lband()
        out["bb_pctb"] = (close - bb.bollinger_lband()) / rng.replace(0, np.nan)
    except Exception:
        out["bb_pctb"] = np.nan

    out["vol_ratio"] = vol / vol.rolling(20).mean()
    out["high_252_prox"] = close / close.rolling(252).max()
    return out


def _market_frame(bench_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Regime features from a benchmark price series (by date)."""
    if bench_df is None or bench_df.empty:
        return None
    close = bench_df["Close"]
    m = pd.DataFrame(index=bench_df.index)
    m["mkt_ret_63"] = close / close.shift(63) - 1
    m["mkt_vol_21"] = close.pct_change().rolling(21).std()
    m.index = m.index.date
    return m


def build_features(
    prices: dict[str, pd.DataFrame],
    benchmarks: dict[str, pd.DataFrame],
    market_of,
    hist_fund_fn=None,
    vix_by_date: Optional[dict] = None,
    step: int = 5,
    warmup: int = 252,
) -> pd.DataFrame:
    """Build the sampled feature matrix.

    Args:
        prices: {ticker: OHLCV df}
        benchmarks: {"US": spy_df, "CA": xic_df}
        market_of: fn(ticker) -> "US"|"CA"
        hist_fund_fn: fn(ticker) -> point-in-time fundamental records (or None to skip)
        vix_by_date: {date: vix_level} (or None)
        step: sample every `step` trading days per ticker
        warmup: skip the first `warmup` bars (need history for 252d features)
    """
    from sterling.research.fundamentals import fundamentals_as_of

    market_frames = {k: _market_frame(v) for k, v in benchmarks.items()}
    rows = []

    for tk, df in prices.items():
        if len(df) < warmup + step:
            continue
        feats = _price_feature_frame(df)
        mkt = market_of(tk)
        mframe = market_frames.get(mkt)
        records = hist_fund_fn(tk) if hist_fund_fn else []

        positions = range(warmup, len(df), step)
        for pos in positions:
            ts = df.index[pos]
            d = ts.date()
            row = {"date": d, "ticker": tk, "market": mkt}
            fr = feats.iloc[pos]
            for c in PRICE_FEATURES:
                row[c] = float(fr[c]) if pd.notna(fr[c]) else np.nan

            price = float(df["Close"].iloc[pos])
            fund = fundamentals_as_of(records, d, price) if records else {}
            for c in FUND_FEATURES:
                row[c] = float(fund[c]) if c in fund and pd.notna(fund[c]) else np.nan

            row["vix"] = float(vix_by_date[d]) if (vix_by_date and d in vix_by_date) else np.nan
            if mframe is not None and d in mframe.index:
                mr = mframe.loc[d]
                row["mkt_ret_63"] = float(mr["mkt_ret_63"]) if pd.notna(mr["mkt_ret_63"]) else np.nan
                row["mkt_vol_21"] = float(mr["mkt_vol_21"]) if pd.notna(mr["mkt_vol_21"]) else np.nan
            else:
                row["mkt_ret_63"] = np.nan
                row["mkt_vol_21"] = np.nan
            rows.append(row)

    df_out = pd.DataFrame(rows)
    logger.info(f"built {len(df_out)} feature rows across {df_out['ticker'].nunique() if len(df_out) else 0} tickers")
    return df_out
