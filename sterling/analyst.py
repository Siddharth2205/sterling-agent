"""Five-axis scoring engine. Each axis 0-100; final = weighted sum."""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import ta

from sterling import data_feed

logger = logging.getLogger(__name__)

# ── Axis weights (3-axis scoring model) ───────────────────────────────────────
# Sentiment and insider are NOT in WEIGHTS — they cannot be backtested on free-tier
# historical data (no dated Finnhub records, no historical pytrends per date).
# They are still fetched in live mode and surfaced in alerts as INFORMATIONAL
# context, but they do not move the confidence number.
# Rationale: a model where two axes silently return 50 (neutral sentinel) 60% of
# the time is not a 5-axis model in practice. A clean 3-axis model with known
# data quality is more honest and more consistently testable.
WEIGHTS = {
    "technical":   0.40,
    "fundamental": 0.40,
    "macro":       0.20,
}

# Axes fetched for informational display but excluded from the score.
INFORMATIONAL_AXES = ("sentiment", "insider")

# ── Signal → recommendation mapping ──────────────────────────────────────────
def _confidence_to_rec(score: float) -> str:
    if score >= 75:
        return "BUY"
    if score >= 60:
        return "ACCUMULATE"
    if score >= 40:
        return "HOLD"
    if score >= 25:
        return "TRIM"
    return "SELL"


# ── Position sizing rules ($1k CAD account) ───────────────────────────────────
MIN_PRICE = 5.0
MIN_AVG_VOLUME = 100_000          # TSX / TSX-V floor
CDR_MIN_AVG_VOLUME = 500_000      # NEO CDR floor — thinner market, higher liquidity bar
MAX_RISK_CAD = 20.0      # 2% of $1,000
MAX_POSITIONS = 4
TARGET_POSITION_CAD = 250.0


def _check_tradeable(ticker: str, price: float, avg_volume: Optional[float]) -> tuple[bool, str]:
    if price < MIN_PRICE:
        return False, f"Price ${price:.2f} < ${MIN_PRICE} minimum"
    if avg_volume is not None and avg_volume < MIN_AVG_VOLUME:
        return False, f"Avg volume {avg_volume:,.0f} < {MIN_AVG_VOLUME:,} minimum"
    return True, ""


def _compute_stop_and_target(price: float, confidence: float, hist: pd.DataFrame) -> tuple[float, float, float]:
    """Derive stop-loss (ATR-based), price target, and R:R ratio."""
    try:
        atr_indicator = ta.volatility.AverageTrueRange(
            high=hist["High"], low=hist["Low"], close=hist["Close"], window=14
        )
        atr = float(atr_indicator.average_true_range().dropna().iloc[-1])
    except Exception:
        atr = price * 0.03  # fallback: 3% of price

    stop_loss = round(price - 2 * atr, 2)
    risk = price - stop_loss

    # Target: scale with confidence (higher confidence → more ambitious target)
    target_multiplier = 1.5 + (confidence - 40) / 100  # 1.5x to 2.1x risk at 100 conf
    target = round(price + risk * max(1.0, target_multiplier), 2)
    rr = round((target - price) / risk, 2) if risk > 0 else 0.0

    return stop_loss, target, rr


def _entry_zone(price: float, atr: Optional[float] = None) -> list:
    spread = atr * 0.5 if atr else price * 0.015
    return [round(price - spread, 2), round(price + spread, 2)]


# ── Axis scorers ─────────────────────────────────────────────────────────────

def score_technical(hist: pd.DataFrame) -> float:
    """RSI, MACD, SMA 50/200 cross, Bollinger position, volume profile → 0-100."""
    if hist is None or len(hist) < 50:
        return 50.0

    signals = []
    close = hist["Close"]
    volume = hist["Volume"] if "Volume" in hist.columns else None

    # RSI(14)
    try:
        rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi().dropna()
        if not rsi_series.empty:
            rsi = float(rsi_series.iloc[-1])
            if rsi < 30:
                signals.append(90)   # oversold — bullish setup
            elif rsi < 45:
                signals.append(70)
            elif rsi < 55:
                signals.append(50)
            elif rsi < 70:
                signals.append(40)
            else:
                signals.append(20)   # overbought
    except Exception:
        pass

    # MACD
    try:
        macd_obj = ta.trend.MACD(close)
        macd_line = macd_obj.macd().dropna()
        signal_line = macd_obj.macd_signal().dropna()
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            cross_up = (macd_line.iloc[-1] > signal_line.iloc[-1]) and (macd_line.iloc[-2] <= signal_line.iloc[-2])
            cross_dn = (macd_line.iloc[-1] < signal_line.iloc[-1]) and (macd_line.iloc[-2] >= signal_line.iloc[-2])
            above = macd_line.iloc[-1] > signal_line.iloc[-1]
            if cross_up:
                signals.append(90)
            elif cross_dn:
                signals.append(10)
            elif above:
                signals.append(65)
            else:
                signals.append(35)
    except Exception:
        pass

    # SMA 50/200 Golden/Death Cross
    try:
        sma50 = ta.trend.SMAIndicator(close, window=50).sma_indicator().dropna()
        sma200 = ta.trend.SMAIndicator(close, window=200).sma_indicator().dropna()
        if not sma50.empty and not sma200.empty:
            if sma50.iloc[-1] > sma200.iloc[-1]:
                signals.append(75)   # golden cross / above 200
            else:
                signals.append(25)   # death cross
    except Exception:
        pass

    # Bollinger Band position
    try:
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        upper = bb.bollinger_hband().dropna()
        lower = bb.bollinger_lband().dropna()
        mid = bb.bollinger_mavg().dropna()
        if not upper.empty and not lower.empty:
            p = close.iloc[-1]
            band_range = float(upper.iloc[-1]) - float(lower.iloc[-1])
            if band_range > 0:
                pct_b = (p - float(lower.iloc[-1])) / band_range
                if pct_b < 0.2:
                    signals.append(80)   # near lower band — potential reversal
                elif pct_b > 0.8:
                    signals.append(30)   # near upper band
                else:
                    signals.append(55)
    except Exception:
        pass

    # Volume trend (is recent volume above 20-day average?)
    try:
        if volume is not None and len(volume) >= 20:
            avg_vol = float(volume.tail(20).mean())
            recent_vol = float(volume.iloc[-1])
            if recent_vol > avg_vol * 1.5:
                signals.append(70)
            elif recent_vol > avg_vol:
                signals.append(55)
            else:
                signals.append(40)
    except Exception:
        pass

    return float(np.mean(signals)) if signals else 50.0


def score_fundamental(fundamentals: dict) -> float:
    """P/E, FCF yield, debt/equity, revenue growth, ROE → 0-100."""
    signals = []

    pe = fundamentals.get("pe")
    if pe is not None:
        if pe < 0:
            signals.append(15)
        elif pe < 15:
            signals.append(80)
        elif pe < 25:
            signals.append(60)
        elif pe < 40:
            signals.append(40)
        else:
            signals.append(20)

    fcf_yield = fundamentals.get("fcf_yield_pct")
    if fcf_yield is not None:
        if fcf_yield > 8:
            signals.append(90)
        elif fcf_yield > 4:
            signals.append(70)
        elif fcf_yield > 0:
            signals.append(50)
        elif fcf_yield < 0:
            signals.append(20)

    de = fundamentals.get("debt_to_equity")
    if de is not None:
        # debt_to_equity from yfinance is already as a ratio (not %)
        de_norm = de / 100 if de > 10 else de  # yfinance returns 0-500 range sometimes
        if de_norm < 0.3:
            signals.append(85)
        elif de_norm < 0.7:
            signals.append(65)
        elif de_norm < 1.5:
            signals.append(45)
        else:
            signals.append(20)

    rev_growth = fundamentals.get("revenue_growth")
    if rev_growth is not None:
        if rev_growth > 0.20:
            signals.append(90)
        elif rev_growth > 0.10:
            signals.append(70)
        elif rev_growth > 0:
            signals.append(55)
        elif rev_growth > -0.10:
            signals.append(40)
        else:
            signals.append(20)

    roe = fundamentals.get("roe")
    if roe is not None:
        if roe > 0.20:
            signals.append(85)
        elif roe > 0.12:
            signals.append(65)
        elif roe > 0:
            signals.append(45)
        else:
            signals.append(20)

    return float(np.mean(signals)) if signals else 50.0


def score_sentiment(ticker: str, finnhub_key: str, skip_trends: bool = False) -> float:
    """News sentiment + Google Trends slope → 0-100.

    When *skip_trends* is True the pytrends call is skipped entirely
    (used by the scan path to avoid 429 rate-limits on ~200 tickers).
    """
    signals = []

    news = data_feed.get_news_sentiment(ticker, finnhub_key)
    sent_score = news.get("sentiment_score", 0.0)
    # Map [-1, 1] → [0, 100]
    signals.append((sent_score + 1) / 2 * 100)

    if not skip_trends:
        trends_slope = data_feed.get_trends_slope(ticker)
        # Map [-1, 1] → [0, 100]
        signals.append((trends_slope + 1) / 2 * 100)

    return float(np.mean(signals)) if signals else 50.0


def score_macro(macro: dict, ticker: str = "") -> float:
    """Global macro context score → 0-100."""
    signals = []

    vix = macro.get("vix")
    if vix is not None:
        if vix < 15:
            signals.append(75)
        elif vix < 20:
            signals.append(60)
        elif vix < 25:
            signals.append(45)
        else:
            signals.append(25)

    tsx_chg = macro.get("tsx_change_pct")
    if tsx_chg is not None:
        if tsx_chg > 1:
            signals.append(75)
        elif tsx_chg > 0:
            signals.append(60)
        elif tsx_chg > -1:
            signals.append(50)
        elif tsx_chg > -2:
            signals.append(35)
        else:
            signals.append(15)

    usd_cad = macro.get("usd_cad")
    if usd_cad is not None:
        # Weaker CAD (higher USD/CAD) benefits exporters; neutral for domestic names
        if usd_cad > 1.40:
            signals.append(40)   # very weak CAD — import-heavy names hurt
        elif usd_cad > 1.35:
            signals.append(50)
        else:
            signals.append(60)

    return float(np.mean(signals)) if signals else 50.0


def score_insider(ticker: str, finnhub_key: str) -> float:
    """Insider buy/sell ratio + volume anomaly → 0-100."""
    signals = []

    insider = data_feed.get_insider_activity(ticker, finnhub_key)
    buy_ratio = insider.get("buy_ratio", 0.5)
    net = insider.get("net_shares", 0)

    if buy_ratio > 0.7:
        signals.append(85)
    elif buy_ratio > 0.5:
        signals.append(65)
    elif buy_ratio > 0.3:
        signals.append(40)
    else:
        signals.append(20)

    if net > 100_000:
        signals.append(80)
    elif net > 0:
        signals.append(60)
    elif net < -100_000:
        signals.append(25)
    else:
        signals.append(45)

    return float(np.mean(signals)) if signals else 50.0


# ── Macro overlay ─────────────────────────────────────────────────────────────

def apply_macro_overlay(confidence: float, macro: dict, ticker: str = "", is_energy: bool = False) -> float:
    """Downgrade BUY signals by 15 points when macro tape is risk-off."""
    risk_off = macro.get("risk_off", False)
    oil_crash = macro.get("oil_crash", False) and is_energy

    if risk_off or oil_crash:
        logger.info(f"Macro overlay: risk-off detected, downgrading {ticker} by 15 pts")
        return max(0.0, confidence - 15.0)

    return confidence


# ── Thesis generator ──────────────────────────────────────────────────────────

def _generate_thesis(ticker: str, rec: str, scores: dict, fundamentals: dict, macro: dict) -> str:
    """Two-sentence plain-English thesis from scoring axes only (technical, fundamental, macro)."""
    scoring_scores = {k: scores[k] for k in WEIGHTS}
    strongest = max(scoring_scores, key=scoring_scores.get)
    weakest = min(scoring_scores, key=scoring_scores.get)

    axis_labels = {
        "technical": "technical setup",
        "fundamental": "fundamental profile",
        "macro": "macro backdrop",
    }

    pe_str = f"P/E of {fundamentals.get('pe'):.1f}" if fundamentals.get("pe") else "valuation"
    rev_str = (
        f"revenue growing {fundamentals.get('revenue_growth', 0)*100:.0f}% YoY"
        if fundamentals.get("revenue_growth") else ""
    )

    s1 = f"{ticker} shows its strongest conviction in {axis_labels[strongest]} ({scoring_scores[strongest]:.0f}/100)"
    if rev_str:
        s1 += f", with {pe_str} and {rev_str}."
    else:
        s1 += "."

    if rec in ("BUY", "ACCUMULATE"):
        s2 = (
            f"Risk-reward is constructive; primary risk is "
            f"{axis_labels[weakest]} ({scoring_scores[weakest]:.0f}/100) — warrants a tight stop."
        )
    elif rec in ("TRIM", "SELL"):
        s2 = (
            f"Weak {axis_labels[weakest]} ({scoring_scores[weakest]:.0f}/100) leads the bear case; "
            "reduce or exit on any bounce."
        )
    else:
        s2 = (
            f"Mixed signals — {axis_labels[weakest]} ({scoring_scores[weakest]:.0f}/100) "
            "is the key watch point; no action until conviction improves."
        )

    return f"{s1} {s2}"


# ── Main scorer ───────────────────────────────────────────────────────────────

def analyze(
    ticker: str,
    finnhub_key: str,
    macro: Optional[dict] = None,
    is_energy: bool = False,
    current_price: Optional[float] = None,
    skip_trends: bool = False,
) -> dict:
    """
    Run the full 5-axis analysis on a ticker.
    Returns the standard signal dict with recommendation, confidence, thesis, etc.
    """
    logger.info(f"Analyzing {ticker}...")
    now = datetime.now(timezone.utc).isoformat()

    # Fetch data
    try:
        hist = data_feed.get_history(ticker, period="1y")
    except Exception as e:
        logger.error(f"Failed to get history for {ticker}: {e}")
        hist = pd.DataFrame()

    fundamentals = data_feed.get_fundamentals(ticker)

    if macro is None:
        try:
            macro = data_feed.get_macro_indicators()
        except Exception:
            macro = {"risk_off": False, "oil_crash": False}

    if current_price is None:
        try:
            q = data_feed.get_quote(ticker)
            current_price = q["price"]
        except Exception:
            current_price = None

    # Score all axes — only WEIGHTS axes drive confidence; others are informational.
    all_scores = {
        "technical":   score_technical(hist),
        "fundamental": score_fundamental(fundamentals),
        "macro":       score_macro(macro, ticker),
        # Informational — fetched for alert context, not weighted in score.
        "sentiment":   score_sentiment(ticker, finnhub_key, skip_trends=skip_trends),
        "insider":     score_insider(ticker, finnhub_key),
    }

    # Clamp all to [0, 100]
    scores = {k: max(0.0, min(100.0, v)) for k, v in all_scores.items()}

    # Weighted confidence (3-axis only)
    confidence_raw = sum(scores[axis] * weight for axis, weight in WEIGHTS.items())
    confidence_raw = apply_macro_overlay(confidence_raw, macro, ticker, is_energy)
    confidence = round(max(0.0, min(100.0, confidence_raw)), 1)

    rec = _confidence_to_rec(confidence)

    # Entry, stop, target
    stop_loss = None
    target = None
    rr = None
    entry_zone = None

    if current_price:
        try:
            atr_obj = ta.volatility.AverageTrueRange(
                high=hist["High"], low=hist["Low"], close=hist["Close"], window=14
            )
            atr_val = float(atr_obj.average_true_range().dropna().iloc[-1]) if not hist.empty else current_price * 0.03
        except Exception:
            atr_val = current_price * 0.03

        stop_loss, target, rr = _compute_stop_and_target(current_price, confidence, hist)
        entry_zone = _entry_zone(current_price, atr_val)

        # Enforce $20 max risk rule
        risk_per_share = current_price - stop_loss
        if risk_per_share > 0:
            max_shares = int(MAX_RISK_CAD / risk_per_share)
            max_cost = max_shares * current_price
            if max_cost < TARGET_POSITION_CAD * 0.5:
                # Stop is too wide for a $250 position — warn
                logger.warning(
                    f"{ticker}: stop distance ${risk_per_share:.2f} limits position to {max_shares} shares "
                    f"(${max_cost:.0f}) — consider wider stop or skip."
                )

    # FX / CDR flags
    if ticker.endswith(".NE"):
        # CDR: CAD-settled, FX-hedged — no FX drag, but note for user context
        fx_warning = "CAD-settled CDR (FX-hedged). No FX drag; check CDR premium/discount to NAV."
    elif ".TO" not in ticker and ".V" not in ticker:
        fx_warning = "USD trade — add 1.5% FX drag to cost estimate."
    else:
        fx_warning = None

    thesis = _generate_thesis(ticker, rec, scores, fundamentals, macro)

    return {
        "ticker": ticker,
        "recommendation": rec,
        "confidence": confidence,
        "thesis": thesis,
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "target": target,
        "risk_reward": rr,
        "signals": {k: round(v, 1) for k, v in scores.items()},
        "current_price": current_price,
        "fx_warning": fx_warning,
        "tradeable": current_price is not None and current_price >= MIN_PRICE,
        "timestamp": now,
    }


def analyze_portfolio(tickers: list, finnhub_key: str) -> list:
    """Analyze all tickers in a list. Returns sorted by confidence descending."""
    macro = None
    try:
        macro = data_feed.get_macro_indicators()
    except Exception:
        macro = {"risk_off": False, "oil_crash": False}

    energy_keywords = {"ENB", "SU", "CVE", "CNQ", "TRP", "PPL", "WCP", "ARX", "PEY", "TVE"}

    results = []
    for ticker in tickers:
        base = ticker.replace(".TO", "").replace(".V", "")
        is_energy = base.upper() in energy_keywords
        try:
            result = analyze(ticker, finnhub_key, macro=macro, is_energy=is_energy)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")
            results.append({"ticker": ticker, "error": str(e), "recommendation": "ERROR", "confidence": 0})

    return sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
