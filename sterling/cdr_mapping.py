"""CIBC Canadian Depositary Receipt (CDR) catalog.

CDRs trade on the NEO Exchange (.NE suffix in yfinance).
Each CDR is:
  - Priced in CAD
  - FX-hedged (no currency drag vs the US underlying)
  - Backed 1:N fractional shares of the underlying US stock

Key properties for Sterling:
  - Price data: use the .NE ticker (NEO exchange)
  - Fundamentals / news: redirect to the US underlying symbol
  - Volume minimum: 500k avg daily shares (higher than TSX 100k floor)
  - No FX drag — CDRs are CAD-settled and hedged
"""

# Minimum average daily volume for CDR tickers.
# Higher than TSX floor (100k) because CDR liquidity is thinner
# and we need a clean price feed for the backtester.
CDR_VOLUME_MIN = 500_000

# CDR_UNIVERSE: {.NE ticker → US underlying yfinance symbol}
# Source: CIBC CDR program roster — verified yfinance-resolvable as of 2026-05.
# Covers tech, financial, healthcare, consumer, industrial, energy, and semiconductor sectors.
CDR_UNIVERSE: dict[str, str] = {
    # ── Mega-cap tech ─────────────────────────────────────────────────────────
    "AMZN.NE": "AMZN",
    "AAPL.NE": "AAPL",
    "META.NE":  "META",
    "MSFT.NE":  "MSFT",
    "NVDA.NE":  "NVDA",
    "NFLX.NE":  "NFLX",
    "TSLA.NE":  "TSLA",
    # ── Software / SaaS ───────────────────────────────────────────────────────
    "ADBE.NE":  "ADBE",
    "CRM.NE":   "CRM",
    "ORCL.NE":  "ORCL",
    "IBM.NE":   "IBM",
    "NOW.NE":   "NOW",
    "PLTR.NE":  "PLTR",
    # ── Cybersecurity ─────────────────────────────────────────────────────────
    "CRWD.NE":  "CRWD",
    "PANW.NE":  "PANW",
    # ── Semiconductors ────────────────────────────────────────────────────────
    "AMD.NE":   "AMD",
    "INTC.NE":  "INTC",
    "QCOM.NE":  "QCOM",
    "AVGO.NE":  "AVGO",
    "TXN.NE":   "TXN",
    "MU.NE":    "MU",
    "AMAT.NE":  "AMAT",
    "ASML.NE":  "ASML",
    "TSM.NE":   "TSM",
    "MRVL.NE":  "MRVL",
    "LRCX.NE":  "LRCX",
    "KLAC.NE":  "KLAC",
    # ── Gig / Platform ────────────────────────────────────────────────────────
    "UBER.NE":  "UBER",
    "ABNB.NE":  "ABNB",
    "PYPL.NE":  "PYPL",
    "COIN.NE":  "COIN",
    "SPOT.NE":  "SPOT",
    # ── Financials ────────────────────────────────────────────────────────────
    "JPM.NE":   "JPM",
    "BAC.NE":   "BAC",
    "GS.NE":    "GS",
    "MS.NE":    "MS",
    "V.NE":     "V",
    "MA.NE":    "MA",
    "BLK.NE":   "BLK",
    "AXP.NE":   "AXP",
    "SPGI.NE":  "SPGI",
    # ── Healthcare ────────────────────────────────────────────────────────────
    "JNJ.NE":   "JNJ",
    "UNH.NE":   "UNH",
    "PFE.NE":   "PFE",
    "MRK.NE":   "MRK",
    "ABBV.NE":  "ABBV",
    "LLY.NE":   "LLY",
    "AMGN.NE":  "AMGN",
    "GILD.NE":  "GILD",
    # ── Consumer / Retail ─────────────────────────────────────────────────────
    "WMT.NE":   "WMT",
    "COST.NE":  "COST",
    "HD.NE":    "HD",
    "SBUX.NE":  "SBUX",
    "NKE.NE":   "NKE",
    "PG.NE":    "PG",
    "PEP.NE":   "PEP",
    "DIS.NE":   "DIS",
    # ── Industrial / Aerospace ────────────────────────────────────────────────
    "BA.NE":    "BA",
    "LMT.NE":   "LMT",
    "RTX.NE":   "RTX",
    "HON.NE":   "HON",
    "DE.NE":    "DE",
    "GE.NE":    "GE",
    "MMM.NE":   "MMM",
    # ── Energy ────────────────────────────────────────────────────────────────
    "XOM.NE":   "XOM",
}

# CDR tickers as a list, for convenience
CDR_TICKERS: list[str] = sorted(CDR_UNIVERSE.keys())

# US underlyings that are energy sector (for oil-crash overlay)
CDR_ENERGY_UNDERLYINGS = {"XOM"}


def is_cdr(ticker: str) -> bool:
    """True if ticker is a NEO-exchange CDR."""
    return ticker.endswith(".NE")


def get_underlying(ticker: str) -> str:
    """Return the US underlying for a CDR ticker; return ticker unchanged if not a CDR."""
    return CDR_UNIVERSE.get(ticker, ticker)


def get_universe(mode: str) -> list[str]:
    """
    Return the ticker list for a given universe mode.

    Modes:
      tsx      — TSX 60 constituents only (default historical behaviour)
      cdr      — CDR tickers only (.NE)
      tsx-cdr  — TSX 60 + CDR tickers combined (recommended for live use)
    """
    from sterling.backtester import TSX60_TICKERS  # avoid circular at module level

    if mode == "tsx":
        return list(TSX60_TICKERS)
    if mode == "cdr":
        return list(CDR_TICKERS)
    if mode == "tsx-cdr":
        seen = set()
        result = []
        for t in list(TSX60_TICKERS) + list(CDR_TICKERS):
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result
    raise ValueError(f"Unknown universe mode: {mode!r}. Choose from: tsx, cdr, tsx-cdr")
