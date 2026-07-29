"""Broad, liquid research universe (US large caps + TSX).

Deliberately NOT the $1k-Wealthsimple-tradeable set — this is for building and testing
a model, so we want breadth and history. ~200 names across sectors.

Survivorship caveat: this is *today's* liquid membership. Names that delisted or shrank
out of relevance over the last decade are absent, which biases historical results upward.
Point-in-time membership isn't freely available; walk-forward validation and honest
reporting are how we keep ourselves honest, not a clean universe.
"""

from __future__ import annotations

# US large caps — S&P 100 core plus liquid additions across sectors.
_US = [
    # Mega-cap tech / comms
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "CSCO", "ACN", "INTC", "AMD", "QCOM", "TXN", "IBM", "NFLX", "INTU",
    "AMAT", "MU", "ADI", "LRCX", "NOW", "PANW", "SNOW", "UBER", "SHOP", "PYPL",
    # Financials
    "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP",
    "SPGI", "CB", "PGR", "MMC", "USB", "PNC", "TFC", "COF",
    # Health care
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "MDT", "GILD", "CVS", "ISRG", "VRTX", "REGN", "ELV",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "HD", "LOW",
    "TGT", "DIS", "CMCSA", "MDLZ", "CL", "MO", "PM", "BKNG",
    # Industrials / materials / energy
    "CAT", "BA", "HON", "GE", "UPS", "RTX", "LMT", "DE", "UNP", "EMR",
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY",
    "LIN", "SHW", "APD", "FCX", "NEM", "NUE",
    # Utilities / real estate / staples extras
    "NEE", "DUK", "SO", "D", "AEP", "PLD", "AMT", "EQIX", "SPG", "O",
]

# Canadian large caps — TSX 60 core across banks, energy, materials, rails, telecom.
_TSX = [
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",            # banks
    "ENB.TO", "TRP.TO", "CNQ.TO", "SU.TO", "CVE.TO", "IMO.TO", "PPL.TO", "TOU.TO",  # energy
    "CNR.TO", "CP.TO",                                                  # rails
    "SHOP.TO", "CSU.TO", "GIB-A.TO", "OTEX.TO",                         # tech
    "ABX.TO", "AEM.TO", "FNV.TO", "WPM.TO", "TECK-B.TO", "NTR.TO",      # materials
    "BCE.TO", "T.TO", "RCI-B.TO",                                       # telecom
    "MFC.TO", "SLF.TO", "GWO.TO", "IFC.TO", "FFH.TO",                   # insurers/financials
    "ATD.TO", "L.TO", "MRU.TO", "DOL.TO", "QSR.TO",                     # consumer/retail
    "BAM.TO", "BN.TO", "TRI.TO", "WCN.TO", "CTC-A.TO", "MG.TO",         # diversified/industrials
    "FTS.TO", "EMA.TO", "H.TO", "CU.TO", "AQN.TO",                      # utilities
    "WSP.TO", "STN.TO", "CAE.TO", "TIH.TO",                             # industrials
]

# Benchmarks used for relative-return labels and reporting (not scored).
BENCHMARKS = {"US": "SPY", "CA": "XIC.TO"}

UNIVERSE = _US + _TSX


def get_universe() -> list[str]:
    """Deduplicated research universe (order-preserving)."""
    seen: set = set()
    out = []
    for t in UNIVERSE:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def market_of(ticker: str) -> str:
    """'CA' for TSX-listed (.TO/.V), else 'US'. Used to pick the right benchmark."""
    return "CA" if (ticker.endswith(".TO") or ticker.endswith(".V")) else "US"
