"""Portfolio CRUD — persists to data/portfolio.json."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).parent.parent / "data"
_PORTFOLIO_FILE = _DATA_DIR / "portfolio.json"


def _load_raw() -> dict:
    _DATA_DIR.mkdir(exist_ok=True)
    if not _PORTFOLIO_FILE.exists():
        return {"holdings": {}, "watchlist": [], "updated": None}
    with open(_PORTFOLIO_FILE, "r") as f:
        return json.load(f)


def _save_raw(data: dict) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    data["updated"] = datetime.utcnow().isoformat()
    with open(_PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _normalize_ticker(ticker: str) -> str:
    """Uppercase and ensure TSX suffix is present if bare Canadian ticker passed."""
    return ticker.strip().upper()


def add_holding(ticker: str, shares: float, avg_cost_cad: float, currency: str = "CAD") -> dict:
    """Add or update a holding. avg_cost_cad is always in CAD equivalent."""
    ticker = _normalize_ticker(ticker)
    data = _load_raw()
    data["holdings"][ticker] = {
        "ticker": ticker,
        "shares": shares,
        "avg_cost_cad": avg_cost_cad,
        "currency": currency.upper(),
        "added": datetime.utcnow().isoformat(),
    }
    _save_raw(data)
    return data["holdings"][ticker]


def remove_holding(ticker: str) -> bool:
    ticker = _normalize_ticker(ticker)
    data = _load_raw()
    if ticker in data["holdings"]:
        del data["holdings"][ticker]
        _save_raw(data)
        return True
    return False


def update_holding(ticker: str, shares: Optional[float] = None, avg_cost_cad: Optional[float] = None) -> Optional[dict]:
    ticker = _normalize_ticker(ticker)
    data = _load_raw()
    if ticker not in data["holdings"]:
        return None
    if shares is not None:
        data["holdings"][ticker]["shares"] = shares
    if avg_cost_cad is not None:
        data["holdings"][ticker]["avg_cost_cad"] = avg_cost_cad
    _save_raw(data)
    return data["holdings"][ticker]


def get_portfolio() -> dict:
    return _load_raw()


def add_to_watchlist(ticker: str) -> list:
    ticker = _normalize_ticker(ticker)
    data = _load_raw()
    if ticker not in data["watchlist"]:
        data["watchlist"].append(ticker)
        _save_raw(data)
    return data["watchlist"]


def remove_from_watchlist(ticker: str) -> list:
    ticker = _normalize_ticker(ticker)
    data = _load_raw()
    data["watchlist"] = [t for t in data["watchlist"] if t != ticker]
    _save_raw(data)
    return data["watchlist"]


def portfolio_summary(current_prices: Optional[dict] = None) -> dict:
    """Return summary with P&L if current_prices dict {ticker: price_cad} provided."""
    data = _load_raw()
    holdings = data["holdings"]
    current_prices = current_prices or {}

    total_cost = 0.0
    total_value = 0.0
    positions = []

    for ticker, h in holdings.items():
        cost_basis = h["shares"] * h["avg_cost_cad"]
        price = current_prices.get(ticker)
        current_value = h["shares"] * price if price else None
        unrealized_pnl = (current_value - cost_basis) if current_value is not None else None
        pnl_pct = (unrealized_pnl / cost_basis * 100) if (unrealized_pnl is not None and cost_basis) else None

        total_cost += cost_basis
        if current_value is not None:
            total_value += current_value

        positions.append({
            "ticker": ticker,
            "shares": h["shares"],
            "avg_cost_cad": h["avg_cost_cad"],
            "cost_basis": round(cost_basis, 2),
            "current_price": price,
            "current_value": round(current_value, 2) if current_value is not None else None,
            "unrealized_pnl": round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "currency": h.get("currency", "CAD"),
        })

    # Compute weights
    for p in positions:
        p["weight_pct"] = round(p["cost_basis"] / total_cost * 100, 1) if total_cost else 0.0

    return {
        "positions": positions,
        "total_cost_cad": round(total_cost, 2),
        "total_value_cad": round(total_value, 2) if total_value else None,
        "total_unrealized_pnl": round(total_value - total_cost, 2) if total_value else None,
        "watchlist": data["watchlist"],
        "position_count": len(positions),
        "updated": data.get("updated"),
    }
