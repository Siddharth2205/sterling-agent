"""Build the dashboard data file (docs/dashboard_data.json) from the search leaderboard.

The dashboard (docs/index.html) reads this JSON and renders a plain-language view of what
the autonomous search has found so far. Regenerated after each search cycle and committed,
so the published page stays current.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from sterling.research import config
from sterling.research.experiment import LEDGER, final_report, HOLDOUT_START

OUT = config.ROOT / "docs" / "dashboard_data.json"


def _plain_config(cfg: dict) -> str:
    """Turn a strategy config into a one-line plain-English description."""
    if not cfg:
        return "—"
    frac = int(cfg.get("long_frac", 0) * 100)
    ls = "buy the best & short the worst" if cfg.get("short_frac") else f"buy the top {frac}%"
    liq = f"names trading ≥ ${int(cfg.get('min_dvol', 0)/1e6)}M/day, price ≥ ${int(cfg.get('min_price',0))}"
    bal = "sector-balanced" if cfg.get("sector_neutral") else "any sector"
    return f"{ls} of {liq}, {bal}"


def build_data() -> dict:
    board = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
    valid = board[board["dev_sharpe"].notna()] if "dev_sharpe" in board.columns else pd.DataFrame()
    rep = final_report()

    candidates = []
    for r in valid.itertuples(index=False):
        candidates.append({
            "dev_sharpe": round(float(r.dev_sharpe), 3) if pd.notna(r.dev_sharpe) else None,
            "holdout_sharpe": round(float(r.holdout_sharpe), 3) if pd.notna(getattr(r, "holdout_sharpe", None)) else None,
            "holdout_alpha": round(float(r.holdout_alpha), 2) if pd.notna(getattr(r, "holdout_alpha", None)) else None,
            "holdout_t": round(float(r.holdout_t), 2) if pd.notna(getattr(r, "holdout_t", None)) else None,
        })

    best_cfg = {}
    if "best_config" in rep:
        try:
            best_cfg = json.loads(rep["best_config"])
        except Exception:
            best_cfg = {}

    data = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "trials": int(rep.get("trials", len(board))),
        "edge_found": bool(rep.get("honest_deployable_edge", False)),
        "best": {
            "holdout_alpha": rep.get("holdout_alpha_pct_yr"),
            "holdout_sharpe": rep.get("holdout_sharpe"),
            "holdout_t": rep.get("holdout_t"),
            "plain": _plain_config(best_cfg),
        },
        "candidates": candidates,
        "holdout_since": str(HOLDOUT_START),
        "deadline": "2026-08-28",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    return data


if __name__ == "__main__":
    print(json.dumps(build_data(), indent=2))
