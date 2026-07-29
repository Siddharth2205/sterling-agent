"""Build the dashboard data file (docs/dashboard_data.json) from the search leaderboard.

The dashboard (docs/index.html) reads this JSON and renders a plain-language, visual view
of what the autonomous search has found. Regenerated after each search cycle and committed,
so the published GitHub Pages site stays current.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone, date

import pandas as pd

from sterling.research import config
from sterling.research.experiment import LEDGER, final_report, HOLDOUT_START
from sterling.research.live import LEDGER as BOOK_LEDGER

OUT = config.ROOT / "docs" / "dashboard_data.json"
DEADLINE = date(2026, 8, 28)
PASS_SHARPE, PASS_T = 0.5, 2.0


def _plain_config(cfg: dict) -> str:
    if not cfg:
        return "—"
    frac = int(cfg.get("long_frac", 0) * 100)
    ls = "buy the best & short the worst" if cfg.get("short_frac") else f"hold the top {frac}%"
    liq = f"≥ ${int(cfg.get('min_dvol', 0)/1e6)}M/day, price ≥ ${int(cfg.get('min_price',0))}"
    bal = "sector-balanced" if cfg.get("sector_neutral") else "any sector"
    wt = "calm-weighted" if cfg.get("inverse_vol") else "equal-weight"
    return f"{ls}, {bal}, {wt} · {liq}"


def _passed(r) -> bool:
    try:
        return float(r.holdout_t) > PASS_T and float(r.holdout_sharpe) > PASS_SHARPE
    except (TypeError, ValueError):
        return False


def _book_picks(n: int = 16) -> dict:
    if not BOOK_LEDGER.exists():
        return {"asof": None, "count": 0, "picks": [], "sectors": []}
    led = pd.read_csv(BOOK_LEDGER)
    if led.empty:
        return {"asof": None, "count": 0, "picks": [], "sectors": []}
    asof = str(led["rebalance_date"].max())
    b = led[led["rebalance_date"].astype(str) == asof]
    picks = [{"ticker": r.ticker, "sector": getattr(r, "sector", "—"),
              "weight": round(float(r.weight) * 100, 2)}
             for r in b.sort_values("weight", ascending=False).head(n).itertuples(index=False)]
    sec = b.groupby("sector")["weight"].sum().sort_values(ascending=False)
    sectors = [{"name": s, "pct": round(float(w) * 100, 1)} for s, w in sec.items()]
    return {"asof": asof, "count": int(len(b)), "picks": picks, "sectors": sectors}


def build_data() -> dict:
    board = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
    valid = board[board["dev_sharpe"].notna()].copy() if "dev_sharpe" in board.columns else pd.DataFrame()
    rep = final_report()

    candidates, progress = [], []
    if not valid.empty:
        valid = valid.sort_values("ts")
        seen = Counter()
        for r in valid.itertuples(index=False):
            try:
                cfg = json.loads(r.config)
            except Exception:
                cfg = {}
            candidates.append({
                "ts": str(getattr(r, "ts", "")),
                "dev_sharpe": round(float(r.dev_sharpe), 3) if pd.notna(r.dev_sharpe) else None,
                "holdout_sharpe": round(float(r.holdout_sharpe), 3) if pd.notna(r.holdout_sharpe) else None,
                "holdout_alpha": round(float(r.holdout_alpha), 2) if pd.notna(r.holdout_alpha) else None,
                "holdout_t": round(float(r.holdout_t), 2) if pd.notna(r.holdout_t) else None,
                "config": _plain_config(cfg),
                "passed": _passed(r),
            })
        # cumulative trials per day
        valid["day"] = valid["ts"].astype(str).str[:10]
        cum = 0
        for day, cnt in valid.groupby("day").size().items():
            cum += int(cnt)
            progress.append({"day": day, "cumulative": cum})

    pass_count = sum(1 for c in candidates if c["passed"])
    best_cfg = {}
    if rep.get("best_config"):
        try:
            best_cfg = json.loads(rep["best_config"])
        except Exception:
            best_cfg = {}

    data = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "trials": int(rep.get("trials", len(candidates))),
        "pass_count": pass_count,
        "edge_found": bool(rep.get("honest_deployable_edge", False)),
        "days_remaining": max(0, (DEADLINE - date.today()).days),
        "deadline": str(DEADLINE),
        "holdout_since": str(HOLDOUT_START),
        "best": {
            "holdout_alpha": rep.get("holdout_alpha_pct_yr"),
            "holdout_sharpe": rep.get("holdout_sharpe"),
            "holdout_t": rep.get("holdout_t"),
            "plain": _plain_config(best_cfg),
        },
        "candidates": candidates,
        "progress": progress,
        "book": _book_picks(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    return data


if __name__ == "__main__":
    d = build_data()
    print(f"dashboard: {d['trials']} trials, {d['pass_count']} passed -> {OUT}")
