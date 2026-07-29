"""Reproducible research pipeline — the orchestration that used to live in ad-hoc
scripts_*.py files. Each step is a function with sane defaults from config, callable
from code or the CLI (`python -m sterling.research ...`).

Steps:
  build_features()  survivorship-free universe -> features -> delisting-aware labels -> pkl
  analyze()         walk-forward gate + net-of-cost portfolio sim -> result json
  run_all()         build_features then analyze
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from sterling.research import (config, store, survivorship as sv, features,
                               model, portfolio_sim as ps, dataset)

logger = logging.getLogger(__name__)


def build_features(horizon: int = config.HORIZON) -> dict:
    """Build the survivorship-free feature+label matrix and cache it to disk."""
    store.ensure_parquet()
    meta = store.load_tickers()
    universe = sv.select_universe(meta)
    delisting = sv.delisting_map(meta)
    logger.info(f"universe={len(universe)}, delisted={sum(t in delisting for t in universe)}")

    prices = sv.load_prices_bulk(universe)
    logger.info(f"price frames: {len(prices)}")

    spy = dataset.download_one("SPY", years=12)
    vixdf = dataset.download_one("^VIX", years=12)
    vix_by_date = {d.date(): float(c) for d, c in vixdf["Close"].items()} if vixdf is not None else {}

    feat = features.build_features(prices, {"US": spy}, market_of=lambda t: "US",
                                   hist_fund_fn=None, vix_by_date=vix_by_date,
                                   step=config.SAMPLE_STEP, warmup=config.WARMUP)
    lab = sv.add_labels(feat, prices, delisting, horizon=horizon)
    lab.to_pickle(config.FEATURES_PKL)

    summary = {
        "rows": len(lab), "tickers": int(lab["ticker"].nunique()),
        "delisted_present": int(sum(t in delisting for t in lab["ticker"].unique())),
        "date_min": str(lab["date"].min()), "date_max": str(lab["date"].max()),
        "pkl": str(config.FEATURES_PKL),
    }
    logger.info(f"features summary: {summary}")
    return summary


def analyze(top_ns=(15, 25), costs_bps=(10, 20), horizon: int = config.HORIZON) -> dict:
    """Walk-forward validation + net-of-cost portfolio sim on the cached features."""
    df = pd.read_pickle(config.FEATURES_PKL)
    wf = model.walk_forward(df, n_folds=config.N_FOLDS, horizon=horizon)
    panel = model.generate_oos_predictions(df, n_folds=config.N_FOLDS, horizon=horizon)
    sims = {}
    for n in top_ns:
        for c in costs_bps:
            sims[f"top{n}_cost{c}"] = ps.simulate(panel, top_n=n, horizon=horizon, cost_bps=c)
    result = {"walkforward": {k: v for k, v in wf.items() if k != "folds"}, "sims": sims}
    json.dump({"walkforward": wf, "sims": sims},
              open(config.RESULT_JSON, "w"), indent=2, default=str)
    logger.info(f"pooled OOS IC={wf.get('pooled_ic')}, ship={wf.get('ship_recommendation')}")
    return result


def run_all() -> dict:
    build_features()
    return analyze()
