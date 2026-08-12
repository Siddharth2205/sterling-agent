"""Autonomous, honest strategy search.

Each *candidate* is a full strategy config: universe/liquidity filter × model settings ×
portfolio construction. It is scored on a DEVELOPMENT period (walk-forward, out-of-sample),
and — critically — the leaderboard's leaders are judged on a LOCKED HOLD-OUT period they
were never selected on. That hold-out is the guard against the exact failure mode of "keep
trying strategies until one looks good": with enough tries something always looks good on
the data you searched, so the only number that counts is performance on data you didn't.

The leaderboard persists across daily runs (committed back to the repo). Each run explores
new candidates — some random, some perturbations of the current leaders — so the search
concentrates where results are promising, while the hold-out keeps it honest.

Nothing here promises a profitable strategy. It promises an *honest* answer.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
import time
from datetime import date, datetime, timezone

import pandas as pd

from sterling.research import config, model as M, construction as C, store, tradeable
from sterling.research.features import ALL_FEATURES

logger = logging.getLogger(__name__)

LEDGER = config.DATA / "experiment_leaderboard.csv"
MATRIX = config.DATA / "experiment_matrix.pkl"
HOLDOUT_START = date(2022, 1, 1)     # last ~4y locked away for the honest final test

# Bump together with the workflow cache key whenever the feature matrix changes.
# Rows from other eras stay on the leaderboard as history but are not comparable —
# dedup, leaderboard and the final report all consider only the current era.
MATRIX_ERA = "v3"

# Search space — the knobs a candidate strategy can turn.
SPACE = {
    "min_price": [3.0, 5.0, 10.0],
    "min_dvol": [500_000, 1_000_000, 5_000_000],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.08],
    "min_samples_leaf": [100, 200, 500],
    "long_frac": [0.1, 0.2, 0.33],
    "short_frac": [0.0, 0.0, 0.2],           # mostly long-only; occasionally long-short
    "sector_neutral": [True, False],
    "inverse_vol": [True, False],
}


FULL_FEATURES = config.DATA / "survivorship_full_features.pkl"


def bootstrap() -> None:
    """Build the UNFILTERED full-history feature matrix from scratch if missing — so the
    cloud can construct everything from just the API key. Downloads the Sharadar bulk
    stocks/tickers, builds features (no liquidity filter — the search varies that itself)."""
    if FULL_FEATURES.exists():
        return
    from sterling.research import survivorship as sv, features as F, dataset
    logger.info("bootstrap: building full-history features from Sharadar bulk...")
    if not config.TICKERS_CSV.exists():
        store.download_bulk("tickers", "full", config.TICKERS_CSV)
    full_csv = config.BULK / "stocks_full.csv"
    full_pq = config.BULK / "stocks_full.parquet"
    if not full_pq.exists():
        if not full_csv.exists():
            store.download_bulk("stocks", "full", full_csv)
        store.csv_to_parquet(full_csv, full_pq)
    config.STOCKS_PARQUET = full_pq          # point the store at full history

    meta = store.load_tickers()
    uni = sv.select_universe(meta)
    delist = sv.delisting_map(meta)
    prices = store.load_prices(uni)
    spy = dataset.download_one("SPY", years=30)
    vixdf = dataset.download_one("^VIX", years=30)
    vix_by_date = {d.date(): float(c) for d, c in vixdf["Close"].items()} if vixdf is not None else {}

    # Point-in-time fundamentals from SEC EDGAR (free; ~2009+ only — earlier rows NaN).
    from sterling.research import edgar
    if not edgar.FACTS_PARQUET.exists():
        edgar.download_companyfacts()
        edgar.convert_companyfacts(ciks=set(edgar.ticker_cik_map().values()) or None)
    hist_fund_fn = edgar.fundamentals_fn(uni)

    feat = F.build_features(prices, {"US": spy}, market_of=lambda t: "US",
                            hist_fund_fn=hist_fund_fn, vix_by_date=vix_by_date,
                            step=config.SAMPLE_STEP, warmup=config.WARMUP)
    lab = sv.add_labels(feat, prices, delist, horizon=config.HORIZON)   # UNFILTERED
    lab.to_pickle(FULL_FEATURES)
    logger.info(f"bootstrap: {len(lab)} unfiltered rows -> {FULL_FEATURES.name}")


def build_matrix(force: bool = False) -> pd.DataFrame:
    """Augment the survivorship-free feature matrix with price / dollar-volume / sector /
    market-cap tier — once — so candidates can filter cheaply. Cached to disk."""
    if MATRIX.exists() and not force:
        return pd.read_pickle(MATRIX)
    bootstrap()
    src = FULL_FEATURES if FULL_FEATURES.exists() else config.FEATURES_PKL
    df = pd.read_pickle(src)
    prices = store.load_prices(set(df["ticker"].unique()))
    liq = tradeable.liquidity_frame(prices)
    df = df.merge(liq, on=["ticker", "date"], how="left")
    meta = store.load_tickers()
    df["sector"] = df["ticker"].map(dict(zip(meta["ticker"], meta["sector"]))).fillna("Unknown")
    df["scale"] = df["ticker"].map(dict(zip(meta["ticker"], meta["scalemarketcap"])))
    df.to_pickle(MATRIX)
    logger.info(f"experiment matrix: {len(df)} rows cached to {MATRIX.name}")
    return df


def enumerate_space() -> list[dict]:
    """Every distinct config in SPACE (duplicate knob values collapse) — 5,832 total."""
    keys = list(SPACE)
    uniq = [sorted(set(SPACE[k])) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*uniq)]


# A candidate is "settled" (never retried) if it produced a real score OR a
# deterministic, data-driven skip. Any OTHER error is a transient/environment crash
# (e.g. a numpy window error on a partially-built matrix) — those MUST be retried,
# never treated as permanently done, or a single bad run poisons the config forever.
_LEGIT_SKIP_MARKERS = ("too few tradeable rows", "insufficient dev/holdout span")


def _is_legit_skip(err) -> bool:
    if err is None or (isinstance(err, float) and pd.isna(err)):
        return False
    return any(m in str(err) for m in _LEGIT_SKIP_MARKERS)


def _era_rows(board: pd.DataFrame) -> pd.DataFrame:
    """Rows scored on the CURRENT feature matrix — the only comparable ones.
    Always preserves the input's columns (a pre-era board has none matching)."""
    if board is None or board.empty:
        return pd.DataFrame()
    if "era" not in board.columns:
        return board.iloc[0:0]          # no era column -> nothing current, columns kept
    return board[board["era"] == MATRIX_ERA]


def _settled_configs(board: pd.DataFrame | None) -> set:
    """Configs with a real score or a legit skip in the current era (won't be retried)."""
    era = _era_rows(board)
    if era.empty:
        return set()
    has_score = era["dev_sharpe"].notna() if "dev_sharpe" in era.columns else pd.Series(False, index=era.index)
    errs = era["error"] if "error" in era.columns else pd.Series(None, index=era.index)
    settled_mask = has_score | errs.map(_is_legit_skip)
    return set(era.loc[settled_mask, "config"])


def untested_configs(board: pd.DataFrame | None) -> list[dict]:
    """Grid configs not yet SETTLED in the current era. A prior transient crash does
    not count as settled — the config is re-attempted (other eras never count; their
    scores came from a different matrix and are not reproducible)."""
    settled = _settled_configs(board)
    return [c for c in enumerate_space() if json.dumps(c, sort_keys=True) not in settled]


def sample_config(rng: random.Random, leaders: pd.DataFrame | None = None) -> dict:
    """A new candidate: perturb a current leader half the time, else fully random."""
    if leaders is not None and len(leaders) and rng.random() < 0.5:
        base = json.loads(leaders.iloc[rng.randrange(min(5, len(leaders)))]["config"])
        cfg = dict(base)
        k = rng.choice(list(SPACE))       # mutate one knob
        cfg[k] = rng.choice(SPACE[k])
        return cfg
    return {k: rng.choice(v) for k, v in SPACE.items()}


def _neutral(panel: pd.DataFrame, cfg: dict) -> dict:
    return C.simulate_neutral(panel, long_frac=cfg["long_frac"], short_frac=cfg["short_frac"],
                              sector_neutral=cfg["sector_neutral"], inverse_vol=cfg["inverse_vol"])


def evaluate(cfg: dict, df: pd.DataFrame) -> dict:
    """Score one candidate: development walk-forward + a locked hold-out test."""
    d = df[(df["price"] >= cfg["min_price"]) & (df["dvol"] >= cfg["min_dvol"])].copy()
    if len(d) < 20_000:
        return {"error": "too few tradeable rows"}
    # cross-sectional excess vs the tradeable universe on each date
    d["label"] = d["fwd_return"] - d.groupby("date")["fwd_return"].transform("mean")

    # Embargo: a dev row's forward window must not reach into the hold-out, or the
    # "untouched" period leaks into training via the labels.
    embargo = pd.Timedelta(days=int(config.HORIZON * 1.6))
    dev = d[d["date"] < (HOLDOUT_START - embargo)]
    hold = d[d["date"] >= HOLDOUT_START]
    if dev["date"].nunique() < 40 or hold["date"].nunique() < 20:
        return {"error": "insufficient dev/holdout span"}

    mp = {"max_depth": cfg["max_depth"], "learning_rate": cfg["learning_rate"],
          "min_samples_leaf": cfg["min_samples_leaf"]}

    # DEVELOPMENT: walk-forward out-of-sample panel + market-neutral book
    wf = M.walk_forward(dev, n_folds=3, model_params=mp)
    dev_panel = M.generate_oos_predictions(dev, n_folds=3, model_params=mp)
    dev_panel = dev_panel.merge(dev[["date", "ticker", "vol_63", "sector"]].drop_duplicates(["date", "ticker"]),
                                on=["date", "ticker"], how="left")
    dev_mn = _neutral(dev_panel, cfg)

    # HOLD-OUT: train on ALL development data, predict the untouched hold-out, then book it
    used = M._usable_features(dev, ALL_FEATURES)
    mdl = M._make_model(mp)
    mdl, used = M.fit_resilient(mdl, dev[used], dev["label"])
    hp = hold.copy()
    hp["pred"] = mdl.predict(hp[used])
    hold_mn = _neutral(hp[["date", "ticker", "fwd_return", "label", "vol_63", "sector", "pred"]], cfg)

    def sharpe(mn):
        h = mn.get("index_hedged_net") if isinstance(mn, dict) else None
        return h.get("sharpe") if isinstance(h, dict) else None

    return {
        "config": json.dumps(cfg, sort_keys=True),
        "rows": len(d), "names": int(d["ticker"].nunique()),
        "dev_ic": wf.get("pooled_ic") if "error" not in wf else None,
        "dev_sharpe": sharpe(dev_mn), "dev_alpha": dev_mn.get("net_alpha_vs_universe_annual_pct"),
        "holdout_sharpe": sharpe(hold_mn), "holdout_alpha": hold_mn.get("net_alpha_vs_universe_annual_pct"),
        "holdout_t": hold_mn.get("net_alpha_t_stat"),
    }


def run_batch(n: int = 20, seed: int | None = None,
              time_budget_min: float | None = None) -> pd.DataFrame:
    """Sweep untested grid configs (current era) and append results to the leaderboard.

    Stops at `n` candidates, at `time_budget_min` minutes, or when the grid is
    exhausted — whichever comes first. Saves after every candidate so a killed run
    loses at most one evaluation.
    """
    rng = random.Random(seed if seed is not None else datetime.now(timezone.utc).microsecond)
    df = build_matrix()
    board = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
    todo = untested_configs(board)
    if not todo:
        logger.info(f"grid exhausted — all {len(enumerate_space())} configs settled in era {MATRIX_ERA}")
        return board
    # Drop this era's stale crash rows for configs we're about to retry, so a retried
    # config ends with exactly one row (its fresh result) instead of crash + result.
    if len(board) and "config" in board.columns:
        todo_keys = {json.dumps(c, sort_keys=True) for c in todo}
        is_stale_crash = (board.get("era") == MATRIX_ERA) & board["config"].isin(todo_keys)
        if is_stale_crash.any():
            logger.info(f"purging {int(is_stale_crash.sum())} stale crash rows for retry")
            board = board[~is_stale_crash].reset_index(drop=True)
    rng.shuffle(todo)
    total = len(enumerate_space())
    logger.info(f"sweep era {MATRIX_ERA}: {total - len(todo)}/{total} done, "
                f"{len(todo)} to go (batch limit n={n}, budget={time_budget_min} min)")

    t0 = time.monotonic()
    for i, cfg in enumerate(todo[:n], 1):
        if time_budget_min is not None and (time.monotonic() - t0) / 60 >= time_budget_min:
            logger.info(f"time budget reached after {i - 1} candidates")
            break
        try:
            row = evaluate(cfg, df)
        except Exception as e:  # noqa: BLE001 — a bad candidate must never kill the run
            row = {"config": json.dumps(cfg, sort_keys=True), "error": repr(e)[:160]}
        row["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row["era"] = MATRIX_ERA
        board = pd.concat([board, pd.DataFrame([row])], ignore_index=True)
        board.to_csv(LEDGER, index=False)
        logger.info(f"[{i}/{min(n, len(todo))}] dev_sharpe={row.get('dev_sharpe')} "
                    f"holdout_sharpe={row.get('holdout_sharpe')}")
    return board


def leaderboard(top: int = 10) -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame()
    b = _era_rows(pd.read_csv(LEDGER))
    if b.empty or "dev_sharpe" not in b.columns:
        return pd.DataFrame()
    return b[b["dev_sharpe"].notna()].sort_values("dev_sharpe", ascending=False).head(top)


def final_report() -> dict:
    """Honest verdict: pick the leader by DEVELOPMENT score, then read its HOLD-OUT number.
    The hold-out is the only estimate that isn't contaminated by the search itself.
    Only current-era rows count — older eras were scored on a different matrix."""
    if not LEDGER.exists():
        return {"error": "no experiments yet"}
    b = _era_rows(pd.read_csv(LEDGER))
    valid = b[b["dev_sharpe"].notna()] if "dev_sharpe" in b.columns else pd.DataFrame()
    n_trials = int(valid["config"].nunique()) if not valid.empty else 0
    if valid.empty:
        return {"trials": n_trials, "verdict": "no valid candidates yet"}
    lead = valid.sort_values("dev_sharpe", ascending=False).iloc[0]
    # Deflate: with many trials, demand the hold-out clear a higher bar than a single test.
    honest_edge = bool(pd.notna(lead.get("holdout_sharpe")) and lead["holdout_sharpe"] > 0.5
                       and pd.notna(lead.get("holdout_t")) and lead["holdout_t"] > 2.0)
    return {
        "trials": n_trials,
        "best_config": lead["config"],
        "dev_sharpe": round(float(lead["dev_sharpe"]), 3),
        "holdout_sharpe": None if pd.isna(lead.get("holdout_sharpe")) else round(float(lead["holdout_sharpe"]), 3),
        "holdout_alpha_pct_yr": None if pd.isna(lead.get("holdout_alpha")) else round(float(lead["holdout_alpha"]), 2),
        "holdout_t": None if pd.isna(lead.get("holdout_t")) else round(float(lead["holdout_t"]), 2),
        "honest_deployable_edge": honest_edge,
        "note": ("Hold-out is the untouched final test; with this many trials, treat anything "
                 "below t>2 on the hold-out as no edge."),
    }
