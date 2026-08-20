import json, random
import pandas as pd
from sterling.research import experiment as X

def test_sample_config_has_all_knobs():
    cfg = X.sample_config(random.Random(0))
    assert set(cfg) == set(X.SPACE)
    for k, v in cfg.items():
        assert v in X.SPACE[k]

def test_sample_config_perturbs_a_leader(monkeypatch):
    leaders = pd.DataFrame({"config": [json.dumps({k: X.SPACE[k][0] for k in X.SPACE})]})
    cfg = X.sample_config(random.Random(1), leaders=leaders)
    assert set(cfg) == set(X.SPACE)   # still a full, valid config

def test_leaderboard_survives_error_only_ledger(tmp_path, monkeypatch):
    # A first batch where every candidate errored writes a CSV with no dev_sharpe
    # column — later runs must not crash on it.
    board = pd.DataFrame([{"config": json.dumps({"x": 1}), "error": "boom", "ts": "t"}])
    p = tmp_path / "lb.csv"; board.to_csv(p, index=False)
    monkeypatch.setattr(X, "LEDGER", p)
    assert X.leaderboard().empty


def test_final_report_flags_no_edge_below_t2(tmp_path, monkeypatch):
    board = pd.DataFrame([
        {"config": json.dumps({"x": 1}), "dev_sharpe": 0.6, "era": X.MATRIX_ERA,
         "holdout_sharpe": 0.83, "holdout_alpha": 12.0, "holdout_t": 1.77},
        {"config": json.dumps({"x": 2}), "dev_sharpe": 0.2, "era": X.MATRIX_ERA,
         "holdout_sharpe": 0.1, "holdout_alpha": 1.0, "holdout_t": 0.3},
    ])
    p = tmp_path / "lb.csv"; board.to_csv(p, index=False)
    monkeypatch.setattr(X, "LEDGER", p)
    rep = X.final_report()
    assert rep["trials"] == 2
    assert rep["holdout_t"] == 1.77            # leader chosen by dev_sharpe
    assert rep["honest_deployable_edge"] is False   # t<2 -> honest no


def test_final_report_ignores_other_eras(tmp_path, monkeypatch):
    # A stellar score from an old matrix era must not become the reported leader.
    board = pd.DataFrame([
        {"config": json.dumps({"x": 1}), "dev_sharpe": 9.9,
         "holdout_sharpe": 9.9, "holdout_alpha": 99.0, "holdout_t": 9.9},   # no era col value
        {"config": json.dumps({"x": 2}), "dev_sharpe": 0.4, "era": X.MATRIX_ERA,
         "holdout_sharpe": 0.3, "holdout_alpha": 2.0, "holdout_t": 0.8},
    ])
    p = tmp_path / "lb.csv"; board.to_csv(p, index=False)
    monkeypatch.setattr(X, "LEDGER", p)
    rep = X.final_report()
    assert rep["trials"] == 1 and rep["holdout_t"] == 0.8


def test_merge_into_ledger_unions_and_prefers_settled(tmp_path, monkeypatch):
    g = X.enumerate_space()
    def row(cfg, dev=None, err=None):
        return {"config": json.dumps(cfg, sort_keys=True), "era": X.MATRIX_ERA,
                "dev_sharpe": dev, "error": err}
    ledger = tmp_path / "lb.csv"
    other = tmp_path / "other.csv"
    # remote (== LEDGER after git reset): cfg0 scored, cfg2 crashed
    pd.DataFrame([row(g[0], 0.5),
                  row(g[2], None, "window shape cannot be larger than input array shape")]
                 ).to_csv(ledger, index=False)
    # ours: cfg2 now scored (retry won), cfg3 new
    pd.DataFrame([row(g[2], 0.3), row(g[3], 0.7)]).to_csv(other, index=False)
    monkeypatch.setattr(X, "LEDGER", ledger)
    X.merge_into_ledger(str(other))
    m = pd.read_csv(ledger)
    assert m["config"].nunique() == 3                       # union of cfg0,2,3
    c2 = m[m["config"] == json.dumps(g[2], sort_keys=True)]
    assert float(c2["dev_sharpe"].iloc[0]) == 0.3            # settled beats crash
    # idempotent
    X.merge_into_ledger(str(other))
    assert pd.read_csv(ledger)["config"].nunique() == 3


def test_grid_status_counts_settled(tmp_path, monkeypatch):
    grid = X.enumerate_space()
    # 2 scored + 1 legit-skip settle; 1 crash does NOT
    rows = [
        {"config": json.dumps(grid[0], sort_keys=True), "era": X.MATRIX_ERA, "dev_sharpe": 0.5},
        {"config": json.dumps(grid[1], sort_keys=True), "era": X.MATRIX_ERA, "dev_sharpe": -0.2},
        {"config": json.dumps(grid[2], sort_keys=True), "era": X.MATRIX_ERA,
         "dev_sharpe": None, "error": "too few tradeable rows"},
        {"config": json.dumps(grid[3], sort_keys=True), "era": X.MATRIX_ERA,
         "dev_sharpe": None, "error": "window shape cannot be larger than input array shape"},
    ]
    p = tmp_path / "lb.csv"; pd.DataFrame(rows).to_csv(p, index=False)
    monkeypatch.setattr(X, "LEDGER", p)
    st = X.grid_status()
    assert st["total"] == 5832 and st["settled"] == 3      # crash not counted
    assert st["remaining"] == 5829 and st["complete"] is False


def test_enumerate_space_is_full_grid():
    grid = X.enumerate_space()
    assert len(grid) == 5832                       # 3^6 x 2 x 2 x 2 distinct configs
    assert len({json.dumps(c, sort_keys=True) for c in grid}) == len(grid)
    for c in grid:
        for k, v in c.items():
            assert v in X.SPACE[k]


def test_untested_handles_board_without_era_column():
    # The real pre-sweep leaderboard has no `era` column at all — everything is
    # untested and nothing may crash (this exact case broke the first sweep run).
    board = pd.DataFrame([{"config": json.dumps({"x": 1}), "dev_sharpe": 0.5, "ts": "t"}])
    assert len(X.untested_configs(board)) == 5832


def test_untested_excludes_current_era_only():
    cfg = X.enumerate_space()[0]
    key = json.dumps(cfg, sort_keys=True)
    # scored in the current era -> settled -> excluded
    board = pd.DataFrame([{"config": key, "era": X.MATRIX_ERA, "dev_sharpe": 0.5}])
    assert key not in {json.dumps(c, sort_keys=True) for c in X.untested_configs(board)}
    # scored only in an old era -> still due for a re-test
    board_old = pd.DataFrame([{"config": key, "era": "v1", "dev_sharpe": 0.5}])
    assert key in {json.dumps(c, sort_keys=True) for c in X.untested_configs(board_old)}


def test_transient_crash_is_retried_but_legit_skip_is_not():
    grid = X.enumerate_space()
    crash_key = json.dumps(grid[0], sort_keys=True)
    skip_key = json.dumps(grid[1], sort_keys=True)
    board = pd.DataFrame([
        # a numpy/env crash — must NOT be treated as done
        {"config": crash_key, "era": X.MATRIX_ERA, "dev_sharpe": None,
         "error": "ValueError('window shape cannot be larger than input array shape')"},
        # a deterministic data-driven skip — legitimately settled, don't retry
        {"config": skip_key, "era": X.MATRIX_ERA, "dev_sharpe": None,
         "error": "too few tradeable rows"},
    ])
    untested = {json.dumps(c, sort_keys=True) for c in X.untested_configs(board)}
    assert crash_key in untested        # crash -> retried
    assert skip_key not in untested     # legit skip -> settled


def test_run_batch_purges_stale_crash_row(tmp_path, monkeypatch):
    grid = X.enumerate_space()
    key = json.dumps(grid[0], sort_keys=True)
    board = pd.DataFrame([{"config": key, "era": X.MATRIX_ERA, "dev_sharpe": None,
                           "error": "ValueError('window shape cannot be larger than input array shape')",
                           "ts": "old"}])
    p = tmp_path / "lb.csv"; board.to_csv(p, index=False)
    monkeypatch.setattr(X, "LEDGER", p)
    monkeypatch.setattr(X, "build_matrix", lambda: pd.DataFrame())
    monkeypatch.setattr(X, "enumerate_space", lambda: [grid[0]])   # only this config in the grid
    # evaluate now succeeds for that config -> the fresh row replaces the crash row
    monkeypatch.setattr(X, "evaluate", lambda cfg, df: {"config": json.dumps(cfg, sort_keys=True),
                                                        "dev_sharpe": 0.7, "holdout_t": 1.0})
    out = X.run_batch(n=1, seed=0)
    rows = out[out["config"] == key]
    assert len(rows) == 1                      # exactly one row, not crash + result
    assert float(rows.iloc[0]["dev_sharpe"]) == 0.7
