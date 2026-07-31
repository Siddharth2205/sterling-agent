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
    # tested in the current era -> excluded
    board = pd.DataFrame([{"config": key, "era": X.MATRIX_ERA}])
    assert key not in {json.dumps(c, sort_keys=True) for c in X.untested_configs(board)}
    # tested only in an old era -> still due for a re-test
    board_old = pd.DataFrame([{"config": key, "era": "v1"}])
    assert key in {json.dumps(c, sort_keys=True) for c in X.untested_configs(board_old)}
