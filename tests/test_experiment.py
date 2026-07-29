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

def test_final_report_flags_no_edge_below_t2(tmp_path, monkeypatch):
    board = pd.DataFrame([
        {"config": json.dumps({"x": 1}), "dev_sharpe": 0.6,
         "holdout_sharpe": 0.83, "holdout_alpha": 12.0, "holdout_t": 1.77},
        {"config": json.dumps({"x": 2}), "dev_sharpe": 0.2,
         "holdout_sharpe": 0.1, "holdout_alpha": 1.0, "holdout_t": 0.3},
    ])
    p = tmp_path / "lb.csv"; board.to_csv(p, index=False)
    monkeypatch.setattr(X, "LEDGER", p)
    rep = X.final_report()
    assert rep["trials"] == 2
    assert rep["holdout_t"] == 1.77            # leader chosen by dev_sharpe
    assert rep["honest_deployable_edge"] is False   # t<2 -> honest no
