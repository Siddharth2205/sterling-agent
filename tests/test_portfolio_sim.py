import numpy as np, pandas as pd
from sterling.research import portfolio_sim as ps

def _panel(n_dates=6, n_tickers=40, signal=0.03, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    rows=[]
    for k in range(n_dates):
        d = (base + pd.Timedelta(days=95*k)).date()
        for t in range(n_tickers):
            pred = rng.normal()
            fwd = signal*pred + rng.normal(0,0.05)   # higher pred -> higher return
            rows.append({"date": d, "ticker": f"T{t}", "pred": pred,
                         "fwd_return": fwd, "bench_fwd": 0.01, "label": fwd-0.01})
    return pd.DataFrame(rows)

class TestSimulate:
    def test_edge_beats_universe_net(self):
        rep = ps.simulate(_panel(signal=0.05, seed=1), top_n=8, horizon=63, cost_bps=10)
        assert rep["net_beats_universe"] is True
        assert rep["net_alpha_vs_universe_pct_per_period"] > 0

    def test_costs_reduce_net_below_gross(self):
        rep = ps.simulate(_panel(seed=2), top_n=8, horizon=63, cost_bps=50)
        assert rep["strategy_net"]["mean_period_pct"] <= rep["strategy_gross"]["mean_period_pct"]

    def test_no_signal_no_alpha(self):
        rep = ps.simulate(_panel(signal=0.0, seed=3), top_n=8, horizon=63, cost_bps=10)
        assert abs(rep["net_alpha_vs_universe_pct_per_period"]) < 0.5
