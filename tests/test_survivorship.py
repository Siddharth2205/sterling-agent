import numpy as np, pandas as pd
from sterling.research import survivorship as sv

def _prices():
    idx = pd.date_range("2022-01-03", periods=120, freq="B")
    def frame(vals): return pd.DataFrame({"Close": vals}, index=idx)
    alive_up  = frame(np.linspace(100, 160, 120))          # winner, still listed
    alive_flat= frame(np.full(120, 100.0))                 # flat, still listed
    # dying name: normal until bar 80 then we mark it delisted at that date
    dying = frame(np.concatenate([np.full(80, 50.0), np.full(40, 20.0)]))
    return {"UP": alive_up, "FLAT": alive_flat, "DIE": dying}, idx

class TestSurvivorshipLabels:
    def test_delisted_name_is_labeled_not_dropped(self):
        prices, idx = _prices()
        die_delist_date = idx[80].date()
        delisting = {"DIE": (die_delist_date, -1.0)}   # total loss on delisting
        # feature rows at bar 40 for each ticker
        rows = [{"date": idx[40].date(), "ticker": tk, "f": 1.0} for tk in prices]
        feat = pd.DataFrame(rows)
        out = sv.add_labels(feat, prices, delisting, horizon=63)
        assert set(out["ticker"]) == {"UP", "FLAT", "DIE"}   # DIE not dropped
        die = out[out["ticker"] == "DIE"].iloc[0]
        # forward window (bar 40 -> 103) crosses delisting at bar 80 → books the loss
        assert die["fwd_return"] < -0.5
        # label is excess vs the equal-weight universe that date
        assert abs(die["label"] - (die["fwd_return"] - out["fwd_return"].mean())) < 1e-9

    def test_winner_beats_universe(self):
        prices, idx = _prices()
        rows = [{"date": idx[40].date(), "ticker": tk, "f": 1.0} for tk in prices]
        out = sv.add_labels(pd.DataFrame(rows), prices, {}, horizon=63)
        up = out[out["ticker"] == "UP"].iloc[0]
        assert up["label"] > 0   # the rising name beats the cross-sectional mean
