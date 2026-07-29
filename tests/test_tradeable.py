import numpy as np, pandas as pd
from sterling.research import tradeable as T

def test_filter_drops_penny_and_illiquid():
    idx = pd.date_range("2022-01-03", periods=60, freq="B")
    prices = {
        "BIG":  pd.DataFrame({"Close": np.full(60, 50.0), "Volume": np.full(60, 1e6)}, index=idx),
        "PENNY":pd.DataFrame({"Close": np.full(60, 0.50), "Volume": np.full(60, 1e6)}, index=idx),
        "THIN": pd.DataFrame({"Close": np.full(60, 50.0), "Volume": np.full(60, 100.0)}, index=idx),
    }
    feat = pd.DataFrame({"ticker": ["BIG","PENNY","THIN"], "date": [idx[40].date()]*3, "x":[1,2,3]})
    out = T.filter_tradeable(feat, prices=prices, min_price=5.0, min_dvol=1e6)
    assert set(out["ticker"]) == {"BIG"}   # penny (price) and thin (volume) dropped
