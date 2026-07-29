import numpy as np, pandas as pd
from sterling.research import signals as S

def _df():
    return pd.DataFrame({
        "date": ["d1","d1","d1","d2","d2","d2"],
        "a": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        "b": [3.0, 2.0, 1.0, 30.0, 20.0, 10.0],
    })

class TestSignals:
    def test_zscore_is_zero_mean_per_date(self):
        z = S.zscore_by_date(_df(), "a")
        d1 = z[:3]
        assert abs(d1.mean()) < 1e-9 and abs(d1.std(ddof=1) - 1) < 1e-6

    def test_combine_averages_opposite_signals_to_zero(self):
        # a and b are exact opposites -> blend cancels to ~0
        c = S.combine(_df(), ["a", "b"])
        assert np.allclose(c.values, 0.0, atol=1e-9)
