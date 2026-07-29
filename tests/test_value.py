import numpy as np, pandas as pd
from sterling.research import value as V

class TestValueScore:
    def test_cheaper_scores_higher(self):
        # same date; T0 cheap (low ratios), T2 expensive (high ratios)
        df = pd.DataFrame({
            "date": ["d1","d1","d1"], "ticker": ["T0","T1","T2"],
            "pe": [8.0, 16.0, 40.0], "pb": [1.0, 2.0, 5.0], "ps": [1.0, 2.0, 6.0],
        })
        s = V.value_score(df)
        assert s.iloc[0] > s.iloc[1] > s.iloc[2]   # cheapest ranks highest

    def test_negative_ratios_ignored(self):
        df = pd.DataFrame({
            "date": ["d1","d1","d1"], "ticker": ["A","B","C"],
            "pe": [-5.0, 10.0, 20.0], "pb": [1.0, 2.0, 4.0], "ps": [1.0, 2.0, 4.0],
        })
        s = V.value_score(df)
        assert np.isfinite(s.iloc[0])   # still scored via pb/ps even with negative pe
