import pandas as pd
from sterling.research import notify

def test_format_book_contains_key_info():
    book = pd.DataFrame({"ticker": ["AAA", "BBB"], "sector": ["Tech", "Energy"],
                         "weight": [0.6, 0.4]})
    msg = notify.format_book("2026-07-24", book, top_n=5)
    assert "AAA" in msg and "2026-07-24" in msg and "60.00%" in msg

def test_send_skips_without_creds(monkeypatch):
    monkeypatch.setattr(notify, "_creds", lambda: (None, None))
    assert notify.send("hi") is False

def test_format_evaluation_handles_empty():
    assert "no paper books" in notify.format_evaluation({"books": []})

def test_format_evaluation_shows_book_vs_equal_weight():
    res = {"as_of": "2026-08-28", "books": [
        {"rebalance_date": "2026-07-24", "names": 466, "days_held": 35,
         "book_return_pct": -0.17, "equal_weight_pct": 0.23, "weighting_alpha_pct": -0.4}]}
    msg = notify.format_evaluation(res)
    assert "2026-07-24" in msg and "-0.17%" in msg and "equal-wt" in msg and "0.23%" in msg
    assert "2026-08-28" in msg and "35d" in msg
