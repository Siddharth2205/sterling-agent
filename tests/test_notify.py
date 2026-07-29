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
