"""Audit 11-07 01:08 #5: buy-side verdicts record the regime entry rule and
the ticker's own sector regime. Record-only."""
from unittest.mock import MagicMock, patch

from agents import analyst


def test_record_names_the_rule_and_the_sector_regime():
    with patch("ingestion.sp500_universe.get_sp500_constituents",
               return_value={"NVDA": {"gics_sector": "Information Technology"}}), \
         patch("db.get_latest_sector_regime",
               return_value={"label": "Strong Risk-Off", "score": 13.0, "regime_type": "gics_sector",
                             "at": "2026-11-06T15:00:00+00:00"}) as q:
        rec = analyst._regime_entry_record("NVDA", "Risk-Off", "OPEN", False)
    assert rec["rule"] == "open_min_c7" and rec["rule_fired"] is False
    assert rec["sector_etf"] == "XLK" and rec["sector_regime"]["label"] == "Strong Risk-Off"
    assert q.call_args.args == ("XLK",)


def test_risk_on_has_no_rule_and_errors_are_recorded():
    with patch("ingestion.sp500_universe.get_sp500_constituents",
               return_value={"NVDA": {"gics_sector": "Information Technology"}}), \
         patch("db.get_latest_sector_regime", side_effect=RuntimeError("pool")):
        rec = analyst._regime_entry_record("NVDA", "Risk-On", "ADD", False)
    assert rec["rule"] == "none" and rec["sector_regime"] == {"error": "RuntimeError"}
    with patch("ingestion.sp500_universe.get_sp500_constituents", return_value={}):
        assert analyst._regime_entry_record("NVDA", "Neutral", "ADD", False)["rule"] == "none"
    with patch("ingestion.sp500_universe.get_sp500_constituents", return_value={}):
        rec = analyst._regime_entry_record("IONQ", "Neutral", "OPEN", True)
    assert rec["sector_etf"] is None and rec["sector_regime"] is None and rec["rule_fired"] is True


def test_sector_regime_query():
    import db
    cur = MagicMock()
    cur.fetchone.return_value = ("Risk-Off", 22.0, "gics_sector", None)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        out = db.get_latest_sector_regime("XLK")
    assert out == {"label": "Risk-Off", "score": 22.0, "regime_type": "gics_sector", "at": None}
    assert "sector_etf=%s" in cur.execute.call_args.args[0]
