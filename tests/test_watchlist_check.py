"""Boot-time watchlist validator: duplicate and unparsed ticker rows are
reported, and nothing about which row loads changes. Log-only."""
import logging
from unittest.mock import patch

import watchlist_check

_MD = """\
### AI Infrastructure
| Ticker | Company | Role | Tier |
| NVDA | Nvidia | Compounder | Core |
| TER | Teradyne | Accelerator | Watch |
| INTC | Intel | Watch | Watch |
### Robotics
| TER | Teradyne | Accelerator | Speculative |
"""


def _loaded():
    return {"NVDA": {"company": "Nvidia", "role": "Compounder", "tier": "Core"},
            "TER": {"company": "Teradyne", "role": "Accelerator", "tier": "Speculative"}}


def test_reports_duplicates_with_the_kept_row_and_unparsed_rows(tmp_path):
    p = tmp_path / "watchlist.md"
    p.write_text(_MD)
    res = watchlist_check.issues(str(p), _loaded())
    assert res["duplicates"] == [{"ticker": "TER", "lines": [4, 7], "kept_line": 7}]
    assert res["unparsed"] == [{"ticker": "INTC", "line": 5}]
    assert res["rows"] == 4 and res["error"] is None


def test_a_missing_file_never_raises(tmp_path):
    res = watchlist_check.issues(str(tmp_path / "nope.md"), {})
    assert res["error"] == "FileNotFoundError" and res["duplicates"] == []


def test_check_and_warn_logs_each_duplicate(tmp_path, caplog):
    p = tmp_path / "watchlist.md"
    p.write_text(_MD)
    with patch("knowledge_loader.load_watchlist", return_value=_loaded()), \
         caplog.at_level(logging.WARNING, logger="watchlist_check"):
        res = watchlist_check.check_and_warn(str(p))
    assert len(res["duplicates"]) == 1
    assert any("TER" in r.message and "[4, 7]" in r.message for r in caplog.records)


def test_the_real_watchlist_reports_ter():
    res = watchlist_check.issues()
    assert res["error"] is None
    assert "TER" in [d["ticker"] for d in res["duplicates"]]


def test_boot_row_carries_the_issues():
    from agents import analyst
    with patch("agents.analyst.boot_record", return_value={"pid": 1, "code_sha": "x"}), \
         patch("watchlist_check.check_and_warn", return_value={"duplicates": [], "unparsed": [],
                                                                 "rows": 0, "error": None}), \
         patch("db.log_boot_activity") as boot:
        analyst.record_boot("test")
    assert boot.call_args.args[1]["watchlist_issues"]["rows"] == 0


def test_boot_row_records_the_bracket_flag():
    from agents import analyst
    with patch("agents.analyst.boot_record", return_value={"pid": 1, "code_sha": "x"}), \
         patch("watchlist_check.check_and_warn", return_value={}), \
         patch("config.USE_BRACKET_ORDERS", False), \
         patch("db.log_boot_activity") as boot:
        analyst.record_boot("test")
    assert boot.call_args.args[1]["use_bracket_orders"] is False


def test_boot_row_records_the_options_backend():
    from agents import analyst
    with patch("agents.analyst.boot_record", return_value={"pid": 1, "code_sha": "x"}), \
         patch("watchlist_check.check_and_warn", return_value={}), \
         patch("ingestion.options_flow._KEY_CONFIGURED", False), \
         patch("db.log_boot_activity") as boot:
        analyst.record_boot("test")
    assert boot.call_args.args[1]["options_backend"] == "free_yfinance"
