"""Routing rows carry the verdict's model and provenance (SEF run 13)."""
from unittest.mock import patch

from agents import crew


def test_route_exit_row_has_verdict_provenance():
    prov = {"code_sha": "abc", "model": "qwen-test", "prompt_sha": "p1"}
    result = {"ticker": "META", "action": "OPEN", "conviction": 6, "provenance": prov}
    with patch("db.log_signal") as log_signal:
        crew._log_route_exit("META", "ROUTE_ZERO_SHARES", result, "c6 < micro floor 7")
    args, kw = log_signal.call_args
    assert args[:2] == ("META", "ROUTE_ZERO_SHARES") and args[2] == 0.6
    assert kw["model"] == "qwen-test" and kw["provenance"] == prov
    assert kw["decision"] == {"route_exit": "ROUTE_ZERO_SHARES", "verdict_action": "OPEN",
                              "verdict_conviction": 6}


def test_route_exit_without_provenance_still_logs():
    with patch("db.log_signal") as log_signal:
        crew._log_route_exit("META", "ROUTE_NO_PRICE", {"conviction": 7}, "no price")
    assert log_signal.call_args.kwargs["provenance"] is None


def test_code_mixed_flags_a_pull_without_restart():
    """Professor M75: HEAD on disk differs from the HEAD read at import."""
    from unittest.mock import patch
    from agents import analyst
    analyst._DISK_CHECK.update(at=0.0, value=None)
    with patch.dict(analyst._CODE_SHA, {"code_sha": "aaa111", "head_sha": "aaa111"}), \
         patch.object(analyst, "_git_code_sha",
                      return_value={"code_sha": "bbb222", "head_sha": "ccc333"}):
        v = analyst._code_mixed_check()
    assert v["code_mixed"] is True and v["disk_code_sha"] == "bbb222"
    analyst._DISK_CHECK.update(at=0.0, value=None)
    # log-only pull: HEAD moved, code_sha didn't (M75 b)
    with patch.dict(analyst._CODE_SHA, {"code_sha": "aaa111", "head_sha": "aaa111"}), \
         patch.object(analyst, "_git_code_sha",
                      return_value={"code_sha": "aaa111", "head_sha": "ddd444"}):
        assert analyst._code_mixed_check()["code_mixed"] is False
    analyst._DISK_CHECK.update(at=0.0, value=None)
    # hot edit: -dirty appears (M75 c)
    with patch.dict(analyst._CODE_SHA, {"code_sha": "aaa111", "head_sha": "aaa111"}), \
         patch.object(analyst, "_git_code_sha",
                      return_value={"code_sha": "aaa111-dirty", "head_sha": "aaa111-dirty"}):
        assert analyst._code_mixed_check()["code_mixed"] is True
    analyst._DISK_CHECK.update(at=0.0, value=None)


def test_missing_model_digest_is_retried():
    """Professor M77: a failed first lookup no longer sticks for the process."""
    from unittest.mock import MagicMock, patch
    from agents import analyst
    saved = analyst._PROVENANCE_STATIC
    analyst._PROVENANCE_STATIC = {"code_sha": "x", "model_digest": None}
    try:
        resp = MagicMock()
        resp.json.return_value = {"models": [{"name": analyst.ANALYST_MODEL, "digest": "abcdef1234567890"}]}
        with patch.object(analyst.requests, "get", return_value=resp), \
             patch.object(analyst, "_code_mixed_check", return_value={}):
            assert analyst._provenance("p")["model_digest"] == "abcdef123456"
    finally:
        analyst._PROVENANCE_STATIC = saved


def test_regime_fallback_filter_follows_the_flag(monkeypatch):
    """A0853-1: default unchanged; with REGIME_OVERALL_ONLY only overall rows."""
    from unittest.mock import MagicMock, patch
    import config
    import db
    for flag, expect in ((False, False), (True, True)):
        monkeypatch.setattr(config, "REGIME_OVERALL_ONLY", flag)
        cur = MagicMock()
        cur.fetchone.return_value = None
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            db.get_latest_regime()
        assert ("regime_type = 'overall'" in cur.execute.call_args[0][0]) is expect


def test_scan_summary_names_the_regime_row_and_last_bar():
    """A0853-1 / A1053-2: the scan line says which regime row it used and
    which day's bar was last scored."""
    import tasks
    line = tasks._scan_summary({"regime": "Risk-Off", "score": 36.0,
                                "regime_source": "gics_sector:XLK",
                                "timestamp": "2026-10-29 13:31:18",
                                "signals": {"last_bar_date": "2026-10-29"}}, [])
    assert "regime source gics_sector:XLK @ 2026-10-29 13:31:18" in line
    assert "last bar 2026-10-29" in line


def test_latest_regime_reports_its_source():
    from datetime import datetime
    from unittest.mock import MagicMock, patch
    import db
    for typ, etf, want in (("gics_sector", "XLK", "gics_sector:XLK"), (None, None, "overall"),
                           ("overall", None, "overall")):
        cur = MagicMock()
        cur.fetchone.return_value = ("Risk-Off", 36, 0.6, {}, {}, "r", datetime(2026, 10, 29), typ, etf)
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
            assert db.get_latest_regime()["regime_source"] == want


def test_knowledge_coverage_lists_cut_headings_and_matches_the_prompt_caps():
    """A1053-1: record which headings the prefix cut drops; the caps must stay
    equal to the slices in _build_prompt."""
    import inspect
    from agents import analyst
    src = inspect.getsource(analyst._build_prompt)
    assert f"thesis_text[:{analyst._THESIS_CAP}]" in src
    assert f"philosophy_text[:{analyst._PHILOSOPHY_CAP}]" in src
    philosophy = "# P\n" + "x" * 2100 + "\n### When to Exit\nrules\n"
    cov = analyst._knowledge_coverage("ZZZZ", "## T\nshort", philosophy)
    assert cov["philosophy"]["cut_headings"] == ["When to Exit"]
    assert cov["philosophy"]["cap_kept"] == 2000 and cov["thesis"]["cut_headings"] == []


def test_fundamentals_history_keeps_revision_fields_and_falls_back():
    """A1253-3: the new columns are written; before migration 60 the old
    insert still runs."""
    from unittest.mock import MagicMock, patch
    import db
    metrics = {"forward_pe": 30.0, "forward_eps": 12.5, "target_mean_price": 600.0,
               "shares_outstanding": 7.4e9, "net_cash": 5e10}
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.log_fundamentals_history("MSFT", metrics)
    sql, params = cur.execute.call_args[0]
    assert "target_mean_price" in sql and 600.0 in params and 12.5 in params
    cur.execute.side_effect = [Exception("column does not exist"), None]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.log_fundamentals_history("MSFT", metrics)
    assert "target_mean_price" not in cur.execute.call_args[0][0]


def test_decision_inputs_record_regime_row_and_implied_growth():
    """C25 / A1253-2: provenance names the regime row used and the reverse-DCF
    numbers, after the call."""
    from unittest.mock import patch
    from agents import analyst
    rp = {"regime": "Risk-Off", "score": 36.0, "regime_source": "gics_sector:XLK",
          "timestamp": "2026-10-29 13:31:18", "signals": {"last_bar_date": "2026-10-29"}}
    with patch("db.get_trade_ladder", return_value={"implied_growth_rate": 0.18,
                                                    "growth_gap_vs_actual": 0.04,
                                                    "growth_assumption_flag": "STRETCHED",
                                                    "computed_at": "2026-10-28 22:00"}):
        rec = analyst._decision_inputs_record("MSFT", rp)
    assert rec["regime_used"] == {"label": "Risk-Off", "score": 36.0, "source": "gics_sector:XLK",
                                  "timestamp": "2026-10-29 13:31:18", "last_bar_date": "2026-10-29"}
    assert rec["implied_growth"]["implied_growth_rate"] == 0.18
    assert rec["implied_growth"]["as_of"] == "2026-10-28 22:00"


def test_ollama_call_keeps_token_counts():
    """C26: prompt_eval_count etc. are kept for provenance; a full context warns."""
    from unittest.mock import MagicMock, patch
    from agents import analyst
    resp = MagicMock()
    resp.json.return_value = {"response": "{}", "prompt_eval_count": 8190, "eval_count": 900,
                              "done_reason": "stop", "total_duration": 9.1e10}
    resp.json.return_value["done_reason"] = "length"
    with patch("agents.analyst.requests.post", return_value=resp), \
         patch.object(analyst.log, "warning") as warn:
        assert analyst._call_ollama("p", num_ctx=8192) == "{}"
    assert analyst._LAST_CALL_STATS["prompt_eval_count"] == 8190
    assert analyst._LAST_CALL_STATS["num_ctx"] == 8192
    assert "num_predict" in warn.call_args[0][0]
    # A failed call clears the previous counts.
    import requests
    with patch("agents.analyst.requests.post", side_effect=requests.exceptions.Timeout()):
        assert analyst._call_ollama("p") is None
    assert analyst._LAST_CALL_STATS == {}


def test_knowledge_coverage_from_the_final_prompt():
    """M82: a heading trimmed after the cap (absent from the prompt) counts as cut."""
    from agents import analyst
    philosophy = "# P\nintro\n## Compounders\nx\n"
    cov = analyst._knowledge_coverage("ZZZZ", "## T\nshort", philosophy, prompt="# P\nintro\n## T")
    assert cov["philosophy"]["cut_headings"] == ["Compounders"]
    assert cov["thesis"]["cut_headings"] == []


def test_scanner_candidate_row_carries_regime_detail_with_fallback():
    """C25 part 2: the row records the regime row used; before migration 61
    the old insert still runs."""
    from unittest.mock import MagicMock, patch
    import db
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    kw = dict(ticker="MSFT", scan_type="daily", composite_score=65, technical_score=25,
              momentum_score=15, catalyst_score=0, thesis_alignment_score=25, priority="LOW",
              catalyst_detail="", passed_to_analyst=False, regime_at_scan="Risk-Off",
              regime_detail={"source": "gics_sector:XLK", "threshold": 80})
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.log_scanner_candidate(**kw)
    assert "regime_detail" in cur.execute.call_args[0][0]
    cur.execute.side_effect = [Exception("column does not exist"), None]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.log_scanner_candidate(**kw)
    assert "regime_detail" not in cur.execute.call_args[0][0]


def test_salvaged_action_is_not_routed():
    """A1453-6: an EXIT regex-pulled from broken JSON becomes a stub."""
    from unittest.mock import patch
    from agents import analyst
    broken = '{"action": "EXIT", "conviction": 2, "reasoning": "x" oops'
    assert analyst._extract_json(broken)["_salvaged"] is True
    with patch("agents.analyst._call_ollama", return_value=broken), \
         patch("agents.analyst._stub_recommendation", return_value={"stub": True}) as stub, \
         patch("knowledge_loader.load_thesis", return_value=""), \
         patch("knowledge_loader.load_philosophy", return_value=""):
        out = analyst.analyze({"ticker": "NVDA"}, situation_brief="b",
                              regime_packet={"regime": "Neutral", "score": 50, "reasoning": ""})
    assert out == {"stub": True}
    assert "salvaged" in stub.call_args[0][1]


def test_stored_prompt_has_no_account_dollars():
    """A1453-3: account dollar figures are redacted in the stored copy; prices stay."""
    from agents import analyst
    p = ("=== CURRENT PORTFOLIO ===\n  NVDA        1 shares | Entry: $180.10 | Current: $195.44 | "
         "P&L: +8.5% | Value: $195\n=== ACCOUNT STATE ===\nPortfolio Value: $1,234.56\n"
         "Cash Available:  $1,000.00\nBuying Power:    $2,000.00")
    out = analyst.redact_account_figures(p)
    assert "1,234" not in out and "1,000.00" not in out and "2,000" not in out and "Value: $195" not in out
    assert "Entry: $180.10" in out and "Current: $195.44" in out
    assert out.count("$[REDACTED]") == 4
