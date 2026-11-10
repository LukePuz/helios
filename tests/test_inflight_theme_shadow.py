"""Log-only: the theme total if buys approved or deferred but not yet filled
were counted (audit 2026-11-07 11:07 #4)."""
import inspect
from unittest.mock import patch

import db
import risk_manager as rm

_WL = {"META": {"sector": "AI Infrastructure — Hyperscalers"},
       "MSFT": {"sector": "AI Infrastructure — Hyperscalers"},
       "ANET": {"sector": "AI Infrastructure — Networking / Optical"},
       "OKLO": {"sector": "Nuclear / Baseload Power"}}


def _shadow(inflight, ticker="MSFT", proposed=0.10, current=0.30):
    with patch("db.get_inflight_buy_approvals", return_value=inflight) as get, \
         patch("knowledge_loader.load_watchlist", return_value=_WL):
        out = rm._inflight_theme_shadow(ticker, proposed, {"current_theme_total": current})
    return out, get


def test_same_theme_inflight_buys_are_added():
    inflight = [{"id": 1, "ticker": "META", "action": "OPEN", "status": "FAILED", "pct": 0.10},
                {"id": 2, "ticker": "OKLO", "action": "OPEN", "status": "PENDING", "pct": 0.05},
                {"id": 3, "ticker": "ANET", "action": "OPEN", "status": "PROCESSING", "pct": 0.08}]
    out, get = _shadow(inflight)
    assert get.call_args.kwargs == {"exclude_ticker": "MSFT"}
    assert out["inflight_n"] == 3 and [r["ticker"] for r in out["same_theme"]] == ["META"]
    assert out["would_theme_total"] == 0.40 and out["would_breach"] is True
    assert out["would_cap_pct"] == round(rm.MAX_SINGLE_THEME_PCT - 0.30, 4)


def test_nothing_inflight_or_no_theme():
    out, _ = _shadow([], current=0.20)
    assert out["would_theme_total"] == 0.20 and out["would_breach"] is False
    out, _ = _shadow([], ticker="XYZ")
    assert out["theme"] is None and out["would_breach"] is None


def test_a_failed_read_is_marked_errored_and_never_raises():
    assert _shadow(None)[0] == {"errored": True}
    with patch("db.get_inflight_buy_approvals", side_effect=RuntimeError("db")):
        assert rm._inflight_theme_shadow("MSFT", 0.1, {})["errored"] is True


def test_query_excludes_approved_and_swept_rows():
    src = inspect.getsource(db.get_inflight_buy_approvals)
    assert "status IN ('PENDING','PROCESSING')" in src and "'APPROVED'" not in src
    assert "NOT LIKE" in src and "action IN ('OPEN','ADD')" in src


def test_crew_records_it_beside_the_theme_shadow():
    from agents import crew
    src = inspect.getsource(crew)
    assert 'sizing["concentration_inflight_shadow"] = _inflight_theme_shadow(' in src


def test_both_concentration_exits_carry_it():
    from agents import crew
    src = inspect.getsource(crew)
    assert src.count('"concentration_inflight_shadow":\n') == 2
    assert src.count('sizing.get("concentration_inflight_shadow")') == 2
