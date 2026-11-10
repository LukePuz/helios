"""Log-only: the theme cap grouped on the watchlist heading prefix, beside
today's sub-heading grouping (audit 2026-11-07 07:07 #5)."""
import risk_manager as rm

_WL = {"NVDA": {"sector": "AI Infrastructure — Compute / Accelerators"},
       "MSFT": {"sector": "AI Infrastructure — Hyperscalers"},
       "ANET": {"sector": "AI Infrastructure — Networking / Optical"},
       "PLTR": {"sector": "AI Software / Agent Economy"},
       "OKLO": {"sector": "Nuclear / Baseload Power"}}


def test_prefix():
    assert rm._theme_prefix("AI Infrastructure — Hyperscalers") == "AI Infrastructure"
    assert rm._theme_prefix("Quantum Computing") == "Quantum Computing"
    assert rm._theme_prefix("") == ""


def test_prefix_grouping_sums_the_sub_headings():
    pos = {"NVDA": 0.20, "MSFT": 0.15, "ANET": 0.10, "PLTR": 0.10, "OKLO": 0.05}
    out = rm._theme_group_shadow("ANET", _WL["ANET"]["sector"], _WL, pos, 0.10)
    assert out["theme"] == "AI Infrastructure" and out["theme_total"] == 0.45
    assert out["would_breach"] is True and out["current_theme_total"] == 0.10
    assert out["would_cap_pct"] == round(max(0.0, rm.MAX_SINGLE_THEME_PCT - 0.35), 4)


def test_off_watchlist_name_is_marked_not_skipped_silently():
    out = rm._theme_group_shadow("XYZ", "", _WL, {}, 0.05)
    assert out == {"group": "heading_prefix", "theme": None, "reason": "no_watchlist_sector"}


def test_never_raises():
    out = rm._theme_group_shadow("NVDA", "AI Infrastructure — X", None, {"NVDA": 0.1}, 0.1)
    assert out["error"]
