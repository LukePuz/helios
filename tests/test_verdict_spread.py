import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import verdict_spread as vs  # noqa: E402


def _row(sym, hour_utc, conv, udr, action="HOLD", day=4):
    return (sym, datetime(2026, 11, day, hour_utc, tzinfo=timezone.utc),
            {"model_conviction": conv, "udr_check": {"model": udr}, "final_action": action})


def test_spread_groups_by_et_day_and_flags_a_straddle():
    rows = [_row("NVDA", 14, 6, 1.0), _row("NVDA", 16, 7, 3.6, "OPEN"), _row("NVDA", 18, 5, 2.4),
            _row("AMD", 14, 6, 2.5), _row("AMD", 15, 6, 3.0), _row("AMD", 16, 6, 2.1),
            _row("PLTR", 14, 6, 1.0), _row("PLTR", 15, 6, 3.0)]   # 2 verdicts: below the minimum
    days = vs.spread(rows, 2.0, 3)
    assert [(d["ticker"], d["n"], d["crosses_floor"]) for d in days] == [("AMD", 3, False), ("NVDA", 3, True)]
    nvda = days[1]
    assert nvda["conviction"] == (5, 7) and nvda["udr"] == (1.0, 3.6) and nvda["actions"] == ["HOLD", "OPEN"]
    assert nvda["actions_differ"] is True and nvda["entry_action"] is True
    assert days[0]["actions_differ"] is False and days[0]["entry_action"] is False


def test_et_day_boundary():
    # 03:00 UTC on 11-05 is 22:00 ET on 11-04
    rows = [_row("X", 3, 6, 1.0, day=5), _row("X", 14, 6, 3.0), _row("X", 15, 6, 3.0)]
    assert vs.spread(rows, 2.0, 3)[0]["day"] == "2026-11-04"


def test_rows_without_udr_check_or_numbers_are_skipped():
    rows = [_row("X", 14, 6, 1.0), _row("X", 15, None, None), ("X", datetime(2026, 11, 4, 16, tzinfo=timezone.utc), {})]
    d = vs.spread(rows, 2.0, 2)[0]
    assert d["n"] == 2 and d["udr"] == (1.0, 1.0) and d["crosses_floor"] is False


def test_render_has_counts_only_and_the_summary():
    days = vs.spread([_row("NVDA", 14, 6, 1.0), _row("NVDA", 16, 7, 3.6), _row("NVDA", 18, 5, 2.4)], 2.0, 3)
    text = vs.render(days, 2.0, 21, 3, "live", "abc1234")
    assert "straddles the 2 entry floor on 1; of those, the final action differs within the day on 0" in text
    assert "| 2026-11-04 | NVDA | 3 | 5-7 | 1.00-3.60 | yes |  |  | HOLD |" in text
    assert "$" not in text
