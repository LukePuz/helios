"""Volatility shadow readout: uses the live helper, only pre-fill bars, no prices or sizes."""
import importlib.util
import os
from datetime import date, datetime, timedelta, timezone

import risk_manager

_spec = importlib.util.spec_from_file_location(
    "vol_shadow_readout", os.path.join(os.path.dirname(__file__), "..", "scripts", "vol_shadow_readout.py"))
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)


def _series(n, start, step):
    d0 = date(2026, 9, 1)
    out, c = [], start
    for i in range(n):
        out.append({"date": (d0 + timedelta(days=i)).isoformat(), "high": c * 1.01, "low": c * 0.99,
                    "close": c})
        c *= 1 + (step if i % 2 == 0 else -step / 2)
    return out


def _setup():
    bars = {"NVDA": _series(60, 187.43, 0.03), "SPY": _series(60, 612.11, 0.004)}
    fill_ts = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)            # NY date 2026-10-01
    fills = [{"symbol": "NVDA", "qty": 7.0, "price": 191.37, "ts": fill_ts, "env": "live",
              "approval_id": 41, "buys_before_qty": 0.0, "sells_before_qty": 0.0}]
    snaps = {"live": [(fill_ts - timedelta(hours=3), 98_765.0)]}
    approvals = [{"id": 41, "ticker": "NVDA", "ts": fill_ts - timedelta(minutes=5), "status": "APPROVED",
                  "stop": 170.0, "bucket": "Compounder", "stop_method": "2.5× ATR (8.55) below entry",
                  "env": "live"}]
    return fills, bars, snaps, approvals


def test_rows_use_only_bars_before_the_fill_and_the_live_helper(monkeypatch):
    fills, bars, snaps, approvals = _setup()
    seen = {}
    real = risk_manager._vol_shadow_record

    def spy(*a, **k):
        seen["bars"] = risk_manager._vol_shadow_bars("NVDA", 20)
        seen["args"] = a
        return real(*a, **k)
    monkeypatch.setattr(risk_manager, "_vol_shadow_record", spy)
    before = risk_manager._vol_shadow_bars
    rows = vr.fill_rows(fills, bars, snaps, approvals)
    assert risk_manager._vol_shadow_bars is before                 # restored after the call
    assert all(b["date"] < "2026-10-01" for b in seen["bars"]) and len(seen["bars"]) == 30
    r = rows[0]
    assert r["linked"] and r["pv_known"] and r["bucket"] == "Compounder"
    assert r["vol_target_shadow"]["read"] == "ok"
    assert r["stop_vol"]["atr_multiple"] == 2.5                    # ATR stop -> bucket midpoint
    assert abs(r["vol_target_shadow"]["order_weight"] - round(7 * 191.37 / 98_765.0, 4)) < 1e-9


def test_fallback_and_unlinked_fills(monkeypatch):
    fills, bars, snaps, approvals = _setup()
    approvals[0]["stop_method"] = "Fallback stop (8% below entry — ATR unavailable)"
    r = vr.fill_rows(fills, bars, snaps, approvals)[0]
    assert r["stop_vol"]["atr_multiple"] is None
    fills[0]["approval_id"] = None
    approvals[0]["ts"] = fills[0]["ts"] - timedelta(days=3)           # outside the 2-day match
    r = vr.fill_rows(fills, bars, {}, approvals)[0]
    assert not r["linked"] and not r["pv_known"]
    assert r["vol_target_shadow"]["order_weight"] is None
    assert r["stop_vol"]["stop_pct"] is None


def test_readout_has_no_prices_sizes_or_account_values():
    fills, bars, snaps, approvals = _setup()
    text = vr.readout(vr.fill_rows(fills, bars, snaps, approvals), "live", "abc123")
    for secret in ("191.37", "187.43", "612.11", "98765", "98,765", "170.0", "| 7 |"):
        assert secret not in text
    assert "| 2026-10-01 | NVDA | live | Compounder |" in text
    assert "not a study result" in text and "Fills: 1; bars ok: 1" in text


def test_fallback_link_requires_the_same_env():
    fills, bars, snaps, approvals = _setup()
    fills[0]["approval_id"] = None                                    # force the fallback match
    assert vr.fill_rows(fills, bars, snaps, approvals)[0]["linked"]
    approvals[0]["env"] = "paper"
    r = vr.fill_rows(fills, bars, snaps, approvals)[0]
    assert not r["linked"] and r["stop_vol"]["stop_pct"] is None
    assert r["vol_target_shadow"]["weights_at"] == "fill"


# --- non-finite closes in x20 --------------------------------------------------

NAN, INF = float("nan"), float("inf")


def _x20_bars():
    d0 = date(2026, 9, 1)
    days = [(d0 + timedelta(days=i)).isoformat() for i in range(30)]
    name = [{"date": d, "close": 100.0 + i} for i, d in enumerate(days)]
    spy = [{"date": d, "close": 400.0} for d in days]
    return name, spy, d0


def test_x20_detail_flags_the_name_or_spy_and_never_returns_nan():
    name, spy, d0 = _x20_bars()
    v, bad = vr.x20_detail(name, spy, d0)
    assert bad is None and abs(v - 0.2) < 1e-12 and vr._x20(name, spy, d0) == v
    for i in (0, 20):
        n2 = [dict(b) for b in name]
        n2[i]["close"] = NAN
        assert vr.x20_detail(n2, spy, d0) == (None, "name") and vr._x20(n2, spy, d0) == vr.NONFINITE
    s2 = [dict(b) for b in spy]
    s2[20]["close"] = INF
    assert vr.x20_detail(name, s2, d0) == (None, "SPY")
    n3 = [dict(b) for b in name]
    n3[5]["close"] = NAN                         # between the two closes: not read
    assert vr.x20_detail(n3, spy, d0)[1] is None
    assert vr.x20_detail(name[:20], spy, d0) == (None, None)        # 20th bar not in yet


def test_rows_and_readout_count_a_nonfinite_x20():
    import datetime as _dt
    fills, bars, snaps, approvals = _setup()
    i = next(k for k, b in enumerate(bars["NVDA"]) if b["date"] >= "2026-10-01")
    bars["NVDA"][i]["close"] = NAN               # the fill-day close x20 starts from
    rows = vr.fill_rows(fills, bars, snaps, approvals)
    assert rows[0]["x20"] is None and rows[0]["x20_nonfinite"] == ["NVDA"]
    now = _dt.datetime(2026, 11, 27, 23, 10, tzinfo=timezone.utc)
    text = vr.readout(rows, "live", "pinned0", now)
    assert "Generated 2026-11-27 23:10 UTC at code pinned0." in text
    assert "x20 non-finite (a NaN or infinite close of the name or SPY; out of the x20 mean, " \
           "no bar changed): 1 (NVDA)." in text
    assert "| - (0) | 1 (NVDA) |" in text and "| non-finite (NVDA) |" in text
    assert "nan" not in text.replace("non-finite", "").replace("NaN", "").lower()
    ok = vr.fill_rows(*_setup())
    assert ok[0]["x20_nonfinite"] == [] and "x20 non-finite" in vr.readout(ok, "live", "h", now)
