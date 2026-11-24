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
