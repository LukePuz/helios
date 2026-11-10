"""c6 bar shadow (log-only).

Three conviction-7 bars stop an OPEN/ADD at conviction 6 on the entry path:
  regime      agents.analyst: OPEN in a Neutral/Risk-Off/Strong Risk-Off
              regime needs conviction >= 7
  moonshot    agents.analyst: a Moonshot OPEN needs conviction >= 7
  tier_floor  risk_manager.size_position: the account tier's minimum
              conviction (config.TIER_MIN_CONVICTION_TO_QUEUE, 7 at micro)

A conviction-6 OPEN/ADD that reached one of them with every earlier gate
passed writes one activity_feed row, stage c6_bar_shadow, kind verdict.
The row says whether, with the bar at 6, the verdict would have been
queued as an approval, and at what size. Analyst gates run before the bars
(UDR, sources, return rate, Trend Template) and a model pass_to_risk of
false end the verdict before any bar, so they write nothing. Routing and
sizing checks after the bars (active approval, rebuy cooldown, catalyst
gate, earnings blackout, the new-fact gate, price, halt, sizing, position
cap, concentration, the micro-tier name cap) are re-checked and name what
else would have stopped it in blocked_also_by (would_queue false). A c6
buy the new-fact gate stopped before sizing is enqueued too, and written
only when a conviction-7 bar (the tier floor) applies to it. Not mirrored:
the swap-for-cash path (a buy it might have funded shows as cash_blocked)
and every execution-stage block (execution's cash/position re-check, the
price-drift abort, the Telegram tap when AUTO_EXECUTE is off, a broker
rejection). would_queue true means an approval would have been logged, not
that an order would have filled.

crew only builds a plain payload and enqueues tasks.c6_bar_shadow (ops
queue, 20/30 s time limits, retry=False so a hung broker can't hold up
routing): no price, ATR or broker call on the routing path. The task sizes
on RiskManager(simulation=True): the halt check only reads, ATR comes from
the price_history cache, no alert is sent. It never connects to IBKR: when
the worker's session isn't already connected, the row is written with
would_queue null and blocked_also_by ["broker_unavailable"] (a connect
attempt retries past the soft time limit and holds an ops worker).

Timing: the routing pre-checks (active approval, cooldown, catalyst,
earnings), the account tier and the sizing are evaluated when the task runs
(evaluated_at), up to 30 minutes (the task's expiry) after the verdict
(as_of); as_of, session and entry_ref_price are from the verdict.

One row per shadow_key: the Redis marker is set before the write, so a row
that errored (error set, would_queue null), hit the soft time limit
(written with error SoftTimeLimitExceeded, then the limit is re-raised) or
found the broker unavailable uses its marker and is not retried. A failed
DB write releases the marker; a task past its expiry writes nothing.

entry_ref_price is a raw quote (ref_price_source brief: the price in the
analyst's brief; routing_quote: the quote crew's sizing used), or, with
neither, the latest cached daily close (price_history_close). price_history
holds yfinance closes adjusted for splits and dividends and rewritten on
each refresh; no unadjusted close is cached. Returns from a raw quote to
adjusted closes (tasks._check_c6_bar_outcomes) are off by any dividend in
the window and wrong across a split.

Redis dedupe markers: helios:c6_bar_shadow:<shadow_key>."""

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

STAGE = "c6_bar_shadow"
MARKER_PREFIX = "helios:c6_bar_shadow:"
_BAR = 6
_MARKER_TTL_S = 40 * 86400
_TASK_EXPIRES_S = 1800

# The analyst gate strings of the two analyst bars (agents/analyst.py).
_ANALYST_BARS = (("moonshot", "moonshot conviction"), ("regime", "regime conviction"))


try:
    from celery.exceptions import SoftTimeLimitExceeded
except Exception:                  # celery missing: nothing to re-raise
    class SoftTimeLimitExceeded(Exception):
        pass


def _broker_connected() -> bool:
    """Whether this worker's IBKR session is already connected. Reads
    ibkr's module state only: never connects."""
    try:
        import ibkr
        ib = ibkr._ib
        return ib is not None and bool(ib.isConnected())
    except SoftTimeLimitExceeded:
        raise
    except Exception:
        return False


def _enabled() -> bool:
    try:
        from config import C6_BAR_SHADOW_ENABLED
        return bool(C6_BAR_SHADOW_ENABLED)
    except Exception:
        return False


def analyst_record(model_action, model_conviction, action_pre_c7, pass_pre_c7, c7_fired,
                   action, conviction, pass_final) -> dict | None:
    """What analyze() keeps in provenance.c6_bar: a conviction-6 OPEN/ADD
    that reached the analyst's conviction-7 bars with pass_to_risk true.
    c7_gates names the bars that turned it into a WATCH (empty when none
    did; the tier floor is checked in risk). None when off or the verdict
    doesn't qualify. Pure: no I/O."""
    if not _enabled() or conviction != _BAR or action_pre_c7 not in ("OPEN", "ADD") \
            or not pass_pre_c7:
        return None
    fired = [name for name, text in _ANALYST_BARS for g in c7_fired if text in g]
    if fired and (action != "WATCH" or pass_final):
        return None            # a bar fired but the verdict still passed: not ours to read
    return {"model_action": model_action, "model_conviction": model_conviction,
            "action": action_pre_c7, "c7_gates": fired}


def session_of(as_of: str) -> str:
    """pre / regular / post (NYSE hours, early closes) or closed (weekend,
    holiday), from an ISO timestamp read in ET."""
    from zoneinfo import ZoneInfo
    from market_calendar import EARLY_CLOSE, NYSE_EARLY_CLOSES, NYSE_HOLIDAYS, RTH_CLOSE, RTH_OPEN
    et = datetime.fromisoformat(as_of).astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5 or et.date() in NYSE_HOLIDAYS:
        return "closed"
    close = EARLY_CLOSE if et.date() in NYSE_EARLY_CLOSES else RTH_CLOSE
    t = et.time()
    return "pre" if t < RTH_OPEN else "regular" if t < close else "post"


def _et_date(as_of: str) -> str:
    from zoneinfo import ZoneInfo
    return datetime.fromisoformat(as_of).astimezone(ZoneInfo("America/New_York")).date().isoformat()


def _payload(result: dict, regime_str: str, c6: dict, c7_gates: list[str], price, price_src,
             prechecks: bool) -> dict:
    """The task's input: plain data only. Pure."""
    from config import TRADING_ENV
    prov = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
    sid = prov.get("signal_id")
    sid = sid if isinstance(sid, int) and not isinstance(sid, bool) else None
    trig = prov.get("trigger") if isinstance(prov.get("trigger"), dict) else {}
    as_of = datetime.now(timezone.utc).isoformat()
    price = float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) \
        and price > 0 else None
    return {"ticker": result["ticker"], "env": TRADING_ENV, "signal_id": sid, "as_of": as_of,
            "session": session_of(as_of), "bucket": result.get("thesis_bucket"),
            "regime": regime_str, "model_action": c6.get("model_action"),
            "model_conviction": c6.get("model_conviction"),
            "conviction_raw": result.get("conviction_raw"),
            "fundamentals_read": prov.get("fundamentals_read"),
            "action": c6.get("action") or result.get("action"), "conviction": _BAR,
            "c7_gates": list(c7_gates), "entry_ref_price": price,
            "ref_price_source": price_src if price else None, "prechecks": prechecks,
            "trigger": {"path": trig.get("path"), "new_fact": trig.get("new_fact")},
            "shadow_key": (f"{TRADING_ENV}:{result['ticker']}:s{sid}:c6" if sid is not None
                           else f"{TRADING_ENV}:{result['ticker']}:d{_et_date(as_of)}:c6")}


def _enqueue(payload: dict) -> None:
    from tasks import c6_bar_shadow
    c6_bar_shadow.apply_async(args=[payload], queue="ops", expires=_TASK_EXPIRES_S, retry=False)


def observe_unrouted(result: dict, regime: dict | None) -> dict | None:
    """A verdict crew does not route: enqueues the shadow when an analyst
    conviction-7 bar was all that stopped a c6 OPEN/ADD. No I/O besides the
    enqueue. Returns the payload or None. Never raises."""
    try:
        if not _enabled() or result.get("pass_to_risk"):
            return None
        prov = result.get("provenance")
        c6 = prov.get("c6_bar") if isinstance(prov, dict) else None
        if not isinstance(c6, dict) or not c6.get("c7_gates"):
            return None
        payload = _payload(result, (regime or {}).get("regime", "Neutral"), c6,
                           c6["c7_gates"], c6.get("ref_price"), "brief", prechecks=True)
        _enqueue(payload)
        return payload
    except Exception as e:
        log.warning(f"[{result.get('ticker')}] c6 bar shadow enqueue failed: {e}")
        return None


def observe_tier_floor(result: dict, regime_str: str, price, sizing: dict) -> dict | None:
    """crew's live sizing returned 0 shares: enqueues the shadow when the
    tier's conviction floor stopped a c6 OPEN/ADD. The routing checks
    before sizing already passed live. Returns the payload or None. Never
    raises."""
    try:
        if not _enabled() or result.get("action") not in ("OPEN", "ADD") \
                or result.get("conviction") != _BAR:
            return None
        if not any("tier minimum" in f for f in (sizing.get("flags") or [])):
            return None
        prov = result.get("provenance")
        c6 = prov.get("c6_bar") if isinstance(prov, dict) else None
        c6 = c6 if isinstance(c6, dict) else {"action": result.get("action")}
        payload = _payload(result, regime_str, c6, ["tier_floor"], price, "routing_quote",
                           prechecks=False)
        _enqueue(payload)
        return payload
    except Exception as e:
        log.warning(f"[{result.get('ticker')}] c6 bar shadow enqueue failed: {e}")
        return None


def observe_new_fact_block(result: dict, regime_str: str) -> dict | None:
    """crew's new-fact gate stopped a c6 OPEN/ADD before sizing: enqueues
    the shadow with no c7 gate yet; the task adds the tier floor when it
    applies and writes nothing when no conviction-7 bar does. Returns the
    payload or None. Never raises."""
    try:
        if not _enabled() or result.get("action") not in ("OPEN", "ADD") \
                or result.get("conviction") != _BAR:
            return None
        prov = result.get("provenance")
        c6 = prov.get("c6_bar") if isinstance(prov, dict) else None
        c6 = c6 if isinstance(c6, dict) else {"action": result.get("action")}
        payload = _payload(result, regime_str, c6, [], c6.get("ref_price"), "brief",
                           prechecks=True)
        payload["gate_blocked"] = "no_new_fact"
        _enqueue(payload)
        return payload
    except Exception as e:
        log.warning(f"[{result.get('ticker')}] c6 bar shadow enqueue failed: {e}")
        return None


def _new_fact_block(payload: dict, rec: dict) -> bool:
    """crew.new_fact_gate on the verdict's trigger; its record goes in
    rec["new_fact_gate"] (absent when the gate is off). True when it blocks."""
    from agents.crew import new_fact_gate
    gate = new_fact_gate(rec["ticker"], rec["action"], payload.get("trigger"))
    if gate is None:
        return False
    rec["new_fact_gate"] = gate
    return not gate["passed"]


def _sizing_block(sizing: dict) -> str:
    """A label for why sizing gave 0 shares. Labels only: the flags carry
    dollar amounts."""
    flags = " ".join(sizing.get("flags") or [])
    if "HALT" in flags:
        return "halted"
    if sizing.get("heat_blocked"):
        return "heat_cap"
    if sizing.get("cash_blocked"):
        return "cash_blocked"
    for text, label in (("tier minimum", "tier_floor"), ("too low for", "bucket_band"),
                        ("concurrent positions", "position_cap"),
                        ("already at/near target", "at_target"), ("Invalid price", "invalid_price"),
                        ("single-share cap", "single_share_cap"),
                        ("per-trade risk cap", "risk_cap"),
                        ("less than 1 share", "under_one_share")):
        if text in flags:
            return label
    return "zero_shares"


def _routing_blocks(ticker: str, action: str, conviction: int) -> list[str]:
    """crew._route_to_approval's checks before sizing, read-only, failing
    the same way it does (closed on an unreadable duplicate-buy guard for a
    buy, as crew does since audit 11-06 21:36 #8; closed for earnings)."""
    import db
    out = []
    try:
        if db.has_active_approval(ticker, strict=True):
            out.append("active_approval")
    except SoftTimeLimitExceeded:
        raise
    except Exception:
        if action in ("OPEN", "ADD"):
            out.append("guard_unreadable")
    try:
        from config import REBUY_COOLDOWN_HOURS
    except ImportError:
        REBUY_COOLDOWN_HOURS = 4
    try:
        prior = db.get_last_executed_approval(ticker, REBUY_COOLDOWN_HOURS, strict=True)
    except SoftTimeLimitExceeded:
        raise
    except Exception:
        prior = None
        if action in ("OPEN", "ADD") and "guard_unreadable" not in out:
            out.append("guard_unreadable")
    if prior and conviction <= prior["conviction"]:
        out.append("rebuy_cooldown")
    try:
        from catalyst_service.gates import catalyst_gate_check
        if not catalyst_gate_check(ticker, days_ahead=45).get("approved"):
            out.append("catalyst_gate")
    except SoftTimeLimitExceeded:
        raise
    except Exception:
        pass
    from config import EARNINGS_ENTRY_BLACKOUT_DAYS
    try:
        near = (db.is_near_earnings(ticker, EARNINGS_ENTRY_BLACKOUT_DAYS, strict=True)
                if EARNINGS_ENTRY_BLACKOUT_DAYS else False)
    except SoftTimeLimitExceeded:
        raise
    except Exception:
        near = None
    if near is not False:
        out.append("earnings_blackout")
    return out


def _cached_close(ticker: str) -> tuple[float | None, str | None]:
    """The latest cached daily close and its date (adjusted; see the module
    docstring)."""
    import db
    bars = db.get_price_history(ticker, days=1)
    return (float(bars[-1]["close"]), bars[-1]["date"]) if bars else (None, None)


def new_record(payload: dict) -> dict:
    """The verdict row before evaluation."""
    rec = {k: payload.get(k) for k in (
        "ticker", "env", "signal_id", "as_of", "session", "bucket", "regime", "model_action",
        "model_conviction", "conviction_raw", "fundamentals_read", "action", "conviction",
        "entry_ref_price", "ref_price_source", "shadow_key")}
    rec.update(kind="verdict", c7_gates=list(payload.get("c7_gates") or []),
               evaluated_at=datetime.now(timezone.utc).isoformat(), account_tier=None,
               would_queue=None, intended_shares=None, intended_notional_pct_of_equity=None,
               blocked_also_by=[])
    try:
        from agents.analyst import shadow_versions
        rec.update(shadow_versions())
    except SoftTimeLimitExceeded:
        raise
    except Exception:
        rec.update(code_sha=None, env_sha=None)
    return rec


def evaluate(payload: dict, client=None, rec: dict | None = None) -> dict:
    """The verdict row: the entry path from the bar on, at conviction 6.
    Runs in the task. Without a client, uses the worker's broker client
    (account and positions only) when it is already connected, else marks
    the row broker_unavailable. Fills rec in place when given."""
    from config import get_active_limits
    rec = new_record(payload) if rec is None else rec
    ticker, blocked = rec["ticker"], rec["blocked_also_by"]
    try:
        if not rec["entry_ref_price"]:
            price, day = _cached_close(ticker)
            if price:
                rec.update(entry_ref_price=price, ref_price_source="price_history_close",
                           ref_close_date=day)
        if payload.get("prechecks"):
            blocked += _routing_blocks(ticker, rec["action"], _BAR)
        if _new_fact_block(payload, rec):
            blocked.append("no_new_fact")
        if client is None:
            if not _broker_connected():
                blocked.append("broker_unavailable")
                return rec
            from ibkr import get_client
            client = get_client()
        from risk_manager import RiskManager
        sim = RiskManager(client, simulation=True)
        pv = sim.portfolio_value()
        limits = get_active_limits(pv)
        rec["account_tier"] = limits.get("tier")
        if payload.get("prechecks") and limits["min_conviction_to_queue"] > _BAR \
                and "tier_floor" not in rec["c7_gates"]:
            rec["c7_gates"].append("tier_floor")
        price = rec["entry_ref_price"]
        if not price:
            blocked.append("no_price")
            rec["would_queue"] = False
            return rec
        sizing = sim.size_position(ticker, _BAR, rec["bucket"], rec["regime"], price,
                                   log_fallback=False, conviction_floor=_BAR)
        shares = int(sizing.get("shares_initial", 0) or 0)
        if shares <= 0:
            blocked.append(_sizing_block(sizing))
        else:
            shares = sim.shares_after_concentration(ticker, sizing, rec["bucket"])
            if shares <= 0:
                blocked.append("concentration")
            else:
                if sim.check_correlation(ticker).get("reduce_by", 0) > 0:
                    shares = max(1, int(shares * 0.75))   # crew's correlation cut
                cap = sim.micro_name_cap(ticker, shares, float(price))   # crew's last cut
                if cap is not None:
                    rec["micro_name_cap"] = cap
                    shares = cap["shares_after"]
                    if shares <= 0:
                        blocked.append("micro_name_cap")
        shares = max(shares, 0)
        rec["intended_shares"] = shares
        rec["intended_notional_pct_of_equity"] = (round(shares * float(price) / pv * 100, 4)
                                                  if pv and pv > 0 else None)
        rec["would_queue"] = not blocked and shares >= 1
    except SoftTimeLimitExceeded:
        raise
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        rec["would_queue"] = None
    return rec


def _emit(rec: dict) -> bool:
    """One activity_feed row per shadow_key (a Redis NX marker; written
    anyway when Redis is down)."""
    import db
    marker = MARKER_PREFIX + rec["shadow_key"]
    r = None
    try:
        import stop_lock
        r = stop_lock._redis()
        if not r.set(marker, "1", nx=True, ex=_MARKER_TTL_S):
            return False
    except SoftTimeLimitExceeded:
        raise
    except Exception:
        r = None
    if rec.get("would_queue"):
        what = (f"would queue {rec['intended_shares']} shares "
                f"({rec['intended_notional_pct_of_equity']}% of equity)")
    elif rec.get("would_queue") is False:
        what = f"not queued ({', '.join(rec['blocked_also_by']) or 'zero shares'})"
    else:
        what = f"not evaluated ({rec.get('error') or 'unknown'})"
    ok = db.log_activity(STAGE, rec["ticker"],
                         f"c6 {rec['action']} stopped by {', '.join(rec['c7_gates'])}; "
                         f"at a bar of 6: {what}", "neutral", rec)
    if not ok and r is not None:
        try:
            r.delete(marker)
        except Exception:
            pass
    return ok


def run(payload: dict, client=None) -> dict | None:
    """tasks.c6_bar_shadow's body. Raises only SoftTimeLimitExceeded."""
    rec = None
    try:
        if not _enabled():
            return None
        rec = new_record(payload)
        evaluate(payload, client, rec)
        if payload.get("gate_blocked") and not rec["c7_gates"]:
            return None            # no conviction-7 bar applied: not a c6-bar case
        _emit(rec)
        return rec
    except SoftTimeLimitExceeded:
        if rec is not None:        # recorded, then the limit goes on
            rec.update(would_queue=None, error="SoftTimeLimitExceeded")
            try:
                _emit(rec)
            except Exception:
                pass
        raise
    except Exception as e:
        log.warning(f"[{payload.get('ticker')}] c6 bar shadow failed: {e}")
        return None
