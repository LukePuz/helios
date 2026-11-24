"""Trim-or-exit rule for held positions. Default off, shadow first.

A fixed rule evaluated after a held-position review, on the verdict the
analyst already wrote. It reads the verdict's action and conviction and
nothing else from the model, and changes nothing the analyst is shown.

  earnings_pre  (print ahead, within TRIM_EXIT_EARNINGS_MAX_DAYS_BEFORE days)
      TRIM -> TRIM;  EXIT -> TRIM when unconfirmed, EXIT when confirmed
  earnings_day  (print dated today, not known to be after the close)
  general       (everything else)
      TRIM/EXIT -> nothing when unconfirmed, the action when confirmed

Confirmed: the newest analyst verdict at least TRIM_EXIT_CONFIRM_MIN_GAP_HOURS
(never under CONFIRM_MIN_GAP_FLOOR_HOURS) older than the evaluation is a TRIM
or EXIT, no older than TRIM_EXIT_CONFIRM_MAX_AGE_DAYS and not from before the
stop record's opened_at. An unreadable history is unconfirmed. An EXIT
verdict on a position with no stop record is left to the model's own pass
flag (exit_no_stop_record, a stand-down).

Switches (config.py, "Sell behaviour"): TRIM_EXIT_ENABLED off means apply()
returns its argument at once, with no read and no row. On by itself is
shadow: one TRIM_EXIT_RULE signals row per evaluation, routing untouched.
TRIM_EXIT_LIVE with a context named in TRIM_EXIT_LIVE_CONTEXTS lets the rule
decide routing for an in-domain verdict, on a copy of the result; the sale
then goes down the existing approval and execution path.

decide() is pure: every input it reads is in the logged row's decision, so
decide(row.decision, row.decision["flags"]) gives the row's outputs again.
The row also carries record-only fields decide() never reads: the latest
stored options implied move (implied_move_last_pct/_date/_dte, read as
agents.crew._gap_risk_shadow reads it) and gap_vs_stop, that move over the
stop distance; the stored event-spanning move (implied_move_event_*),
gap_vs_stop_event (it over the stop distance) and gap_vs_stop_basis, which
of the two moves spans the print under review.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone

log = logging.getLogger(__name__)

RULE_VERSION = "1"
# The signals kind (db._NON_ANALYST_SIGNALS). _log_row passes it as a literal
# so tests/test_signal_env_filter.py finds it.
KIND = "TRIM_EXIT_RULE"
SELL_ACTIONS = ("TRIM", "EXIT")
CONTEXTS = ("earnings_pre", "earnings_day", "general")
# The confirmation gap never goes below this many hours, whatever
# TRIM_EXIT_CONFIRM_MIN_GAP_HOURS says: at 0 the verdict under evaluation,
# written seconds earlier, would confirm itself.
CONFIRM_MIN_GAP_FLOOR_HOURS = 1.0
# Blocks that stand the rule down: the verdict routes as it did before.
# exit_no_stop_record: an EXIT verdict on a position whose stop record was
# read and is missing. The rule can neither confirm it (no opened_at) nor trim
# (no stop to resize), so the model's own pass flag decides, as before.
STAND_DOWN = ("shadow", "observe", "exempt", "context_not_live", "exit_no_stop_record")
# Blocks that mean no sale under the rule. row_not_logged is added by apply()
# after the row write, so it never appears in a fully written row.
NO_SALE = ("no_position", "cooldown", "min_hold", "no_stop_record", "pending_sell",
           "active_approval", "market_closed", "verdict_not_logged", "row_not_logged")
# The keys decide() writes; _log_row stores them beside the inputs.
OUTPUT_KEYS = ("mode", "context", "is_sell", "in_domain", "confirmed", "confirm_reason",
               "confirming_verdict", "action_if_confirmed", "action_if_unconfirmed",
               "would_action", "would_qty", "legacy_qty", "remaining_qty", "reason",
               "blocked_by", "owns_pass_flag", "routed")
# Record only (decide() never reads them): the stored event-spanning
# straddle, as agents.crew._read_event_move fills it.
_EVENT_MOVE_KEYS = ("implied_move_event_pct", "implied_move_event_expiry", "implied_move_event_dte",
                    "implied_move_event_earnings_date", "implied_move_event_date")


def _cfg() -> dict:
    """Every flag the rule reads, from config at call time, in JSON-safe
    form. Logged whole with each row."""
    import config
    return {
        "TRIM_EXIT_ENABLED": bool(config.TRIM_EXIT_ENABLED),
        "TRIM_EXIT_LIVE": bool(config.TRIM_EXIT_LIVE),
        "TRIM_EXIT_LIVE_CONTEXTS": sorted(str(c).strip().lower() for c in config.TRIM_EXIT_LIVE_CONTEXTS),
        "TRIM_EXIT_EXEMPT_TICKERS": sorted(str(t).strip().upper() for t in config.TRIM_EXIT_EXEMPT_TICKERS),
        "TRIM_EXIT_MIN_CONVICTION": int(config.TRIM_EXIT_MIN_CONVICTION),
        "TRIM_EXIT_EARNINGS_MAX_DAYS_BEFORE": int(config.TRIM_EXIT_EARNINGS_MAX_DAYS_BEFORE),
        "TRIM_EXIT_CONFIRM_MIN_GAP_HOURS": float(config.TRIM_EXIT_CONFIRM_MIN_GAP_HOURS),
        "TRIM_EXIT_CONFIRM_MAX_AGE_DAYS": int(config.TRIM_EXIT_CONFIRM_MAX_AGE_DAYS),
        "TRIM_EXIT_COOLDOWN_DAYS": int(config.TRIM_EXIT_COOLDOWN_DAYS),
        "TRIM_EXIT_MIN_HOLD_HOURS": int(config.TRIM_EXIT_MIN_HOLD_HOURS),
        "TRIM_EXIT_ONE_SHARE_TRIM": str(config.TRIM_EXIT_ONE_SHARE_TRIM).strip().lower(),
        "TRIM_EXIT_MIN_REMAINING_SHARES": int(config.TRIM_EXIT_MIN_REMAINING_SHARES),
        "TRIM_EXIT_FRACTION": float(config.TRIM_EXIT_FRACTION),
        # The model-flag TRIM's fraction, for legacy_qty.
        "TRIM_FRACTION": float(config.TRIM_FRACTION),
    }


def _num(v) -> float | None:
    """v as a finite float, else None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _ts(v) -> datetime | None:
    """A datetime or ISO string as an aware UTC datetime, else None. A naive
    value is read as UTC."""
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v)
        except ValueError:
            return None
    if not isinstance(v, datetime):
        return None
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.astimezone(timezone.utc)


def _iso(v) -> str | None:
    """A datetime or date as an ISO string (datetimes in UTC), else None."""
    if isinstance(v, datetime):
        return _ts(v).isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return v if isinstance(v, str) else None


def _as_date(v) -> date | None:
    """A date, datetime or ISO string as a calendar date, else None."""
    if isinstance(v, str):
        try:
            v = date.fromisoformat(v[:10])
        except ValueError:
            return None
    if isinstance(v, datetime):
        return v.date()
    return v if isinstance(v, date) else None


def trim_qty(owned, cfg: dict | None = None) -> int:
    """Whole shares a rule TRIM sells from `owned`: the fraction, at least
    one, never so many that fewer than TRIM_EXIT_MIN_REMAINING_SHARES stay.
    0 when no such sale exists (one share held with one to remain). cfg is
    the flag snapshot; read from config when not given."""
    cfg = cfg or _cfg()
    owned = _num(owned)
    if owned is None or owned <= 0:
        return 0
    qty = min(max(1, int(owned * cfg["TRIM_EXIT_FRACTION"])),
              int(owned - cfg["TRIM_EXIT_MIN_REMAINING_SHARES"]))
    return qty if qty >= 1 else 0


def _unknown_contexts(cfg: dict) -> list:
    """Names in TRIM_EXIT_LIVE_CONTEXTS that are not contexts, so never live
    (a misspelling such as "earnings-pre")."""
    return [c for c in cfg["TRIM_EXIT_LIVE_CONTEXTS"] if c not in CONTEXTS]


def _context(inputs: dict, cfg: dict) -> str:
    """earnings_pre, earnings_day or general, from the review event's
    earnings phase and the days to the print."""
    phase = inputs.get("earnings_phase")
    days = inputs.get("days_to_print")
    if (phase == "pre" and isinstance(days, int)
            and 0 <= days <= cfg["TRIM_EXIT_EARNINGS_MAX_DAYS_BEFORE"]):
        return "earnings_pre"
    if phase == "day":
        return "earnings_day"
    return "general"


def _confirmation(inputs: dict, cfg: dict, now: datetime | None) -> tuple[bool, str, dict | None]:
    """(confirmed, why, the verdict looked at). Anything unreadable or
    missing is unconfirmed."""
    reads = inputs.get("reads") or {}
    if reads.get("verdicts") != "ok" or now is None:
        return False, "verdicts_unreadable", None
    gap = max(CONFIRM_MIN_GAP_FLOOR_HOURS, cfg["TRIM_EXIT_CONFIRM_MIN_GAP_HOURS"])
    newest_allowed = now - timedelta(hours=gap)
    prior, prior_at = None, None
    for v in inputs.get("verdicts") or []:
        at = _ts((v or {}).get("timestamp"))
        if at is None or at > newest_allowed:
            continue
        if prior_at is None or at > prior_at:
            prior, prior_at = v, at
    if prior is None:
        return False, "no_earlier_verdict", None
    if prior_at < now - timedelta(days=cfg["TRIM_EXIT_CONFIRM_MAX_AGE_DAYS"]):
        return False, "too_old", prior
    if str(prior.get("action") or "").upper() not in SELL_ACTIONS:
        return False, "earlier_not_sell", prior
    opened_at = _ts(inputs.get("opened_at"))
    if reads.get("stop") != "ok" or opened_at is None:
        return False, "opened_at_unknown", prior
    if prior_at < opened_at:
        return False, "before_opened_at", prior
    return True, "confirmed", prior


def decide(inputs: dict, cfg: dict) -> dict:
    """The rule. Pure: no I/O, and the clock is inputs["evaluated_at"].
    inputs is what _gather returns (or a logged row's decision); cfg is the
    flag snapshot. A read that is not marked "ok" in inputs["reads"] fails
    closed. Returns the OUTPUT_KEYS."""
    reads = inputs.get("reads") or {}
    now = _ts(inputs.get("evaluated_at"))
    ticker = str(inputs.get("ticker") or "").upper()
    action = str(inputs.get("action") or "").upper()
    conviction = _num(inputs.get("conviction"))
    owned = _num(inputs.get("owned")) or 0.0
    observe = bool(inputs.get("observe"))
    live = bool(cfg["TRIM_EXIT_LIVE"])
    context = _context(inputs, cfg)
    is_sell = action in SELL_ACTIONS
    in_domain = is_sell and conviction is not None and conviction >= cfg["TRIM_EXIT_MIN_CONVICTION"]

    out = {"mode": "observe" if observe else "live" if live else "shadow",
           "context": context, "is_sell": is_sell, "in_domain": in_domain,
           "confirmed": False, "confirm_reason": None, "confirming_verdict": None,
           "action_if_confirmed": None, "action_if_unconfirmed": None,
           "would_action": None, "would_qty": None, "legacy_qty": None, "remaining_qty": None,
           "reason": None, "blocked_by": [], "owns_pass_flag": False, "routed": False}

    blocked = []
    if not live:
        blocked.append("shadow")
    if observe:
        blocked.append("observe")
    if ticker in cfg["TRIM_EXIT_EXEMPT_TICKERS"]:
        blocked.append("exempt")
    if context not in cfg["TRIM_EXIT_LIVE_CONTEXTS"]:
        blocked.append("context_not_live")

    if not is_sell:
        out.update(reason="not_sell_verdict", blocked_by=blocked)
        return out

    confirmed, why, prior = _confirmation(inputs, cfg, now)
    if context == "earnings_pre":
        if_confirmed, if_unconfirmed = action, "TRIM"
    else:
        if_confirmed, if_unconfirmed = action, None
    out.update(confirmed=confirmed, confirm_reason=why, confirming_verdict=prior,
               action_if_confirmed=if_confirmed, action_if_unconfirmed=if_unconfirmed,
               legacy_qty=(owned if action == "EXIT"
                           else min(owned, max(1, int(owned * cfg["TRIM_FRACTION"])))))

    would, reason = None, "below_min_conviction"
    if in_domain:
        would = if_confirmed if confirmed else if_unconfirmed
        if would is None:
            reason = "wait_confirm"
        elif context == "earnings_pre" and not confirmed:
            reason = "earnings_trim" if action == "TRIM" else "earnings_exit_unconfirmed"
        elif context == "earnings_pre" and action == "TRIM":
            reason = "earnings_trim"
        else:
            reason = "confirmed"
        if would == "TRIM" and owned > 0 and trim_qty(owned, cfg) == 0:
            # No partial sale exists. hold: nothing. exit: an EXIT request,
            # which sells only when confirmed.
            if cfg["TRIM_EXIT_ONE_SHARE_TRIM"] == "exit" and confirmed:
                would, reason = "EXIT", "trim_one_share_exit"
            else:
                would, reason = None, "trim_one_share"
    if would is not None and owned > 0:
        qty = owned if would == "EXIT" else float(trim_qty(owned, cfg))
        out.update(would_qty=qty, remaining_qty=round(owned - qty, 4))

    if owned <= 0:
        blocked.append("no_position")
    # A rule TRIM, and a one-share TRIM turned into an EXIT, wait out the
    # cooldown; an EXIT verdict does not.
    if (would == "TRIM" or reason == "trim_one_share_exit") and cfg["TRIM_EXIT_COOLDOWN_DAYS"] > 0 and (
            reads.get("last_sell") != "ok" or inputs.get("last_sell")):
        blocked.append("cooldown")
    if cfg["TRIM_EXIT_MIN_HOLD_HOURS"] > 0 and (
            reads.get("last_buy") != "ok" or inputs.get("last_buy")):
        blocked.append("min_hold")
    if reads.get("stop") != "ok":
        blocked.append("no_stop_record")
    elif not inputs.get("stop_record"):
        blocked.append("exit_no_stop_record" if action == "EXIT" else "no_stop_record")
    if reads.get("pending_sell") != "ok" or inputs.get("pending_sell") is not False:
        blocked.append("pending_sell")
    if reads.get("active_approval") != "ok" or inputs.get("active_approval") is not False:
        blocked.append("active_approval")
    if reads.get("market") != "ok" or inputs.get("market_open") is not True:
        blocked.append("market_closed")
    if inputs.get("verdict_signal_id") is None:
        blocked.append("verdict_not_logged")

    owns = in_domain and not any(b in STAND_DOWN for b in blocked)
    out.update(would_action=would, reason=reason, blocked_by=blocked, owns_pass_flag=owns,
               routed=bool(owns and would is not None and not any(b in NO_SALE for b in blocked)))
    return out


def _read(reads: dict, name: str, fn):
    """fn(), with reads[name] set to "ok" or "error:<class>". None on error."""
    try:
        value = fn()
        reads[name] = "ok"
        return value
    except Exception as e:
        reads[name] = f"error:{type(e).__name__}"
        return None


def _pending_sell(ticker: str, now: datetime) -> dict:
    """Whether one of the bot's own SELLs on the ticker is still open, from
    the three reads the stop reconcile uses. Raises if any read fails."""
    import db
    from agents.catalyst_time import ET
    today = now.astimezone(ET).date()
    sources = []
    if any(o.get("ticker") == ticker and o.get("side") == "SELL"
           and o.get("created_at") is not None
           and o["created_at"].astimezone(ET).date() == today
           for o in db.get_unconfirmed_orders()):
        sources.append("unconfirmed")
    if ticker in db.awaiting_fill_sell_tickers():
        sources.append("awaiting_fill")
    if ticker in db.trade_pending_sell_tickers():
        sources.append("trade_pending")
    return {"pending": bool(sources), "sources": sources}


def _utcnow() -> datetime:
    """The evaluation clock (tests patch it)."""
    return datetime.now(timezone.utc)


def _implied_move_fields(ticker: str, et_date: date) -> dict:
    """Record only: the ticker's latest stored options implied move, read as
    agents.crew._gap_risk_shadow reads it (no broker or network call).
    Raises when the state can't be read."""
    from agents.crew import _read_implied_move
    got: dict = {}
    _read_implied_move(ticker, et_date, got)
    return {k: got.get(k) for k in ("implied_move_last_pct", "implied_move_last_date",
                                    "implied_move_last_dte", *_EVENT_MOVE_KEYS)}


def _gap_vs_stop_event(inputs: dict, earnings_date: date | None) -> dict:
    """Record only: gap_vs_stop_event, the stored event-spanning move over
    the stop distance, and gap_vs_stop_basis, "event_spanning" when that
    move exists and the print under review (earnings_date) is on or before
    its expiry, else "nearest" (gap_vs_stop's move). Never raises: a failure
    gives basis "error"."""
    try:
        ev, dist = _num(inputs.get("implied_move_event_pct")), _num(inputs.get("stop_distance_pct"))
        expiry = _as_date(inputs.get("implied_move_event_expiry"))
        return {"gap_vs_stop_event": (round(ev / dist, 4) if ev is not None and ev > 0
                                      and dist is not None and dist > 0 else None),
                "gap_vs_stop_basis": ("event_spanning" if ev is not None and expiry is not None
                                      and earnings_date is not None and earnings_date <= expiry
                                      else "nearest")}
    except Exception as e:
        log.debug(f"gap_vs_stop_event failed: {e}")
        return {"gap_vs_stop_event": None, "gap_vs_stop_basis": "error"}


def _gather(result: dict, cfg: dict, now: datetime, *, path: str, event: dict | None,
            earnings_date, positions: dict | None, observe: bool) -> dict:
    """Everything decide() reads, JSON-safe, plus record-only fields decide()
    ignores (the implied moves, gap_vs_stop, gap_vs_stop_event and its
    basis, the earnings calendar row). The
    market clock, the stop record and the implied move are read for every
    evaluation; the other database reads only for a TRIM/EXIT verdict. Each
    read's outcome is recorded in "reads", and a failed one leaves its value
    None, which decide() fails closed on."""
    from agents.catalyst_time import today_et
    ticker = str(result.get("ticker") or "").upper()
    action = str(result.get("action") or "").upper()
    prov = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
    pos = (positions or {}).get(ticker) or {}
    owned = abs(_num(pos.get("qty")) or 0.0)
    price = _num(pos.get("current_price"))
    entry = _num(pos.get("entry_price"))
    event = event if isinstance(event, dict) else {}
    timing = event.get("earnings_timing") if isinstance(event.get("earnings_timing"), dict) else {}
    earnings_date = _as_date(earnings_date)
    et_date = today_et(now)
    days = (earnings_date - et_date).days if earnings_date is not None else None
    reads: dict = {}
    inputs = {
        "ticker": ticker, "path": path, "observe": bool(observe),
        "evaluated_at": now.isoformat(), "et_date": et_date.isoformat(),
        "market_session": None, "market_open": None,
        "action": action, "conviction": result.get("conviction"),
        # Record only. Not "model_pass_to_risk": that is the verdict row's
        # key, and readouts that walk every kind's decision would count it twice.
        "verdict_pass_to_risk": bool(result.get("pass_to_risk")),
        "verdict_signal_id": prov.get("signal_id"),
        "earnings_date": _iso(earnings_date), "earnings_phase": event.get("earnings_phase"),
        "days_to_print": days, "earnings_time_of_day": timing.get("time_of_day"),
        "earnings_calendar_date": None, "earnings_calendar_confirmed": None,
        "owned": owned, "price": price, "entry_price": entry,
        # crew._get_positions falls back to the entry price when the broker
        # gives no current price.
        "price_equals_entry": price is not None and price == entry,
        "stop_record": None, "stop_price": None, "stop_distance_pct": None, "opened_at": None,
        "implied_move_last_pct": None, "implied_move_last_date": None,
        "implied_move_last_dte": None, "gap_vs_stop": None,
        **{k: None for k in _EVENT_MOVE_KEYS}, "gap_vs_stop_event": None, "gap_vs_stop_basis": None,
        "verdicts": None, "last_buy": None, "last_sell": None,
        "pending_sell": None, "pending_sell_sources": None, "active_approval": None,
        "reads": reads,
    }

    def _market():
        import market_data
        return market_data.is_market_hours(now), market_data.market_session(now)
    market = _read(reads, "market", _market)
    if market is not None:
        inputs["market_open"], inputs["market_session"] = bool(market[0]), market[1]

    import db
    stop = _read(reads, "stop", lambda: db.get_position_stop(ticker, strict=True))
    if reads["stop"] == "ok":
        inputs["stop_record"] = stop is not None
    if isinstance(stop, dict):
        stop_price = _num(stop.get("stop_price"))
        inputs["stop_price"] = stop_price
        inputs["opened_at"] = _iso(stop.get("opened_at"))
        if price and price > 0 and stop_price is not None:
            inputs["stop_distance_pct"] = round((price - stop_price) / price * 100, 4)

    # Record only (decide() never reads these).
    move = _read(reads, "implied_move", lambda: _implied_move_fields(ticker, et_date))
    if move is not None:
        inputs.update(move)
    last_pct, dist = _num(inputs["implied_move_last_pct"]), inputs["stop_distance_pct"]
    if last_pct is not None and last_pct > 0 and dist is not None and dist > 0:
        inputs["gap_vs_stop"] = round(last_pct / dist, 4)
    inputs.update(_gap_vs_stop_event(inputs, earnings_date))
    inputs.update(_held_trend_fields(ticker, now, price, entry, reads))
    inputs["held_exit_candidates"] = _held_exit_fields(ticker, now, entry, inputs.get("opened_at"),
                                                       reads)

    if action not in SELL_ACTIONS:
        return inputs

    rows = _read(reads, "verdicts", lambda: db.get_recent_verdicts(
        ticker, hours=cfg["TRIM_EXIT_CONFIRM_MAX_AGE_DAYS"] * 24, limit=50, strict=True))
    if rows is not None:
        inputs["verdicts"] = [{"timestamp": _iso(r.get("timestamp")), "action": r.get("signal"),
                               "conviction": r.get("conviction")} for r in rows]

    if cfg["TRIM_EXIT_MIN_HOLD_HOURS"] > 0:
        buy = _read(reads, "last_buy", lambda: db.get_last_executed_approval(
            ticker, cfg["TRIM_EXIT_MIN_HOLD_HOURS"], strict=True))
        if buy:
            inputs["last_buy"] = {"conviction": buy.get("conviction"),
                                  "decided_at": _iso(buy.get("decided_at"))}
    else:
        reads["last_buy"] = "off"

    if cfg["TRIM_EXIT_COOLDOWN_DAYS"] > 0:
        inputs["last_sell"] = _read(reads, "last_sell", lambda: db.sell_trades_since(
            ticker, now - timedelta(days=cfg["TRIM_EXIT_COOLDOWN_DAYS"])))
    else:
        reads["last_sell"] = "off"

    pending = _read(reads, "pending_sell", lambda: _pending_sell(ticker, now))
    if pending is not None:
        inputs["pending_sell"] = pending["pending"]
        inputs["pending_sell_sources"] = pending["sources"]

    approval = _read(reads, "active_approval", lambda: db.has_active_approval(ticker, strict=True))
    if approval is not None:
        inputs["active_approval"] = bool(approval)

    if earnings_date is not None:
        # Record only: whether the calendar marks the date confirmed.
        row = _read(reads, "earnings_calendar", lambda: db.get_next_earnings_row(ticker))
        if row:
            inputs["earnings_calendar_date"] = _iso(row[0])
            inputs["earnings_calendar_confirmed"] = bool(row[1])
    return inputs


def _held_trend_fields(ticker: str, now: datetime, price, entry, reads: dict) -> dict:
    """Record only (decide() never reads these): the held name's price
    structure on cached daily bars, read as the live entry gate reads it
    (agents.crew._entry_trend_check, one price_history read, no fetch), so a
    later held-position exit rule mirroring that gate can be scored on rows
    written before it exists. held_below_both_sma is True only on an ok read
    with the last close below both the 50- and 200-day averages, None when
    either is unknown. unrealized_pct is price over the broker entry price,
    None without both or when the price equals the entry (crew._get_positions
    falls back to the entry price when the broker gives none).
    reads["held_trend"] records the read's outcome. Never raises: a
    record-only field must not withhold a sale through apply's except."""
    out = {"held_trend": None, "held_below_both_sma": None, "unrealized_pct": None}
    try:
        if (isinstance(price, (int, float)) and isinstance(entry, (int, float))
                and entry > 0 and price > 0 and price != entry):
            out["unrealized_pct"] = round((price / entry - 1) * 100, 4)
        from agents.crew import _entry_trend_check
        trend = _entry_trend_check(ticker, now=now)
        if not isinstance(trend, dict):
            reads["held_trend"] = f"error:{type(trend).__name__}"
            return out
        read = str(trend.get("read"))
        reads["held_trend"] = read.replace("error: ", "error:")
        out["held_trend"] = trend
        b50, b200 = trend.get("below_sma50"), trend.get("below_sma200")
        if read == "ok" and isinstance(b50, bool) and isinstance(b200, bool):
            out["held_below_both_sma"] = b50 and b200
    except Exception as e:
        reads["held_trend"] = f"error:{type(e).__name__}"
    return out


_HELD_EXIT_BAR_ROWS = 260


def _held_exit_fields(ticker: str, now: datetime, entry, opened_at, reads: dict) -> dict | None:
    """Record only (decide() never reads it): agents.held_exit.candidates on
    the cached daily bars (one db.get_price_history read, today's unfinished
    bar dropped, non-finite bars carried forward or dropped as the entry
    trend check does), the broker entry price, the stop record's opened_at
    as the entry's ET date (kept across ADDs, so the first fill), and the
    active trade plan (one db.get_active_plan read). reads["held_exit"] and
    reads["held_exit_plan"] record each read's outcome. None when
    HELD_EXIT_SHADOW_ENABLED is off. Never raises: a record-only field must
    not withhold a sale through apply's except."""
    try:
        import config
        if not getattr(config, "HELD_EXIT_SHADOW_ENABLED", False):
            reads["held_exit"] = "off"
            return None
        import db
        from agents import held_exit
        from agents.catalyst_time import ET
        from agents.market_regime import completed_bars
        from agents.trade_ladder import _sanitize_bars
        try:
            raw = list(db.get_price_history(ticker, days=_HELD_EXIT_BAR_ROWS, strict=True) or [])
            bars = _sanitize_bars(completed_bars(raw, now=now))
            bars = [b for b in bars if all(isinstance(b.get(k), (int, float)) and math.isfinite(b[k])
                                           for k in ("high", "low", "close"))]
            reads["held_exit"] = "ok" if bars else "no_cached_bars"
        except Exception as e:
            bars, reads["held_exit"] = [], f"error:{type(e).__name__}"
        try:
            plan, plan_read = db.get_active_plan(ticker, strict=True), "ok"
        except Exception as e:
            plan, plan_read = None, f"error:{type(e).__name__}"
        reads["held_exit_plan"] = plan_read
        opened = _ts(opened_at)
        rec = held_exit.candidates(
            bars, entry_price=entry, entry_date=opened.astimezone(ET).date() if opened else None,
            plan=plan, plan_read=plan_read,
            time_stop_sessions=int(config.HELD_EXIT_TIME_STOP_SESSIONS),
            drawdown_atr=float(config.HELD_EXIT_DRAWDOWN_ATR))
        rec["read"] = reads["held_exit"]
        return rec
    except Exception as e:
        reads["held_exit"] = f"error:{type(e).__name__}"
        return {"version": None, "read": reads["held_exit"]}


def _log_row(inputs: dict, outputs: dict, cfg: dict) -> int | None:
    """Write the TRIM_EXIT_RULE signals row. Returns its id, or None when the
    full row was not written."""
    import db
    import config
    from agents.analyst import shadow_versions
    ticker = inputs["ticker"]
    would = outputs.get("would_action")
    reasoning = (f"trim/exit rule ({outputs.get('mode')}, {outputs.get('context')}): verdict "
                 f"{inputs.get('action') or 'none'} c{inputs.get('conviction')}, would "
                 f"{would or 'do nothing'} ({outputs.get('reason')})"
                 + (f"; blocked by {', '.join(outputs['blocked_by'])}" if outputs.get("blocked_by") else "")
                 + ("; routed" if outputs.get("routed") else "; not routed by the rule"))
    decision = {
        "rule_version": RULE_VERSION,
        **inputs, **outputs,
        "options_shadow_key": f"{config.TRADING_ENV}:{ticker}:{inputs.get('et_date')}",
        # Record only: TRIM_EXIT_LIVE_CONTEXTS names that match no context.
        "unknown_live_contexts": _unknown_contexts(cfg),
        "flags": dict(cfg),
    }
    provenance = {"verdict_signal_id": inputs.get("verdict_signal_id"),
                  "path": inputs.get("path"), **shadow_versions()}
    return db.log_signal(ticker, "TRIM_EXIT_RULE", 0.0, reasoning[:500],
                         provenance=provenance, decision=decision)


def _stand_down_known(result: dict, event, earnings_date, observe: bool) -> bool:
    """After a failure in apply(): True when the rule is known not to own
    this verdict's routing (shadow, an observe site, not an in-domain sell,
    an exempt ticker, or a context that is not live). False when it owns it
    or that cannot be told. Never raises."""
    try:
        from agents.catalyst_time import today_et
        cfg = _cfg()
        if observe or not cfg["TRIM_EXIT_LIVE"]:
            return True
        action = str(result.get("action") or "").upper()
        conviction = _num(result.get("conviction"))
        if action not in SELL_ACTIONS:
            return True
        if conviction is not None and conviction < cfg["TRIM_EXIT_MIN_CONVICTION"]:
            return True
        if str(result.get("ticker") or "").upper() in cfg["TRIM_EXIT_EXEMPT_TICKERS"]:
            return True
        earnings_date = _as_date(earnings_date)
        days = (earnings_date - today_et(_utcnow())).days if earnings_date is not None else None
        phase = event.get("earnings_phase") if isinstance(event, dict) else None
        context = _context({"earnings_phase": phase, "days_to_print": days}, cfg)
        return context not in cfg["TRIM_EXIT_LIVE_CONTEXTS"]
    except Exception:
        return False


def apply(result: dict, *, path: str, event: dict | None = None, earnings_date=None,
          positions: dict | None = None, observe: bool = False) -> dict:
    """Evaluate the rule on an analyst result and return the dict to route on.

    TRIM_EXIT_ENABLED off: returns `result` itself, at once; no read, no row.
    Otherwise one TRIM_EXIT_RULE row is written (at an observe site only for a
    TRIM/EXIT verdict) and:
      - when the rule owns the verdict's routing (TRIM_EXIT_LIVE, an in-domain
        sell, not exempt, a live context, not an observe site, not an EXIT
        on a position with no stop record), a shallow copy
        is returned: routed with the rule's action, pass_to_risk True and a
        "trim_exit" tag, or pass_to_risk False when it is blocked or waiting
        for confirmation;
      - in every other case `result` itself is returned, untouched.
    `result` is never mutated. Never raises: on a failure the original comes
    back unless the rule owns the routing or that cannot be told, where a
    copy with pass_to_risk False comes back (no sale)."""
    try:
        import config
        if not getattr(config, "TRIM_EXIT_ENABLED", False):
            return result
    except Exception:
        return result
    try:
        action = str(result.get("action") or "").upper()
        if observe and action not in SELL_ACTIONS:
            return result
        cfg = _cfg()
        if _unknown_contexts(cfg):
            log.warning(f"trim/exit rule: TRIM_EXIT_LIVE_CONTEXTS names no context: "
                        f"{', '.join(_unknown_contexts(cfg))} (known: {', '.join(CONTEXTS)})")
        now = _utcnow()
        inputs = _gather(result, cfg, now, path=path, event=event, earnings_date=earnings_date,
                         positions=positions, observe=observe)
        outputs = decide(inputs, cfg)
        row_id = None
        try:
            row_id = _log_row(inputs, outputs, cfg)
        except Exception as e:
            log.warning(f"[{inputs['ticker']}] trim/exit rule row not written: {type(e).__name__}: {e}")
        if not outputs["owns_pass_flag"]:
            return result
        routed = dict(result)
        if outputs["routed"] and row_id is not None:
            routed["action"] = outputs["would_action"]
            routed["pass_to_risk"] = True
            routed["trim_exit"] = {
                "rule_version": RULE_VERSION, "row_id": row_id, "context": outputs["context"],
                "verdict_action": action, "action": outputs["would_action"],
                "confirmed": outputs["confirmed"], "reason": outputs["reason"],
                "would_qty": outputs["would_qty"],
            }
            log.info(f"[{inputs['ticker']}] trim/exit rule routes {outputs['would_action']} "
                     f"({outputs['context']}, {outputs['reason']})")
        else:
            blocked = list(outputs["blocked_by"])
            if outputs["routed"]:
                blocked.append("row_not_logged")
            routed["pass_to_risk"] = False
            log.info(f"[{inputs['ticker']}] trim/exit rule: no sale ({outputs['reason']}"
                     + (f"; blocked by {', '.join(blocked)}" if blocked else "") + ")")
        return routed
    except Exception as e:
        try:
            ticker = result.get("ticker") if isinstance(result, dict) else None
            log.error(f"[{ticker}] trim/exit rule failed: {type(e).__name__}: {e}", exc_info=True)
            if _stand_down_known(result, event, earnings_date, observe):
                return result
            closed = dict(result)
            closed["pass_to_risk"] = False
            return closed
        except Exception:
            return result


def execution_block(ticker: str, action: str, shares, owned, tag) -> str | None:
    """Recheck for a sale the rule routed, just before execution touches the
    stop: a reason not to send it, or None. `shares` is the quantity about to
    be sold and `owned` the position, both as execution holds them; `tag` is
    the sale's trim_exit tag (risk_json["trim_exit"]). A tag without a live
    context fails the check."""
    cfg = _cfg()
    if not (cfg["TRIM_EXIT_ENABLED"] and cfg["TRIM_EXIT_LIVE"]):
        return "switched off since routing"
    if str(ticker or "").upper() in cfg["TRIM_EXIT_EXEMPT_TICKERS"]:
        return "ticker exempt since routing"
    tag = tag if isinstance(tag, dict) else {}
    if tag.get("context") not in cfg["TRIM_EXIT_LIVE_CONTEXTS"]:
        return "context not live"
    action = str(action or "").upper()
    if action == "TRIM":
        shares, owned = _num(shares), _num(owned)
        if shares is None or owned is None:
            return "quantity unreadable"
        if round(owned - shares, 4) < cfg["TRIM_EXIT_MIN_REMAINING_SHARES"]:
            return "trim would leave too few shares"
    elif action == "EXIT" and tag.get("reason") == "trim_one_share_exit":
        # The one-share EXIT stands only while TRIM_EXIT_ONE_SHARE_TRIM is
        # still "exit" and no partial sale exists: shares bought since the
        # review would all go with it.
        if cfg["TRIM_EXIT_ONE_SHARE_TRIM"] != "exit":
            return "one-share exit no longer enabled"
        owned = _num(owned)
        if owned is None:
            return "quantity unreadable"
        if trim_qty(owned, cfg) > 0:
            return "one-share exit, but a partial sale exists now"
    return None
