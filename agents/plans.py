"""Trade plans, shadow only (roadmap 1; Luke 10-27, "plan, then follow it").

The analyst already writes most of a plan in every verdict (thesis, catalyst
and date, bull target, downside scenario, primary risk), but nothing kept it,
so each verdict re-decided from scratch. This keeps one active plan per
ticker in trade_plans and checks every later verdict against it:

  NEW_PLAN    an OPEN/ADD with no active plan wrote one
  SEEDED      a held position with no plan got one from its stop record
  FOLLOW      the verdict agrees with what the plan says now
  DIVERGE     the verdict disagrees (e.g. TRIM while the plan says hold)
  REVISE      an OPEN/ADD moved the target or invalidation by >= 10%, or the
              catalyst date: the plan is superseded by a new revision
  INVALIDATE  price is at or below the invalidation level (plan closed)
  NO_PLAN     no plan and nothing to build one from

The result goes in signals.provenance["plan"]. Nothing here reads the plan
back into the prompt or changes an order: plan-in-brief is a later,
default-off step.
"""
import logging
import re
from datetime import date, timedelta

log = logging.getLogger(__name__)

# How long a plan lives without a dated catalyst, by bucket (days).
_HORIZON_DAYS = {"Compounder": 180, "Accelerator": 90, "Moonshot": 60}
_CATALYST_GRACE_DAYS = 5      # a plan keyed to a catalyst runs this long past it
_REVISE_LEVEL_PCT = 0.10      # target/invalidation move that counts as a revision
_COALESCE_HOURS = 4           # FOLLOWs this close together add no decision

_PRICE_RE = re.compile(r"Current Price: \$([\d,]+(?:\.\d+)?)")
# A per-share price, not "$2B" or "$40 million" (professor M94).
_DOLLAR_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)(?![\d,]*\d)(?!\.\d)(?!\s*(?:[BMKT]\b|bn|mn|billion|million|thousand|trillion))", re.I)

_BUY, _HOLD, _SELL = "buy", "hold", "sell"
_DIRECTION = {"OPEN": _BUY, "ADD": _BUY, "HOLD": _HOLD, "WATCH": _HOLD, "PASS": _HOLD,
              "TRIM": _SELL, "EXIT": _SELL}


def brief_price(brief: str) -> float | None:
    m = _PRICE_RE.search(brief or "")
    return float(m.group(1).replace(",", "")) if m else None


def _dollars(text) -> float | None:
    if isinstance(text, (int, float)):
        return float(text) if text > 0 else None
    m = _DOLLAR_RE.search(str(text or ""))
    try:
        v = float(m.group(1).replace(",", "")) if m else None
        return v if v else None
    except ValueError:
        return None


def _catalyst_date(text) -> date | None:
    """YYYY-MM-DD, or YYYY-MM read as that month's last day."""
    s = str(text or "")
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    try:
        if m:
            return date(int(m[1]), int(m[2]), int(m[3]))
        m = re.search(r"(\d{4})-(\d{2})(?!\d)", s)
        if m:
            nxt = date(int(m[1]) + (m[2] == "12"), int(m[2]) % 12 + 1, 1)
            return nxt - timedelta(days=1)
    except ValueError:
        pass
    return None


def plan_from_verdict(ticker: str, parsed: dict, conviction: int, price: float,
                      today: date) -> dict:
    """A plan from the analyst's own fields. Invalidation falls back to the
    stated downside percent when downside_scenario has no dollar figure."""
    val = parsed.get("valuation") or {}
    cat = parsed.get("catalyst") or {}
    gap = parsed.get("expectation_gap") or {}
    target = _dollars(val.get("bull_case_target"))
    invalid = _dollars(val.get("downside_scenario"))
    if invalid is None and isinstance(parsed.get("expected_downside_pct"), (int, float)):
        invalid = round(price * (1 + parsed["expected_downside_pct"] / 100), 2)
    if invalid is not None and invalid >= price:
        invalid = None          # a "downside" above the price isn't an invalidation level
    if target is not None and target <= price:
        target = None
    cdate = _catalyst_date(cat.get("date"))
    bucket = parsed.get("thesis_bucket")
    if cdate and cdate >= today:
        horizon = cdate + timedelta(days=_CATALYST_GRACE_DAYS)
    else:
        horizon = today + timedelta(days=_HORIZON_DAYS.get(bucket, 90))
    return {
        "ticker": ticker, "source": "analyst", "bucket": bucket, "conviction": conviction,
        "entry_price": price, "invalidation_price": invalid, "target_price": target,
        "catalyst_event": (cat.get("event") or "")[:300] or None, "catalyst_date": cdate,
        "horizon_end": horizon, "thesis": (parsed.get("thesis_summary") or "")[:500] or None,
        "primary_risk": (gap.get("primary_risk") or "")[:300] or None, "revision": 0,
    }


def plan_says(plan: dict, price: float | None, today: date, held: bool) -> tuple[str, str]:
    """What the plan calls for now: (direction, why)."""
    if price is not None and plan.get("invalidation_price") and price <= plan["invalidation_price"]:
        return _SELL, "price at or below invalidation"
    if price is not None and plan.get("target_price") and price >= plan["target_price"]:
        return _SELL, "target reached"
    if plan.get("horizon_end") and today > plan["horizon_end"]:
        return _HOLD, "plan expired: review"
    if held:
        return _HOLD, "within plan"
    entry = plan.get("entry_price")
    if price is not None and entry and price <= entry * 1.05:
        return _BUY, "within 5% of the plan entry"
    return _HOLD, "above the plan entry"


def _moved(old, new) -> bool:
    return bool(old and new and abs(new - old) / old >= _REVISE_LEVEL_PCT)


def _month_end(d: date) -> bool:
    return (d + timedelta(days=1)).day == 1


def _revise_triggers(plan: dict, new: dict) -> dict:
    """Log-only: which REVISE conditions fired ("target", "invalidation",
    "catalyst_date"), and catalyst_month_only_pair, true when the catalyst
    date is the only trigger and the two dates share a month with one of
    them on the month's last day: a cheap flag for a month-only "YYYY-MM"
    (read as the last day by _catalyst_date) against an exact date. It
    decides nothing; the REVISE rule in classify is unchanged."""
    fired = []
    if _moved(plan.get("target_price"), new.get("target_price")):
        fired.append("target")
    if _moved(plan.get("invalidation_price"), new.get("invalidation_price")):
        fired.append("invalidation")
    a, b = plan.get("catalyst_date"), new.get("catalyst_date")
    if a and b and a != b:
        fired.append("catalyst_date")
    pair = bool(fired == ["catalyst_date"] and (a.year, a.month) == (b.year, b.month)
                and (_month_end(a) or _month_end(b)))
    return {"revise_triggers": fired, "catalyst_month_only_pair": pair}


def classify(plan: dict, action: str, new: dict | None, price: float | None,
             today: date, held: bool) -> dict:
    """Relation of one verdict to an active plan. `new` is the plan this
    verdict would write (OPEN/ADD only), used to detect a revision."""
    says, why = plan_says(plan, price, today, held)
    out = {"plan_id": plan["id"], "plan_says": says, "plan_why": why,
           "verdict_dir": _DIRECTION.get(action, _HOLD)}
    if why == "price at or below invalidation":
        out["relation"] = "INVALIDATE"
        # Log-only (audit 2026-11-08 21:30 #3g): a buy below the plan's own
        # invalidation is the contradiction a plan should record.
        out["action"] = action
        out["buy_below_invalidation"] = action in ("OPEN", "ADD")
        return out
    if new and action in ("OPEN", "ADD") and (
            _moved(plan.get("target_price"), new.get("target_price"))
            or _moved(plan.get("invalidation_price"), new.get("invalidation_price"))
            or (plan.get("catalyst_date") and new.get("catalyst_date")
                and plan["catalyst_date"] != new["catalyst_date"])):
        out["relation"] = "REVISE"
        try:   # log-only: a surprise here must not cost the observation
            out.update(_revise_triggers(plan, new))
        except Exception as e:
            log.warning(f"plan {plan.get('id')} revise_triggers failed: {e}")
        return out
    agree = (out["verdict_dir"] == says
             or (says == _BUY and out["verdict_dir"] == _HOLD)   # waiting is allowed
             or (says == _HOLD and out["verdict_dir"] == _BUY and held))  # adding within plan
    out["relation"] = "FOLLOW" if agree else "DIVERGE"
    return out


def _would_revise(plan: dict, parsed: dict, conviction: int, price: float, today: date,
                  ticker: str) -> dict:
    """Log-only (audit A1031-0053-1a): REVISE needs an OPEN/ADD, and held
    names get almost only HOLDs, so their plans never revise. This says
    whether this verdict's own levels moved enough that an OPEN/ADD would
    have revised the plan. A held_seed plan's invalidation is the stop, not
    a model level, so only its target and catalyst date are compared."""
    cand = plan_from_verdict(ticker, parsed, conviction, price, today)
    why = []
    if _moved(plan.get("target_price"), cand.get("target_price")):
        why.append("target")
    if plan.get("source") != "held_seed" and _moved(plan.get("invalidation_price"),
                                                    cand.get("invalidation_price")):
        why.append("invalidation")
    if plan.get("catalyst_date") and cand.get("catalyst_date") \
            and plan["catalyst_date"] != cand["catalyst_date"]:
        why.append("catalyst_date")
    return {"would_revise": bool(why), "would_revise_why": why}


def _with_levels(plan: dict, parsed: dict, price: float, ticker: str, today: date) -> dict:
    """The plan to insert, with compile_plan's named levels (migration 82),
    each level's source and the ladder's computed_at (migration 84) added:
    record only, nothing reads them for trading. On any failure the
    plan is returned without them."""
    try:
        import db
        ladder = db.get_trade_ladder(ticker)
        lv = compile_plan(parsed, price, ladder, None, today)
        return {**plan, "structural_stop": lv.get("structural_stop"), "t1": lv.get("t1"),
                "t2": lv.get("t2"), "add_trigger": lv.get("add_trigger"),
                "levels_source": lv.get("source"), "level_sources": lv.get("level_sources"),
                "ladder_computed_at": (ladder or {}).get("computed_at")}
    except Exception as e:
        log.warning(f"[{ticker}] plan levels not compiled, plan stored without them: {e}")
        return plan


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) < 1e6 else None


def _level_dispersion(ticker: str, parsed: dict) -> dict:
    """Log-only (audit 2026-11-08 19:19 #2): how far the model's own levels
    move across this ticker's verdicts over the last 24 h, this verdict
    included, in percent of price (no prices recorded). target_noise_frac
    is the relative bull-target move that the typical gap between two draws
    of expected_return_pct implies (target = price x (1 + r/100), so a gap
    of D pp moves it by about D / (100 + median r); D is the median absolute
    pairwise difference, not the range); revise_within_noise says whether
    that reaches the REVISE bar (_REVISE_LEVEL_PCT), i.e. whether a REVISE
    on this ticker-day could be sampling noise. Nothing reads it to decide.
    Never raises."""
    import statistics
    out = {"window_h": 24, "n": 0, "status": "ok"}
    try:
        import db
        rows = db.get_recent_verdict_levels(ticker, hours=24)
    except Exception as e:
        rows = []
        out["status"] = f"read_error:{type(e).__name__}"
    cur = {"expected_return_pct": parsed.get("expected_return_pct"),
           "expected_downside_pct": parsed.get("expected_downside_pct"),
           "catalyst_date": (parsed.get("catalyst") or {}).get("date")}
    rows = [cur, *rows]
    rets = [x for x in (_num(r.get("expected_return_pct")) for r in rows) if x is not None]
    downs = [x for x in (_num(r.get("expected_downside_pct")) for r in rows) if x is not None]
    dates = {str(r.get("catalyst_date")).strip() for r in rows if r.get("catalyst_date")}
    out.update(n=len(rows), n_return=len(rets), catalyst_dates_distinct=len(dates))
    if len(rets) >= 2:
        med = statistics.median(rets)
        rng = max(rets) - min(rets)
        # The noise measure is the median absolute difference between two
        # draws: a REVISE compares one verdict with one plan, and the full
        # range grows with the number of draws (audit 2026-11-08 21:30 #4).
        pair = statistics.median(abs(a - b) for i, a in enumerate(rets) for b in rets[i + 1:])
        out.update(return_min=round(min(rets), 2), return_max=round(max(rets), 2),
                   return_range_pp=round(rng, 2), return_median=round(med, 2),
                   return_pairwise_median_pp=round(pair, 2))
        noise = pair / (100 + med) if (100 + med) > 0 else None
        out["target_noise_frac"] = round(noise, 4) if noise is not None else None
        out["revise_within_noise"] = (noise >= _REVISE_LEVEL_PCT) if noise is not None else None
    else:
        out.update(target_noise_frac=None, revise_within_noise=None)
    if len(downs) >= 2:
        out["downside_range_pp"] = round(max(downs) - min(downs), 2)
    return out


def observe(ticker: str, parsed: dict, action: str, conviction: int, brief: str,
            now=None) -> dict:
    """_observe plus level_dispersion on every result (log-only). Never raises."""
    out = _observe(ticker, parsed, action, conviction, brief, now=now)
    try:
        out["level_dispersion"] = _level_dispersion(ticker, parsed or {})
    except Exception as e:
        out["level_dispersion"] = {"status": f"error:{type(e).__name__}"}
    return out


def _observe(ticker: str, parsed: dict, action: str, conviction: int, brief: str,
             now=None) -> dict:
    """Record one verdict against the ticker's plan. Never raises.

    now sets the trading date and bounds only the exit_sells read. The stop
    record, broker position, plan and closed-outcome reads are current
    state, so a call with a past now is not a point-in-time replay
    (professor m581; d4d8703's message says more than this)."""
    try:
        import config
        import db
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        now = now or datetime.now(timezone.utc)
        et = ZoneInfo("America/New_York")
        today = now.astimezone(et).date()   # trading dates are ET, not UTC
        price = brief_price(brief)
        # "held" is still "has a stop row": what FOLLOW/DIVERGE and
        # plan_live mean is a metric choice (Luke's). The broker's view is
        # recorded beside it, log-only, so a broker position with no stop row
        # (or a lingering row) shows up (audit A1031-0453-2).
        # Strict: a failed read is an ERROR row, not "not held", which would
        # close a held_seed plan as "position closed" (SEF 139g).
        stop = db.get_position_stop(ticker, strict=True)
        held = stop is not None
        try:
            import tasks
            broker, broker_reason, _ = tasks._broker_held(ticker)
        except Exception:
            broker, broker_reason = None, "error"
        held_info = {"held_broker": broker, "held_broker_reason": broker_reason,
                     "held_mismatch": broker is not None and broker != held}
        plan = db.get_active_plan(ticker)
        new = (plan_from_verdict(ticker, parsed, conviction, price, today)
               if price and action in ("OPEN", "ADD") else None)

        # Close a plan that ran past its horizon, or a plan whose position
        # is gone, before judging this verdict (professor M93).
        close_check = {"would_close_on_exit": None, "close_outcome_id": None,
                       "close_check_status": "not_checked"}
        if plan is not None:
            reason = None
            if plan.get("horizon_end") and today > plan["horizon_end"]:
                reason = "expired"
            elif plan.get("source") == "held_seed" and not held:
                reason = "position closed"
            elif not held and plan.get("created_at"):
                # An OPEN/ADD plan whose position has since closed (stop,
                # EXIT, profit-take). A plan that never filled has no closed
                # trade after it, so it stays open. On a lookup error the
                # plan is kept. The check runs with PLAN_CLOSE_ON_EXIT off
                # too and is recorded (would_close_on_exit), so the switch's
                # effect can be measured without turning it on (professor
                # M315); only the switch closes the plan.
                try:
                    # A revision postdates its trade's entry, so the entry
                    # is bounded by the chain's first plan (M282).
                    entered = (db.plan_root_created_at(plan["id"]) if plan.get("revision")
                               else None) or plan["created_at"]
                    closed = db.outcome_closed_since(ticker, plan["created_at"], entered_since=entered)
                    close_check = {"would_close_on_exit": bool(closed),
                                   "close_outcome_id": (closed or {}).get("id"),
                                   "close_check_status": "ok"}
                    if closed and config.PLAN_CLOSE_ON_EXIT:
                        reason = "position closed"
                except Exception as e:
                    close_check["close_check_status"] = "error"
                    log.warning(f"[{ticker}] plan {plan['id']} closed-trade lookup failed: {e}")
            if reason:
                db.touch_plan(plan["id"], "CLOSED", close_reason=reason)
                if reason == "expired":
                    # The plan's time stop: this verdict is the decision at
                    # its horizon, though a held name re-seeds at once
                    # (audit A1031-0053-1b). Recorded only.
                    held_info.update(decision_at_horizon=True, expired_plan_id=plan["id"])
                plan = None

        if plan is None:
            if new:
                return {"relation": "NEW_PLAN",
                        "plan_id": db.insert_plan(_with_levels(new, parsed, price, ticker, today)),
                        "price": price,
                        "held": held, "plan_live": held, **held_info}
            if stop and price:
                seed = plan_from_verdict(ticker, parsed, conviction, price, today)
                seed.update(source="held_seed", entry_price=stop["entry_price"],
                            invalidation_price=stop["stop_price"],
                            bucket=stop.get("thesis_bucket") or seed["bucket"])
                return {"relation": "SEEDED", "plan_id": db.insert_plan(seed), "price": price,
                        "held": held, "plan_live": True, **held_info}
            return {"relation": "NO_PLAN", "price": price, "held": held, **held_info}

        out = classify(plan, action, new, price, today, held)
        out["price"] = price
        # A plan written by an OPEN that never filled (sized to 0 shares) is
        # not a live plan: WATCH/HOLD on it reads as FOLLOW (audit A0100-4).
        # Count FOLLOW/DIVERGE on plan_live rows only.
        out.update(held=held, plan_source=plan.get("source"), plan_live=held, **held_info,
                   **close_check)
        out["plan_age_days"] = ((today - plan["created_at"].astimezone(et).date()).days
                                if plan.get("created_at") else None)
        # The plan's invalidation beside the stop record (audit 11-06 03:34
        # #1): the plan says "wrong below X", the recorded stop is Y. Under a
        # native trail the broker ratchets its own trigger and the record can
        # lag (execution._whole_position_stop), so trailing_active is kept
        # beside it (professor 59q). Log-only; which level governs is Luke's.
        out["invalidation_vs_stop"] = None
        if held:
            try:
                inv, sp = plan.get("invalidation_price"), (stop or {}).get("stop_price")
                inv = float(inv) if inv is not None else None
                sp = float(sp) if sp is not None else None
                out["invalidation_vs_stop"] = {
                    "invalidation": inv, "stop": sp,
                    "gap_frac": round(inv / sp - 1, 4) if inv and sp else None,
                    "stop_below_invalidation": (sp < inv) if inv and sp else None,
                    "trailing_active": (stop or {}).get("trailing_active")}
            except Exception:
                pass
        # With PLAN_CLOSE_ON_EXIT off (the default), an analyst plan stays
        # active after its position is sold, so a re-entry is judged against
        # the plan the exit ended (audit 11-05 21:30 #1). Record-only: whether
        # such a plan should close is a metric choice (packet G75).
        # exit_sells names the sales (count, last id, write time, reason) so
        # a TRIM can be told from the sale that closed the position; the
        # status says whether it was checked (professor m580).
        out.update(exited_since_plan=None, exit_sells=None, exited_since_plan_status="not_checked")
        if not held and plan.get("created_at"):
            try:
                # Unknown on a DB error, not "not sold" (professor m578).
                sells = db.sell_trades_since(ticker, plan["created_at"], now)
                out.update(exited_since_plan=sells is not None, exit_sells=sells,
                           exited_since_plan_status="ok")
            except Exception as e:
                out["exited_since_plan_status"] = "error"
                log.debug(f"[{ticker}] exited_since_plan failed: {e}")
        last = plan.get("last_seen_at") or plan.get("created_at")
        out["would_coalesce"] = bool(
            out["relation"] == "FOLLOW" and last
            and (now - last).total_seconds() < _COALESCE_HOURS * 3600)
        if held and price and out["relation"] not in ("REVISE", "INVALIDATE"):
            # Its own guard: malformed model fields must not skip the
            # touch_plan below (review of eed0fcb).
            try:
                out.update(_would_revise(plan, parsed, conviction, price, today, ticker))
            except Exception as e:
                log.debug(f"[{ticker}] would_revise failed: {e}")
                out["would_revise"] = None
        if out["relation"] == "REVISE":
            new.update(revision=(plan.get("revision") or 0) + 1, replaces_id=plan["id"])
            db.touch_plan(plan["id"], "REVISE")
            out["new_plan_id"] = db.insert_plan(_with_levels(new, parsed, price, ticker, today))
        elif out["relation"] == "INVALIDATE":
            db.touch_plan(plan["id"], "INVALIDATE", close_reason="price at or below invalidation")
        else:
            db.touch_plan(plan["id"], out["relation"])
        return out
    except Exception as e:
        log.error(f"[{ticker}] plan shadow failed: {e}")   # error level so the log monitor sees it
        return {"relation": "ERROR", "error": str(e)[:200]}


def compile_plan(parsed: dict, price: float, ladder: dict | None, actual_stop,
                 today: date) -> dict:
    """A concrete trade plan from the analyst's view plus the stock's price
    structure (trade_ladder rungs), shadow only (audit A2114-1): today the
    stop, targets and trailing come from generic rules and ignore both.

      structural_stop  below the nearest support rung (by 0.5 ATR) that sits
                       at least 1 ATR under price; the analyst's invalidation
                       when there is no ladder
      t1               the nearest resistance rung at least 1 ATR above price
      t2               the analyst's bull target, else the blended fair value
                       or the next resistance
      add_trigger      a close above t1's band (breakout through resistance)
      time_stop        the plan horizon (catalyst + grace, or bucket days)

    level_sources names where each of structural_stop, t1, t2 and add_trigger
    came from: "ladder_support", "analyst_invalidation", "ladder_resistance",
    "analyst_target", "fair_value", or "none" when the level is None.

    Logged next to the stop sizing actually places, so the gap between the
    plan and the trade can be measured before anything is wired in."""
    base = plan_from_verdict("", parsed, 0, price, today)
    out = {"source": "ladder" if ladder else "analyst_only",
           "invalidation_analyst": base["invalidation_price"], "time_stop": str(base["horizon_end"]),
           "catalyst_date": str(base["catalyst_date"]) if base["catalyst_date"] else None}
    atr = float((ladder or {}).get("atr") or 0) or None
    rungs = sorted((ladder or {}).get("rungs") or [], key=lambda r: r["price"])
    supports = [r for r in rungs if atr and r["price"] <= price - atr]
    resist = [r for r in rungs if atr and r["price"] >= price + atr]
    stop = round(supports[-1]["price"] - 0.5 * atr, 2) if supports else base["invalidation_price"]
    t1 = resist[0]["price"] if resist else None
    t2 = base["target_price"]
    t2_from = "analyst_target"
    if t2 is None:
        fv = (ladder or {}).get("fair_value_blended")
        fv = float(fv) if fv and float(fv) > price else None
        t2, t2_from = (fv, "fair_value") if fv else \
            (resist[1]["price"] if len(resist) > 1 else None, "ladder_resistance")
    out.update(
        structural_stop=stop, t1=t1, t2=round(t2, 2) if t2 else None,
        add_trigger=resist[0].get("band_high") if resist else None,
        support_types=supports[-1]["types"] if supports else None,
        resistance_types=resist[0]["types"] if resist else None,
        actual_stop=float(actual_stop) if actual_stop else None,
    )
    if stop and stop < price:
        out["structural_stop_pct"] = round((price - stop) / price * 100, 2)
        if t1:
            out["r_to_t1"] = round((t1 - price) / (price - stop), 2)
    if out["actual_stop"] and stop:
        out["stop_gap_pct"] = round((out["actual_stop"] - stop) / price * 100, 2)  # + = actual is tighter
    # Where each level came from ("none" when the level is None); "source"
    # above only says whether a ladder row existed.
    from_ = {"structural_stop": "ladder_support" if supports else "analyst_invalidation",
             "t1": "ladder_resistance", "t2": t2_from, "add_trigger": "ladder_resistance"}
    out["level_sources"] = {k: v if out[k] is not None else "none" for k, v in from_.items()}
    return out


def level_hits(plan: dict, price: float | None, today: date) -> list[tuple[str, float | str]]:
    """Plan levels or dates the price or calendar has reached: what a level
    watcher would re-review on (audit A2114-4/-7). Pure; log-only caller."""
    hits = []
    if price is not None and plan.get("invalidation_price") and price <= plan["invalidation_price"]:
        hits.append(("invalidation", plan["invalidation_price"]))
    if price is not None and plan.get("target_price") and price >= plan["target_price"]:
        hits.append(("target", plan["target_price"]))
    if plan.get("horizon_end") and today > plan["horizon_end"]:
        hits.append(("horizon_end", str(plan["horizon_end"])))
    elif plan.get("catalyst_date") and today > plan["catalyst_date"]:
        hits.append(("catalyst_passed", str(plan["catalyst_date"])))
    return hits
