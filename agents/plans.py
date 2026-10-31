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


def classify(plan: dict, action: str, new: dict | None, price: float | None,
             today: date, held: bool) -> dict:
    """Relation of one verdict to an active plan. `new` is the plan this
    verdict would write (OPEN/ADD only), used to detect a revision."""
    says, why = plan_says(plan, price, today, held)
    out = {"plan_id": plan["id"], "plan_says": says, "plan_why": why,
           "verdict_dir": _DIRECTION.get(action, _HOLD)}
    if why == "price at or below invalidation":
        out["relation"] = "INVALIDATE"
        return out
    if new and action in ("OPEN", "ADD") and (
            _moved(plan.get("target_price"), new.get("target_price"))
            or _moved(plan.get("invalidation_price"), new.get("invalidation_price"))
            or (plan.get("catalyst_date") and new.get("catalyst_date")
                and plan["catalyst_date"] != new["catalyst_date"])):
        out["relation"] = "REVISE"
        return out
    agree = (out["verdict_dir"] == says
             or (says == _BUY and out["verdict_dir"] == _HOLD)   # waiting is allowed
             or (says == _HOLD and out["verdict_dir"] == _BUY and held))  # adding within plan
    out["relation"] = "FOLLOW" if agree else "DIVERGE"
    return out


def observe(ticker: str, parsed: dict, action: str, conviction: int, brief: str,
            now=None) -> dict:
    """Record one verdict against the ticker's plan. Never raises."""
    try:
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
        stop = db.get_position_stop(ticker)
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

        # Close a plan that ran past its horizon, or a held-position plan
        # whose position is gone, before judging this verdict (professor M93).
        if plan is not None:
            reason = None
            if plan.get("horizon_end") and today > plan["horizon_end"]:
                reason = "expired"
            elif plan.get("source") == "held_seed" and not held:
                reason = "position closed"
            if reason:
                db.touch_plan(plan["id"], "CLOSED", close_reason=reason)
                plan = None

        if plan is None:
            if new:
                return {"relation": "NEW_PLAN", "plan_id": db.insert_plan(new), "price": price,
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
        out.update(held=held, plan_source=plan.get("source"), plan_live=held, **held_info)
        out["plan_age_days"] = ((today - plan["created_at"].astimezone(et).date()).days
                                if plan.get("created_at") else None)
        last = plan.get("last_seen_at") or plan.get("created_at")
        out["would_coalesce"] = bool(
            out["relation"] == "FOLLOW" and last
            and (now - last).total_seconds() < _COALESCE_HOURS * 3600)
        if out["relation"] == "REVISE":
            new.update(revision=(plan.get("revision") or 0) + 1, replaces_id=plan["id"])
            db.touch_plan(plan["id"], "REVISE")
            out["new_plan_id"] = db.insert_plan(new)
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
    if t2 is None:
        fv = (ladder or {}).get("fair_value_blended")
        t2 = (float(fv) if fv and float(fv) > price else None) or (resist[1]["price"] if len(resist) > 1 else None)
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
