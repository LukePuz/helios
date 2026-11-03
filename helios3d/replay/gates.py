"""Pure lift of the analyst's deterministic gates, for replay.

agents/analyst.py _analyze_inner turns the model's JSON into the final
verdict through a fixed chain of rules. This module repeats that chain on a
stored model output (signals.decision.model_output), with every input the
live code read through I/O passed in instead. The gate strings are built
with the same f-strings as live, so a replayed gate list can be compared
with signals.decision.gates text for text.
tests/test_helios3d_replay_parity.py runs the live analyze() (I/O mocked)
and replay_gates() over a grid of synthetic outputs and asserts the same
gates, final action, conviction and pass_to_risk.

Order of the chain (as in live):
  unheld coercion, UDR floor, source count, expected-return rate,
  Trend Template, Moonshot conviction, regime selectivity, fundamentals
  conviction cap, pass_to_risk validation, low-conviction sell.

Inputs (keys of ``inp``; a key that is ABSENT means "unknown"):
  held                 candidate["held"] at the gate (True/False/None)
  bucket               the bucket the gates used (watchlist role, or the
                       live setup category when SETUP_GATES_ENABLED)
  regime               regime_packet["regime"] (may be None)
  fund_read            provenance.fundamentals_read: "ok", "missing" or
                       "read error: X"
  is_etf               ticker in knowledge_loader.get_etf_tickers()
  trend_template_pass  compute_trend_template(ticker)["pass"]; None = the
                       function returned None (insufficient history: passes)
  prev_verdict         db.get_latest_signal_for(ticker) before this verdict:
                       {"confidence", "timestamp"} or None
  as_of                the gate time (datetime, aware)
  min_return_per_week  config MIN_EXPECTED_RETURN_PCT_PER_WEEK
  month_midpoint       config CATALYST_MONTH_MIDPOINT_ENABLED
  etf_exempt           config ETF_FUNDAMENTALS_CAP_EXEMPT
  (the harness sets the three config keys only from provenance.config; one
  not recorded leaves the step that needs it oracle, never today's value)

Return-rate reason: matches live from 453a950 on ("no expected_return_pct"
for a dated catalyst with no expected_return_pct). Before that commit live
raised TypeError on that input and logged ANALYST_FAILED instead of a
verdict; the harness marks every ANALYST_FAILED row not_replayable as a
whole decision, so the newer branch cannot produce a false differ for them.

Catalyst normalisation: matches live from the analyst commits that read a
non-object catalyst as {} and a non-string catalyst.date as its text (int
or float) or None. Before them, such outputs made live raise (ANALYST_FAILED)
or fail to write the verdict row at all (_catalyst_check raising inside the
db.log_signal arguments). Neither leaves a replayable record: ANALYST_FAILED
rows are not_replayable and a missing row is never read, so the
normalisation cannot create a false differ on decisions recorded before it.

When an input a step needs is unknown, the step is an "oracle" step: it takes
whether the gate fired (and its text) from the recorded gate list instead of
deciding it. Such a step is reported as oracle, never as replayed.
"""
from __future__ import annotations

import copy
import re
from datetime import datetime, timezone

# Literals from agents/analyst.py (module constants or inline numbers in
# _analyze_inner). test_literals_match_live checks the named ones.
LOW_CONVICTION_SELL_MAX = 4
EXIT_CONVICTION_MAX = 2
LOW_CONVICTION_CONFIRM_MAX_AGE_DAYS = 7
BUY_ACTIONS = ("OPEN", "ADD")
SELL_ACTIONS = ("TRIM", "EXIT")
ACTIONABLE = ("OPEN", "ADD", "TRIM", "EXIT")
CAUTIOUS_REGIMES = ("Neutral", "Risk-Off", "Strong Risk-Off")
RATE_BUCKETS = ("Accelerator", "Moonshot")

# Statuses of one step.
REPLAYED = "replayed"
ORACLE = "oracle"
NOT_APPLICABLE = "not_applicable"
NOT_REPLAYABLE = "not_replayable"

STEP_ORDER = ("unheld", "udr", "sources", "return_rate", "trend_template", "moonshot",
              "regime", "fundamentals", "pass_validation", "low_conviction_sell")


# ---------------------------------------------------------------------------
# Lifted pure helpers (parity-tested against agents.analyst)
# ---------------------------------------------------------------------------

def count_distinct_sources(sources: list) -> int:
    """agents.analyst.count_distinct_sources."""
    return len({
        (s.get("type"), (s.get("detail") or "")[:80])
        for s in (sources or []) if isinstance(s, dict) and s.get("type")
    })


def compute_weeks_to_catalyst(catalyst_date: str | None, as_of: datetime,
                              month_midpoint: bool = False) -> float | None:
    """agents.analyst.compute_weeks_to_catalyst with ``as_of`` and the
    month-midpoint switch as arguments (live: now and config)."""
    if not catalyst_date:
        return None
    text = catalyst_date.strip().lower()
    if text == "ongoing":
        return None
    now = as_of
    parsed = None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            parsed = datetime.strptime(catalyst_date.strip(), fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        return None
    if month_midpoint and re.fullmatch(r"\d{4}-\d{1,2}", catalyst_date.strip()):
        parsed = parsed.replace(day=15)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    days = (parsed - now).total_seconds() / 86400
    return max(days, 1.0) / 7.0


def expected_return_per_week(expected_return_pct: float | None,
                             weeks_to_catalyst: float | None) -> float | None:
    """agents.analyst.expected_return_per_week."""
    if expected_return_pct is None or weeks_to_catalyst is None or weeks_to_catalyst <= 0:
        return None
    return expected_return_pct / weeks_to_catalyst


def optional_float(parsed: dict, field: str) -> float | None:
    """_analyze_inner's _optional_float."""
    value = parsed.get(field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def signal_age_days(signal: dict, now: datetime) -> float:
    """agents.analyst._signal_age_days with ``now`` as an argument."""
    ts = signal.get("timestamp")
    if not isinstance(ts, datetime):
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 86400


def low_conviction_sell(action: str, conviction: int, prev: dict | None,
                        now: datetime) -> tuple[str, bool, str]:
    """agents.analyst._low_conviction_sell with the previous verdict (live:
    db.get_latest_signal_for) and ``now`` as arguments."""
    if prev and signal_age_days(prev, now) > LOW_CONVICTION_CONFIRM_MAX_AGE_DAYS:
        return action, False, (f"Low-conviction {action} held for confirmation: previous "
                               f"verdict is older than {LOW_CONVICTION_CONFIRM_MAX_AGE_DAYS} days")
    prev_conv = round(prev["confidence"] * 10) if prev else None
    if prev_conv is None or prev_conv > LOW_CONVICTION_SELL_MAX:
        return action, False, (f"Low-conviction {action} held for confirmation: previous "
                               f"verdict conviction {prev_conv}, needs <= {LOW_CONVICTION_SELL_MAX}")
    if action == "EXIT" and conviction > EXIT_CONVICTION_MAX:
        return "TRIM", True, (f"EXIT at conviction {conviction} reduced to TRIM "
                              f"(EXIT needs <= {EXIT_CONVICTION_MAX}); previous verdict {prev_conv}")
    return action, True, f"Low-conviction {action} confirmed: previous verdict conviction {prev_conv}"


# ---------------------------------------------------------------------------
# Gate kinds (for comparing gate lists at the rule level)
# ---------------------------------------------------------------------------

def gate_kind(text: str) -> str:
    t = str(text)
    if t.startswith("unheld "):
        return "unheld"
    if t.startswith("udr "):
        return "udr"
    if t.startswith("sources "):
        return "sources"
    if t.startswith("return rate"):
        return "return_rate"
    if t.startswith("trend template"):
        return "trend_template"
    if t.startswith("moonshot conviction"):
        return "moonshot"
    if " regime conviction " in t:
        return "regime"
    if t.startswith("no fundamentals:") or t.startswith("fundamentals "):
        return "fundamentals_etf_exempt" if "cap not applied" in t else "fundamentals_cap"
    if t.startswith("pass_to_risk dropped"):
        return "pass_dropped"
    if t.startswith("low-conviction sell"):
        return "low_conviction_sell"
    return "other"


_LOW_SELL_RE = re.compile(r"^low-conviction sell (\w+) c(\d+) -> (\w+), pass=(True|False)")


class _Recorded:
    """The recorded gate list, consumed kind by kind by oracle steps."""

    def __init__(self, gates):
        self.gates = [str(g) for g in (gates or [])]
        self.used: set[int] = set()

    def take(self, *kinds: str) -> str | None:
        for i, g in enumerate(self.gates):
            if i not in self.used and gate_kind(g) in kinds:
                self.used.add(i)
                return g
        return None


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------

def normalise_catalyst(parsed: dict) -> None:
    """_analyze_inner's catalyst normalisation (in place, on the working
    copy; model_output stays raw): a catalyst that is neither an object nor
    None becomes {}, and a non-string catalyst.date becomes its text when it
    is an int or float (not a bool), else None."""
    if not isinstance(parsed.get("catalyst", {}), (dict, type(None))):
        parsed["catalyst"] = {}
    cat = parsed.get("catalyst")
    if isinstance(cat, dict) and cat.get("date") is not None and not isinstance(cat["date"], str):
        raw = cat["date"]
        cat["date"] = (str(raw) if isinstance(raw, (int, float))
                       and not isinstance(raw, bool) else None)


def parse_model_output(model_output: dict) -> tuple[str, int, float]:
    """(action, conviction, udr) exactly as _analyze_inner reads them. Raises
    as live does on a value live could not parse."""
    action = str(model_output.get("action", "WATCH")).upper()
    conviction = max(1, min(10, int(model_output.get("conviction", 1) or 1)))
    udr = float(model_output.get("upside_downside_ratio", 0) or 0)
    return action, conviction, udr


def replay_gates(model_output: dict, inp: dict, recorded_gates=()) -> dict:
    """Run the gate chain on a stored model output. Pure. Returns the
    model and final action/conviction/pass_to_risk, the gate list and one
    entry per step (status, fired, reason)."""
    parsed = copy.deepcopy(model_output)
    rec = _Recorded(recorded_gates)
    steps: list[dict] = []
    gates: list[str] = []

    def step(name, status, fired=False, reason=None):
        steps.append({"step": name, "status": status, "fired": bool(fired), "reason": reason})

    action, conviction, udr = parse_model_output(parsed)
    normalise_catalyst(parsed)
    model_action, model_conviction = action, conviction
    model_pass = bool(parsed.get("pass_to_risk", False))
    bucket_known = "bucket" in inp
    bucket = inp.get("bucket")

    # 1. unheld ADD/HOLD
    if action in ("ADD", "HOLD"):
        coerced = "OPEN" if action == "ADD" else "WATCH"
        if "held" in inp:
            fired = inp["held"] is False
            text = f"unheld {action} -> {coerced}"
            step("unheld", REPLAYED, fired)
        else:
            text = rec.take("unheld")
            fired = text is not None
            step("unheld", ORACLE, fired, "held flag at the gate not recorded")
        if fired:
            gates.append(text)
            action = coerced
    else:
        step("unheld", NOT_APPLICABLE)

    # 2. UDR floor
    if udr < 2.0 and action in BUY_ACTIONS:
        gates.append(f"udr {udr:.1f} < 2.0: {action} -> WATCH")
        action = "WATCH"
        parsed["pass_to_risk"] = False
        step("udr", REPLAYED, True)
    else:
        step("udr", REPLAYED if action in BUY_ACTIONS else NOT_APPLICABLE)

    # 3. distinct sources
    sources = parsed.get("sources") or []
    source_count = count_distinct_sources(sources)
    if source_count < 2 and action in BUY_ACTIONS:
        gates.append(f"sources {source_count} < 2: {action} -> WATCH")
        action = "WATCH"
        parsed["pass_to_risk"] = False
        step("sources", REPLAYED, True)
    else:
        step("sources", REPLAYED if action in BUY_ACTIONS else NOT_APPLICABLE)

    # 4. expected-return rate (Accelerator/Moonshot buys).
    rate_cfg = "min_return_per_week" in inp and "month_midpoint" in inp
    if action in BUY_ACTIONS:
        if bucket_known and "as_of" in inp and rate_cfg:
            if bucket in RATE_BUCKETS:
                catalyst_date = (parsed.get("catalyst") or {}).get("date")
                weeks = compute_weeks_to_catalyst(catalyst_date, inp["as_of"],
                                                  bool(inp["month_midpoint"]))
                per_week = expected_return_per_week(optional_float(parsed, "expected_return_pct"),
                                                    weeks)
                floor = inp["min_return_per_week"]
                fired = per_week is None or per_week < floor
                if fired:
                    reason = ("no computable catalyst timeframe" if weeks is None
                              else "no expected_return_pct" if per_week is None
                              else f"{per_week:.2f}%/week < {floor}%/week minimum")
                    gates.append(f"return rate ({reason}): {action} -> WATCH")
                    action = "WATCH"
                    parsed["pass_to_risk"] = False
                step("return_rate", REPLAYED, fired)
            else:
                step("return_rate", NOT_APPLICABLE, False, f"bucket {bucket}")
        elif bucket_known and bucket not in RATE_BUCKETS:
            step("return_rate", NOT_APPLICABLE, False, f"bucket {bucket}")
        else:
            text = rec.take("return_rate")
            if text is not None:
                gates.append(text)
                action = "WATCH"
                parsed["pass_to_risk"] = False
            step("return_rate", ORACLE, text is not None,
                 "bucket not recorded" if not bucket_known else
                 "gate time unknown" if "as_of" not in inp else
                 "rate config (MIN_EXPECTED_RETURN_PCT_PER_WEEK / CATALYST_MONTH_MIDPOINT_ENABLED) "
                 "not recorded")
    else:
        # Live's Accelerator/Moonshot HOLD branch only flags a stale catalyst
        # (log only, no gate); after the catalyst normalisation it cannot
        # raise, so it is not replayed.
        step("return_rate", NOT_APPLICABLE)

    # 5. Trend Template (Accelerator buys)
    if action in BUY_ACTIONS and (not bucket_known or bucket == "Accelerator"):
        if bucket_known and "trend_template_pass" in inp:
            tt = inp["trend_template_pass"]
            fired = tt is not None and not tt
            if fired:
                gates.append(f"trend template: {action} -> WATCH")
                action = "WATCH"
                parsed["pass_to_risk"] = False
            step("trend_template", REPLAYED, fired)
        else:
            text = rec.take("trend_template")
            if text is not None:
                gates.append(text)
                action = "WATCH"
                parsed["pass_to_risk"] = False
            step("trend_template", ORACLE, text is not None,
                 "bucket not recorded" if not bucket_known else
                 "compute_trend_template needs point-in-time price bars; its result is not recorded")
    else:
        step("trend_template", NOT_APPLICABLE)

    # 6. Moonshot OPEN needs conviction >= 7
    if action == "OPEN" and conviction < 7:
        if bucket_known:
            fired = bucket == "Moonshot"
            if fired:
                gates.append(f"moonshot conviction {conviction} < 7: OPEN -> WATCH")
                action = "WATCH"
                parsed["pass_to_risk"] = False
            step("moonshot", REPLAYED, fired)
        else:
            text = rec.take("moonshot")
            if text is not None:
                gates.append(text)
                action = "WATCH"
                parsed["pass_to_risk"] = False
            step("moonshot", ORACLE, text is not None, "bucket not recorded")
    else:
        step("moonshot", NOT_APPLICABLE)

    # 7. regime selectivity for new positions
    if action == "OPEN" and conviction < 7:
        if "regime" in inp:
            regime_str = inp["regime"]
            fired = regime_str in CAUTIOUS_REGIMES
            if fired:
                gates.append(f"{regime_str} regime conviction {conviction} < 7: OPEN -> WATCH")
                action = "WATCH"
                parsed["pass_to_risk"] = False
            step("regime", REPLAYED, fired)
        else:
            text = rec.take("regime")
            if text is not None:
                gates.append(text)
                action = "WATCH"
                parsed["pass_to_risk"] = False
            step("regime", ORACLE, text is not None, "regime label not recorded")
    else:
        step("regime", NOT_APPLICABLE)

    # 8. fundamentals conviction cap
    if action in BUY_ACTIONS and conviction > 6:
        if "fund_read" in inp:
            fund_read = inp["fund_read"]
            missing = fund_read != "ok"
            if missing:
                fund_label = "no fundamentals" if fund_read == "missing" else f"fundamentals {fund_read}"
                prior = [g for g in rec.gates if gate_kind(g).startswith("fundamentals")]
                notes = []
                if "is_etf" in inp:
                    is_etf, etf_status = bool(inp["is_etf"]), REPLAYED
                else:
                    is_etf, etf_status = any("ETF" in g for g in prior), ORACLE
                    notes.append("ETF membership at the time not recorded")
                if not is_etf:
                    exempt = False
                elif "etf_exempt" in inp:
                    exempt = bool(inp["etf_exempt"])
                else:
                    exempt, etf_status = any("cap not applied" in g for g in prior), ORACLE
                    notes.append("ETF_FUNDAMENTALS_CAP_EXEMPT not recorded")
                if is_etf and exempt:
                    gates.append(f"{fund_label}: ETF, cap not applied (conviction {conviction})")
                else:
                    gates.append(f"{fund_label}: conviction {conviction} -> 6"
                                 + (" (ETF; exemption off)" if is_etf else ""))
                    conviction = 6
                step("fundamentals", etf_status, True, "; ".join(notes) or None)
            else:
                step("fundamentals", REPLAYED, False)
        else:
            text = rec.take("fundamentals_cap", "fundamentals_etf_exempt")
            if text is not None:
                gates.append(text)
                if gate_kind(text) == "fundamentals_cap":
                    conviction = 6
            step("fundamentals", ORACLE, text is not None, "fundamentals read not recorded")
    else:
        step("fundamentals", NOT_APPLICABLE)

    # 9. pass_to_risk validation
    actionable = action in ACTIONABLE
    if parsed.get("pass_to_risk") and not (actionable and conviction >= 5):
        gates.append(f"pass_to_risk dropped ({action} c{conviction})")
        parsed["pass_to_risk"] = False
        step("pass_validation", REPLAYED, True)
    else:
        step("pass_validation", REPLAYED, False)

    # 10. low-conviction sells
    if action in SELL_ACTIONS and conviction <= LOW_CONVICTION_SELL_MAX:
        sell_from = action
        if "prev_verdict" in inp and "as_of" in inp:
            action, sell_passes, sell_note = low_conviction_sell(action, conviction,
                                                                 inp["prev_verdict"], inp["as_of"])
            gates.append(f"low-conviction sell {sell_from} c{conviction} -> {action}, "
                         f"pass={sell_passes}" + (f" ({sell_note})" if sell_note else ""))
            parsed["pass_to_risk"] = sell_passes
            step("low_conviction_sell", REPLAYED, True)
        else:
            text = rec.take("low_conviction_sell")
            m = _LOW_SELL_RE.match(text or "")
            if m:
                gates.append(text)
                action, parsed["pass_to_risk"] = m.group(3), m.group(4) == "True"
                step("low_conviction_sell", ORACLE, True, "previous verdict not supplied")
            else:
                parsed["pass_to_risk"] = False
                step("low_conviction_sell", NOT_REPLAYABLE, False,
                     "previous verdict not supplied and no recorded gate to take the outcome from")
    else:
        step("low_conviction_sell", NOT_APPLICABLE)

    return {"model_action": model_action, "model_conviction": model_conviction,
            "model_pass_to_risk": model_pass, "final_action": action,
            "final_conviction": conviction,
            "final_pass_to_risk": bool(parsed.get("pass_to_risk", False)),
            "gates": gates, "steps": steps, "source_count": source_count}
