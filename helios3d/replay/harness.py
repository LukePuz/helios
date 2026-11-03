"""Replay of recorded decisions with the stored model output (LLM stub mode).

For every decision in a date range (ET calendar days, inclusive), the
harness takes the model's stored JSON (signals.decision.model_output) in
place of a model call, reruns the code downstream of the model, and diffs
the result against what was recorded. Nothing calls a model, so a run is
deterministic: the same database gives a byte-identical report.

STEPS
- gates (helios3d.replay.gates): the analyst's deterministic chain, lifted
  pure and parity-tested against agents.analyst. Inputs the live code read
  through I/O come from the record: held flag (provenance.brief_exposure.
  held_at_gate), regime label (provenance.regime_used.label), fundamentals
  read (provenance.fundamentals_read), config switches (provenance.config),
  the bucket (the "Portfolio Role:" line of the stored prompt, read by a SQL
  substring so the prompt text never leaves the database, else the approval's
  thesis_bucket), and the previous verdict for the low-conviction-sell rule
  (public.signals, the db.get_latest_signal_for query bounded at the
  verdict's timestamp, ties broken by id). Config switches come only from
  provenance.config; a switch not recorded leaves the step that needs it
  oracle (SETUP_GATES_ENABLED not recorded leaves the bucket unknown). A step whose input is missing takes its outcome from
  the recorded gate list and is reported as "oracle", not "replayed".
  Trend Template is always oracle: compute_trend_template read price bars
  that public.price_history rewrites in place, and its result is not stored.
- route: whether the verdict goes to routing (crew routes iff pass_to_risk)
  is replayed. Which branch routing took is not_replayable: it read live
  state (active approvals, rebuy cooldown, catalyst and earnings calendars,
  the broker price and account). The recorded branch is reported: an
  approval (h3d.decision_record.approval_id) or the first terminal routing
  row (ROUTE_*, EARNINGS_BLOCKED, CATALYST_BLOCKED, COOLDOWN_SKIPPED,
  SELL_NO_POSITION, ...) of the same ticker after the verdict and before the
  ticker's next verdict, dropping rows whose provenance names another
  verdict's prompt_sha.
- sizing: for an OPEN/ADD that reached an approval, size_position is rerun
  through helios3d.portfolio.limits.size_legacy with the approval's price,
  stop and bucket and the nearest portfolio snapshot at or before the
  verdict. It runs only with sizing=True (CLI --sizing), because the snapshot
  holds account values; the report carries share counts and reason codes,
  never account values or dollar amounts. Otherwise not_replayable, with the
  reason.

STATUS of a decision
- not_replayable: no stored model output (ANALYST_FAILED, rows before the
  decision column, salvaged output) or an output the live parse rejects.
- differ: any of model_action, model_conviction, model_pass_to_risk,
  final_action, final_conviction, final_pass_to_risk, the gate kinds in
  order, or routed differs from the record.
- match: none differs. gate_text_equal says whether the gate strings are
  also identical (the rate gate prints a %/week figure that moves with the
  gate time; replay uses signals.timestamp, which is a little later).
  fully_replayed says whether every applicable gate step was replayed (no
  oracle step); route_branch and sizing are reported apart.
The gates are today's rules; a decision made under older code can differ
because the rule changed (code_sha is in the report and the summary).

Reads only, inside READ ONLY transactions. Writes the report file only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from helios3d.replay import gates as G

ET = ZoneInfo("America/New_York")
REPLAY_FORMAT = 1
LLM_MODE = "stub"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO / "logs" / "helios3d"

# Routing rows that end a routing attempt (crew._route_to_approval,
# scan_guard), in the order they can occur.
ROUTE_TERMINAL = ("ROUTE_APPROVAL_PENDING", "COOLDOWN_SKIPPED", "SELL_POSITION_UNREADABLE",
                  "SELL_NO_POSITION", "CATALYST_BLOCKED", "EARNINGS_BLOCKED",
                  "ROUTE_RISK_INIT_FAILED", "ROUTE_NO_PRICE", "ROUTE_ZERO_SHARES",
                  "ROUTE_CONCENTRATION_ZERO", "ROUTE_NO_BROKER")
# Rows written on the way that do not end routing.
ROUTE_NOTES = ("CATALYST_GATE_ERROR", "SHADOW_SWAP", "SELL_SKIPPED_CLOSED")
ROUTE_KINDS = ROUTE_TERMINAL + ROUTE_NOTES
# config keys size_legacy reads that provenance.config records.
SIZING_CONFIG_KEYS = ("TIER_MAX_CONCURRENT_POSITIONS", "MAX_TRADE_PCT_OF_CASH_RESERVE",
                      "ONE_SHARE_MAX_PCT_OF_CASH", "ALLOW_ONE_SHARE_ABOVE_ALLOCATION")
BUCKETS = ("Compounder", "Accelerator", "Moonshot")
COMPARED = ("model_action", "model_conviction", "model_pass_to_risk", "final_action",
            "final_conviction", "final_pass_to_risk", "gate_kinds", "routed")
# How long after the range end routing rows and next verdicts are read.
ROUTE_LOOKAHEAD = timedelta(days=3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_dict(v) -> dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else {}
        except ValueError:
            return {}
    return {}


def _aware(ts) -> datetime | None:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str) and ts.strip():
        try:
            return _aware(datetime.fromisoformat(ts.strip()))
        except ValueError:
            return None
    return None


def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def range_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    """[start 00:00 ET, end+1 00:00 ET) as UTC datetimes."""
    if end < start:
        raise ValueError("end is before start")
    lo = datetime.combine(start, time(0), tzinfo=ET).astimezone(timezone.utc)
    hi = datetime.combine(end + timedelta(days=1), time(0), tzinfo=ET).astimezone(timezone.utc)
    return lo, hi


def replay_code_sha() -> str:
    """Hash of the replay code itself, so a report names the rules it ran."""
    h = hashlib.sha256()
    for name in ("gates.py", "harness.py"):
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:12]


def _current_config(key, default=None):
    import config
    return getattr(config, key, default)


# ---------------------------------------------------------------------------
# Inputs for the gate chain
# ---------------------------------------------------------------------------

def gate_inputs(rec: dict, sig: dict, approval: dict | None) -> tuple[dict, dict]:
    """(inp, sources): the replay_gates inputs recoverable for one decision,
    and where each came from. A key left out of inp is unknown."""
    prov = _as_dict(sig.get("provenance"))
    cfg = prov.get("config") if isinstance(prov.get("config"), dict) else {}
    inp: dict = {}
    src: dict = {}
    as_of = _aware(rec.get("as_of_ts"))
    if as_of is not None:
        inp["as_of"], src["as_of"] = as_of, "signals.timestamp"
    # Config switches only as recorded; a missing one leaves its step oracle.
    for key, name in (("min_return_per_week", "MIN_EXPECTED_RETURN_PCT_PER_WEEK"),
                      ("month_midpoint", "CATALYST_MONTH_MIDPOINT_ENABLED"),
                      ("etf_exempt", "ETF_FUNDAMENTALS_CAP_EXEMPT")):
        if name in cfg and cfg[name] is not None:
            inp[key], src[key] = cfg[name], "provenance.config"
        else:
            src[key] = "not recorded"
    be = prov.get("brief_exposure")
    if isinstance(be, dict) and "held_at_gate" in be:
        inp["held"], src["held"] = be["held_at_gate"], "provenance.brief_exposure.held_at_gate"
    ru = prov.get("regime_used")
    if isinstance(ru, dict) and ru.get("label") is not None:
        inp["regime"], src["regime"] = ru["label"], "provenance.regime_used.label"
    if isinstance(prov.get("fundamentals_read"), str):
        inp["fund_read"], src["fund_read"] = prov["fundamentals_read"], "provenance.fundamentals_read"
    setup_on = cfg.get("SETUP_GATES_ENABLED")
    src["setup_gates_enabled"] = ("provenance.config" if setup_on is not None else "not recorded")
    if setup_on is None:
        src["bucket"] = "unknown: SETUP_GATES_ENABLED not recorded"
    elif setup_on:
        src["bucket"] = "unknown: SETUP_GATES_ENABLED (classify_setup at the time)"
    elif sig.get("prompt_role") in BUCKETS:
        inp["bucket"], src["bucket"] = sig["prompt_role"], "prompt Portfolio Role line"
    elif approval and approval.get("bucket") in BUCKETS:
        inp["bucket"], src["bucket"] = approval["bucket"], "pending_approvals.thesis_bucket"
    else:
        src["bucket"] = "unknown"
    return inp, src


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def assign_routes(verdicts: list[dict], route_rows: list[dict],
                  verdict_ids: dict[str, list[int]]) -> dict[int, dict]:
    """The recorded routing of each verdict. verdicts: {signal_id, ticker,
    trading_env, prompt_sha, approval_id}; route_rows: {id, ticker, kind,
    trading_env, prompt_sha}; verdict_ids: ticker -> sorted verdict ids."""
    by_ticker: dict[str, list[dict]] = {}
    for r in sorted(route_rows, key=lambda r: r["id"]):
        by_ticker.setdefault(r["ticker"], []).append(r)
    out = {}
    for v in verdicts:
        sid, t = v["signal_id"], v["ticker"]
        later = [i for i in verdict_ids.get(t, []) if i > sid]
        nxt = min(later) if later else None
        rows = [r for r in by_ticker.get(t, [])
                if r["id"] > sid and (nxt is None or r["id"] < nxt)
                and (not r.get("trading_env") or not v.get("trading_env")
                     or r["trading_env"] == v["trading_env"])
                and (not r.get("prompt_sha") or not v.get("prompt_sha")
                     or r["prompt_sha"] == v["prompt_sha"])]
        method = ("prompt_sha" if any(r.get("prompt_sha") and r["prompt_sha"] == v.get("prompt_sha")
                                      for r in rows) else "id_window" if rows else None)
        terminal = [r["kind"] for r in rows if r["kind"] in ROUTE_TERMINAL]
        if v.get("approval_id") is not None:
            outcome = "approval"
        elif terminal:
            outcome = terminal[0]
        else:
            outcome = "none"
        out[sid] = {"outcome": outcome, "method": method, "rows": [r["kind"] for r in rows],
                    "row_ids": [r["id"] for r in rows],
                    "next_verdict_id": nxt}
    return out


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def sizing_step(rec: dict, replayed: dict, route: dict, approval: dict | None,
                cfg: dict, regime: str | None, portfolio: dict | None,
                sizing_enabled: bool) -> dict:
    """size_legacy rerun for an OPEN/ADD that reached an approval. Returns
    {status, reason, replayed_shares, recorded_shares, replayed_reason}."""
    out = {"status": G.NOT_APPLICABLE, "reason": None, "replayed_shares": None,
           "recorded_shares": None, "replayed_reason": None}
    if replayed["final_action"] not in G.BUY_ACTIONS or not replayed["final_pass_to_risk"]:
        return out
    out["status"] = G.NOT_REPLAYABLE
    if route["outcome"] != "approval" or not approval:
        out["reason"] = (f"routing ended at {route['outcome']}; no sized price is recorded "
                         "for it")
        return out
    out["recorded_shares"] = approval.get("shares_initial")
    unrecorded = [k for k in ("SETUP_GATES_ENABLED",) + SIZING_CONFIG_KEYS if k not in cfg]
    if unrecorded:
        out["reason"] = "config not recorded: " + ", ".join(unrecorded)
        return out
    if cfg.get("SETUP_GATES_ENABLED"):
        out["reason"] = "setup-gate sizing path: needs classify_setup and the trade ladder at the time"
        return out
    drift = sorted(k for k in SIZING_CONFIG_KEYS if cfg[k] != _jsonable(_current_config(k)))
    if drift:
        out["reason"] = "config changed since the decision: " + ", ".join(drift)
        return out
    price, bucket = approval.get("price_at_sizing"), approval.get("bucket")
    if not price or bucket is None or regime is None:
        out["reason"] = "approval lacks price_at_sizing or bucket, or the regime label is unknown"
        return out
    if not sizing_enabled:
        out["reason"] = "account state not read (run with --sizing to read portfolio_snapshots)"
        return out
    if not portfolio:
        out["reason"] = "no portfolio snapshot at or before the decision"
        return out
    from helios3d.portfolio import limits
    positions = portfolio.get("positions") or {}
    held = _num((positions.get(rec["ticker"]) or {}).get("market_value")) or 0.0
    r = limits.size_legacy(conviction=int(approval.get("conviction") or replayed["final_conviction"]),
                           bucket=bucket, regime=regime, price=float(price),
                           pv=float(portfolio["pv"]), cash=float(portfolio["cash"]),
                           held_value=held, open_count=len(positions),
                           stop_price=approval.get("stop_loss_price"))
    out.update(status=G.REPLAYED, replayed_shares=int(r["shares_initial"]),
               replayed_reason=r["reason"])
    rec_shares = out["recorded_shares"]
    out["match"] = rec_shares is not None and int(rec_shares) == out["replayed_shares"]
    if not out["match"]:
        out["reason"] = ("recorded shares are after the concentration cap"
                         + (" (flagged)" if approval.get("size_capped") else "")
                         + " and the correlation cut; the account state is the nearest "
                           "5-minute snapshot, not the broker read at sizing")
    return out


def _jsonable(v):
    """A config value as it reads back from provenance JSON (tuples become
    lists, dict keys strings)."""
    return json.loads(json.dumps(v, default=str))


# ---------------------------------------------------------------------------
# One decision
# ---------------------------------------------------------------------------

def _recorded(dec: dict) -> dict | None:
    if "final_action" not in dec or "model_output" not in dec:
        return None
    gates = [str(g) for g in (dec.get("gates") or [])]
    return {"model_action": dec.get("model_action"), "model_conviction": dec.get("model_conviction"),
            "model_pass_to_risk": dec.get("model_pass_to_risk"),
            "final_action": dec.get("final_action"), "final_conviction": dec.get("final_conviction"),
            "final_pass_to_risk": dec.get("final_pass_to_risk"), "gates": gates,
            "gate_kinds": [G.gate_kind(g) for g in gates]}


def replay_one(rec: dict, sig: dict | None, approval: dict | None, route: dict,
               prev_lookup, portfolio_lookup, sizing_enabled: bool) -> dict:
    """The report entry for one decision. prev_lookup(rec) -> previous
    verdict dict or None (raises when unreadable); portfolio_lookup(rec) ->
    {pv, cash, positions} or None."""
    as_of = _aware(rec.get("as_of_ts"))
    entry = {"signal_id": rec["signal_id"], "decision_id": rec.get("decision_id"),
             "ticker": rec["ticker"], "trading_env": rec.get("trading_env"),
             "as_of_ts": as_of.astimezone(timezone.utc).isoformat() if as_of else None,
             "code_sha": rec.get("code_sha"), "trigger": rec.get("trigger"),
             "recorded_signal": rec.get("action"),
             "route": {"recorded": route["outcome"], "method": route["method"],
                       "rows": route["rows"]},
             "reasons": []}

    def not_replayable(reason):
        entry.update(status="not_replayable", reasons=[reason])
        return entry

    if rec.get("action") in ("ANALYST_FAILED",):
        return not_replayable("no verdict: ANALYST_FAILED")
    if sig is None:
        return not_replayable("signals row not found")
    dec = _as_dict(sig.get("decision"))
    recorded = _recorded(dec)
    if recorded is None:
        return not_replayable("no decision.model_output / final fields on the signals row")
    mo = dec.get("model_output")
    if not isinstance(mo, dict):
        return not_replayable("decision.model_output is not a JSON object")
    try:
        G.parse_model_output(mo)
    except Exception as e:
        return not_replayable(f"stored model output fails the live parse ({type(e).__name__})")

    inp, sources = gate_inputs(rec, sig, approval)
    try:
        out = G.replay_gates(mo, inp, recorded["gates"])
        lcs = next(s for s in out["steps"] if s["step"] == "low_conviction_sell")
        if lcs["status"] != G.NOT_APPLICABLE and prev_lookup is not None and "as_of" in inp:
            try:
                inp["prev_verdict"] = prev_lookup(rec)
                sources["prev_verdict"] = "public.signals before as_of_ts"
            except Exception as e:
                sources["prev_verdict"] = f"unreadable ({type(e).__name__})"
            if "prev_verdict" in inp:
                out = G.replay_gates(mo, inp, recorded["gates"])
    except Exception as e:
        return not_replayable(f"gate replay raised {type(e).__name__}")

    replayed = {k: out[k] for k in ("model_action", "model_conviction", "model_pass_to_risk",
                                    "final_action", "final_conviction", "final_pass_to_risk",
                                    "gates")}
    replayed["gate_kinds"] = [G.gate_kind(g) for g in out["gates"]]
    replayed["routed"] = bool(out["final_pass_to_risk"])
    recorded["routed"] = route["outcome"] != "none"

    diff = {k: [recorded[k], replayed[k]] for k in COMPARED if recorded[k] != replayed[k]}
    steps = out["steps"] + [{"step": "route_branch",
                             "status": G.NOT_REPLAYABLE if replayed["routed"] else G.NOT_APPLICABLE,
                             "fired": False,
                             "reason": ("branch read live state: active approvals, cooldown, "
                                        "catalyst and earnings calendars, broker price and account")
                             if replayed["routed"] else None}]
    prov = _as_dict(sig.get("provenance"))
    cfg = prov.get("config") if isinstance(prov.get("config"), dict) else {}
    sizing = sizing_step(rec, out, route, approval, cfg, inp.get("regime"),
                         portfolio_lookup(rec) if (sizing_enabled and portfolio_lookup
                                                   and route["outcome"] == "approval"
                                                   and out["final_action"] in G.BUY_ACTIONS)
                         else None, sizing_enabled)
    steps.append({"step": "sizing", "status": sizing["status"],
                  "fired": False, "reason": sizing["reason"]})

    if rec.get("action") != recorded["final_action"]:
        entry["reasons"].append("signals.signal differs from decision.final_action")
    entry.update(
        status="differ" if diff else "match",
        recorded={k: recorded[k] for k in COMPARED + ("gates",)},
        replayed={k: replayed[k] for k in COMPARED + ("gates",)},
        diff=diff,
        gate_text_equal=recorded["gates"] == replayed["gates"],
        fully_replayed=not any(s["status"] in (G.ORACLE, G.NOT_REPLAYABLE)
                               for s in out["steps"]),
        steps=steps,
        input_sources=sources,
        sizing=sizing)
    if diff:
        entry["reasons"] += [f"{k} differs" for k in diff]
    return entry


# ---------------------------------------------------------------------------
# Summary and report
# ---------------------------------------------------------------------------

def _inc(d: dict, k, n=1):
    d[k] = d.get(k, 0) + n


def summarize(entries: list[dict]) -> dict:
    s = {"decisions": len(entries), "status": {}, "not_replayable_reasons": {},
         "differ_fields": {}, "match_fully_replayed": 0, "match_with_oracle": 0,
         "match_gate_text_differs": 0, "steps": {}, "route_recorded": {},
         "route_method": {}, "sizing": {}, "by_code_sha": {}}
    for e in entries:
        st = e["status"]
        _inc(s["status"], st)
        cs = s["by_code_sha"].setdefault(str(e.get("code_sha")), {})
        _inc(cs, st)
        _inc(s["route_recorded"], e["route"]["recorded"])
        _inc(s["route_method"], str(e["route"]["method"]))
        if st == "not_replayable":
            for r in e["reasons"]:
                _inc(s["not_replayable_reasons"], r)
            continue
        for k in e["diff"]:
            _inc(s["differ_fields"], k)
        if st == "match":
            _inc(s, "match_fully_replayed" if e["fully_replayed"] else "match_with_oracle")
            if not e["gate_text_equal"]:
                _inc(s, "match_gate_text_differs")
        for stp in e["steps"]:
            d = s["steps"].setdefault(stp["step"], {})
            _inc(d, stp["status"])
            if stp["status"] in (G.ORACLE, G.NOT_REPLAYABLE) and stp.get("reason"):
                _inc(d.setdefault("reasons", {}), stp["reason"])
        sz = e["sizing"]
        key = sz["status"] if sz["status"] != G.REPLAYED else (
            "replayed_match" if sz.get("match") else "replayed_differ")
        _inc(s["sizing"], key)
    return s


def build_report(start: date, end: date, records: list[dict], source_name: str,
                 signals: dict[int, dict], approvals: dict[int, dict], routes: dict[int, dict],
                 prev_lookup=None, portfolio_lookup=None, sizing_enabled: bool = False) -> dict:
    entries = []
    for rec in sorted(records, key=lambda r: r["signal_id"]):
        appr = approvals.get(rec.get("approval_id")) if rec.get("approval_id") is not None else None
        route = routes.get(rec["signal_id"]) or {"outcome": "none", "method": None, "rows": []}
        entries.append(replay_one(rec, signals.get(rec["signal_id"]), appr, route,
                                  prev_lookup, portfolio_lookup, sizing_enabled))
    lo, hi = range_bounds(start, end)
    return {"format": REPLAY_FORMAT, "llm_mode": LLM_MODE, "replay_code_sha": replay_code_sha(),
            "range": {"start": start.isoformat(), "end": end.isoformat(),
                      "from_utc": lo.isoformat(), "to_utc": hi.isoformat(), "calendar": "ET"},
            "record_source": source_name, "sizing_enabled": sizing_enabled,
            "summary": summarize(entries), "decisions": entries}


def _json_default(o):
    if isinstance(o, datetime):
        return _aware(o).astimezone(timezone.utc).isoformat()
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    return str(o)


def report_bytes(report: dict) -> bytes:
    return (json.dumps(report, sort_keys=True, indent=1, default=_json_default,
                       ensure_ascii=True) + "\n").encode()


# ---------------------------------------------------------------------------
# Database source (read-only)
# ---------------------------------------------------------------------------

H3D_RECORDS_SQL = (
    "SELECT signal_id, decision_id, as_of_ts, ticker, trading_env, action, conviction, "
    "code_sha, trigger, approval_id "
    "FROM h3d.decision_record WHERE as_of_ts >= %s AND as_of_ts < %s ORDER BY signal_id")

# Only the role word is taken from the stored prompt; the text stays in the DB.
SIGNALS_BY_ID_SQL = (
    "SELECT id, signal, confidence, decision, provenance, "
    "substring(prompt from E'\\nPortfolio Role: (Compounder|Accelerator|Moonshot)\\n') "
    "FROM public.signals WHERE id = ANY(%s)")

APPROVALS_SQL = (
    "SELECT id, ticker, action, conviction, thesis_bucket, trading_env, "
    "risk_json->>'shares_initial', risk_json->>'price_at_sizing', risk_json->>'stop_loss_price', "
    "COALESCE(risk_json->'flags', '[]'::jsonb)::text LIKE %s "
    "FROM public.pending_approvals WHERE id = ANY(%s)")

ROUTE_ROWS_SQL = (
    "SELECT id, upper(trim(symbol)), signal, trading_env, provenance->>'prompt_sha' "
    "FROM public.signals WHERE upper(trim(symbol)) = ANY(%s) AND signal IN %s "
    "AND id > %s AND timestamp < %s ORDER BY id")

VERDICT_IDS_SQL = (
    "SELECT id, upper(trim(symbol)) FROM public.signals "
    "WHERE upper(trim(symbol)) = ANY(%s) AND signal NOT IN %s AND id > %s AND timestamp < %s "
    "ORDER BY id")

# db.get_latest_signal_for, bounded at the verdict's own timestamp.
PREV_VERDICT_SQL = (
    "SELECT signal, confidence, timestamp FROM public.signals "
    "WHERE symbol = %s AND signal NOT IN %s "
    "AND (trading_env = %s OR (trading_env IS NULL AND COALESCE(reasoning, '') NOT LIKE %s)) "
    "AND timestamp < %s AND id <> %s ORDER BY timestamp DESC, id DESC LIMIT 1")


class DbSource:
    """Reads for one replay, each inside a READ ONLY transaction."""

    def __init__(self, conn):
        self.conn = conn

    def _rows(self, sql, params):
        from helios3d.schema.conn import read_only
        with read_only(self.conn) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def records(self, lo: datetime, hi: datetime) -> tuple[list[dict], str]:
        exists = self._rows("SELECT to_regclass('h3d.decision_record')", None)
        if exists and exists[0][0] is not None:
            rows = self._rows(H3D_RECORDS_SQL, (lo, hi))
            return [{"signal_id": int(r[0]), "decision_id": str(r[1]), "as_of_ts": _aware(r[2]),
                     "ticker": str(r[3]).strip().upper(), "trading_env": r[4], "action": r[5],
                     "conviction": r[6], "code_sha": r[7], "trigger": r[8],
                     "approval_id": r[9]} for r in rows], "h3d.decision_record"
        return self._build_records(lo, hi), "built in memory (h3d.decision_record absent)"

    def _build_records(self, lo: datetime, hi: datetime) -> list[dict]:
        from helios3d.record import decision as D
        from helios3d.schema.conn import read_only
        top = 2 ** 62
        after = D.page_start(self.conn, top, lo)
        out = []
        while True:
            with read_only(self.conn) as cur:
                sigs = D.fetch_signal_page(cur, after, top, lo)
                if not sigs:
                    break
                ctx = D.fetch_context(cur, sigs)
            for s in sigs:
                if lo <= s["as_of_ts"] < hi:
                    r = D.build_record(s, ctx, None)
                    out.append({k: r[k] for k in ("signal_id", "decision_id", "as_of_ts", "ticker",
                                                  "trading_env", "action", "conviction",
                                                  "code_sha", "trigger", "approval_id")})
            after = sigs[-1]["signal_id"]
            if len(sigs) < D.PAGE_SIZE:
                break
        return out

    def signals(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        rows = self._rows(SIGNALS_BY_ID_SQL, (sorted(ids),))
        return {int(r[0]): {"signal": r[1], "confidence": r[2], "decision": r[3],
                            "provenance": r[4], "prompt_role": r[5]} for r in rows}

    def approvals(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        rows = self._rows(APPROVALS_SQL, ("%Size capped%", sorted(ids)))
        out = {}
        for r in rows:
            shares = _num(r[6])
            out[int(r[0])] = {"ticker": r[1], "action": r[2], "conviction": r[3], "bucket": r[4],
                              "trading_env": r[5],
                              "shares_initial": int(shares) if shares is not None else None,
                              "price_at_sizing": _num(r[7]), "stop_loss_price": _num(r[8]),
                              "size_capped": bool(r[9])}
        return out

    def routes(self, verdicts: list[dict], hi: datetime) -> dict[int, dict]:
        if not verdicts:
            return {}
        import db
        tickers = sorted({v["ticker"] for v in verdicts})
        first = min(v["signal_id"] for v in verdicts)
        until = hi + ROUTE_LOOKAHEAD
        rows = self._rows(ROUTE_ROWS_SQL, (tickers, ROUTE_KINDS, first, until))
        route_rows = [{"id": int(r[0]), "ticker": r[1], "kind": r[2], "trading_env": r[3],
                       "prompt_sha": r[4]} for r in rows]
        vids: dict[str, list[int]] = {}
        for r in self._rows(VERDICT_IDS_SQL, (tickers, tuple(db._NON_ANALYST_SIGNALS), first, until)):
            vids.setdefault(r[1], []).append(int(r[0]))
        return assign_routes(verdicts, route_rows, vids)

    def prev_verdict(self, rec: dict) -> dict | None:
        import db
        env = rec.get("trading_env") or _current_config("TRADING_ENV", "paper")
        rows = self._rows(PREV_VERDICT_SQL, (rec["ticker"], tuple(db._NON_ANALYST_SIGNALS), env,
                                             db.DEGRADED_SIGNAL_PREFIX + "%",
                                             rec["as_of_ts"], rec["signal_id"]))
        if not rows:
            return None
        return {"signal": rows[0][0], "confidence": float(rows[0][1] or 0),
                "timestamp": _aware(rows[0][2])}

    def portfolio(self, rec: dict) -> dict | None:
        from helios3d.features.store import portfolio_snapshot
        res = portfolio_snapshot(None, rec["as_of_ts"], rec.get("trading_env"), conn=self.conn)
        if not res.rows:
            return None
        r = res.rows[0]
        return {"pv": float(r["total_value"]), "cash": float(r["cash"]),
                "positions": _as_dict(r.get("positions_json"))}


def run(start: date, end: date, source, sizing: bool = False) -> dict:
    """Read the range through ``source`` (DbSource or a fake with the same
    methods) and build the report. Writes nothing."""
    lo, hi = range_bounds(start, end)
    records, source_name = source.records(lo, hi)
    signals = source.signals([r["signal_id"] for r in records])
    approvals = source.approvals(sorted({r["approval_id"] for r in records
                                         if r.get("approval_id") is not None}))
    verdicts = []
    for r in records:
        prov = _as_dict((signals.get(r["signal_id"]) or {}).get("provenance"))
        verdicts.append({"signal_id": r["signal_id"], "ticker": r["ticker"],
                         "trading_env": r.get("trading_env"),
                         "prompt_sha": prov.get("prompt_sha"), "approval_id": r.get("approval_id")})
    routes = source.routes(verdicts, hi)
    return build_report(start, end, records, source_name, signals, approvals, routes,
                        prev_lookup=source.prev_verdict,
                        portfolio_lookup=source.portfolio if sizing else None,
                        sizing_enabled=sizing)


def default_out(start: date, end: date) -> Path:
    return DEFAULT_OUT_DIR / f"replay_{start.isoformat()}_{end.isoformat()}.json"


def write_report(report: dict, path: Path) -> str:
    """Write the report; returns its sha256 (first 16 hex)."""
    data = report_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()[:16]
