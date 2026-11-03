"""The shadow panel runner: tails h3d.decision_record and runs the panel.

Plug-in point 1 of the readiness audit: no live code changes. The runner
reads decision records (already built from public.signals and scrubbed by
helios3d.record.decision); it never reads public.signals, the stored live
prompt or any account table.

PER DECISION (signal_id order, above the cursor)
1. Budget rule (helios3d.panel.config): off runs nothing; full runs every
   decision; shortlist runs the decisions matching the rule. A decision the
   rule excludes is passed over and the cursor moves past it.
2. Four analyst calls, back to back (shared prompt prefix), each on its own
   data slice read as of the record's as_of_ts.
3. One PM call on the analyst outputs: thesis, agreement, plan block,
   verdict, conviction.
4. One red-team call when the PM verdict is a would-be buy (OPEN/ADD).
5. One transaction: the panel_output rows, the pm_verdict row and the
   cursor. A decision is either fully written or not at all.
Before every model call the live-lane gate runs (helios3d.panel.gate). If
the lane does not clear within the wait, or the kill file appears, the run
stops: the decision in progress is not written and the cursor stays, so the
next run starts with it again.

IDEMPOTENCE: by default a decision is read only when it has no pm_verdict
row (under any panel_config_sha, failure rows included) and no skip marker
(h3d.panel_skip); inserts use ON CONFLICT DO NOTHING. The read looks back
LOOKBACK_IDS signal ids below the cursor, so a record committed out of order
is still picked up. Changing a template, model or call option changes
panel_config_sha but does not re-panel old decisions: that takes
--rerun-config, which picks decisions with no successful pm_verdict for the
current config (write-failure rows count as not done), after
--from-signal-id (default 0). The lookback also re-checks decisions the
budget rule excluded against the current rule.

POISON PILLS: bounded strings are cut to their column widths (_WIDTHS). If
writing a decision fails, it is recorded as a failed pm_verdict row (error
"write: <type>: ...") and the cursor moves on; if that also fails, the
decision gets a skip marker in h3d.panel_skip (or, if even that fails, a
logged reason) and the cursor moves. Only a failure to move the cursor stops
the run.

RETRIES: --retry-failed reads decisions whose only pm_verdict rows are
write failures, and skipped decisions, after --from-signal-id (default 0).
Writing a decision (success or a new failure) first deletes its old
write-failure rows and its skip marker in the same transaction, so a retry
replaces the failure instead of adding a second row.

A role whose call fails (timeout, HTTP error, unparseable reply) is stored
with parse_ok false and its error; the PM runs on the rest. An unreachable
server (LLMUnavailable) stops the run like a closed gate.

CLI (dry-run is the default: reads records and features, renders prompts,
prints what would run; no model call, no write):
    python -m helios3d.panel run [--limit N] [--from-signal-id ID] [--cursor NAME]
        [--rerun-config] [--retry-failed]
    python -m helios3d.panel run --apply [--limit N] ...
--apply needs HELIOS3D_PANEL_BUDGET (full or shortlist), HELIOS3D_OLLAMA_URL
and a model for every role (see helios3d/panel/config.py). It also applies
pending h3d migrations. Not scheduled; helios3d/deploy/helios3d-panel.service
is a commented-out example unit.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone

from helios3d.panel import roles, schemas
from helios3d.panel.config import (ANALYST_ROLES, PM_ROLE, RED_TEAM_ROLE, PanelSettings,
                                   SettingsError)
from helios3d.panel.gate import GateResult, LiveLaneGate
from helios3d.panel.llm import LLMCallError, LLMUnavailable, OllamaClient, extract_json
from helios3d.record.decision import redact_text

log = logging.getLogger(__name__)

DEFAULT_CURSOR = "panel"
LOOKBACK_IDS = 200
PAGE_SIZE = 100
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CURSOR_SQL = ("SELECT last_signal_id FROM h3d.panel_cursor WHERE name = %s")
_RECORDS_HEAD = (
    "SELECT d.decision_id::text, d.signal_id, d.as_of_ts, d.ticker, d.trading_env, d.trigger, "
    "d.action, d.conviction, d.inputs->'trigger' "
    "FROM h3d.decision_record d "
    "WHERE d.signal_id > %s "
    "AND (%s::text[] IS NULL OR d.trading_env = ANY(%s::text[])) ")
_RECORDS_TAIL = "ORDER BY d.signal_id LIMIT %s"
# A pm_verdict row that marks a failed write, not a panel ('%%' is a
# literal '%' in a parameterised query).
FAILED_ROW = "p.error LIKE 'write:%%'"
_NOT_DONE_ANY = ("AND NOT EXISTS (SELECT 1 FROM h3d.pm_verdict p "
                 "WHERE p.decision_id = d.decision_id) ")
_NOT_DONE_OK = ("AND NOT EXISTS (SELECT 1 FROM h3d.pm_verdict p "
                "WHERE p.decision_id = d.decision_id AND (p.error IS NULL OR NOT "
                + FAILED_ROW + ")) ")
_NOT_DONE_OK_CFG = ("AND NOT EXISTS (SELECT 1 FROM h3d.pm_verdict p "
                    "WHERE p.decision_id = d.decision_id AND p.panel_config_sha = %s "
                    "AND (p.error IS NULL OR NOT " + FAILED_ROW + ")) ")
_NOT_SKIPPED = ("AND NOT EXISTS (SELECT 1 FROM h3d.panel_skip k "
                "WHERE k.decision_id = d.decision_id) ")


def records_sql(rerun_config: bool = False, retry_failed: bool = False,
                has_panel: bool = True, has_skip: bool = True) -> tuple[str, bool]:
    """The record read for a mode, and whether it takes the config sha as a
    parameter (after the env pair, before the limit). Built from fixed
    fragments only."""
    if not has_panel:               # before migration 5: nothing is done
        return _RECORDS_HEAD + _RECORDS_TAIL, False
    if rerun_config:
        body, cfg = _NOT_DONE_OK_CFG, True
    elif retry_failed:
        body, cfg = _NOT_DONE_OK, False
    else:
        body, cfg = _NOT_DONE_ANY, False
    if has_skip and not retry_failed:
        body += _NOT_SKIPPED
    return _RECORDS_HEAD + body + _RECORDS_TAIL, cfg


RECORDS_SQL = records_sql()[0]
RECORDS_SQL_RERUN = records_sql(rerun_config=True)[0]
RECORDS_SQL_RETRY = records_sql(retry_failed=True)[0]
RECORDS_SQL_NO_PANEL = records_sql(has_panel=False)[0]

_OUT_COLS = ("decision_id", "signal_id", "ticker", "as_of_ts", "role", "model", "panel_config_sha",
             "templates_sha", "prompt_sha", "data_slice", "output_json", "score", "confidence",
             "parse_ok", "error", "latency_s", "prompt_tokens", "output_tokens", "done_reason",
             "panel_code_sha", "budget")
_PM_COLS = ("decision_id", "signal_id", "ticker", "as_of_ts", "live_action", "live_conviction",
            "model", "panel_config_sha", "templates_sha", "prompt_sha", "thesis", "agreement",
            "agreement_rationale", "agreement_facts", "verdict", "conviction", "plan",
            "plan_issues", "red_team_ran", "red_team_objection", "red_team_case_strength",
            "analysts_ok", "parse_ok", "error", "latency_s", "prompt_tokens", "output_tokens",
            "panel_code_sha", "budget")
_JSON_COLS = ("data_slice", "output_json", "agreement_facts", "plan", "plan_issues")
# VARCHAR widths in migration 5 (a test checks they match).
_WIDTHS = {"ticker": 16, "role": 24, "model": 80, "panel_config_sha": 16, "templates_sha": 16,
           "prompt_sha": 64, "error": 200, "done_reason": 20, "panel_code_sha": 48,
           "budget": 16, "live_action": 30, "agreement": 16, "verdict": 16,
           "red_team_objection": 16}


def _insert_sql(table: str, cols, conflict: str) -> str:
    vals = ", ".join("%s::uuid" if c == "decision_id" else "%s::jsonb" if c in _JSON_COLS else "%s"
                     for c in cols)
    return (f"INSERT INTO h3d.{table} ({', '.join(cols)}) VALUES ({vals}) "
            f"ON CONFLICT ON CONSTRAINT {conflict} DO NOTHING")


OUTPUT_INSERT_SQL = _insert_sql("panel_output", _OUT_COLS, "panel_output_decision_role_cfg_uq")
PM_INSERT_SQL = _insert_sql("pm_verdict", _PM_COLS, "pm_verdict_decision_cfg_uq")
CURSOR_UPSERT_SQL = (
    "INSERT INTO h3d.panel_cursor (name, last_signal_id, updated_at) VALUES (%s, %s, NOW()) "
    "ON CONFLICT (name) DO UPDATE SET last_signal_id = GREATEST(h3d.panel_cursor.last_signal_id, "
    "EXCLUDED.last_signal_id), updated_at = NOW()")
CLEAR_FAILED_SQL = ("DELETE FROM h3d.pm_verdict p WHERE p.decision_id = %s::uuid AND "
                    + FAILED_ROW)
CLEAR_SKIP_SQL = "DELETE FROM h3d.panel_skip WHERE decision_id = %s::uuid"
SKIP_UPSERT_SQL = (
    "INSERT INTO h3d.panel_skip (decision_id, signal_id, panel_config_sha, reason, skipped_at) "
    "VALUES (%s::uuid, %s, %s, %s, NOW()) ON CONFLICT (decision_id) DO UPDATE SET "
    "signal_id = EXCLUDED.signal_id, panel_config_sha = EXCLUDED.panel_config_sha, "
    "reason = EXCLUDED.reason, skipped_at = NOW()")


class Yielded(Exception):
    """The live lane did not clear, the kill file is present, or the server
    is unreachable: stop the run, keep the cursor."""

    def __init__(self, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.reason, self.detail = reason, detail or {}


def panel_code_sha(repo: str = _REPO) -> str | None:
    """HEAD of this checkout, with -dirty when helios3d/panel has local
    changes. Read with git; no live module is imported."""
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "helios3d/panel"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (head + ("-dirty" if dirty else "")) or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def _aware(ts):
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def read_cursor(conn, name: str) -> int:
    from helios3d.schema.conn import read_only
    with read_only(conn) as cur:
        cur.execute("SELECT to_regclass('h3d.panel_cursor')")
        if cur.fetchone()[0] is None:
            return 0
        cur.execute(CURSOR_SQL, (name,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def fetch_records(conn, after_id: int, config_sha: str, trading_envs=(), limit: int = PAGE_SIZE,
                  rerun_config: bool = False, retry_failed: bool = False):
    from helios3d.schema.conn import read_only
    envs = list(trading_envs) or None
    with read_only(conn) as cur:
        cur.execute("SELECT to_regclass('h3d.decision_record')")
        if cur.fetchone()[0] is None:
            return []
        cur.execute("SELECT to_regclass('h3d.pm_verdict')")
        has_panel = cur.fetchone()[0] is not None
        cur.execute("SELECT to_regclass('h3d.panel_skip')")
        has_skip = cur.fetchone()[0] is not None
        sql, with_cfg = records_sql(rerun_config, retry_failed, has_panel, has_skip)
        cur.execute(sql, (after_id, envs, envs) + ((config_sha,) if with_cfg else ()) + (limit,))
        rows = cur.fetchall()
    out = []
    for did, sid, ts, tkr, env, trig, action, conv, trig_json in rows:
        if isinstance(trig_json, str):
            try:
                trig_json = json.loads(trig_json)
            except ValueError:
                trig_json = None
        out.append({"decision_id": did, "signal_id": int(sid), "as_of_ts": _aware(ts),
                    "ticker": str(tkr).strip().upper(), "trading_env": env, "trigger": trig,
                    "action": action, "conviction": conv,
                    "inputs": {"trigger": trig_json} if isinstance(trig_json, dict) else {}})
    return out


# ---------------------------------------------------------------------------
# One decision
# ---------------------------------------------------------------------------

def _ids_only(obj):
    """Model output and model/parse error text: account ids redacted in every
    string, nothing else. The models never see account data (no role reads an
    account source), so figure and label redaction would only remove company
    figures (enterprise value, NAV, ...) before the PM reads them."""
    if isinstance(obj, dict):
        return {k: _ids_only(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ids_only(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj, figures=False)
    return obj


class PanelRunner:
    def __init__(self, settings: PanelSettings, conn, llm=None, gate=None, feature_get=None,
                 code_sha: str | None = None, apply: bool = False, cursor_name: str = DEFAULT_CURSOR):
        self.s = settings
        self.conn = conn
        self.apply = apply
        self.llm = llm
        self.gate = gate
        self.get = feature_get
        self.code_sha = code_sha
        self.cursor_name = cursor_name
        self.templates_sha = roles.templates_sha()
        self.config_sha = settings.config_sha(self.templates_sha)

    # -- calls ------------------------------------------------------------
    def _gate(self):
        res: GateResult = self.gate.wait_clear()
        if not res.ok:
            raise Yielded(res.reason, {**res.detail, "waited_s": res.waited_s})

    def _call(self, role: str, prompt: str, parser) -> dict:
        """Gate, call, parse. Returns the stored-row fields for the role."""
        self._gate()
        model = self.s.model_for(role)
        row = {"role": role, "model": model, "prompt_sha": roles.prompt_sha(prompt),
               "parsed": None, "error": None, "latency_s": None, "prompt_tokens": None,
               "output_tokens": None, "done_reason": None}
        try:
            r = self.llm.generate(model, prompt, num_predict=self.s.num_predict_for(role),
                                  num_ctx=self.s.num_ctx, temperature=self.s.temperature,
                                  seed=self.s.seed, timeout_s=self.s.timeout_s)
        except LLMUnavailable as e:
            raise Yielded("llm_unavailable", {"error": str(e)[:120]}) from None
        except LLMCallError as e:
            row["error"] = redact_text(str(e), figures=False)[:200]
            return row
        row.update(latency_s=r.latency_s, prompt_tokens=r.prompt_tokens,
                   output_tokens=r.output_tokens, done_reason=(r.done_reason or None))
        try:
            row["parsed"] = _ids_only(parser(extract_json(r.text)))
        except (ValueError, schemas.SchemaError) as e:
            row["error"] = ("parse: " + redact_text(str(e), figures=False))[:200]
        return row

    def run_decision(self, rec: dict) -> dict:
        """The panel for one record; returns rows to write. Raises Yielded."""
        t, as_of = rec["ticker"], rec["as_of_ts"]
        trig = roles.trigger_text(rec.get("trigger"), rec.get("inputs"))
        out_rows, analysts = [], {}
        for role in ANALYST_ROLES:
            try:
                slc = roles.build_slice(role, t, as_of, conn=self.conn, get=self.get)
            except Exception as e:
                out_rows.append(self._out_row(rec, {"role": role, "model": self.s.model_for(role),
                                                    "error": f"slice: {type(e).__name__}"[:200]},
                                              {"role": role, "error": type(e).__name__}))
                analysts[role] = None
                continue
            row = self._call(role, roles.render_analyst(role, t, as_of, trig, slc),
                             schemas.parse_analyst)
            analysts[role] = row["parsed"]
            out_rows.append(self._out_row(rec, row, slc))
        try:
            ctx = roles.scrub_slice(roles.price_context(t, as_of, conn=self.conn, get=self.get))
        except Exception as e:
            ctx = {"error": type(e).__name__}
        pm = self._call(PM_ROLE, roles.render_pm(t, as_of, trig, analysts, ctx), schemas.parse_pm)
        red = None
        if schemas.is_would_be_buy(pm["parsed"]):
            red = self._call(RED_TEAM_ROLE, roles.render_red_team(t, as_of, pm["parsed"], ctx),
                             schemas.parse_red_team)
            out_rows.append(self._out_row(rec, red, {"role": RED_TEAM_ROLE, "data": ctx}))
        return {"outputs": out_rows, "pm": self._pm_row(rec, pm, analysts, red, ctx)}

    # -- rows -------------------------------------------------------------
    def _base(self, rec):
        return {"decision_id": rec["decision_id"], "signal_id": rec["signal_id"],
                "ticker": rec["ticker"], "as_of_ts": rec["as_of_ts"],
                "panel_config_sha": self.config_sha, "templates_sha": self.templates_sha,
                "panel_code_sha": self.code_sha, "budget": self.s.budget}

    def _out_row(self, rec, row, slc) -> dict:
        p = row.get("parsed")
        score = None
        if p:
            score = p.get("case_strength") if row["role"] == RED_TEAM_ROLE else p.get("score")
        return {**self._base(rec), "role": row["role"], "model": row.get("model"),
                # Stored as rendered: build_slice already applied scrub_slice.
                "prompt_sha": row.get("prompt_sha"), "data_slice": slc,
                "output_json": p, "score": score,
                "confidence": p.get("confidence") if p else None, "parse_ok": p is not None,
                "error": row.get("error"), "latency_s": row.get("latency_s"),
                "prompt_tokens": row.get("prompt_tokens"),
                "output_tokens": row.get("output_tokens"), "done_reason": row.get("done_reason")}

    def _pm_row(self, rec, pm, analysts, red, ctx) -> dict:
        p = pm.get("parsed") or {}
        rp = (red or {}).get("parsed") or {}
        return {**self._base(rec), "live_action": rec.get("action"),
                "live_conviction": rec.get("conviction"), "model": pm.get("model"),
                "prompt_sha": pm.get("prompt_sha"), "thesis": p.get("thesis"),
                "agreement": p.get("agreement"), "agreement_rationale": p.get("agreement_rationale"),
                "agreement_facts": roles.agreement_facts(analysts), "verdict": p.get("verdict"),
                "conviction": p.get("conviction"), "plan": p.get("plan"),
                "plan_issues": p.get("plan_issues"), "red_team_ran": red is not None,
                "red_team_objection": rp.get("objection"),
                "red_team_case_strength": rp.get("case_strength"),
                "analysts_ok": sum(1 for a in analysts.values() if a),
                "parse_ok": bool(pm.get("parsed")), "error": pm.get("error"),
                "latency_s": pm.get("latency_s"), "prompt_tokens": pm.get("prompt_tokens"),
                "output_tokens": pm.get("output_tokens")}

    # -- writes (h3d only) -------------------------------------------------
    def _failed_pm_row(self, rec, error: str) -> dict:
        """A pm_verdict row marking a decision whose rows could not be
        written. The default read counts it as done; --retry-failed and
        --rerun-config read the decision again, and the next write of the
        decision deletes this row (CLEAR_FAILED_SQL)."""
        row = {c: None for c in _PM_COLS}
        row.update(self._base(rec), live_action=rec.get("action"),
                   live_conviction=rec.get("conviction"), agreement_facts={},
                   red_team_ran=False, analysts_ok=0, parse_ok=False, error=error)
        return row

    @staticmethod
    def _params(row: dict, cols) -> tuple:
        out = []
        for c in cols:
            v = row.get(c)
            if c in _JSON_COLS and v is not None:
                v = json.dumps(v, sort_keys=True, default=str, allow_nan=False)
            elif isinstance(v, str) and c in _WIDTHS:
                v = v[:_WIDTHS[c]]
            out.append(v)
        return tuple(out)

    def write(self, result: dict | None, signal_id: int, skip: tuple | None = None) -> None:
        """One transaction: for a result, drop the decision's old
        write-failure rows and skip marker, then insert its rows; for
        ``skip`` = (rec, reason), upsert a skip marker; then the cursor.
        Neither: cursor only."""
        try:
            with self.conn.cursor() as cur:
                if result is not None:
                    did = result["pm"]["decision_id"]
                    cur.execute(CLEAR_FAILED_SQL, (did,))
                    cur.execute(CLEAR_SKIP_SQL, (did,))
                    for r in result["outputs"]:
                        cur.execute(OUTPUT_INSERT_SQL, self._params(r, _OUT_COLS))
                    cur.execute(PM_INSERT_SQL, self._params(result["pm"], _PM_COLS))
                if skip is not None:
                    rec, reason = skip
                    cur.execute(SKIP_UPSERT_SQL, (rec["decision_id"], rec["signal_id"],
                                                  self.config_sha, str(reason)[:200]))
                cur.execute(CURSOR_UPSERT_SQL, (self.cursor_name, signal_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def write_or_fail(self, rec: dict, result: dict, summary: dict) -> bool:
        """write(), with the fallbacks in POISON PILLS. True when the
        decision's rows were written."""
        sid = rec["signal_id"]
        try:
            self.write(result, sid)
            return True
        except Exception as e:
            summary["write_failures"] += 1
            reason = f"write: {type(e).__name__}: {redact_text(str(e))[:120]}"
            log.warning(f"panel: signal {sid}: {reason}; recording it as failed")
        try:
            self.write({"outputs": [], "pm": self._failed_pm_row(rec, reason)}, sid)
            return False
        except Exception as e:
            log.warning(f"panel: signal {sid}: the failure row did not write either "
                        f"({type(e).__name__}); skipping it")
            summary["skipped"].append({"signal_id": sid, "reason": reason[:200]})
        try:
            self.write(None, sid, skip=(rec, reason))
        except Exception as e:
            log.warning(f"panel: signal {sid}: the skip marker did not write "
                        f"({type(e).__name__}); moving the cursor only")
            self.write(None, sid)   # raises when even the cursor can't move (DB down)
        return False

    # -- job ----------------------------------------------------------------
    def run(self, limit: int = 0, from_signal_id: int | None = None,
            rerun_config: bool = False, retry_failed: bool = False) -> dict:
        summary = {"mode": "apply" if self.apply else "dry-run", "budget": self.s.budget,
                   "panel_config_sha": self.config_sha, "templates_sha": self.templates_sha,
                   "cursor_name": self.cursor_name, "records": 0, "excluded_by_budget": 0,
                   "would_run": 0, "ran": 0, "red_team": 0, "role_failures": 0,
                   "yielded": None, "migrations_applied": [], "write_failures": 0,
                   "skipped": [], "rerun_config": rerun_config, "retry_failed": retry_failed}
        if self.apply:
            self.s.check(calling=True)
            from helios3d.schema import migrate
            summary["migrations_applied"] = migrate.apply(self.conn)
        else:
            self.s.check(calling=False)
        cursor = read_cursor(self.conn, self.cursor_name)
        summary["cursor"] = cursor
        if from_signal_id is not None:
            after = int(from_signal_id)
        elif rerun_config or retry_failed:
            after = 0
        else:
            after = max(0, cursor - LOOKBACK_IDS)
        sample = []
        while True:
            recs = fetch_records(self.conn, after, self.config_sha, self.s.trading_envs, PAGE_SIZE,
                                 rerun_config=rerun_config, retry_failed=retry_failed)
            if not recs:
                break
            for rec in recs:
                if limit and (summary["ran"] if self.apply else summary["records"]) >= limit:
                    return self._done(summary, sample)
                after = rec["signal_id"]
                summary["records"] += 1
                if not self.s.shortlisted(rec.get("action"), rec.get("conviction")) \
                        and self.s.budget != "off":
                    summary["excluded_by_budget"] += 1
                    if self.apply:
                        self.write(None, rec["signal_id"])
                    continue
                if not self.apply:
                    summary["would_run"] += 1 if self.s.budget != "off" else 0
                    if len(sample) < 3:
                        sample.append(self._dry_sample(rec))
                    continue
                try:
                    result = self.run_decision(rec)
                except Yielded as y:
                    summary["yielded"] = {"reason": y.reason, **y.detail,
                                          "at_signal_id": rec["signal_id"]}
                    return self._done(summary, sample)
                if not self.write_or_fail(rec, result, summary):
                    continue
                summary["ran"] += 1
                summary["red_team"] += int(result["pm"]["red_team_ran"])
                summary["role_failures"] += sum(1 for r in result["outputs"] if not r["parse_ok"]) \
                    + int(not result["pm"]["parse_ok"])
            if len(recs) < PAGE_SIZE:
                break
        return self._done(summary, sample)

    def _dry_sample(self, rec) -> dict:
        """What a dry run shows for a record: prompt sizes and data sources
        per role. The prompts themselves are not printed."""
        t, as_of = rec["ticker"], rec["as_of_ts"]
        trig = roles.trigger_text(rec.get("trigger"), rec.get("inputs"))
        out = {"signal_id": rec["signal_id"], "ticker": t, "live_action": rec.get("action"),
               "roles": {}}
        for role in ANALYST_ROLES:
            try:
                slc = roles.build_slice(role, t, as_of, conn=self.conn, get=self.get)
                p = roles.render_analyst(role, t, as_of, trig, slc)
                out["roles"][role] = {"prompt_chars": len(p), "sources": slc["sources"]}
            except Exception as e:
                out["roles"][role] = {"error": type(e).__name__}
        return out

    @staticmethod
    def _done(summary, sample):
        if summary["mode"] == "dry-run":
            summary["sample"] = sample
        return summary


def run(apply: bool = False, limit: int = 0, from_signal_id: int | None = None,
        cursor_name: str = DEFAULT_CURSOR, settings: PanelSettings | None = None, conn=None,
        llm=None, gate=None, feature_get=None, rerun_config: bool = False,
        retry_failed: bool = False) -> dict:
    """Open a connection when none is given, build the client and gate from
    the settings when not injected, and run."""
    from helios3d.panel.config import from_env
    s = settings or from_env()
    if conn is None:
        from helios3d.schema.conn import connection
        with connection() as c:
            return run(apply, limit, from_signal_id, cursor_name, s, c, llm, gate, feature_get,
                       rerun_config, retry_failed)
    if apply:
        s.check(calling=True)
        llm = llm or OllamaClient(s.ollama_url)
        gate = gate or LiveLaneGate(s)
    return PanelRunner(s, conn, llm=llm, gate=gate, feature_get=feature_get,
                       code_sha=panel_code_sha(), apply=apply,
                       cursor_name=cursor_name).run(limit=limit, from_signal_id=from_signal_id,
                                                    rerun_config=rerun_config,
                                                    retry_failed=retry_failed)


__all__ = ["PanelRunner", "Yielded", "run", "SettingsError", "fetch_records", "read_cursor"]
