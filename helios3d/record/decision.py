"""Decision records for analyst verdicts (Helios 3D, Phase 1).

One row in h3d.decision_record per analyst verdict in public.signals, built
by tailing that table, so the live code is not touched. A record names the
code, prompt and model that made the verdict, the regime row it used, the
approval, orders, outcome rows and trade plan it led to (the decision_id
join public has no column for), and the structured inputs recoverable from
signals.provenance and signals.decision.

Research tooling only: it stores what happened. Which records count as the
study's verdicts is the analysis's filter, not this module's (trading_env and
inputs.degraded are stored so the labels.py filter can be applied).

ROWS: every public.signals row whose signal is not in db._NON_ANALYST_SIGNALS,
plus ANALYST_FAILED rows (an analysis that produced no verdict). All
trading_env values are kept.

FIELDS
- decision_id: UUIDv5 of "public.signals:<id>", the same on every rebuild.
- as_of_ts = signals.timestamp; ticker = signals.symbol (upper);
  action = signals.signal; conviction = round(signals.confidence * 10).
- trigger = provenance.trigger.path (daily_scan, event, ...).
- code_sha, head_sha, prompt_template_sha, brief_template_sha, prompt_sha,
  model_digest from provenance; model = signals.model, else provenance.model
  (ANALYST_FAILED rows store NULL in signals.model on purpose).
- inputs: a fixed list of provenance/decision keys (see _PROV_KEYS,
  _DECISION_KEYS, _MODEL_SCORE_KEYS), scrubbed by scrub(); keys that were not
  copied are listed by name in inputs._not_copied and scrubbed paths in
  inputs._dropped_keys. The full rows stay in public.signals under signal_id.

NEVER COPIED: the signals.prompt column (only `prompt IS NOT NULL` is read),
signals.reasoning, the model's free text (decision.model_output is reduced to
its scalar scores; only action and thesis_bucket stay as strings, capped at
40 chars), and any key that names an account figure: the fragments in
_DENY_SUBSTRINGS (cash, balance, equity, account, acct, pnl, netliq, funds,
margin, mktvalue, ...) matched in the squashed lower-cased key, so camelCase
and IBKR tags such as TotalCashValue match. UPPER_SNAKE config thresholds
are exempt only under inputs.versions.config. In strings, IBKR account ids
and dollar figures next to an account word are redacted, and error text and
decision.failure.reason are capped at 120 chars after redaction.

JOINS (each one's method, status, candidates and reason are in `links`;
status is matched, none, ambiguous, conflict, missing or skipped; a column
is set only for a row that was found and checked):
- approval_id: pending_approvals.signal_id = signals.id (migration 77), or
  analyst_json.provenance.signal_id = signals.id. When neither exists, a
  pending_approvals row with the same ticker, trading_env and
  analyst_json.provenance.prompt_sha, and no signal id of its own, matches
  only if it is the only such approval and the prompt_sha belongs to no
  other verdict. Two or more candidates, or an approval whose two signal ids
  disagree, leave approval_id NULL with the reason.
- order_ids: every orders row whose approval_id is the matched approval.
- trade_outcome_ids: every trade_outcomes row whose approval_id is the
  matched approval; more than one is stored and flagged ambiguous.
- trade_plan_id / trade_plan_new_id: provenance.plan.plan_id and
  .new_plan_id (written by agents/plans.observe), kept when the trade_plans
  row exists and has the verdict's ticker.
- regime_row_id: the market_regime_log row whose timestamp equals
  provenance.regime_used.timestamp exactly and whose regime_type/sector_etf
  match regime_used.source; zero or several rows leave it NULL.
No join falls back to a time window.

RUNS: the cursor is MAX(signal_id) in h3d.decision_record (0 when the table
is absent or empty, which makes the first run the backfill). Each run reads
signals with id above the cursor, plus every signal from the last
--relink-days days, since approvals, orders and outcome rows arrive after
the verdict (this re-read also catches a signal id committed out of order).
Paging starts at min(cursor, first id in the relink window), not at 0.
A record is upserted on signal_id and rewritten only when its content_sha
changes, so a re-run with nothing new writes nothing.

After any change to the builder or the scrubber (key lists, deny rules,
redaction, caps), run once with --full --apply: an incremental run only
rebuilds new and recent signals, so older records would keep the old rules.

Reads public tables only inside READ ONLY transactions. Writes only to
h3d.decision_record. Not scheduled.

CLI (dry-run is the default; it reads, builds and prints, and writes nothing):
    python -m helios3d.record.decision [--dry-run] [--full] [--relink-days N]
    python -m helios3d.record.decision --apply [--full] [--relink-days N]
"""
import argparse
import hashlib
import json
import logging
import re
import sys
import uuid
from datetime import date, datetime, timedelta, timezone

from helios3d.record.verdict_rows import DECISION_KEEP_KINDS, DECISION_RULE

# Fixed namespace for decision ids; never change it (ids would all move).
DECISION_NAMESPACE = uuid.UUID("5b1e7c9a-3d2f-4e1b-9a6c-0d4f2e8b7a31")
RECORD_VERSION = 1
DEFAULT_RELINK_DAYS = 14
PAGE_SIZE = 500
# Rows recorded although db._NON_ANALYST_SIGNALS lists them.
_KEEP_NON_ANALYST = DECISION_KEEP_KINDS

# provenance keys copied into inputs.
_PROV_KEYS = ("trigger", "regime_used", "implied_growth", "fundamentals_read", "brief_exposure",
              "gate_flags", "knowledge_coverage", "plan", "llm_call", "provenance_error")
# provenance keys copied into inputs.versions.
_VERSION_KEYS = ("model", "ollama_version", "model_parameters", "model_template_sha",
                 "model_quantization", "prompt_template_full_sha", "brief_template_full_sha",
                 "config", "config_env", "ollama_identity_at")
# provenance keys stored as columns (so not repeated in inputs).
_PROV_COLUMN_KEYS = ("code_sha", "head_sha", "prompt_template_sha", "brief_template_sha",
                     "prompt_sha", "model_digest", "signal_id")
_DECISION_KEYS = ("model_action", "model_conviction", "model_pass_to_risk", "final_action",
                  "final_conviction", "final_pass_to_risk", "gates", "catalyst_check",
                  "prior_verdicts_72h", "rule_shadows", "trim_shadow", "hold_fails_entry_rule",
                  "pending_entry", "udr_check", "udr_structural", "event_burst",
                  "nonfinite_replaced", "nonfinite_paths", "failure", "downgraded_reason", "malformed_fields",
                  "return_malformed", "malformed_sell_alert",
                  "held_review", "imputed_fields", "trend_template", "parse_mode")
# Scalar fields of decision.model_output kept as inputs.model_scores.
_MODEL_SCORE_KEYS = ("action", "conviction", "thesis_strength", "catalyst_quality",
                     "valuation_risk_reward", "data_completeness", "thesis_bucket",
                     "upside_downside_ratio", "expected_return_pct", "expected_downside_pct",
                     "pass_to_risk", "options_opportunity")

# Keys never copied at any depth: free text that can carry anything.
_DENY_EXACT = frozenset({"prompt", "brief", "situation_brief", "raw", "raw_reply",
                         "raw_reply_head", "reasoning", "text", "thesis_text", "philosophy_text",
                         "model_output", "positions_json", "positions"})
# Fragments that name an account figure, matched as substrings of the key
# lower-cased with separators removed, so snake_case, camelCase and squashed
# IBKR tags (TotalCashValue, cashBalance, EquityWithLoanValue, acctId,
# DailyPnL, mktValue) all match.
_DENY_SUBSTRINGS = ("cash", "balance", "equity", "account", "acct", "pnl", "netliq", "liquidation",
                    "liquidity", "funds", "margin", "mktvalue", "marketvalue", "buyingpower",
                    "portfoliovalue", "totalvalue", "positionvalue", "unrealized",
                    "networth", "capital", "notional", "positionsize", "dollars")
# Whole words only (as a substring "nav" would hit "unavailable").
_DENY_WORDS = frozenset({"nav"})
# The one subtree where an UPPER_SNAKE key with a scalar value is a config.py
# threshold (MAX_TRADE_PCT_OF_CASH_RESERVE), not an account figure.
_CONFIG_PATH = "versions.config"
# Values: IBKR account ids, and dollar amounts next to an account word in
# either order (covers the labels agents.analyst.redact_account_figures
# removes from the stored prompt: Portfolio Value, Cash Available, Buying
# Power, "| Value:").
# Individual (U), paper (DU) and advisor (F, DF) account ids.
# Any case (du1234567), and joined with "_" (acct_U1234567), where a \b
# boundary does not fire: no letter or digit before, no digit after. Glued
# straight onto a letter it is caught only after "acct" or "account"
# (acctU1234567, accountDU1234567); after any other letter it is left alone,
# since that would also match ISINs and codes such as US1234567890 or
# AU12345678.
_ACCOUNT_ID_RE = re.compile(
    r"(?i)(?:(?<![A-Za-z0-9])|(?<=acct)|(?<=account))(?:DU|DF|U|F)\d{6,9}(?![0-9])")
# A figure with no "$" after an account word, also inside a camelCase tag:
# NetLiquidation=99999, TotalCashValue: 5,000, "balance: 5,000 USD". Only
# numbers of 3+ digits (or comma-grouped), not followed by % or more digits.
_FIGURE_NO_DOLLAR_RE = re.compile(
    # Labels: the account words, and value/NAV only in account-qualified
    # forms (account value, total account value, portfolio value; NAV after
    # account/portfolio/my/our). "net liquidation value" is net\s*liq.
    # Company figures (enterprise, fair, book, market, deal value; a bare
    # NAV or "Value:") are not labels.
    r"(?i)((?:cash|balance|equity|net\s*liq|funds|margin|(?:account|portfolio)\s*value"
    r"|\b(?:account|portfolio|my|our)\s+nav\b)"
    r"\w{0,24}[^\d$\n\[]{0,12}?)"
    # A bare 19xx/20xx year (no '.', ',' or digit after it) is not a figure,
    # unless it is assigned (NetLiquidation=2025).
    r"(?<![\d.])(?:(?<==)|(?<==\s)|(?!(?:19|20)\d\d(?!\d|[.,]\d)))"
    # A comma-grouped figure always counts (cash 5,000/month, 7,500-ish); a
    # plain digit run followed by '-' or '/' and a digit is a date or id.
    # A '.' or ',' after the figure ends it (sentence punctuation) unless a
    # digit follows: "NetLiquidation=99999." and "cash 5,000, then" count.
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d%]|[.,]\d)|\d{3,}(?:\.\d+)?(?![\d%]|[.,]\d)(?![-/]\d))")
# Strings are cut to this many characters before redaction (bounds regex work).
_TEXT_CAP = 4096
# Public text (article title, link of the trigger): only account ids are
# redacted there. Matched on the full path, not the key name at any depth.
_PUBLIC_TEXT_PATHS = frozenset({"trigger.title", "trigger.url"})
_FIGURE_AFTER_WORD_RE = re.compile(
    r"(?i)(\b(?:cash|value|balance|equity|net\s*liq\w*|buying\s*power|funds|margin"
    r"|(?:account|portfolio|my|our)\s+nav)\b[^$\n]{0,24}?)"
    r"\$\s?[\d,]+(?:\.\d+)?[kmb]?")
_FIGURE_BEFORE_WORD_RE = re.compile(
    r"(?i)\$\s?[\d,]+(?:\.\d+)?[kmb]?(?=[^$\n]{0,24}?\b(?:cash|balance|equity|buying\s*power)\b)")
# Free-text caps, applied after redaction (by key name, or by exact path).
_ERROR_CAP = 120
_PATH_CAPS = {"decision.failure.reason": 120, "decision.catalyst_check.model_date": 40}
_SCORE_STRING_KEYS = ("action", "thesis_bucket")
_SCORE_STRING_CAP = 40


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def decision_uuid(signal_id: int) -> str:
    return str(uuid.uuid5(DECISION_NAMESPACE, f"public.signals:{int(signal_id)}"))


def _is_config_name(key: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]*", key))


def _denied(key: str, value, parent: str = "") -> bool:
    """Whether a key is dropped. `parent` is the path of the dict holding it."""
    k = str(key)
    low = k.lower()
    if low in _DENY_EXACT:
        return True
    if (parent == _CONFIG_PATH and _is_config_name(k)
            and (value is None or isinstance(value, (bool, int, float)))):
        return False
    if set(re.split(r"[^a-z0-9]+", low)) & _DENY_WORDS:
        return True
    squashed = re.sub(r"[^a-z0-9]", "", low)
    return any(p in squashed for p in _DENY_SUBSTRINGS)


def redact_text(s: str, figures: bool = True) -> str:
    """s (cut to _TEXT_CAP chars) without IBKR account ids and, unless
    figures is False, without figures next to an account word."""
    s = _ACCOUNT_ID_RE.sub("[REDACTED_ACCT]", s[:_TEXT_CAP])
    if not figures:
        return s
    s = _FIGURE_AFTER_WORD_RE.sub(r"\1$[REDACTED]", s)
    s = _FIGURE_BEFORE_WORD_RE.sub("$[REDACTED]", s)
    return _FIGURE_NO_DOLLAR_RE.sub(r"\1[REDACTED]", s)


def _cap_for(key: str, path: str) -> int | None:
    if path in _PATH_CAPS:
        return _PATH_CAPS[path]
    return _ERROR_CAP if str(key).lower().endswith("error") else None


def scrub(obj, path: str = "", dropped: list | None = None, key: str = ""):
    """A copy of obj without denied keys at any depth, with account ids and
    figures redacted in strings and error/reason text capped; each dropped
    key's path (or redacted string's path) is appended to `dropped`."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if _denied(k, v, path):
                if dropped is not None:
                    dropped.append(p)
                continue
            out[str(k)] = scrub(v, p, dropped, str(k))
        return out
    if isinstance(obj, (list, tuple)):
        return [scrub(v, f"{path}[{i}]", dropped, key) for i, v in enumerate(obj)]
    if isinstance(obj, str):
        if len(obj) > _TEXT_CAP and dropped is not None:
            dropped.append(f"{path} (truncated from {len(obj)} chars)")
        obj = obj[:_TEXT_CAP]
        red = redact_text(obj, figures=path not in _PUBLIC_TEXT_PATHS)
        if red != obj and dropped is not None:
            dropped.append(f"{path} (figure redacted)")
        cap = _cap_for(key, path)
        if cap is not None and len(red) > cap:
            if dropped is not None:
                dropped.append(f"{path} (truncated from {len(red)} chars)")
            red = red[:cap]
        return red
    return obj


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


def recover_inputs(provenance, decision, prompt_stored: bool | None = None,
                   degraded: bool | None = None) -> dict:
    """The structured inputs a record keeps, from signals.provenance and
    signals.decision. Pure."""
    prov, dec = _as_dict(provenance), _as_dict(decision)
    inputs = {"record_version": RECORD_VERSION, "prompt_stored": prompt_stored,
              "degraded": degraded}
    for k in _PROV_KEYS:
        if k in prov:
            inputs[k] = prov[k]
    versions = {k: prov[k] for k in _VERSION_KEYS if k in prov}
    if versions:
        inputs["versions"] = versions
    dsub = {k: dec[k] for k in _DECISION_KEYS if k in dec}
    if dsub:
        inputs["decision"] = dsub
    mo = dec.get("model_output")
    if isinstance(mo, dict):
        scores = {}
        for k in _MODEL_SCORE_KEYS:
            v = mo.get(k)
            if k not in mo:
                continue
            if v is None or isinstance(v, (bool, int, float)):
                scores[k] = v
            elif isinstance(v, str) and k in _SCORE_STRING_KEYS:
                scores[k] = v[:_SCORE_STRING_CAP]
        if isinstance(mo.get("sources"), list):
            scores["source_count"] = len(mo["sources"])
        inputs["model_scores"] = scores
    copied = set(_PROV_KEYS) | set(_VERSION_KEYS) | set(_PROV_COLUMN_KEYS)
    not_copied = sorted([f"provenance.{k}" for k in prov if k not in copied]
                        + [f"decision.{k}" for k in dec if k not in _DECISION_KEYS])
    dropped: list = []
    inputs = scrub(inputs, "", dropped)
    inputs["_not_copied"] = not_copied
    inputs["_dropped_keys"] = dropped
    return inputs


def _link(status: str, method: str | None = None, candidates=(), reason: str | None = None,
          **extra) -> dict:
    out = {"status": status, "method": method, "candidates": sorted(candidates), "reason": reason}
    out.update(extra)
    return out


def _int_or_none(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v.strip())
    return None


def _approval_claims(pa: dict) -> set[int]:
    """The signal ids an approval row names (its column and its provenance)."""
    return {s for s in (_int_or_none(pa.get("signal_id")), _int_or_none(pa.get("prov_signal_id")))
            if s is not None}


def resolve_approval(sig: dict, approvals: list[dict],
                     sha_signal_ids: dict[str, set[int]]) -> tuple[int | None, dict]:
    """The pending_approvals row this verdict led to, or None with the reason.
    `approvals`: the candidate rows read for this page; `sha_signal_ids`:
    prompt_sha -> the verdict signal ids that carry it."""
    sid = sig["signal_id"]
    claiming = [pa for pa in approvals if sid in _approval_claims(pa)]
    conflicted = [pa["id"] for pa in claiming if len(_approval_claims(pa)) > 1]
    if conflicted:
        return None, _link("conflict", "signal_id", [pa["id"] for pa in claiming],
                           "approval's signal_id column and provenance.signal_id differ")
    if len(claiming) == 1:
        pa = claiming[0]
        method = ("signal_id_column" if _int_or_none(pa.get("signal_id")) == sid
                  else "provenance_signal_id")
        return pa["id"], _link("matched", method, [pa["id"]], is_test=pa.get("is_test"))
    if len(claiming) > 1:
        return None, _link("ambiguous", "signal_id", [pa["id"] for pa in claiming],
                           "several approvals carry this signal_id")
    if sig.get("action") in _KEEP_NON_ANALYST:
        return None, _link("none", None, (), f"{sig.get('action')} is not a verdict")
    sha = sig.get("prompt_sha")
    if not sha:
        return None, _link("none", None, (), "no approval names this signal_id; no prompt_sha")
    cands = [pa for pa in approvals
             if pa.get("prov_prompt_sha") == sha and not _approval_claims(pa)
             and (pa.get("ticker") or "").strip().upper() == sig["ticker"]
             and (pa.get("trading_env") is None or sig.get("trading_env") is None
                  or pa.get("trading_env") == sig.get("trading_env"))]
    if not cands:
        return None, _link("none", None, (), "no approval names this signal_id or prompt_sha")
    if len(cands) > 1:
        return None, _link("ambiguous", "prompt_sha", [pa["id"] for pa in cands],
                           "several approvals share this prompt_sha")
    sharers = sha_signal_ids.get(sha, set()) | {sid}
    if len(sharers) > 1:
        return None, _link("ambiguous", "prompt_sha", [cands[0]["id"]],
                           f"prompt_sha shared by {len(sharers)} verdicts",
                           sharing_signal_ids=sorted(sharers))
    pa = cands[0]
    return pa["id"], _link("matched", "prompt_sha", [pa["id"]], is_test=pa.get("is_test"))


def resolve_children(approval_id: int | None, approval_link: dict, rows: list[dict],
                     kind: str) -> tuple[list[int], dict]:
    """orders or trade_outcomes rows of the matched approval."""
    if approval_id is None:
        return [], _link("skipped", None, (), f"approval {approval_link['status']}")
    ids = sorted(r["id"] for r in rows if r.get("approval_id") == approval_id)
    if not ids:
        return [], _link("none", "approval_id", (), f"no {kind} row for the approval")
    if kind == "trade_outcomes" and len(ids) > 1:
        return ids, _link("ambiguous", "approval_id", ids, "several outcome rows for one approval")
    return ids, _link("matched", "approval_id", ids)


def resolve_plan(sig: dict, plans: dict[int, dict]) -> tuple[int | None, int | None, dict]:
    plan = sig.get("plan")
    if not isinstance(plan, dict):
        return None, None, _link("none", None, (), "no provenance.plan")
    relation = plan.get("relation")
    out_ids, notes = [], []
    for key in ("plan_id", "new_plan_id"):
        pid = _int_or_none(plan.get(key))
        if pid is None:
            out_ids.append(None)
            continue
        row = plans.get(pid)
        if row is None:
            notes.append(f"{key} {pid} not in trade_plans")
            out_ids.append(None)
        elif (row.get("ticker") or "").strip().upper() != sig["ticker"]:
            notes.append(f"{key} {pid} is for {row.get('ticker')}")
            out_ids.append(None)
        elif (row.get("trading_env") and sig.get("trading_env")
              and row["trading_env"] != sig["trading_env"]):
            notes.append(f"{key} {pid} is for trading_env {row['trading_env']}")
            out_ids.append(None)
        else:
            out_ids.append(pid)
    named = [i for i in (_int_or_none(plan.get("plan_id")), _int_or_none(plan.get("new_plan_id")))
             if i is not None]
    if not named:
        return None, None, _link("none", None, (), f"plan relation {relation}", relation=relation)
    status = ("matched" if not notes else
              "conflict" if any("is for" in n for n in notes) else "missing")
    return out_ids[0], out_ids[1], _link(status, "provenance_plan_id", named,
                                         "; ".join(notes) or None, relation=relation)


def _parse_ts(v) -> datetime | None:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        t = datetime.fromisoformat(v.strip())
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _regime_source(source) -> tuple[str | None, str | None] | None:
    """('overall', None), ('gics_sector', 'XLK'), or None when the source does
    not name a stored row type (fresh, missing)."""
    if not isinstance(source, str) or not source or source == "fresh":
        return None
    head, _, etf = source.partition(":")
    return head, (etf or None)


def resolve_regime(regime_used, regimes: list[dict]) -> tuple[int | None, dict]:
    ru = regime_used if isinstance(regime_used, dict) else {}
    ts = _parse_ts(ru.get("timestamp"))
    if ts is None:
        return None, _link("none", None, (), "regime_used has no row timestamp",
                           source=ru.get("source"))
    rows = [r for r in regimes if _parse_ts(r.get("timestamp")) == ts]
    want = _regime_source(ru.get("source"))
    if want is not None:
        rtype, etf = want
        rows = [r for r in rows
                if (r.get("regime_type") or "overall") == rtype and (r.get("sector_etf") or None) == etf]
    ids = [r["id"] for r in rows]
    if len(ids) == 1:
        return ids[0], _link("matched", "timestamp_and_type", ids)
    if not ids:
        return None, _link("none", "timestamp_and_type", (),
                           "no market_regime_log row at that timestamp and type",
                           source=ru.get("source"))
    return None, _link("ambiguous", "timestamp_and_type", ids,
                       "several market_regime_log rows at that timestamp", source=ru.get("source"))


# Columns written, in order (decision_id .. builder_code_sha minus the times).
_COLS = ("decision_id", "signal_id", "as_of_ts", "ticker", "trading_env", "trigger", "action",
         "conviction", "code_sha", "head_sha", "prompt_template_sha", "brief_template_sha",
         "prompt_sha", "model", "model_digest", "regime_row_id", "approval_id", "order_ids",
         "trade_outcome_ids", "trade_plan_id", "trade_plan_new_id", "links", "inputs",
         "content_sha", "builder_code_sha")
# Not part of content_sha: who built it and the hash itself.
_NON_CONTENT = ("content_sha", "builder_code_sha")


def _json_default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    return str(o)


def content_sha(rec: dict) -> str:
    """Hash of the record's content. as_of_ts is taken in UTC, so the same
    instant read under another session time zone hashes the same."""
    body = {k: rec.get(k) for k in _COLS if k not in _NON_CONTENT}
    ts = _parse_ts(body.get("as_of_ts"))
    if ts is not None:
        body["as_of_ts"] = ts.astimezone(timezone.utc)
    body = json.dumps(body,
                      sort_keys=True, default=_json_default)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _short(v, n: int):
    return v[:n] if isinstance(v, str) else None


def build_record(sig: dict, ctx: dict, builder_sha: str | None) -> dict:
    """The h3d.decision_record row for one signal. `ctx` holds the page's
    join rows: approvals, sha_signal_ids, orders, outcomes, plans, regimes.
    Pure."""
    prov, dec = _as_dict(sig.get("provenance")), _as_dict(sig.get("decision"))
    trig = prov.get("trigger") if isinstance(prov.get("trigger"), dict) else {}
    s = dict(sig, prompt_sha=prov.get("prompt_sha"), plan=prov.get("plan"))
    approval_id, l_app = resolve_approval(s, ctx.get("approvals", []), ctx.get("sha_signal_ids", {}))
    order_ids, l_ord = resolve_children(approval_id, l_app, ctx.get("orders", []), "orders")
    outcome_ids, l_out = resolve_children(approval_id, l_app, ctx.get("outcomes", []),
                                          "trade_outcomes")
    plan_id, plan_new_id, l_plan = resolve_plan(s, ctx.get("plans", {}))
    regime_id, l_reg = resolve_regime(prov.get("regime_used"), ctx.get("regimes", []))
    conf = sig.get("confidence")
    rec = {
        "decision_id": decision_uuid(sig["signal_id"]),
        "signal_id": sig["signal_id"],
        "as_of_ts": sig["as_of_ts"],
        "ticker": sig["ticker"],
        "trading_env": sig.get("trading_env"),
        "trigger": _short(trig.get("path"), 40),
        "action": sig["action"],
        "conviction": round(float(conf) * 10) if conf is not None else None,
        "code_sha": _short(prov.get("code_sha"), 48),
        "head_sha": _short(prov.get("head_sha"), 48),
        "prompt_template_sha": _short(prov.get("prompt_template_sha"), 64),
        "brief_template_sha": _short(prov.get("brief_template_sha"), 64),
        "prompt_sha": _short(prov.get("prompt_sha"), 64),
        "model": _short(sig.get("model") or prov.get("model"), 80),
        "model_digest": _short(prov.get("model_digest"), 80),
        "regime_row_id": regime_id,
        "approval_id": approval_id,
        "order_ids": order_ids,
        "trade_outcome_ids": outcome_ids,
        "trade_plan_id": plan_id,
        "trade_plan_new_id": plan_new_id,
        "links": {"approval": l_app, "orders": l_ord, "trade_outcomes": l_out,
                  "trade_plan": l_plan, "regime": l_reg,
                  "model_source": "signals.model" if sig.get("model") else
                  ("provenance.model" if prov.get("model") else None)},
        "inputs": recover_inputs(prov, dec, sig.get("prompt_stored"), sig.get("degraded")),
        "builder_code_sha": builder_sha,
    }
    rec["content_sha"] = content_sha(rec)
    return rec


# ---------------------------------------------------------------------------
# Reads (public tables, READ ONLY)
# ---------------------------------------------------------------------------

CURSOR_SQL = "SELECT COALESCE(MAX(signal_id), 0) FROM h3d.decision_record"

# The prompt column is read only as a flag; reasoning only for the degraded
# prefix test. Neither text leaves the database.
SIGNALS_SQL = (
    "SELECT id, symbol, timestamp, signal, confidence, trading_env, model, provenance, decision, "
    "(prompt IS NOT NULL) AS prompt_stored, "
    "(trading_env IS NULL AND COALESCE(reasoning, '') LIKE %s) AS degraded "
    "FROM public.signals "
    "WHERE " + DECISION_RULE +  # verdict_rows.DECISION_RULE
    "AND id > %s AND (id > %s OR timestamp >= %s) "
    "ORDER BY id LIMIT %s"
)

APPROVALS_SQL = (
    "SELECT id, signal_id, ticker, trading_env, is_test, "
    "analyst_json->'provenance'->>'signal_id', analyst_json->'provenance'->>'prompt_sha' "
    "FROM public.pending_approvals "
    "WHERE signal_id = ANY(%s) OR analyst_json->'provenance'->>'signal_id' = ANY(%s) "
    "OR analyst_json->'provenance'->>'prompt_sha' = ANY(%s)"
)

# Verdict signals carrying each prompt_sha (for the prompt_sha join only).
SHA_SHARERS_SQL = (
    "SELECT provenance->>'prompt_sha', array_agg(id) FROM public.signals "
    "WHERE provenance->>'prompt_sha' = ANY(%s) AND signal NOT IN %s "
    "GROUP BY 1"
)

ORDERS_SQL = "SELECT id, approval_id FROM public.orders WHERE approval_id = ANY(%s)"
OUTCOMES_SQL = "SELECT id, approval_id FROM public.trade_outcomes WHERE approval_id = ANY(%s)"
PLANS_SQL = "SELECT id, ticker, trading_env FROM public.trade_plans WHERE id = ANY(%s)"
# First signal id inside the relink window: paging starts just below it.
WINDOW_START_SQL = "SELECT MIN(id) FROM public.signals WHERE timestamp >= %s"
REGIMES_SQL = ("SELECT id, timestamp, regime_type, sector_etf FROM public.market_regime_log "
               "WHERE timestamp = ANY(%s)")


def read_cursor(conn) -> int:
    """MAX(signal_id) already recorded; 0 when h3d.decision_record is absent."""
    from helios3d.schema.conn import read_only
    with read_only(conn) as cur:
        cur.execute("SELECT to_regclass('h3d.decision_record')")
        if cur.fetchone()[0] is None:
            return 0
        cur.execute(CURSOR_SQL)
        return int(cur.fetchone()[0] or 0)


def page_start(conn, cursor_id: int, relink_since: datetime) -> int:
    """The id paging starts after: min(cursor, first id in the relink window
    - 1), so an incremental run doesn't scan the table from id 0."""
    from helios3d.schema.conn import read_only
    with read_only(conn) as cur:
        cur.execute(WINDOW_START_SQL, (relink_since,))
        row = cur.fetchone()
    first = row[0] if row else None
    return cursor_id if first is None else min(cursor_id, int(first) - 1)


def fetch_signal_page(cur, after_id: int, cursor_id: int, relink_since: datetime,
                      limit: int = PAGE_SIZE) -> list[dict]:
    import db
    cur.execute(SIGNALS_SQL, (db.DEGRADED_SIGNAL_PREFIX + "%", tuple(db._NON_ANALYST_SIGNALS),
                              _KEEP_NON_ANALYST, after_id, cursor_id, relink_since, limit))
    out = []
    for (sid, sym, ts, sig, conf, env, model, prov, dec, prompt_stored, degraded) in cur.fetchall():
        out.append({"signal_id": int(sid), "ticker": str(sym).strip().upper(),
                    "as_of_ts": ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
                    "action": sig, "confidence": conf, "trading_env": env, "model": model,
                    "provenance": prov, "decision": dec, "prompt_stored": bool(prompt_stored),
                    "degraded": bool(degraded)})
    return out


def fetch_context(cur, sigs: list[dict]) -> dict:
    """The join rows for one page of signals."""
    ids = [s["signal_id"] for s in sigs]
    provs = [_as_dict(s.get("provenance")) for s in sigs]
    shas = sorted({p.get("prompt_sha") for p in provs if isinstance(p.get("prompt_sha"), str)})
    cur.execute(APPROVALS_SQL, (ids, [str(i) for i in ids], shas))
    approvals = [{"id": r[0], "signal_id": r[1], "ticker": r[2], "trading_env": r[3],
                  "is_test": r[4], "prov_signal_id": r[5], "prov_prompt_sha": r[6]}
                 for r in cur.fetchall()]
    sha_signal_ids: dict[str, set[int]] = {}
    sha_needed = sorted({pa["prov_prompt_sha"] for pa in approvals
                         if pa["prov_prompt_sha"] and not _approval_claims(pa)})
    if sha_needed:
        import db
        cur.execute(SHA_SHARERS_SQL, (sha_needed, tuple(db._NON_ANALYST_SIGNALS)))
        sha_signal_ids = {r[0]: {int(x) for x in (r[1] or [])} for r in cur.fetchall()}
    approval_ids = sorted({pa["id"] for pa in approvals})
    orders, outcomes = [], []
    if approval_ids:
        cur.execute(ORDERS_SQL, (approval_ids,))
        orders = [{"id": r[0], "approval_id": r[1]} for r in cur.fetchall()]
        cur.execute(OUTCOMES_SQL, (approval_ids,))
        outcomes = [{"id": r[0], "approval_id": r[1]} for r in cur.fetchall()]
    plan_ids = sorted({i for p in provs if isinstance(p.get("plan"), dict)
                       for i in (_int_or_none(p["plan"].get("plan_id")),
                                 _int_or_none(p["plan"].get("new_plan_id"))) if i is not None})
    plans = {}
    if plan_ids:
        cur.execute(PLANS_SQL, (plan_ids,))
        plans = {r[0]: {"id": r[0], "ticker": r[1], "trading_env": r[2]} for r in cur.fetchall()}
    stamps = sorted({t for p in provs for t in [_parse_ts(_as_dict(p.get("regime_used")).get("timestamp"))]
                     if t is not None})
    regimes = []
    if stamps:
        cur.execute(REGIMES_SQL, (stamps,))
        regimes = [{"id": r[0], "timestamp": r[1], "regime_type": r[2], "sector_etf": r[3]}
                   for r in cur.fetchall()]
    return {"approvals": approvals, "sha_signal_ids": sha_signal_ids, "orders": orders,
            "outcomes": outcomes, "plans": plans, "regimes": regimes}


# ---------------------------------------------------------------------------
# Write (h3d.decision_record only)
# ---------------------------------------------------------------------------

_CASTS = {"decision_id": "::uuid", "order_ids": "::integer[]", "trade_outcome_ids": "::integer[]",
          "links": "::jsonb", "inputs": "::jsonb"}
_UPDATABLE = [c for c in _COLS if c not in ("decision_id", "signal_id")]

UPSERT_SQL = (
    "INSERT INTO h3d.decision_record AS d (" + ", ".join(_COLS) + ", built_at, updated_at) VALUES ("
    + ", ".join(f"%s{_CASTS.get(c, '')}" for c in _COLS) + ", NOW(), NOW()) "
    "ON CONFLICT (signal_id) DO UPDATE SET "
    + ", ".join(f"{c}=EXCLUDED.{c}" for c in _UPDATABLE) + ", updated_at=NOW() "
    "WHERE d.content_sha IS DISTINCT FROM EXCLUDED.content_sha "
    "RETURNING (xmax = 0) AS inserted"
)


def _params(rec: dict) -> tuple:
    out = []
    for c in _COLS:
        v = rec[c]
        if c in ("links", "inputs"):
            v = json.dumps(v, sort_keys=True, default=_json_default)
        out.append(v)
    return tuple(out)


def upsert_records(conn, recs: list[dict]) -> dict:
    """Insert new records and rewrite changed ones, in one transaction.
    Returns counts of inserted, updated and unchanged."""
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    if not recs:
        return counts
    try:
        with conn.cursor() as cur:
            for rec in recs:
                cur.execute(UPSERT_SQL, _params(rec))
                row = cur.fetchone()
                if row is None:
                    counts["unchanged"] += 1
                elif row[0]:
                    counts["inserted"] += 1
                else:
                    counts["updated"] += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return counts


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def _tally(summary: dict, rec: dict) -> None:
    for name, link in rec["links"].items():
        if isinstance(link, dict):
            key = f"{name}:{link['status']}"
            summary["links"][key] = summary["links"].get(key, 0) + 1


def run(apply: bool = False, full: bool = False, relink_days: int = DEFAULT_RELINK_DAYS,
        now: datetime | None = None, conn=None, page_size: int = PAGE_SIZE) -> dict:
    """Build records for signals above the cursor (all of them with full)
    and for the last relink_days; upsert only when apply."""
    from helios3d.schema.conn import connection, read_only
    if conn is None:
        with connection() as c:
            return run(apply=apply, full=full, relink_days=relink_days, now=now, conn=c,
                       page_size=page_size)
    from helios3d.record.labels import code_sha
    now = now or datetime.now(timezone.utc)
    cursor_id = 0 if full else read_cursor(conn)
    relink_since = now - timedelta(days=max(0, relink_days))
    summary = {"mode": "apply" if apply else "dry-run", "cursor": cursor_id,
               "relink_since": relink_since.isoformat(timespec="seconds"), "signals": 0,
               "inserted": 0, "updated": 0, "unchanged": 0, "links": {},
               "migrations_applied": []}
    if apply:
        from helios3d.schema import migrate
        summary["migrations_applied"] = migrate.apply(conn)
    sha = code_sha()
    summary["builder_code_sha"] = sha
    after = 0 if full else page_start(conn, cursor_id, relink_since)
    summary["page_start"] = after
    sample = []
    while True:
        with read_only(conn) as cur:
            sigs = fetch_signal_page(cur, after, cursor_id, relink_since, page_size)
            if not sigs:
                break
            ctx = fetch_context(cur, sigs)
        recs = [build_record(s, ctx, sha) for s in sigs]
        for r in recs:
            _tally(summary, r)
        summary["signals"] += len(recs)
        if apply:
            for k, v in upsert_records(conn, recs).items():
                summary[k] += v
        elif len(sample) < 5:
            sample += [{k: r[k] for k in ("signal_id", "ticker", "action", "trigger",
                                          "approval_id", "order_ids", "regime_row_id")}
                       for r in recs[:5 - len(sample)]]
        after = sigs[-1]["signal_id"]
        if len(sigs) < page_size:
            break
    if not apply:
        summary["sample"] = sample
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Decision records for analyst verdicts (h3d).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="build and print; write nothing (default)")
    g.add_argument("--apply", action="store_true", help="upsert into h3d.decision_record")
    p.add_argument("--full", action="store_true", help="ignore the cursor; rebuild every signal")
    p.add_argument("--relink-days", type=int, default=DEFAULT_RELINK_DAYS,
                   help=f"also rebuild signals from the last N days (default {DEFAULT_RELINK_DAYS})")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run(apply=bool(args.apply), full=args.full, relink_days=args.relink_days)
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
