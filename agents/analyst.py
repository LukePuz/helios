"""
Analyst Agent — deep thesis evaluation using the local LLM.

Evaluates scanner candidates and event-triggered opportunities.
Produces a structured recommendation matching analyst_agent.md spec:
  action: OPEN | ADD | HOLD | TRIM | EXIT | WATCH | PASS
  conviction: 1-10 integer
  thesis_bucket: Compounder | Accelerator | Moonshot
  expectation_gap, catalyst, valuation, upside_downside_ratio, key_risks

Designed for 72B model reasoning quality; degrades gracefully on smaller models.
"""
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

try:
    from config import MIN_EXPECTED_RETURN_PCT_PER_WEEK, SETUP_GATES_ENABLED
except ImportError:
    MIN_EXPECTED_RETURN_PCT_PER_WEEK = 1.0
    SETUP_GATES_ENABLED = False

OLLAMA_HOST   = os.getenv("OLLAMA_HOST",    "http://localhost:11434")
ANALYST_MODEL = os.getenv("ANALYST_MODEL",  "qwen2.5:72b")

# qwen2.5 3b/7b/72b all support this context length natively — raised from
# the previous 8192, which risked silent truncation once briefs got large.
_NUM_CTX = 32768
_BRIEF_MAX_CHARS = 24_000  # ~6k tokens; the token-budget guard handles the rest


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SUBSCORE_LABELS = {
    "thesis_strength":       "thesis_strength",
    "catalyst_quality":      "catalyst_quality",
    "valuation_risk_reward": "valuation_risk_reward",
    "data_completeness":     "data_completeness",
}


def _build_track_record_text() -> str:
    """
    Realized win-rate and alpha-vs-SPY by conviction level AND by each 1-5
    sub-score, so the model can weigh its own stated confidence — and each
    individual component of it — against how it has actually performed.
    Returns "" when there isn't enough closed-trade history yet.

    The per-dimension breakdown is the actual point of having sub-scores at
    all: "conviction-8 calls win 55%" doesn't say why, but "catalyst_quality
    -5 calls win 70%, valuation_risk_reward-5 calls only win 45%" tells the
    model specifically which kind of judgment to trust more.
    """
    try:
        import db
        calibration = db.get_conviction_calibration()
    except Exception:
        return ""

    rows = [c for c in calibration if c.get("total")]
    if not rows:
        return ""

    lines = ["Historical calibration — realized outcome by conviction you stated at entry:"]
    for c in rows:
        alpha = c.get("avg_alpha_pct")
        alpha_str = f"{alpha:+.1%} alpha vs SPY" if alpha is not None else "alpha unknown"
        lines.append(
            f"  Conviction {c['conviction']}/10: {c['total']} closed trades, "
            f"{c['win_rate']:.0%} win rate, {alpha_str}"
        )

    for field, label in _SUBSCORE_LABELS.items():
        try:
            sub_rows = [r for r in db.get_subscore_calibration(field) if r.get("total")]
        except Exception:
            sub_rows = []
        if not sub_rows:
            continue
        lines.append(f"\nBy {label} (1-5):")
        for r in sub_rows:
            alpha = r.get("avg_alpha_pct")
            alpha_str = f"{alpha:+.1%} alpha vs SPY" if alpha is not None else "alpha unknown"
            lines.append(
                f"  {label}={r['score']}: {r['total']} closed trades, "
                f"{r['win_rate']:.0%} win rate, {alpha_str}"
            )

    lines.append(
        "\nWeigh this record when you state your own conviction and sub-scores — if past "
        "conviction-9 calls realized a much lower win rate than 9/10 implies, or one specific "
        "sub-score consistently overstates its dimension, that overconfidence should be "
        "corrected now, specifically on the dimension the data shows is miscalibrated."
    )
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Rough heuristic (~4 chars/token) — good enough for a budget guard, not exact."""
    return len(text) // 4


# Reserved for the model's own output (num_predict) plus a safety margin so
# the budget check doesn't cut it razor-thin.
_NUM_PREDICT = 1600  # was 1200; headroom now that action/conviction come last
_OUTPUT_TOKEN_RESERVE = _NUM_PREDICT + 300


_PROVENANCE_STATIC: dict | None = None


def _git_code_sha() -> dict:
    """Commit identity of the checkout, read once at import so it names the
    code this process loaded, not whatever is checked out when the first
    analysis runs (professor 0k, 2026-10-28).

    code_sha: last commit that touched anything outside research-log/ (from
    2026-10-28; before, it was HEAD, usually the research-log commit after
    the code commit). head_sha: HEAD. Both get -dirty if the checkout has
    uncommitted tracked changes outside research-log/ or untracked .py files (untracked modules
    running live was the C2 failure mode; professor C3-residual)."""
    import subprocess
    repo = str(Path(__file__).resolve().parent.parent)

    def git(*args):
        return subprocess.run(["git", "-C", repo, *args],
                              capture_output=True, text=True, timeout=5).stdout
    try:
        head = git("rev-parse", "--short=12", "HEAD").strip()
        code = git("log", "-1", "--format=%h", "--abbrev=12", "--", ".",
                   ":(exclude)research-log").strip()
        # Not stripped: porcelain's leading space is part of the XY column.
        status = git("status", "--porcelain").splitlines()
        dirty = "-dirty" if [l for l in status
                             if not l[3:].startswith("research-log/")
                             and (not l.startswith("??") or l.rstrip().endswith(".py"))] else ""
        return {"code_sha": (code + dirty) if code else None,
                "head_sha": (head + dirty) if head else None}
    except Exception:
        return {"code_sha": None, "head_sha": None}


_CODE_SHA = _git_code_sha()
_DISK_CHECK: dict = {"at": 0.0, "value": None}


def _code_mixed_check() -> dict:
    """HEAD on disk now vs the HEAD this process read at import. Modules
    imported later come from the files on disk, so a difference means this
    process may be running a mix of commits (professor M75, 2026-10-29: a
    pull without a restart). Cached 60 s; one git call per minute at most."""
    import time
    now = time.monotonic()
    if _DISK_CHECK["value"] is None or now - _DISK_CHECK["at"] > 60:
        disk_all = _git_code_sha()
        # code_sha (last commit outside research-log/, -dirty kept) so a
        # log-only pull isn't flagged and a hot edit is (professor M75 b, c).
        disk, boot = disk_all.get("code_sha"), _CODE_SHA.get("code_sha")
        _DISK_CHECK.update(at=now, value={
            "disk_head_sha": disk_all.get("head_sha"),
            "disk_code_sha": disk,
            "code_mixed": bool(disk and boot and disk != boot),
        })
    return _DISK_CHECK["value"]


def _rule_shadows(ticker: str, action: str, brief: str) -> dict | None:
    """Log-only checks of rules Luke wants watched before they can block
    (D13-D15, 10-29): a buy soon after selling the name, a buy with no
    earnings date on file, and a Moonshot ADD below the recorded entry
    (average down). Recorded on OPEN/ADD verdicts only. Never raises."""
    if action not in ("OPEN", "ADD"):
        return None
    out = {}
    try:
        import db
        from datetime import datetime, timezone
        sell = db.last_sell_trade(ticker, days=14)
        out["recent_exit_days"] = ((datetime.now(timezone.utc) - sell["timestamp"]).days
                                   if sell else None)
        out["recent_exit_reason"] = (sell["reason"] or "")[:80] if sell else None
        out["earnings_date_missing"] = db.get_next_earnings_row(ticker) is None
        stop = db.get_position_stop(ticker)
        from agents.plans import brief_price
        price = brief_price(brief)
        out["average_down_moonshot"] = bool(
            action == "ADD" and stop and stop.get("thesis_bucket") == "Moonshot"
            and price and price < stop["entry_price"])
    except Exception as e:
        out["error"] = str(e)[:120]
    return out


_TRIM_SHADOW_STREAK = 3   # reviews in a row with UDR below 1


def _trim_shadow(ticker: str, action: str, parsed: dict, brief: str) -> dict | None:
    """Log-only (audit A0100-3): a held name the model keeps at HOLD/WATCH
    while its own upside/downside is below 1 on several reviews in a row, or
    whose price has reached its plan target. Asymmetry is only checked at
    entry today, so a fully priced holding is never trimmed. Never raises."""
    if action not in ("HOLD", "WATCH"):
        return None
    try:
        import db
        from agents.plans import brief_price
        if db.get_position_stop(ticker) is None:
            return None
        udr = parsed.get("upside_downside_ratio")
        udr = float(udr) if isinstance(udr, (int, float)) else None
        history = [udr] + db.recent_model_udrs(ticker, _TRIM_SHADOW_STREAK - 1)
        streak = 0
        for v in history:
            if v is None or v >= 1:
                break
            streak += 1
        plan = db.get_active_plan(ticker)
        price = brief_price(brief)
        at_target = bool(plan and plan.get("target_price") and price and price >= plan["target_price"])
        return {"udr": udr, "udr_below_1_streak": streak, "at_plan_target": at_target,
                "would_trim_review": streak >= _TRIM_SHADOW_STREAK or at_target}
    except Exception as e:
        return {"error": str(e)[:120]}


def _udr_check(parsed: dict, brief: str) -> dict:
    """Log-only (audit A0700-5): the gate uses the UDR the model writes,
    which swings 1.1-2.9 on one name within minutes. Recompute it from the
    model's own % fields and from its dollar targets against the brief's
    price, and flag when the 2.0 gate would decide differently. Never raises."""
    import math

    def num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    try:
        from agents.plans import _dollars, brief_price
        model = num(parsed.get("upside_downside_ratio"))
        up, down = num(parsed.get("expected_return_pct")), num(parsed.get("expected_downside_pct"))
        from_pcts = round(up / abs(down), 2) if up is not None and down else None
        val = parsed.get("valuation") if isinstance(parsed.get("valuation"), dict) else {}
        target, floor = _dollars(val.get("bull_case_target")), _dollars(val.get("downside_scenario"))
        price = brief_price(brief)
        from_targets = (round((target - price) / (price - floor), 2)
                        if price and target and floor and target > price > floor else None)
        computed = [v for v in (from_pcts, from_targets) if v is not None]
        # Price at which the model's own target/floor give 2:1 (audit A2059-2):
        # a WATCH under the gate becomes "buy at X"; touches are scored offline.
        p2 = round((target + 2 * floor) / 3, 2) if target and floor and target > floor else None
        return {"model": model, "from_pcts": from_pcts, "from_targets": from_targets,
                "gate_disagrees": (model is not None and bool(computed)
                                   and any((v >= 2.0) != (model >= 2.0) for v in computed)),
                "price_for_2r": p2,
                "price_for_2r_pct": round((p2 / price - 1) * 100, 2) if p2 and price else None}
    except Exception as e:
        return {"error": str(e)[:120]}


_DEBOUNCE_SHADOW_MIN = 15


def _event_burst(ticker: str, trigger: dict | None, gates: list[str]) -> dict | None:
    """Log-only (audit A0700-2): events on one ticker are analysed one at a
    time, so several reports minutes apart never corroborate each other
    (five MSFT target raises in 7 minutes on 10-29, five separate WATCHes).
    Records the other event verdicts on this ticker in the last 15 minutes,
    whether a debounce would have merged this one with them, and whether the
    2-source gate downgraded it. Never raises."""
    if not trigger or trigger.get("path") != "event":
        return None
    try:
        import db
        prior = db.recent_event_verdicts(ticker, _DEBOUNCE_SHADOW_MIN)
        keys = {p.get("url") or p.get("title") for p in prior} - {None, ""}
        this = trigger.get("url") or trigger.get("title")
        return {"window_min": _DEBOUNCE_SHADOW_MIN,
                "window_end": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "published_at": trigger.get("published_at"), "prior_events": len(prior),
                "other_distinct_events": len(keys - {this}),
                "would_merge": bool(keys - {this}),
                "source_gate_hit": any(g.startswith("sources ") for g in gates)}
    except Exception as e:
        return {"error": str(e)[:120]}


def _brief_template_sha() -> str | None:
    try:
        import hashlib
        import inspect
        import context_engine
        return hashlib.sha256(inspect.getsource(context_engine).encode()).hexdigest()[:12]
    except Exception:
        return None


def _treatment_config() -> dict:
    import config
    return {k: getattr(config, k, None) for k in (
        "EARNINGS_ENTRY_BLACKOUT_DAYS", "ALLOW_ONE_SHARE_ABOVE_ALLOCATION", "SETUP_GATES_ENABLED",
        "ANALYST_MODEL", "SCREENER_MODEL", "MIN_EXPECTED_RETURN_PCT_PER_WEEK",
        "BRIEF_PRICE_STRUCTURE", "BRIEF_POSITION_CONTEXT", "REGIME_OVERALL_ONLY",
        # Added 10-29 (professor M96): switches that shipped that evening.
        "REGIME_COMPLETED_BARS", "QUIVER_ENABLED", "ONE_SHARE_MAX_PCT_OF_CASH",
        "TIER_MAX_CONCURRENT_POSITIONS", "MAX_TRADE_PCT_OF_CASH_RESERVE",
        "ETF_FUNDAMENTALS_CAP_EXEMPT", "PLAN_LEVEL_REVIEW_ENABLED",
        # Added 10-31 (A1031-0253-1): the month-only catalyst reading.
        "CATALYST_MONTH_MIDPOINT_ENABLED")}


def _provenance(prompt: str) -> dict:
    """What produced this decision (professor C3/M7, 2026-10-27): code commit
    (with -dirty if the checkout has uncommitted tracked changes), the
    model's digest rather than its tag, a hash of the prompt template, and a
    hash of the full rendered prompt. Stored with the result in
    pending_approvals.analyst_json and the thesis snapshots, so a decision
    can be tied to the exact code, model and prompt that made it."""
    global _PROVENANCE_STATIC
    if _PROVENANCE_STATIC is None:
        import hashlib
        import inspect
        try:
            tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).json().get("models", [])
            digest = next((m.get("digest") for m in tags if m.get("name") == ANALYST_MODEL), None)
        except Exception:
            digest = None
        _PROVENANCE_STATIC = {
            "code_sha": _CODE_SHA["code_sha"],
            "head_sha": _CODE_SHA["head_sha"],
            # Env-overridable treatment switches: the same code_sha can run
            # with different values (professor C3-residual, 2026-10-28).
            "config": _treatment_config(),
            "model": ANALYST_MODEL,
            "model_digest": digest[:12] if digest else None,
            "prompt_template_sha": hashlib.sha256(
                inspect.getsource(_build_prompt).encode()).hexdigest()[:12],
            # The brief builder changes what the model sees without touching
            # _build_prompt (professor C29, 10-29).
            "brief_template_sha": _brief_template_sha(),
        }
    if _PROVENANCE_STATIC.get("model_digest") is None:
        # The first /api/tags call failed (Ollama still loading): don't stamp
        # every decision of this process without a digest (professor M77).
        try:
            tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).json().get("models", [])
            digest = next((m.get("digest") for m in tags if m.get("name") == ANALYST_MODEL), None)
            if digest:
                _PROVENANCE_STATIC["model_digest"] = digest[:12]
        except Exception:
            pass
    import hashlib
    return {**_PROVENANCE_STATIC, **_code_mixed_check(),
            "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12]}


# The prefix caps in _build_prompt (kept out of it: its source is hashed into
# prompt_template_sha). Keep these equal to the slices there.
_THESIS_CAP, _PHILOSOPHY_CAP = 6000, 4500

# Sections of portfolio_philosophy.md every verdict gets, plus the stock's own
# bucket. The prefix cut kept 2,000 of 8,458 chars, so the exit, cash and
# bucket rules never reached the model (A1053-1; Luke D10, 10-29).
_PHILOSOPHY_KEEP = ("Primary Mission", "Core Beliefs", "Investment Style", "Concentration Philosophy",
                    "Cash Management", "Risk Tolerance", "When to Hold Through Drawdown",
                    "When to Exit Despite Thesis", "Conviction Tiers")
_BUCKET_HEADINGS = {"Compounder": "Compounders", "Accelerator": "Accelerators", "Moonshot": "Moonshots"}


def _philosophy_sections(text: str, role: str) -> str:
    """The kept sections, in file order, with their headings. Falls back to
    the full text if the file has none of them (the prompt's cap then applies)."""
    import re
    heads = [(m.start(), m.group(1).strip()) for m in re.finditer(r"^#{2,3}\s+(.+)$", text, re.M)]
    bucket = _BUCKET_HEADINGS.get(role, "Accelerators")
    parts = []
    for i, (pos, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        if title in _PHILOSOPHY_KEEP or title.startswith(bucket):
            parts.append(text[pos:end].strip())
    return "\n\n".join(parts) if parts else text


def _knowledge_coverage(ticker: str, thesis_text: str, philosophy_text: str,
                        prompt: str | None = None) -> dict:
    """Log-only record of how much of each knowledge file the prompt kept
    and which headings the prefix cut dropped (audit 10:53, A1053-1). With
    the final prompt, a heading counts as cut if it isn't in the prompt, so
    the later token-budget trim is covered too (professor M82)."""
    import re

    def one(text, cap):
        heads = [(m.start(), m.group(0).strip()) for m in re.finditer(r"^#{1,4}\s+(.+)$", text, re.M)]
        if prompt is not None:
            cut = [h.lstrip("#").strip() for _, h in heads if h not in prompt]
        else:
            cut = [h.lstrip("#").strip() for pos, h in heads if pos >= cap]
        return {"chars": len(text), "cap_kept": min(len(text), cap), "cut_headings": cut}
    try:
        from knowledge_loader import load_watchlist
        entry = load_watchlist().get(ticker.upper())
        thesis_file = entry["thesis_file"] if entry else "ai_infrastructure.md (fallback)"
    except Exception:
        thesis_file = None
    return {"thesis_file": thesis_file, "thesis": one(thesis_text, _THESIS_CAP),
            "philosophy": one(philosophy_text, _PHILOSOPHY_CAP)}


_ACCOUNT_FIGURE_RE = re.compile(
    r"((?:Portfolio Value|Cash Available|Buying Power):\s*|\| Value: )\$[\d,]+(?:\.\d+)?")


def redact_account_figures(prompt: str) -> str:
    """The stored copy of a prompt goes into research data: drop the account's
    dollar figures (portfolio value, cash, buying power, position values).
    prompt_sha is still taken from the unredacted prompt (audit 14:53,
    A1453-3). The model itself still sees them."""
    return _ACCOUNT_FIGURE_RE.sub(r"\1$[REDACTED]", prompt or "")


def _gate_flags(ticker: str, action: str, conviction: int, source_count: int,
                passes: bool, model_action: str | None = None, model_conviction: int | None = None,
                now=None) -> dict:
    """Log-only counters for gate questions that are Luke's (audits
    A1031-0253-2, A1031-0453-3/-4), on the final action and conviction:
    model_sell_pass_unconfirmed  a c>=5 TRIM/EXIT passing on the model's own
                                 flag, with no confirming verdict;
    rubric_gate_conflict         the model's own OPEN/ADD at c7 citing >=2
                                 sources, before the gates (the rubric's 7 is
                                 one piece of evidence; review of 7756153);
    verdicts_today               earlier verdicts on the ticker this ET day,
                                 i.e. how many draws came before this one."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    out = {"model_sell_pass_unconfirmed": bool(passes and action in ("TRIM", "EXIT")
                                               and conviction > LOW_CONVICTION_SELL_MAX),
           "rubric_gate_conflict": model_action in ("OPEN", "ADD") and model_conviction == 7
                                   and source_count >= 2}
    try:
        import db
        et = ZoneInfo("America/New_York")
        today = (now or datetime.now(timezone.utc)).astimezone(et).date()
        out["verdicts_today"] = sum(
            1 for v in db.get_recent_verdicts(ticker, hours=24, limit=200, strict=True)
            if v.get("timestamp") and v["timestamp"].astimezone(et).date() == today)
    except Exception:
        out["verdicts_today"] = None
    return out


def _brief_exposure(ticker: str, prompt: str) -> dict:
    """Log-only: what the brief showed the model about its own position and
    past verdicts, and whether the broker holds the ticker (audit A1031-0453
    -1/-2). Read from the prompt text, so it is what the model saw. Since
    BRIEF_POSITION_CONTEXT, repeats on held names see their last verdicts and
    aren't independent draws; the section keys on a stop row, not the broker.
    held_broker is None when unknown; held_broker_reason says why."""
    out = {"position_context_shown": False, "prior_verdicts_shown": []}
    # Only inside the position section: event text above it is raw news.
    in_section = False
    for line in (prompt or "").splitlines():
        if line.startswith("=== POSITION CONTEXT"):
            in_section = out["position_context_shown"] = True
        elif line.startswith("==="):
            in_section = False
        elif in_section and line.startswith("Recent verdicts: "):
            out["prior_verdicts_shown"] = [v.strip() for v in line[len("Recent verdicts: "):].split(";")
                                           if v.strip()]
    try:
        import db
        out["has_stop_row"] = db.get_position_stop(ticker, strict=True) is not None
    except Exception:
        out["has_stop_row"] = None
    try:
        import tasks
        out["held_broker"], out["held_broker_reason"], _ = tasks._broker_held(ticker)
    except Exception:
        out["held_broker"], out["held_broker_reason"] = None, "error"
    return out


def _decision_inputs_record(ticker: str, regime_packet: dict | None) -> dict:
    """Log-only, after the model call: which regime row this verdict used
    (professor C25; a sector row here is A0853-1) and the nightly reverse-DCF
    numbers it was not shown, to compare with its own current_price_implies
    text (audit 12:53, A1253-2)."""
    out = {}
    rp = regime_packet or {}
    out["regime_used"] = {
        "label": rp.get("regime"), "score": rp.get("score"),
        "source": rp.get("regime_source") or ("fresh" if rp.get("agent") else None),
        "timestamp": str(rp.get("timestamp")) if rp.get("timestamp") else None,
        "last_bar_date": (rp.get("signals") or {}).get("last_bar_date"),
    }
    try:
        import db
        ladder = db.get_trade_ladder(ticker) or {}
        out["implied_growth"] = {
            k: ladder.get(k) for k in ("implied_growth_rate", "growth_gap_vs_actual",
                                       "growth_assumption_flag")}
        if ladder.get("computed_at"):
            out["implied_growth"]["as_of"] = str(ladder["computed_at"])
    except Exception as e:
        log.debug(f"[{ticker}] implied growth record failed: {e}")
    return out


def _build_prompt(
    candidate: dict,
    situation_brief: str,
    thesis_text: str,
    regime_packet: dict,
    philosophy_text: str,
    num_ctx: int = _NUM_CTX,
) -> str:
    ticker = candidate.get("ticker", "UNKNOWN")
    role   = candidate.get("role",   "Accelerator")
    tier   = candidate.get("tier",   "Watch")
    sector = candidate.get("sector", "")

    regime    = regime_packet.get("regime",   "Neutral")
    reg_score = regime_packet.get("score",    50)

    _markov = regime_packet.get("signals", {}).get("markov", {})
    markov_line = ""
    if _markov.get("state") is not None and _markov.get("stickiness") is not None and _markov.get("signal") is not None:
        markov_line = (
            f"\nSPY Markov state: {_markov['state']} "
            f"(stickiness {_markov['stickiness']:.0%}, directional signal {_markov['signal']:+.2f})"
        )

    # Generous caps — num_ctx was raised from 8192, and the budget loop below
    # trims further if the assembled prompt still doesn't fit.
    thesis_excerpt      = thesis_text[:6000]
    philosophy_excerpt  = philosophy_text[:4500]
    # Was a silent 8,000-char cut that dropped the brief's later sections
    # (fundamentals, memory) before the token guard below ever ran (limitations
    # audit 2026-10-27, finding 12). The guard trims by priority and logs it.
    if situation_brief and len(situation_brief) > _BRIEF_MAX_CHARS:
        log.warning(f"[{ticker}] Situation brief {len(situation_brief):,} chars, cut to {_BRIEF_MAX_CHARS:,}")
    brief_text          = situation_brief[:_BRIEF_MAX_CHARS] if situation_brief else "No additional context available."
    log.info(f"[{ticker}] Situation brief: {len(situation_brief or ''):,} chars")
    track_record        = _build_track_record_text()

    def _render() -> str:
        track_record_section = f"\n=== YOUR TRACK RECORD ===\n{track_record}\n" if track_record else ""
        return f"""You are a disciplined investment analyst evaluating whether to open, add, hold, trim, exit, or pass on a stock position.

=== PORTFOLIO PHILOSOPHY (excerpt) ===
{philosophy_excerpt}

=== SECTOR THESIS ===
{thesis_excerpt}

=== MARKET REGIME ===
Current Regime: {regime} (score: {reg_score}/100){markov_line}
{regime_packet.get("reasoning", "")[:500]}
{track_record_section}
=== CANDIDATE DETAILS ===
Ticker: {ticker}
Portfolio Role: {role}
Conviction Tier: {tier}
Sector: {sector}
Scanner composite score: {candidate.get("composite_score", 0)}/100

=== SITUATION BRIEF ===
{brief_text}

=== YOUR TASK ===
Evaluate {ticker} and produce a structured investment recommendation.

Answer these questions in order:
1. Which sector thesis does this fit? Is the thesis intact or has it weakened?
2. What does the current price IMPLY the market expects? What is MORE LIKELY to happen?
3. What is the specific catalyst? When? What happens if it disappoints?
4. Is the upside/downside ratio at least 2:1?
5. What are the top 2-3 risks?

Valuation grounding: if the situation brief includes a FUNDAMENTALS section
with real numbers, derive "current_multiple", "peer_range", and the
upside/downside estimates FROM THOSE FIGURES — do not invent a multiple.
If the FUNDAMENTALS section says data isn't cached yet, say so explicitly
in your reasoning and cap conviction at 6 for OPEN/ADD, since the valuation
call would otherwise be an unsupported guess.

Every dollar figure you write — bull_case_target, downside_scenario, any
price mentioned in reasoning or thesis_summary — MUST be computed relative
to the "Current Price" stated at the top of the situation brief, never an
assumed or remembered price. If Current Price is marked UNKNOWN, do not
write a dollar figure anywhere in your output; describe valuation only in
relative terms (e.g. "trades below peer multiples") instead of a price.

Conviction scale — each number below is a distinct bar, not a paired range.
Count the citable, independent pieces of evidence in the brief that support
an edge (a specific mispricing, an under-appreciated catalyst) — do not
assign a score by how confident the narrative sounds:
  1: Thesis is directly contradicted by evidence in the brief. Do not hold.
  2: Thesis mostly broken — only a remote chance the story still plays out.
  3: Speculative — a plausible story but zero concrete supporting evidence in the brief yet.
  4: Speculative, but with exactly one concrete supporting data point; still mostly narrative.
  5: Solid, thesis intact, fairly priced — no identifiable edge over consensus. The default for "good company, no edge."
  6: Same as 5, plus one minor edge that doesn't yet rise to a full mispricing case (e.g. a catalyst that isn't confirmed yet, or a small valuation discount with no clear reason it closes).
  7: Exactly ONE clear, citable mispricing or edge, backed by a single hard piece of evidence from the brief, with a dated catalyst.
  8: TWO OR MORE independent, citable pieces of evidence supporting the same mispricing, a dated catalyst, and a quantified upside/downside.
  9: Overwhelming evidence — three or more independent confirmations, asymmetric payoff, near-term dated catalyst, no meaningful counter-evidence in the brief.
  10: Everything in 9, plus a specific, explainable reason the market hasn't already corrected this (a structural or temporary reason, not "it just hasn't happened yet").
Do not default to 7 as a safe middle score, and do not aim for any particular
number to "qualify" for a decision — score honestly per the table above and
let downstream position-sizing logic decide what's actionable. If you find
yourself about to write 7 out of habit, recount the actual evidence first.

In addition to the overall conviction, rate these four components separately,
each 1-5 — conviction should be a defensible synthesis of these, not an
independent gut feel. Keep each scale coarse (1-5, not finer): the point is
that each dimension can be individually tracked against real outcomes later,
which needs an honest, differentiated rating on each axis rather than
four copies of the same overall confidence.
  thesis_strength (1-5): 1 = generic story anyone could tell about this sector; 5 = a specific, differentiated insight that isn't obviously already priced in.
  catalyst_quality (1-5): 1 = no concrete catalyst, or timing is vague/ongoing; 5 = a specific, dated event with a clear outcome that will confirm or break the thesis.
  valuation_risk_reward (1-5): 1 = poor risk/reward even if the thesis plays out exactly; 5 = exceptional asymmetry. Should track your upside_downside_ratio, not contradict it.
  data_completeness (1-5): 1 = little concrete evidence in the brief or fundamentals, mostly inference; 5 = comprehensive, fresh, well-sourced data backing the call.

Then output ONLY valid JSON in exactly this format (no markdown, no explanation outside the JSON).
Fill the fields in the order shown: the evidence, scenarios, valuation, risks and
reasoning come first, and your scores and action come last, decided from what you
wrote above them:

{{
  "sources": [
    {{"type": "news or edgar_filing or options_flow or insider_trade or congressional_trade or analyst_rating or fundamentals_data or price_action or arxiv_paper or other", "detail": "The specific fact this source contributes -- not a repeat of the catalyst description"}}
  ],
  "expectation_gap": {{
    "bull_case": "What needs to happen for upside",
    "current_price_implies": "What growth/outcome is already priced in",
    "upside_exists_if": "Condition under which upside materializes",
    "primary_risk": "Single most likely thing that makes this wrong"
  }},
  "catalyst": {{
    "event": "Description of catalyst",
    "date": "The MOST PRECISE date you actually know: YYYY-MM-DD if a specific day is confirmed (e.g. a scheduled earnings date, or 'the next session' after a report you're reading right now), YYYY-MM if only the month is known, or 'ongoing' if there's no defined timing at all. Use day-level precision whenever the brief supports it -- 'YYYY-MM' for something you actually know happens tomorrow understates how soon it is.",
    "expected_upside_if_positive": "+X to +Y%",
    "expected_downside_if_negative": "-X to -Y%",
    "is_priced_in": false
  }},
  "valuation": {{
    "current_multiple": "Xx NTM earnings or revenue",
    "peer_range": "Range for comparable peers",
    "bull_case_target": "$X (12 months)",
    "downside_scenario": "$X"
  }},
  "expected_return_pct": <float -- your single-number bull-case % upside from the Current Price to bull_case_target, e.g. 25.0 for +25%. Same figure as catalyst.expected_upside_if_positive / valuation.bull_case_target, just as one clean number instead of a range string, so it can be tracked against the actual outcome later.>,
  "expected_downside_pct": <float -- same idea for the downside_scenario, as a negative number, e.g. -15.0>,
  "upside_downside_ratio": <your computed ratio: bull-case upside percent divided by downside percent>,
  "key_risks": ["Risk 1", "Risk 2", "Risk 3"],
  "reasoning": "2-3 sentence narrative explaining the recommendation, citing specific numbers and their source per the rule below.",
  "thesis_bucket": "Compounder or Accelerator or Moonshot",
  "thesis_summary": "One sentence why this position exists.",
  "thesis_strength": <integer 1-5>,
  "catalyst_quality": <integer 1-5>,
  "valuation_risk_reward": <integer 1-5>,
  "data_completeness": <integer 1-5>,
  "conviction": <integer 1-10 per the scale above>,
  "action": "OPEN or ADD or HOLD or TRIM or EXIT or WATCH or PASS",
  "options_opportunity": false,
  "options_notes": "Brief note or null",
  "pass_to_risk": true
}}

Rules:
- Set pass_to_risk to true ONLY if action is OPEN or ADD AND conviction >= 5 AND upside_downside_ratio >= 2.0. For TRIM or EXIT, conviction is your conviction in the thesis (a low number means it is weakening or broken); the system decides pass_to_risk for low-conviction sells itself
- For Moonshot tickers: conviction >= 7 required to recommend OPEN; never exceed 2% position
- Score conviction on the evidence alone, regardless of regime — regime-based selectivity is applied downstream by position sizing, not by inflating or suppressing your stated score
{'- Your stated conviction has a track record above — calibrate against it, not against how confident you feel' if track_record else ''}
- reasoning and thesis_summary must be concrete, not persuasive prose: cite actual figures from the SITUATION BRIEF or FUNDAMENTALS section (a growth rate, a multiple, a dollar figure, a date) and name where each came from (e.g. "per Q2 fundamentals", "per the [source] article in the brief"). A sentence with no number and no named source is not acceptable reasoning — if the brief genuinely has no hard data to cite, say that explicitly instead of writing confident-sounding prose to fill the gap.
- List every genuinely distinct piece of evidence you actually relied on in "sources" — not a single article's opinion restated in your own words as if it were confirmation. Two sources of the same type (e.g. two separate news articles independently reporting the same event) DO count as two if they are actually distinct reports, not the same fact cited twice. List real items only — never pad this list to clear the threshold below; an empty or short list is correct and expected when the brief genuinely only supports one piece of evidence, and OPEN/ADD will be downgraded to WATCH in that case rather than the call being forced through.
- Output ONLY the JSON object, nothing else
"""

    prompt = _render()

    # Token-budget guard: if the assembled prompt won't fit num_ctx minus the
    # output reserve, trim the lowest-priority sections first — the situation
    # brief's tail (context_engine builds it with the most important sections
    # first, so its tail is the least critical), then the thesis excerpt,
    # then the philosophy excerpt. Every trim is logged; nothing is silently
    # truncated by Ollama without us knowing.
    budget = num_ctx - _OUTPUT_TOKEN_RESERVE
    est = _estimate_tokens(prompt)
    if est > budget:
        original_est = est
        for _ in range(20):
            if est <= budget:
                break
            if len(brief_text) > 500:
                brief_text = brief_text[: int(len(brief_text) * 0.75)]
            elif len(thesis_excerpt) > 500:
                thesis_excerpt = thesis_excerpt[: int(len(thesis_excerpt) * 0.75)]
            elif len(philosophy_excerpt) > 300:
                philosophy_excerpt = philosophy_excerpt[: int(len(philosophy_excerpt) * 0.75)]
            else:
                break
            prompt = _render()
            est = _estimate_tokens(prompt)
        log.warning(
            f"[{ticker}] Prompt trimmed to fit context budget: "
            f"~{original_est} -> ~{est} tokens (budget {budget})"
        )

    return prompt


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _conviction_distribution(tokens) -> dict | None:
    """The model's probability over the conviction digit it wrote, from the
    same call's top-5 token logprobs: an expected value and P(>=7) instead of
    one integer that clusters at 5 and 7 (SEF research R5, after G-Eval).
    Log-only. None when the field or logprobs are missing."""
    import math
    if not tokens:
        return None
    text = ""
    for i, tok in enumerate(tokens):
        if re.search(r'"conviction"\s*:\s*$', text) and (tok.get("token") or "").strip():
            probs = {}
            for alt in tok.get("top_logprobs") or [tok]:
                t = (alt.get("token") or "").strip()
                if t.isdigit() and 1 <= int(t) <= 9:
                    probs[int(t)] = probs.get(int(t), 0.0) + math.exp(alt.get("logprob", -99))
            total = sum(probs.values())
            if not total:
                return None
            nxt = tokens[i + 1].get("token", "") if i + 1 < len(tokens) else ""
            return {
                "chosen": (tok.get("token") or "").strip() + ("0" if nxt.startswith("0") else ""),
                "expected": round(sum(v * p for v, p in probs.items()) / total, 3),
                "p_ge7": round(sum(p for v, p in probs.items() if v >= 7) / total, 3),
                "probs": {str(v): round(p / total, 3) for v, p in sorted(probs.items())},
                "mass": round(total, 3),     # share of probability on 1-9 digits
                # A 10 is two tokens; its "1" counts as 1 above (review 5 #3).
                "first_digit_only": nxt.startswith("0"),
            }
        text += tok.get("token") or ""
    return None


def _call_ollama(prompt: str, num_ctx: int = _NUM_CTX) -> str | None:
    _LAST_CALL_STATS.clear()  # a failed call must not inherit the last one's counts
    payload = {
        "model":   ANALYST_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {
            "temperature":    0.2,
            "num_predict":    _NUM_PREDICT,
            "num_ctx":        num_ctx,
            "repeat_penalty": 1.1,
        },
        # Token probabilities, for the log-only conviction distribution
        # (SEF research R5, 10-29). Doesn't change sampling.
        "logprobs":     True,
        "top_logprobs": 5,
    }
    try:
        # Real 72B completions on this hardware routinely land at 182-195s
        # (CPU-offloaded portion of the model, ~47GB not fitting in VRAM) —
        # the old 180s timeout was cutting off almost every call right at
        # the finish line, silently falling back to a stub recommendation
        # (conviction=1, pass_to_risk=False) on ~100% of calls for days
        # (found 2026-07-17 investigating an unrelated chat feature — grep
        # helios logs for "Stub recommendation" going back to 07-15).
        # 400s gives real headroom above the observed ~195s ceiling.
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=400,
        )
        resp.raise_for_status()
        _LAST_CALL_FAILURE["kind"] = None
        body = resp.json()
        # What Ollama actually did with the prompt (professor C26): if
        # prompt_eval_count reaches num_ctx, the head of the prompt (the
        # instructions and schema) was silently dropped.
        _LAST_CALL_STATS.update({k: body.get(k) for k in (
            "prompt_eval_count", "eval_count", "done_reason", "total_duration")})
        _LAST_CALL_STATS["num_ctx"] = num_ctx
        try:   # log-only: must never fail a finished call (review 5 #3)
            _LAST_CALL_STATS["conviction_logprobs"] = _conviction_distribution(body.get("logprobs"))
        except Exception as e:
            _LAST_CALL_STATS["conviction_logprobs"] = None
            log.debug(f"conviction logprobs unreadable: {e}")
        if body.get("done_reason") == "length":
            log.warning(f"Ollama reply hit num_predict ({_NUM_PREDICT}); the JSON may be cut off")
        return body.get("response", "")
    except requests.exceptions.Timeout:
        # Was reported as "unreachable" (limitations audit 12:53, #8).
        _LAST_CALL_FAILURE["kind"] = "timeout"
        log.warning(f"Ollama call timed out after 400 s at {OLLAMA_HOST}")
        return None
    except requests.exceptions.ConnectionError:
        _LAST_CALL_FAILURE["kind"] = "unreachable"
        log.warning(f"Ollama not reachable at {OLLAMA_HOST}")
        return None
    except Exception as e:
        _LAST_CALL_FAILURE["kind"] = "error"
        log.warning(f"Ollama call failed: {e}")
        return None


# Why the last _call_ollama returned None, for the stub's reason text.
_LAST_CALL_FAILURE: dict = {"kind": None}
# Token counts from the last successful call, recorded in provenance (C26).
_LAST_CALL_STATS: dict = {}
_STUB_REASONS = {"timeout": "Ollama timed out", "unreachable": "Ollama unreachable",
                 "error": "Ollama call failed"}


def count_distinct_sources(sources: list) -> int:
    """Pure helper for the multi-source corroboration gate (2026-08-30,
    user request) -- deduplicates the model's "sources" list on
    (type, detail-prefix) so it can't clear the >=2 threshold in analyze()
    by listing the same fact twice under different wording. Extracted as
    its own function (same pattern as agents/chase_risk.py and
    agents/price_action.py) so the counting logic is unit-testable without
    mocking the Ollama call."""
    return len({
        (s.get("type"), (s.get("detail") or "")[:80])
        for s in (sources or []) if isinstance(s, dict) and s.get("type")
    })


def _prior_verdicts(ticker: str) -> list | None:
    """Shadow record for buy verdicts (limitations audit 2026-10-28 14:53,
    #1b): the ticker's verdicts in the previous 72 h, so whether a buy agreed
    with recent passes can be measured later. Logged only; nothing gates on
    it. Read before this verdict is written, so it excludes itself."""
    try:
        import db
        return [{"t": str(v["timestamp"])[:16], "signal": v["signal"],
                 "c": v["conviction"], "udr": v["udr"]} for v in db.get_recent_verdicts(ticker)]
    except Exception:
        return None


def _catalyst_check(ticker: str, parsed: dict) -> dict:
    """Shadow record of how the model's catalyst date relates to the calendar
    (limitations audit 2026-10-28 14:53, #5). Logged only; the rate gate
    is unchanged. past_or_today: the date floors to 1 day in
    compute_weeks_to_catalyst, the most favourable rate. month_only: "YYYY-MM",
    read as the 15th."""
    cat = parsed.get("catalyst") or {}
    raw = str(cat.get("date") or "").strip()
    out = {"model_date": raw or None, "past_or_today": None,
           "month_only": bool(re.fullmatch(r"\d{4}-\d{2}", raw)),
           "calendar_next_earnings": None, "days_model_vs_calendar": None}
    try:
        d = None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            d = datetime.strptime(raw, "%Y-%m-%d").date()
        elif out["month_only"]:
            d = datetime.strptime(raw + "-15", "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        if d is not None:
            out["past_or_today"] = d <= today
        import db
        nxt = db.get_next_earnings(ticker)
        if nxt is not None:
            out["calendar_next_earnings"] = str(nxt)
            if d is not None:
                out["days_model_vs_calendar"] = (d - nxt).days
    except Exception as e:
        out["error"] = str(e)[:120]
    return out


def compute_weeks_to_catalyst(catalyst_date: str | None, as_of: datetime | None = None,
                              month_midpoint: bool | None = None) -> float | None:
    """Pure helper for the minimum-expected-return-rate gate (2026-08-30,
    user request). Parses the model's catalyst.date field, trying the
    most precise format first:
      - "YYYY-MM-DD": exact date.
      - "YYYY-MM": month known, day unknown -- read as the 1st (strptime's
        default), or the 15th when month_midpoint (default:
        CATALYST_MONTH_MIDPOINT_ENABLED, off) is set (audit A1031-0253-1).
      - "ongoing", empty, or anything unparseable: no defined timeframe
        at all -- returns None. Callers must treat None as "can't compute
        a rate," not as passing or failing the gate on its own; analyze()
        chooses to fail closed (see its own comment) since "ongoing" is
        also literally what catalyst_quality's own 1/5 rubric calls a
        weak catalyst.
    A catalyst dated today or in the past is floored to 1 day (1/7 week)
    rather than returning zero or a negative number -- "reading an
    earnings report and expecting a move the next session" is meant to
    describe an imminent, not-yet-priced-in catalyst, not a broken one;
    the floor keeps the resulting rate large and finite instead of
    infinite or negative."""
    if not catalyst_date:
        return None
    text = catalyst_date.strip().lower()
    if text == "ongoing":
        return None
    now = as_of or datetime.now(timezone.utc)

    parsed = None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            parsed = datetime.strptime(catalyst_date.strip(), fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        return None
    if month_midpoint is None:
        try:
            from config import CATALYST_MONTH_MIDPOINT_ENABLED as month_midpoint
        except ImportError:
            month_midpoint = False
    if month_midpoint and re.fullmatch(r"\d{4}-\d{1,2}", catalyst_date.strip()):
        parsed = parsed.replace(day=15)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    days = (parsed - now).total_seconds() / 86400
    return max(days, 1.0) / 7.0


def expected_return_per_week(expected_return_pct: float | None, weeks_to_catalyst: float | None) -> float | None:
    """Pure helper: the rate the minimum-expected-return-rate gate
    actually checks. Returns None (not a default of 0 or an exception)
    if either input is missing -- callers decide how to treat that,
    same "None means can't compute, don't guess" posture as
    compute_weeks_to_catalyst above."""
    if expected_return_pct is None or weeks_to_catalyst is None or weeks_to_catalyst <= 0:
        return None
    return expected_return_pct / weeks_to_catalyst


# ---------------------------------------------------------------------------
# JSON extraction (robust to 3B model quirks)
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None

    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: pull individual fields from malformed output
    log.warning("JSON parse failed — attempting field extraction")
    extracted: dict = {}
    for field, pattern in [
        ("action",        r'"action"\s*:\s*"(\w+)"'),
        ("conviction",    r'"conviction"\s*:\s*(\d+)'),
        ("thesis_strength", r'"thesis_strength"\s*:\s*(\d+)'),
        ("catalyst_quality", r'"catalyst_quality"\s*:\s*(\d+)'),
        ("valuation_risk_reward", r'"valuation_risk_reward"\s*:\s*(\d+)'),
        ("data_completeness", r'"data_completeness"\s*:\s*(\d+)'),
        ("thesis_bucket", r'"thesis_bucket"\s*:\s*"([^"]+)"'),
        ("reasoning",     r'"reasoning"\s*:\s*"([^"]+)"'),
        ("pass_to_risk",  r'"pass_to_risk"\s*:\s*(true|false)'),
        ("upside_downside_ratio", r'"upside_downside_ratio"\s*:\s*([0-9.]+)'),
        ("expected_return_pct", r'"expected_return_pct"\s*:\s*(-?[0-9.]+)'),
        ("expected_downside_pct", r'"expected_downside_pct"\s*:\s*(-?[0-9.]+)'),
    ]:
        m = re.search(pattern, cleaned, re.IGNORECASE)
        if m:
            val = m.group(1)
            if field in ("conviction", "thesis_strength", "catalyst_quality",
                         "valuation_risk_reward", "data_completeness"):
                extracted[field] = int(val)
            elif field == "pass_to_risk":
                extracted[field] = val.lower() == "true"
            elif field in ("upside_downside_ratio", "expected_return_pct", "expected_downside_pct"):
                extracted[field] = float(val)
            else:
                extracted[field] = val

    if extracted:
        extracted["_salvaged"] = True  # see analyze(): never routes an action
    return extracted if extracted else None


# ---------------------------------------------------------------------------
# Stub for unavailable model
# ---------------------------------------------------------------------------

LOW_CONVICTION_SELL_MAX = 4   # TRIM passes at or below this, if confirmed
EXIT_CONVICTION_MAX = 2       # EXIT (sell everything) only at or below this
LOW_CONVICTION_CONFIRM_MAX_AGE_DAYS = 7  # confirming verdict must be this recent


def _signal_age_days(signal: dict) -> float:
    from datetime import datetime, timezone
    ts = signal.get("timestamp")
    if not isinstance(ts, datetime):
        return 0.0  # unknown age: don't block on it
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 86400


def _low_conviction_sell(ticker: str, action: str, conviction: int) -> tuple[str, bool, str]:
    """Decide a TRIM/EXIT at conviction <= 4. Returns (action, pass_to_risk, note).
    Needs the previous analyst verdict on this ticker to be <= 4 too, so one
    noisy reading can't sell; db.get_latest_signal_for skips routing rows and
    runs before this analysis is logged."""
    try:
        import db
        prev = db.get_latest_signal_for(ticker)
    except Exception as e:
        log.warning(f"[{ticker}] low-conviction sell: previous verdict lookup failed: {e}")
        prev = None
    if prev and _signal_age_days(prev) > LOW_CONVICTION_CONFIRM_MAX_AGE_DAYS:
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


def _stub_recommendation(candidate: dict, reason: str) -> dict:
    ticker = candidate.get("ticker", "UNKNOWN")
    log.warning(f"[{ticker}] Stub recommendation — {reason}")
    try:
        from feed import emit
        emit("analyst", ticker, f"No verdict ({reason}) — skipping", "neutral")
    except Exception:
        pass
    # This used to be a WARNING-level log line and nothing else — a real
    # multi-day outage (every single analysis silently falling back to a
    # stub from 2026-07-15 through 2026-07-17, ~137 calls, because the old
    # 180s timeout was too tight) went completely unnoticed because nobody
    # was grepping logs for it. Throttled to once/hour so a genuinely
    # unhealthy model gets surfaced fast without spamming on every call.
    try:
        import db
        from datetime import datetime as _dt, timezone as _tz
        last = db.get_state("stub_recommendation_alert_at")
        now = _dt.now(_tz.utc)
        due = True
        if last:
            try:
                due = (now - _dt.fromisoformat(last)).total_seconds() >= 3600
            except ValueError:
                pass
        if due:
            db.set_state("stub_recommendation_alert_at", now.isoformat())
            import alerts
            alerts._send(
                f"⚠️ <b>Analyst falling back to stub</b> — {ticker}: {reason}\n"
                f"The 72B model didn't return a usable response, so this and any "
                f"other recent analyses defaulted to WATCH/no-action instead of a "
                f"real recommendation. Check Ollama health if this keeps happening."
            )
    except Exception as e:
        log.warning(f"stub_recommendation alert failed: {e}")

    # Auto-recovery driven by REAL production failures, not a synthetic probe
    # — found 2026-07-21: the periodic check_ollama_gpu_health task kept
    # reporting healthy (its own probe always finished under 300s) while
    # real analyst calls kept timing out at 400s all day, because that
    # probe itself was adding load to the 72B server's single concurrency
    # slot. This reacts to what's actually failing: 2+ stub fallbacks
    # within a 20-minute window means something is genuinely wrong right
    # now, independent of what any synthetic health check believes.
    try:
        import db
        from datetime import datetime as _dt, timezone as _tz
        now = _dt.now(_tz.utc)
        streak_start_raw = db.get_state("stub_streak_start")
        streak_start = None
        if streak_start_raw:
            try:
                streak_start = _dt.fromisoformat(streak_start_raw)
            except ValueError:
                pass
        if streak_start is None or (now - streak_start).total_seconds() > 1200:
            streak_start = now
            streak_count = 1
        else:
            streak_count = int(db.get_state("stub_streak_count") or "0") + 1
        db.set_state("stub_streak_start", streak_start.isoformat())
        db.set_state("stub_streak_count", str(streak_count))

        if streak_count >= 2:
            db.set_state("stub_streak_count", "0")  # avoid re-triggering every call while throttled
            from tasks import restart_ollama_and_alert
            restart_ollama_and_alert(
                f"{streak_count} stub fallbacks in the last 20 minutes (latest: {ticker}, {reason})"
            )
    except Exception as e:
        log.warning(f"stub streak auto-recovery check failed: {e}")

    # Still real (deterministic, not model-derived), unlike everything else
    # in this stub -- no reason to throw away a genuine price-based signal
    # just because Ollama is down.
    try:
        from agents.chase_risk import compute_chase_risk
        chase = compute_chase_risk(ticker)
    except Exception:
        chase = None
    try:
        from agents.price_action import compute_price_action
        price_action = compute_price_action(ticker)
    except Exception:
        price_action = None

    return {
        "ticker":         ticker,
        "action":         "WATCH",
        "conviction":     1,
        "thesis_strength": 1,
        "catalyst_quality": 1,
        "valuation_risk_reward": 1,
        "data_completeness": 1,
        "chase_risk": chase["score"] if chase else None,
        "chase_risk_pct_from_kijun": chase["pct_from_kijun"] if chase else None,
        "price_action": price_action["score"] if price_action else None,
        "price_action_structure": price_action["structure"] if price_action else None,
        "thesis_bucket":  candidate.get("role", "Accelerator"),
        "thesis_summary": f"{ticker} on watchlist — model unavailable.",
        "expectation_gap": {
            "bull_case": "Unknown", "current_price_implies": "Unknown",
            "upside_exists_if": "Unknown", "primary_risk": "Unknown",
        },
        "catalyst": {
            "event": "Unknown", "date": "Unknown",
            "expected_upside_if_positive": "Unknown",
            "expected_downside_if_negative": "Unknown",
            "is_priced_in": False,
        },
        "valuation": {
            "current_multiple": "Unknown", "peer_range": "Unknown",
            "bull_case_target": "Unknown", "downside_scenario": "Unknown",
        },
        "upside_downside_ratio": 0.0,
        "key_risks":       [f"No verdict: {reason}"],
        "sources": [],
        "source_count": 0,
        "expected_return_pct": None,
        "expected_downside_pct": None,
        "weeks_to_catalyst": None,
        "expected_return_per_week": None,
        "trend_template": None,
        "options_opportunity": False,
        "options_notes":   None,
        "pass_to_risk":    False,
        "reasoning":       f"Stub: {reason}. Model ({ANALYST_MODEL}) not available.",
        "model_used":      f"stub",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main analyst function
# ---------------------------------------------------------------------------

def trigger_record(path: str, event: dict | None = None) -> dict:
    """What started this analysis, for the verdict's provenance (audit 16:53,
    A1653-2). Log-only: never passed to the prompt."""
    e = event or {}
    rec = {"path": path, "source": e.get("source"), "type": e.get("type"),
           "title": (e.get("title") or "")[:200] or None, "url": e.get("url"),
           "published_at": str(e.get("timestamp") or e.get("published_at") or "") or None}
    if isinstance(e.get("text"), str) and e["text"]:
        # The brief shows only the first EVENT_TEXT_CAP characters (SEF data QA).
        rec["text_len"] = len(e["text"])
        try:
            from context_engine import EVENT_TEXT_CAP
            rec["text_truncated"] = len(e["text"]) > EVENT_TEXT_CAP
        except Exception:
            pass
    return rec


def analyze(
    candidate: dict,
    situation_brief: str = "",
    regime_packet: dict | None = None,
    trigger: dict | None = None,
) -> dict:
    """
    Evaluate a scanner candidate and produce an investment recommendation.

    Args:
        candidate:       Scanner candidate dict (must have 'ticker', 'role', 'tier').
        situation_brief: Assembled context string from context_engine.
        regime_packet:   Market regime output packet.

    Returns:
        Full analyst recommendation dict matching analyst_agent.md spec.
    """
    ticker = candidate.get("ticker", "UNKNOWN")
    log.info(f"[{ticker}] Analyzing with {ANALYST_MODEL}...")

    try:
        from feed import emit
        emit("analyst", ticker, f"Starting deep analysis with {ANALYST_MODEL}…", "neutral")
    except Exception:
        pass

    if regime_packet is None:
        try:
            from agents.market_regime import get_cached_regime
            regime_packet = get_cached_regime()
        except Exception:
            regime_packet = {"regime": "Neutral", "score": 50, "reasoning": ""}

    try:
        from knowledge_loader import load_thesis, load_philosophy
        thesis_text     = load_thesis(ticker)
        philosophy_text = load_philosophy()
    except Exception as e:
        log.warning(f"[{ticker}] Knowledge load failed: {e}")
        thesis_text     = ""
        philosophy_text = ""

    prompt = _build_prompt(candidate, situation_brief, thesis_text, regime_packet,
                           _philosophy_sections(philosophy_text, candidate.get("role", "Accelerator")))

    raw_response = _call_ollama(prompt)
    if not raw_response:
        return _stub_recommendation(
            candidate, _STUB_REASONS.get(_LAST_CALL_FAILURE["kind"], "Ollama returned nothing"))

    parsed = _extract_json(raw_response)
    if not parsed:
        log.warning(f"[{ticker}] Could not parse model response")
        return _stub_recommendation(candidate, "JSON parse failed")
    if parsed.pop("_salvaged", False) and str(parsed.get("action", "")).upper() in (
            "OPEN", "ADD", "TRIM", "EXIT"):
        # Fields regex-pulled from broken JSON carry no reasoning, sources or
        # risks; a sell gate that ignores pass_to_risk could still act on
        # them (audit 14:53, A1453-6). Never route one.
        log.warning(f"[{ticker}] Salvaged {parsed.get('action')} from malformed JSON; not routed")
        return _stub_recommendation(candidate, "JSON parse failed (fields salvaged, not routed)")

    action     = str(parsed.get("action", "WATCH")).upper()
    conviction = max(1, min(10, int(parsed.get("conviction", 1) or 1)))
    udr        = float(parsed.get("upside_downside_ratio", 0) or 0)
    # Decision record (roadmap 0e #8): the model's own verdict and full
    # output before any gate, plus each gate that changed it. signals kept
    # only the final action and 500 chars of reasoning, so a WATCH couldn't
    # be told apart from an OPEN that a gate downgraded.
    model_output = json.loads(json.dumps(parsed, default=str))
    model_pass = bool(parsed.get("pass_to_risk", False))
    model_action, model_conviction = action, conviction
    gates: list[str] = []

    # The model doesn't reliably know what's held (ADD/HOLD on unheld MSFT
    # and AMZN, 10-15 and 10-26). An ADD on nothing is an OPEN and must pass
    # the OPEN-only gates below; a HOLD on nothing is a WATCH. 2026-10-27.
    if candidate.get("held") is False and action in ("ADD", "HOLD"):
        coerced = "OPEN" if action == "ADD" else "WATCH"
        log.info(f"[{ticker}] {action} on a ticker not held -> {coerced}")
        parsed["reasoning"] = parsed.get("reasoning", "") + f" [{action} on an unheld ticker treated as {coerced}]"
        gates.append(f"unheld {action} -> {coerced}")
        action = coerced

    # Sub-scores (1-5 each) — decompose overall conviction into components
    # that can be individually calibrated against real outcomes later (see
    # db.get_subscore_calibration / _build_track_record_text), instead of
    # only ever knowing "conviction-8 calls win X% of the time" without
    # knowing whether it's the catalyst calls or the valuation calls dragging
    # that number down. `... or 3` here would silently turn a genuine 0 (bad
    # output, should floor to 1) into the neutral default 3 instead — missing
    # key and "explicitly 0" need different handling, not the same fallback.
    def _sub_score(field: str) -> int:
        value = parsed.get(field)
        if value is None:
            value = 3
        return max(1, min(5, int(value)))

    def _optional_float(field: str) -> float | None:
        value = parsed.get(field)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            log.debug(f"[{ticker}] Non-numeric {field}: {value!r}")
            return None

    thesis_strength        = _sub_score("thesis_strength")
    catalyst_quality       = _sub_score("catalyst_quality")
    valuation_risk_reward  = _sub_score("valuation_risk_reward")
    data_completeness      = _sub_score("data_completeness")

    # Chase risk — deterministic, not LLM-estimated (see agents/chase_risk.py
    # docstring for why). Informational only for now: doesn't affect action,
    # conviction, or sizing, just surfaces "good thesis, bad entry price" as
    # its own visible number in trade_outcomes and Telegram alerts.
    try:
        from agents.chase_risk import compute_chase_risk, KIJUN_PERIOD
        chase = compute_chase_risk(ticker)
    except Exception as e:
        log.debug(f"[{ticker}] chase_risk computation failed: {e}")
        chase = None
        KIJUN_PERIOD = 26
    chase_risk_score = chase["score"] if chase else None
    if chase:
        parsed["reasoning"] = (
            parsed.get("reasoning", "") +
            f" [Chase risk {chase['score']}/5 — price is {chase['pct_from_kijun']:+.1f}% "
            f"from its {KIJUN_PERIOD}-day equilibrium (Kijun-sen).]"
        )

    # Price action / volume — deterministic (agents/price_action.py),
    # already wired into the scanner's composite_score (swing structure +
    # 200DMA slope + OBV volume confirmation). Surfaced here too so the
    # analyst's own reasoning can reference the same real signal instead of
    # restating "50DMA above 200DMA" as if that were the whole picture.
    try:
        from agents.price_action import compute_price_action
        price_action = compute_price_action(ticker)
    except Exception as e:
        log.debug(f"[{ticker}] price_action computation failed: {e}")
        price_action = None
    price_action_score = price_action["score"] if price_action else None
    if price_action:
        parsed["reasoning"] = (
            parsed.get("reasoning", "") +
            f" [Price action {price_action['score']}/5 — {price_action['summary']}]"
        )

    # Bucket is watchlist-authoritative, not an LLM call — strategy ("strategy
    # is data, not code") lives in knowledge/watchlist.md, not in whatever a
    # given inference happens to output. A DDOG recommendation was sized as
    # Compounder (wider caps, wider ATR stop) purely because the model said so
    # in its JSON, despite being told the watchlist role was Accelerator —
    # silently accepting that let a weak/misconfigured model quietly loosen
    # risk limits. The LLM's suggestion is kept for audit but never used for
    # sizing; update the watchlist by hand if a reclassification is deliberate.
    #
    # Swing-trading cutover, Phase 4/5 (2026-10-11): when SETUP_GATES_ENABLED,
    # agents.setup_gates.classify_setup(ticker)'s LIVE classification (Breakout/
    # Catalyst/Continuation, or None if nothing currently matches) becomes the
    # sizing/stop authority for this ticker INSTEAD OF the static watchlist
    # role, per the design: the three setup categories are dynamic per-moment
    # classifications, not a static per-ticker label the way the old buckets
    # were. knowledge/watchlist.md's "role" column is untouched and used as
    # the fallback when nothing currently qualifies (e.g. a legacy position
    # the gates haven't classified, or a genuinely quiet period — the gates
    # ran at ~0% qualification across the watchlist for multiple consecutive
    # days in choppy shadow-mode conditions, which is expected). When the
    # flag is False, none of this runs and behavior is unchanged.
    watchlist_role = candidate.get("role", "Accelerator")
    llm_bucket     = parsed.get("thesis_bucket", watchlist_role)
    live_setup_best = None
    live_setup_gates = {}
    if SETUP_GATES_ENABLED:
        try:
            from agents.setup_gates import classify_setup
            live_setup_best, live_setup_gates = classify_setup(ticker)
        except Exception as e:
            log.warning(f"[{ticker}] classify_setup failed ({e}) — falling back to watchlist role")
        bucket = live_setup_best["category"] if live_setup_best else watchlist_role
        bucket_mismatch = llm_bucket != bucket
        if bucket_mismatch:
            log.info(
                f"[{ticker}] Analyst suggested bucket '{llm_bucket}'; live setup "
                f"classification is '{bucket}' — using the live classification for sizing."
            )
    else:
        bucket_mismatch = llm_bucket != watchlist_role
        if bucket_mismatch:
            log.warning(
                f"[{ticker}] Analyst suggested bucket '{llm_bucket}' but watchlist role "
                f"is '{watchlist_role}' — using the watchlist role for sizing. Update "
                f"knowledge/watchlist.md if this reclassification should stick."
            )
        bucket = watchlist_role

    # Enforce minimum UDR for actionable recommendations
    if udr < 2.0 and action in ("OPEN", "ADD"):
        log.info(f"[{ticker}] UDR {udr:.1f} < 2.0 — downgrading to WATCH")
        gates.append(f"udr {udr:.1f} < 2.0: {action} -> WATCH")
        action = "WATCH"
        parsed["pass_to_risk"] = False
        parsed["reasoning"] = (parsed.get("reasoning", "") +
                               f" [Downgraded: UDR {udr:.1f} < 2.0 minimum]")

    # Multi-source corroboration gate (2026-08-30, user request): a thesis
    # resting on a single piece of evidence -- one article's opinion, one
    # filing, one options-flow blip -- isn't validated, it's a guess with
    # conviction attached. The prompt asks the model to list every distinct
    # piece of evidence it actually relied on in "sources"; this is the
    # deterministic enforcement, same pattern as the UDR gate above -- a
    # prompt instruction alone isn't reliable enough on its own (the
    # fundamentals-missing conviction cap exists for the identical reason).
    # Distinctness is checked on (type, detail) pairs, not raw list length,
    # so the model can't clear the bar by listing the same fact twice under
    # different wording.
    sources = parsed.get("sources") or []
    source_count = count_distinct_sources(sources)
    if source_count < 2 and action in ("OPEN", "ADD"):
        log.info(f"[{ticker}] Only {source_count} distinct source(s) cited — downgrading to WATCH")
        gates.append(f"sources {source_count} < 2: {action} -> WATCH")
        action = "WATCH"
        parsed["pass_to_risk"] = False
        parsed["reasoning"] = (parsed.get("reasoning", "") +
                               f" [Downgraded: {source_count} distinct source(s) cited, need >=2]")

    # Minimum expected-return RATE gate (2026-08-30, user request) --
    # scoped to Accelerator/Moonshot only, see config.py's
    # MIN_EXPECTED_RETURN_PCT_PER_WEEK docstring for the full rationale
    # and worked examples. A catalyst with no defined timeframe at all
    # ("ongoing") can't have a rate computed -- fails CLOSED (downgrades)
    # rather than passing by default, consistent with catalyst_quality's
    # own rubric already treating vague/ongoing timing as weak (1/5).
    weeks_to_catalyst = None
    per_week = None
    catalyst_stale = False
    if bucket in ("Accelerator", "Moonshot") and action in ("OPEN", "ADD"):
        catalyst_date = (parsed.get("catalyst") or {}).get("date")
        weeks_to_catalyst = compute_weeks_to_catalyst(catalyst_date)
        per_week = expected_return_per_week(_optional_float("expected_return_pct"), weeks_to_catalyst)
        if per_week is None or per_week < MIN_EXPECTED_RETURN_PCT_PER_WEEK:
            reason = ("no computable catalyst timeframe" if weeks_to_catalyst is None
                      else f"{per_week:.2f}%/week < {MIN_EXPECTED_RETURN_PCT_PER_WEEK}%/week minimum")
            log.info(f"[{ticker}] Expected-return rate check failed ({reason}) — downgrading to WATCH")
            gates.append(f"return rate ({reason}): {action} -> WATCH")
            action = "WATCH"
            parsed["pass_to_risk"] = False
            parsed["reasoning"] = parsed.get("reasoning", "") + f" [Downgraded: {reason}]"
    elif bucket in ("Accelerator", "Moonshot") and action == "HOLD":
        # 2026-09-15, user request: found reviewing 4 real held positions
        # that this whole discipline silently stopped applying the day
        # after entry -- a HOLD review never recomputed it, so a position
        # could coast indefinitely on a catalyst that had gone stale
        # ("ongoing" milestones, no date) without the same 1%/week bar a
        # fresh entry has to clear ever being re-checked. Deliberately
        # informational only here, NOT a downgrade -- WATCH/EXIT has no
        # meaning for an already-held position the way it does for a
        # fresh candidate, and auto-converting a HOLD into a TRIM that
        # could auto-execute is a live-order behavior change, not a
        # visibility fix. catalyst_stale is surfaced in the result for
        # thesis-drift tracking (db.compute_thesis_drift) and reporting
        # to actually force human attention instead.
        catalyst_date = (parsed.get("catalyst") or {}).get("date")
        weeks_to_catalyst = compute_weeks_to_catalyst(catalyst_date)
        per_week = expected_return_per_week(_optional_float("expected_return_pct"), weeks_to_catalyst)
        if per_week is None or per_week < MIN_EXPECTED_RETURN_PCT_PER_WEEK:
            catalyst_stale = True
            reason = ("no computable catalyst timeframe" if weeks_to_catalyst is None
                      else f"{per_week:.2f}%/week < {MIN_EXPECTED_RETURN_PCT_PER_WEEK}%/week minimum")
            log.info(f"[{ticker}] HOLD carries a stale catalyst ({reason}) — flagged, not auto-downgraded")
            parsed["reasoning"] = parsed.get("reasoning", "") + f" [STALE CATALYST ON HOLD: {reason}]"

    # Trend Template gate (2026-09-11, user request) -- Mark Minervini's
    # 8-point Stage-2-uptrend criteria, see agents/price_action.py's
    # compute_trend_template() docstring for the full rule set and the
    # documented RS-percentile proxy. Scoped to Accelerator only:
    # Minervini's methodology targets established momentum leaders, which
    # is Accelerator's role here, not Moonshot's (pre-uptrend speculative
    # plays would almost never pass this by design) or Compounder's
    # (long-term thesis, not a momentum-timing entry). Same "prompt
    # instruction alone isn't reliable enough" reasoning as the other
    # deterministic gates above -- the LLM's own price/trend commentary
    # isn't a substitute for computing the actual 8 criteria.
    # Swing-trading cutover, Phase 4/5 (2026-10-11): this gate is scoped to
    # `bucket == "Accelerator"` below, which already means it's naturally
    # SKIPPED whenever SETUP_GATES_ENABLED gave this ticker a live Breakout/
    # Catalyst/Continuation match -- `bucket` is literally that category
    # name in that case (set above), not "Accelerator". No extra condition
    # needed: agents.setup_gates already ran compute_trend_template
    # internally (via agents.trade_ladder's nightly trend_stage field) for
    # any Breakout/Continuation match, so recomputing it here would be
    # redundant. The fallback case (live_setup_best is None -> bucket falls
    # back to the watchlist role) means the gates DIDN'T classify this
    # ticker, so this inline check is still the only place Trend Template
    # runs for it.
    trend_template = None
    if bucket == "Accelerator" and action in ("OPEN", "ADD"):
        try:
            from agents.price_action import compute_trend_template
            trend_template = compute_trend_template(ticker)
        except Exception as e:
            log.debug(f"[{ticker}] Trend Template check failed: {e}")
        if trend_template is not None and not trend_template["pass"]:
            log.info(f"[{ticker}] Trend Template failed ({trend_template['reasoning']}) — downgrading to WATCH")
            gates.append(f"trend template: {action} -> WATCH")
            action = "WATCH"
            parsed["pass_to_risk"] = False
            parsed["reasoning"] = parsed.get("reasoning", "") + f" [Downgraded: Trend Template — {trend_template['reasoning']}]"
        # trend_template is None (insufficient history) passes through
        # un-gated rather than failing closed -- unlike the expected-
        # return-rate gate, a genuinely new listing without 252 days of
        # history isn't a red flag the way "no catalyst timeframe" is; it's
        # just data the gate can't evaluate yet, and Accelerator names are
        # by definition established enough that this should be rare.

    # Moonshots need conviction >= 7 for OPEN
    if bucket == "Moonshot" and action == "OPEN" and conviction < 7:
        log.info(f"[{ticker}] Moonshot conviction {conviction} < 7 — downgrading to WATCH")
        gates.append(f"moonshot conviction {conviction} < 7: OPEN -> WATCH")
        action = "WATCH"
        parsed["pass_to_risk"] = False

    # Regime-based selectivity for new positions — enforced here instead of
    # asked of the model. The prompt used to say "raise threshold to
    # conviction 7+" for cautious regimes, which measurably caused the model
    # to anchor on exactly 7 as the passing score rather than reasoning
    # independently (2026-07-25 finding: 69% of every actionable decision
    # ever made was conviction=7, with none ever below 5 or above 8). The
    # model now scores conviction on evidence alone; this applies the actual
    # selectivity policy deterministically instead of via prompt anchoring.
    regime_str = regime_packet.get("regime", "Neutral")
    if action == "OPEN" and regime_str in ("Neutral", "Risk-Off", "Strong Risk-Off") and conviction < 7:
        log.info(f"[{ticker}] {regime_str} regime — conviction {conviction} < 7 for new position — downgrading to WATCH")
        gates.append(f"{regime_str} regime conviction {conviction} < 7: OPEN -> WATCH")
        action = "WATCH"
        parsed["pass_to_risk"] = False

    # Deterministic guardrail for ungrounded valuation: the prompt instructs
    # the model to cap its own conviction at 6 when fundamentals aren't
    # cached, but a prompt instruction isn't reliable enough on its own —
    # enforce it here too so an OPEN/ADD can't ride on an invented multiple.
    try:
        import db
        fund = db.get_fundamentals(ticker)
        fundamentals_missing = not fund or fund.get("market_cap") is None
    except Exception:
        fundamentals_missing = True
    conviction_raw = conviction
    is_etf = False
    if fundamentals_missing and action in ("OPEN", "ADD") and conviction > 6:
        try:
            from knowledge_loader import get_etf_tickers
            is_etf = ticker in get_etf_tickers()
        except Exception:
            pass
    if fundamentals_missing and action in ("OPEN", "ADD") and conviction > 6:
        from config import ETF_FUNDAMENTALS_CAP_EXEMPT
        if is_etf and ETF_FUNDAMENTALS_CAP_EXEMPT:
            gates.append(f"no fundamentals: ETF, cap not applied (conviction {conviction})")
        else:
            log.info(f"[{ticker}] No cached fundamentals — capping conviction {conviction} -> 6")
            gates.append(f"no fundamentals: conviction {conviction} -> 6"
                         + (" (ETF; exemption off)" if is_etf else ""))
            conviction = 6

    # Validate pass_to_risk
    actionable = action in ("OPEN", "ADD", "TRIM", "EXIT")
    if parsed.get("pass_to_risk") and not (actionable and conviction >= 5):
        gates.append(f"pass_to_risk dropped ({action} c{conviction})")
        parsed["pass_to_risk"] = False

    # Low-conviction sells (2026-10-27). The conviction >= 5 rule above meant
    # a TRIM/EXIT with a weakening or broken thesis (conviction 1-4) could
    # never pass: from Aug 13 to Oct 27 all 6 TRIMs were dropped (QBTS at
    # c3-c4 five times, then it hit its stop). Sells now pass on a confirmed
    # low reading, conservatively: the previous analyst verdict on the ticker
    # must also be <= 4, EXIT needs <= 2, and an EXIT at 3-4 becomes a TRIM.
    # UDR doesn't apply to sells.
    if action in ("TRIM", "EXIT") and conviction <= LOW_CONVICTION_SELL_MAX:
        sell_from = action
        action, sell_passes, sell_note = _low_conviction_sell(ticker, action, conviction)
        gates.append(f"low-conviction sell {sell_from} c{conviction} -> {action}, "
                     f"pass={sell_passes}" + (f" ({sell_note})" if sell_note else ""))
        parsed["pass_to_risk"] = sell_passes
        if sell_note:
            parsed["reasoning"] = parsed.get("reasoning", "") + f" [{sell_note}]"

    provenance = _provenance(prompt)
    try:
        provenance["knowledge_coverage"] = _knowledge_coverage(ticker, thesis_text, philosophy_text,
                                                               prompt)
    except Exception as e:
        log.debug(f"[{ticker}] knowledge coverage record failed: {e}")
    provenance.update(_decision_inputs_record(ticker, regime_packet))
    try:
        provenance["brief_exposure"] = _brief_exposure(ticker, prompt)
    except Exception as e:
        log.debug(f"[{ticker}] brief exposure record failed: {e}")
    try:
        provenance["gate_flags"] = _gate_flags(ticker, action, conviction, source_count,
                                               bool(parsed.get("pass_to_risk")),
                                               model_action, model_conviction)
    except Exception as e:
        log.debug(f"[{ticker}] gate flags record failed: {e}")
    provenance["trigger"] = trigger
    est = _estimate_tokens(prompt)
    llm = {**_LAST_CALL_STATS, "prompt_tokens_est": est}
    # Ollama may cut the prompt's head below num_ctx to keep room for the
    # reply, so compare what it read with our crude estimate: a flag, not
    # proof (professor C26 residual).
    if llm.get("prompt_eval_count") and est:
        llm["truncation_suspect"] = llm["prompt_eval_count"] / est < 0.8
        if llm["truncation_suspect"]:
            log.warning(f"[{ticker}] Ollama read {llm['prompt_eval_count']} prompt tokens vs "
                        f"~{est} estimated; the prompt's start may have been cut")
    llm["reply_cut"] = llm.get("done_reason") == "length"
    provenance["llm_call"] = llm
    # Shadow trade plan (roadmap 1): log-only, nothing reads it back.
    try:
        from agents.plans import observe as _plan_observe
        provenance["plan"] = _plan_observe(ticker, parsed, action, conviction, situation_brief)
    except Exception as e:
        log.warning(f"[{ticker}] plan shadow unavailable: {e}")
    try:
        import db
        db.log_signal(
            symbol=ticker,
            signal=action,
            confidence=conviction / 10.0,
            reasoning=parsed.get("reasoning", "")[:500],
            model=ANALYST_MODEL,
            provenance=provenance,
            decision={
                "model_action": model_action, "model_conviction": model_conviction,
                "model_pass_to_risk": model_pass,
                "final_action": action, "final_conviction": conviction,
                "final_pass_to_risk": bool(parsed.get("pass_to_risk", False)),
                "gates": gates, "model_output": model_output,
                "catalyst_check": _catalyst_check(ticker, parsed),
                "prior_verdicts_72h": _prior_verdicts(ticker) if action in ("OPEN", "ADD") else None,
                "rule_shadows": _rule_shadows(ticker, action, situation_brief),
                "trim_shadow": _trim_shadow(ticker, action, parsed, situation_brief),
                "udr_check": _udr_check(parsed, situation_brief),
                "event_burst": _event_burst(ticker, trigger, gates),
            },
            prompt=redact_account_figures(prompt),
        )
    except Exception as e:
        log.warning(f"[{ticker}] Signal log failed: {e}")

    result = {
        "ticker":    ticker,
        "action":    action,
        "conviction": conviction,
        # Pre-guardrail conviction as the model stated it — kept for audit and
        # calibration so a fundamentals-cache outage capping 7s to 6 doesn't
        # poison the conviction→outcome track record with manufactured 6s.
        "conviction_raw": conviction_raw,
        "thesis_strength": thesis_strength,
        "catalyst_quality": catalyst_quality,
        "valuation_risk_reward": valuation_risk_reward,
        "data_completeness": data_completeness,
        "chase_risk": chase_risk_score,
        "chase_risk_pct_from_kijun": chase["pct_from_kijun"] if chase else None,
        "price_action": price_action_score,
        "price_action_structure": price_action["structure"] if price_action else None,
        "thesis_bucket": bucket,
        "llm_suggested_bucket": llm_bucket if bucket_mismatch else None,
        "thesis_summary": parsed.get("thesis_summary", ""),
        "expectation_gap": parsed.get("expectation_gap", {}),
        "catalyst":  parsed.get("catalyst", {}),
        "valuation": parsed.get("valuation", {}),
        "expected_return_pct": _optional_float("expected_return_pct"),
        "expected_downside_pct": _optional_float("expected_downside_pct"),
        "weeks_to_catalyst": weeks_to_catalyst,
        "expected_return_per_week": per_week,
        "catalyst_stale": catalyst_stale,
        "trend_template": trend_template,
        "upside_downside_ratio": udr,
        "key_risks": parsed.get("key_risks", []),
        "sources": sources,
        "source_count": source_count,
        "options_opportunity": bool(parsed.get("options_opportunity", False)),
        "options_notes": parsed.get("options_notes"),
        "pass_to_risk": bool(parsed.get("pass_to_risk", False)),
        "reasoning": parsed.get("reasoning", ""),
        "model_used": ANALYST_MODEL,
        "provenance": provenance,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log.info(
        f"[{ticker}] → {action} | conviction={conviction}/10 | "
        f"UDR={udr:.1f} | pass_to_risk={result['pass_to_risk']}"
    )

    try:
        from feed import emit
        reasoning_snip = result.get("reasoning", "")[:140]
        if action in ("OPEN", "ADD"):
            sentiment = "bullish"
        elif action in ("TRIM", "EXIT"):
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        pass_flag = " — sending to Risk Manager" if result["pass_to_risk"] else ""
        emit("analyst", ticker,
             f"{action} | conviction {conviction}/10 | UDR {udr:.1f}x{pass_flag} — {reasoning_snip}",
             sentiment)
    except Exception:
        pass

    return result


def analyze_batch(
    candidates: list[dict],
    regime_packet: dict | None = None,
    max_candidates: int = 10,
    positions: dict[str, dict] | None = None,
    trading_client=None,
) -> list[dict]:
    """
    Analyze a batch of scanner candidates.
    Limits to max_candidates to control inference time on smaller models.

    Each candidate gets a full situation brief (context_engine.build_scan_brief)
    — macro, technicals, options flow, fundamentals, and vector-memory
    retrieval — the same context quality the rare event-triggered path gets.
    Previously this called analyze() with no brief at all, so the daily scan
    (the main decision flow) ran on scanner scores alone.
    """
    top = sorted(
        candidates[:max_candidates],
        key=lambda c: (c.get("priority") == "HIGH", c.get("composite_score", 0)),
        reverse=True,
    )

    results = []
    for candidate in top:
        ticker = candidate.get("ticker")
        brief = ""
        try:
            from context_engine import build_scan_brief
            from agents.crew import _get_price
            price = _get_price(ticker, trading_client)
            brief = build_scan_brief(ticker, positions, trading_client, current_price=price)
        except Exception as e:
            log.warning(f"[{ticker}] Scan brief build failed, analyzing without it: {e}")

        try:
            result = analyze(candidate, situation_brief=brief, regime_packet=regime_packet,
                             trigger=trigger_record("daily_scan", {"source": "scanner"}))
            results.append(result)
        except Exception as e:
            log.error(f"[{ticker}] Analyze failed: {e}")
            results.append(_stub_recommendation(candidate, f"Exception: {e}"))

    return results
