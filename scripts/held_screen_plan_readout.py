"""Held-name screen-to-plan readout: every screen of a held name beside the
active plan's primary_risk and the event verdict it led to, where one
exists (audit 2026-11-26 18:53 #1). Read-only engineering readout; counts
only.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/held_screen_plan_readout.py [--days 14] [--env live] \
        [--out FILE] [--export FILE.json] [--watchlist knowledge/watchlist.md]

Rows: signals SCREENED_IN and SCREENED_OUT rows in --env over the window
whose provenance.held is true (the held flag the screener was given).
Per row, as stored:
  - title (decision.title), title_tone, new_fact (provenance), the screen
    outcome, would_pass_held_negative (SCREENED_OUT only), override,
    material_event_terms (SCREENED_OUT rows carry it);
  - the plan: the ticker's trade_plans row created at or before the screen
    and not closed by then (closed_at NULL or later), the newest such, in
    --env or with a NULL env; its id, status and primary_risk;
  - the verdict (SCREENED_IN only): the first analyst verdict on the event
    path for the same ticker within VERDICT_WINDOW_H hours after the screen
    whose trigger url (first 300 characters, as the screen row stores it)
    equals the screen's url, or whose trigger title equals the screen's
    title when the screen has no url; its id and final / model action.

Two labels sit beside the raw text, both open (a labelling choice, Luke's):
  - ADVERSE_TERMS (version, sha): named word groups (credit, governance,
    allegation, legal, investigation, export / regulatory) matched on the
    title. The screener's own tone lists are not changed or read here.
  - risk words (RISK_MATCH_VERSION, sha): title words shared with the plan's
    primary_risk, after RISK_STOPWORDS, the ticker and the watchlist company
    name are dropped; a word of RISK_PREFIX_LEN characters or more also
    matches on its first RISK_PREFIX_LEN characters.
Neither is a measure; any other list or rule can be applied to the exported
rows. The report is counts only: no rate is chosen, and no threshold.

--export writes the rows (JSON) with titles and primary_risk text masked:
every dollar figure becomes $X. The markdown never prints a title. No
prices or account data are read.

SUPABASE_PORT must be 6543 (the transaction pooler); one read-only
transaction with a statement timeout, rolled back. Imports: db and the
standard library only. Not loaded by weekly_readouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REQUIRED_PORT = "6543"
STATEMENT_TIMEOUT_MS = 60000
VERDICT_WINDOW_H = 24
ET = ZoneInfo("America/New_York")
OPEN_LABEL = "open (labelling choice, Luke's)"

ADVERSE_TERMS_VERSION = 1
# (group, pattern), matched case-insensitively on the title. "spy" is matched
# only as written in lower or title case (SPY is the ETF).
ADVERSE_TERMS = (
    ("credit", r"\bcredit\s*(?:watch|warning|rating|ratings|negative|outlook)\b|\brating\s+"
               r"(?:review|cut|watch)\b|\bjunk\b"),
    ("governance", r"\bgovernance\b|\bfired\b|\bfire[sd]?\s+(?:its|the)\s+ceo\b|\boust(?:s|ed|er)?\b|"
                   r"\bresign(?:s|ed|ation)?\b|\bsuccession\b|\bsteps?\s+down\b"),
    ("allegation", r"\balleg(?:e|es|ed|edly|ation|ations)\b|\baccus(?:e|es|ed|ation|ations)\b|"
                   r"\bfraud\b|\bwhistleblower\b|\bscandal\b|(?-i:\b[Ss]py(?:ing)?\b)"),
    ("legal", r"\bsue[sd]?\b|\blawsuits?\b|\blitigation\b|\bsubpoena(?:s|ed)?\b|"
              r"\bindict(?:s|ed|ment)?\b|\bclass\s+action\b"),
    ("investigation", r"\binvestigat(?:e|es|ed|ion|ions|ing)\b|\bprobes?\b|\binquiry\b|\bantitrust\b"),
    ("export_regulatory", r"\bexports?\b|\bcurbs?\b|\brestrictions?\b|\bbans?\b|\bbanned\b|"
                          r"\bsanctions?\b|\btariffs?\b|\bembargo\b|\bblacklist(?:s|ed)?\b|"
                          r"\bentity\s+list\b|\bregulat(?:or|ors|ory|ion|ions)\b"),
)
_ADVERSE = tuple((g, re.compile(p, re.I)) for g, p in ADVERSE_TERMS)
ADVERSE_TERMS_SHA = hashlib.sha256(
    json.dumps([list(t) for t in ADVERSE_TERMS], separators=(",", ":")).encode()).hexdigest()[:12]

RISK_MATCH_VERSION = 1
RISK_PREFIX_LEN = 6
RISK_STOPWORDS = frozenset("""a an the and or of to in on for as at by with from is are was were be
been its it this that these those after before over into amid why what how will should could
would can may might stock stocks shares share inc corp company nasdaq nyse report reports says
said new up down vs than more most just now here there their his her our your you who has have
had not but if so about still yahoo finance cnbc barron reuters bloomberg marketwatch investor
business daily motley fool zacks benzinga seeking alpha wsj live updates today week year month
first last top buy sell hold price prices market markets ai risk risks potential impact
thesis undermine primary further which lead leading due including key such other""".split())
RISK_MATCH_SHA = hashlib.sha256(
    (" ".join(sorted(RISK_STOPWORDS)) + f"|prefix={RISK_PREFIX_LEN}").encode()).hexdigest()[:12]
_WORD = re.compile(r"[a-z][a-z0-9&\-]{2,}")
_WATCH_ROW = re.compile(r"^\|\s*([A-Z]{1,6})\s*\|\s*([^|]+?)\s*\|")
_DOLLAR = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:[BMKT]\b|bn\b|mn\b|billion\b|million\b|"
                     r"thousand\b|trillion\b))?", re.I)

_SQL = """
SELECT s.id, s.timestamp, s.symbol, s.signal,
       s.decision->>'title', s.decision->>'url',
       s.provenance->>'title_tone', s.provenance->'new_fact', s.provenance->>'override',
       s.provenance->'would_pass_held_negative', s.provenance->'material_event_terms',
       s.provenance->>'model', s.provenance->>'source',
       p.id, p.status, p.primary_risk,
       v.id, v.timestamp, v.decision->>'final_action', v.decision->>'model_action'
FROM signals s
LEFT JOIN LATERAL (
    SELECT tp.id, tp.status, tp.primary_risk FROM trade_plans tp
    WHERE tp.ticker = s.symbol AND (tp.trading_env = %(env)s OR tp.trading_env IS NULL)
      AND tp.created_at <= s.timestamp AND (tp.closed_at IS NULL OR tp.closed_at > s.timestamp)
    ORDER BY tp.created_at DESC LIMIT 1) p ON true
LEFT JOIN LATERAL (
    SELECT sv.id, sv.timestamp, sv.decision FROM signals sv
    WHERE s.signal = 'SCREENED_IN' AND sv.symbol = s.symbol
      AND sv.timestamp >= s.timestamp
      AND sv.timestamp < s.timestamp + make_interval(hours => %(window_h)s)
      AND sv.decision ? 'model_output' AND sv.provenance->'trigger'->>'path' = 'event'
      AND (sv.trading_env = %(env)s OR sv.trading_env IS NULL)
      AND ((s.decision->>'url' IS NOT NULL
            AND left(sv.provenance->'trigger'->>'url', 300) = s.decision->>'url')
           OR (s.decision->>'url' IS NULL
               AND sv.provenance->'trigger'->>'title' = s.decision->>'title'))
    ORDER BY sv.timestamp LIMIT 1) v ON true
WHERE s.signal IN ('SCREENED_IN', 'SCREENED_OUT') AND s.trading_env = %(env)s
  AND s.timestamp >= %(since)s AND s.provenance->>'held' = 'true'
ORDER BY s.timestamp, s.id
"""


def words(text: str | None) -> list[str]:
    """Lower-case words of text, in order. Pure."""
    return _WORD.findall((text or "").lower())


def company_names(path: str | None = None) -> dict[str, frozenset]:
    """ticker -> the words of its company name, from every ticker row of
    knowledge/watchlist.md. {} when the file can't be read."""
    try:
        with open(path or os.path.join(ROOT, "knowledge", "watchlist.md")) as f:
            lines = f.read().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        m = _WATCH_ROW.match(line)
        if m and m.group(1) != "TICKER":
            out.setdefault(m.group(1), frozenset(words(m.group(2))))
    return out


def mask_dollars(text: str | None) -> str | None:
    """text with every dollar figure replaced by $X. Pure."""
    return _DOLLAR.sub("$X", text) if isinstance(text, str) else None


def adverse_groups(title: str | None) -> list[str]:
    """The ADVERSE_TERMS groups the title matches, in list order. Pure."""
    return [g for g, rx in _ADVERSE if rx.search(title or "")]


def _content(text: str | None, ticker: str | None, company: frozenset) -> list[str]:
    seen, out = set(), []
    tick = (ticker or "").lower()
    for w in words(text):
        w = w.rstrip("-")
        if len(w) < 3 or w in RISK_STOPWORDS or w == tick or w in company or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def risk_words(title: str | None, primary_risk: str | None, ticker: str | None,
               company: frozenset = frozenset()) -> list[str] | None:
    """Title words shared with primary_risk (RISK_MATCH_VERSION rule, module
    docstring), in title order; None when there is no primary_risk. Pure."""
    if not isinstance(primary_risk, str) or not primary_risk.strip():
        return None
    risk = _content(primary_risk, ticker, company)
    rs, pre = set(risk), {w[:RISK_PREFIX_LEN] for w in risk if len(w) >= RISK_PREFIX_LEN}
    return [w for w in _content(title, ticker, company)
            if w in rs or (len(w) >= RISK_PREFIX_LEN and w[:RISK_PREFIX_LEN] in pre)]


def _json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _bool(v):
    v = _json(v)
    return v if isinstance(v, bool) else None


def build_rows(raw, names: dict[str, frozenset]) -> list[dict]:
    """One record per fetched row (the _SQL column order). Pure."""
    out = []
    for (sid, ts, sym, sig, title, url, tone, nf, override, wph, met, model, source,
         pid, pstatus, prisk, vid, vts, vfinal, vmodel) in raw:
        wph = _json(wph)
        company = names.get(sym or "", frozenset())
        out.append({
            "signal_id": sid,
            "et": ts.astimezone(ET).strftime("%Y-%m-%d %H:%M") if ts else None,
            "symbol": sym, "outcome": sig, "title": title, "url": url,
            "title_tone": tone, "new_fact": _bool(nf), "override": override,
            "would_pass_held_negative": (wph.get("would_pass") if isinstance(wph, dict)
                                         and isinstance(wph.get("would_pass"), bool) else None),
            "override_skipped": wph.get("override_skipped") if isinstance(wph, dict) else None,
            "material_event_terms": _json(met), "model": model, "source": source,
            "plan_id": pid, "plan_status": pstatus, "primary_risk": prisk,
            "verdict_id": vid,
            "verdict_et": vts.astimezone(ET).strftime("%Y-%m-%d %H:%M") if vts else None,
            "verdict_action": vfinal, "verdict_model_action": vmodel,
            "adverse_groups": adverse_groups(title),
            "risk_words": risk_words(title, prisk, sym, company),
        })
    return out


def _k(v) -> str:
    return "none" if v is None else str(v).lower() if isinstance(v, bool) else str(v)


def _table(title: str, counts: Counter, cols: list[str], rows: list[str]) -> list[str]:
    out = [f"### {title}", "", "| | " + " | ".join(cols) + " | all |",
           "|---|" + "---:|" * (len(cols) + 1)]
    for r in rows:
        cells = [counts.get((r, c), 0) for c in cols]
        out.append(f"| {r} | " + " | ".join(str(x) for x in cells) + f" | {sum(cells)} |")
    tot = [sum(counts.get((r, c), 0) for r in rows) for c in cols]
    out += ["| all | " + " | ".join(str(x) for x in tot) + f" | {sum(tot)} |", ""]
    return out


def _cross(rows, rkey, ckey) -> tuple[Counter, list[str], list[str]]:
    c = Counter((rkey(r), ckey(r)) for r in rows)
    return c, sorted({k[0] for k in c}), sorted({k[1] for k in c})


def render(rows: list[dict], env: str, days: int, since: datetime, head: str) -> str:
    tones = lambda r: _k(r["title_tone"])
    outc = lambda r: r["outcome"]
    lines = ["# Held-name screen-to-plan readout", "",
             f"Env {env}; held-name screens since {since.astimezone(ET):%Y-%m-%d %H:%M} ET "
             f"({days} days); code {head}. Counts only; no measure is chosen. "
             f"Adverse terms v{ADVERSE_TERMS_VERSION} (sha {ADVERSE_TERMS_SHA}) and risk words "
             f"v{RISK_MATCH_VERSION} (sha {RISK_MATCH_SHA}) are {OPEN_LABEL}.", "",
             f"Rows: {len(rows)} (SCREENED_IN {sum(r['outcome'] == 'SCREENED_IN' for r in rows)}, "
             f"SCREENED_OUT {sum(r['outcome'] == 'SCREENED_OUT' for r in rows)}). "
             f"With an active plan: {sum(r['plan_id'] is not None for r in rows)}; of those with "
             f"a primary_risk: {sum(r['risk_words'] is not None for r in rows)}.", ""]
    c, rr, cc = _cross(rows, tones, outc)
    lines += _table("title_tone by screen outcome", c, cc, rr)
    c, rr, cc = _cross(rows, lambda r: _k(r["new_fact"]), outc)
    lines += _table("new_fact by screen outcome", c, cc, rr)
    out_rows = [r for r in rows if r["outcome"] == "SCREENED_OUT"]
    c, rr, cc = _cross(out_rows, tones, lambda r: _k(r["would_pass_held_negative"]))
    lines += _table("SCREENED_OUT: title_tone by would_pass_held_negative", c, cc, rr)
    c, rr, cc = _cross(out_rows, tones, lambda r: _k(r["override_skipped"]))
    lines += _table("SCREENED_OUT: title_tone by override_skipped", c, cc, rr)
    in_rows = [r for r in rows if r["outcome"] == "SCREENED_IN"]
    c, rr, cc = _cross(in_rows, lambda r: _k(r["override"]),
                       lambda r: "no verdict" if r["verdict_id"] is None else _k(r["verdict_action"]))
    lines += _table("SCREENED_IN: override by verdict final action (no verdict: none matched)",
                    c, cc, rr)
    c, rr, cc = _cross(in_rows, tones,
                       lambda r: "no verdict" if r["verdict_id"] is None else _k(r["verdict_action"]))
    lines += _table("SCREENED_IN: title_tone by verdict final action", c, cc, rr)
    c, rr, cc = _cross(rows, tones,
                       lambda r: r["outcome"] + (" adverse" if r["adverse_groups"] else " none"))
    lines += _table(f"title_tone by outcome and any adverse-term group (v{ADVERSE_TERMS_VERSION}, "
                    f"{OPEN_LABEL})", c, cc, rr)
    g = Counter((grp, r["outcome"]) for r in rows for grp in r["adverse_groups"])
    lines += _table("adverse-term group by outcome (a row can hit several)", g,
                    sorted({k[1] for k in g}), [x for x, _ in ADVERSE_TERMS if any(
                        k[0] == x for k in g)])
    with_risk = [r for r in rows if r["risk_words"] is not None]
    c, rr, cc = _cross(with_risk, tones,
                       lambda r: r["outcome"] + (" shared" if r["risk_words"] else " none"))
    lines += _table(f"rows with a plan primary_risk: title_tone by outcome and any shared risk "
                    f"word (v{RISK_MATCH_VERSION}, {OPEN_LABEL})", c, cc, rr)
    return "\n".join(lines).rstrip() + "\n"


def export_rows(rows: list[dict]) -> list[dict]:
    """The rows for --export: title and primary_risk with dollar figures
    masked, the url left out (it can carry a figure). Pure."""
    keep = [k for k in (rows[0] if rows else {}) if k != "url"]
    return [{**{k: r[k] for k in keep}, "title": mask_dollars(r["title"]),
             "primary_risk": mask_dollars(r["primary_risk"])} for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--env", default="live")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--export", help="write the rows (JSON, dollar figures masked) here")
    ap.add_argument("--watchlist", help="watchlist.md for company names "
                    "(default: knowledge/watchlist.md beside this script)")
    args = ap.parse_args(argv)
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        print(f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction pooler); not reading.",
              file=sys.stderr)
        return 2
    since = datetime.now(timezone.utc) - timedelta(days=args.days)

    import db
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SET LOCAL statement_timeout = %s", (STATEMENT_TIMEOUT_MS,))
            cur.execute(_SQL, {"since": since, "env": args.env, "window_h": VERDICT_WINDOW_H})
            raw = cur.fetchall()
    finally:
        try:
            conn.rollback()
        finally:
            db._put_conn(conn)
    rows = build_rows(raw, company_names(args.watchlist))
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    text = render(rows, args.env, args.days, since, head or "unknown")
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    if args.export:
        with open(args.export, "w") as f:
            json.dump({"adverse_terms_version": ADVERSE_TERMS_VERSION,
                       "adverse_terms_sha": ADVERSE_TERMS_SHA,
                       "risk_match_version": RISK_MATCH_VERSION, "risk_match_sha": RISK_MATCH_SHA,
                       "label_status": OPEN_LABEL, "verdict_window_h": VERDICT_WINDOW_H,
                       "rows": export_rows(rows)}, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
