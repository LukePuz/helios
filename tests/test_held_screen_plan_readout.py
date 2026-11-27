"""scripts/held_screen_plan_readout.py (audit 2026-11-26 18:53 #1): the open
adverse-term and risk-word labels, the row builder, the counts-only report,
the masked export and the read-only guard. No DB.

The titles in AUDIT_TITLES are the held NVDA screens the audit named
(11-26, live), dollar figures masked as $X."""
import importlib.util
import json
import os
import types
from collections import Counter
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "held_screen_plan_readout.py")
_spec = importlib.util.spec_from_file_location("held_screen_plan_readout", _PATH)
src = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src)

AUDIT_TITLES = {
    105511: ("Does S&P's Credit Warning Matter For Nvidia Stock?", ["credit"]),
    105664: ("Nvidia’s Reliance on Huang Could Now Hurt Its Credit Rating", ["credit"]),
    105895: ("Nvidia faces credit rating review as S&P flags governance concerns",
             ["credit", "governance"]),
    105546: ("US Chip Curbs Can't Shake China's Appetite For Nvidia", ["export_regulatory"]),
    105567: ("Jensen Huang 'absolutely could be fired again,' legal expert says", ["governance"]),
    106169: ("Did Jensen Huang’s Alleged Chinese Spy Get Him to Do Anything Nvidia Didn’t Want?",
             ["allegation"]),
}
RISK57 = "Regulatory scrutiny in China or a slowdown in hyperscaler capex could undermine the thesis."
TS = datetime(2026, 11, 26, 12, 24, 0, tzinfo=timezone.utc)
VTS = datetime(2026, 11, 26, 12, 27, 0, tzinfo=timezone.utc)


def _raw(sid=1, sig="SCREENED_IN", title="Nvidia faces credit rating review", tone="neutral",
         nf="true", override=None, wph=None, prisk=RISK57, vid=10, vfinal="HOLD", sym="NVDA"):
    return (sid, TS, sym, sig, title, "https://x/y", tone, nf, override,
            json.dumps(wph) if wph is not None else None, None, "qwen", "yfinance",
            57, "active", prisk, vid, VTS if vid else None, vfinal if vid else None,
            vfinal if vid else None)


# ── labels ──────────────────────────────────────────────────────────────────

def test_versions_and_shas():
    assert src.ADVERSE_TERMS_VERSION == 1 and src.RISK_MATCH_VERSION == 1
    assert len(src.ADVERSE_TERMS_SHA) == 12 and len(src.RISK_MATCH_SHA) == 12
    body = json.dumps([list(t) for t in src.ADVERSE_TERMS], separators=(",", ":"))
    assert src.ADVERSE_TERMS_SHA == src.hashlib.sha256(body.encode()).hexdigest()[:12]
    assert [g for g, _ in src.ADVERSE_TERMS] == ["credit", "governance", "allegation", "legal",
                                                 "investigation", "export_regulatory"]
    assert "Luke" in src.OPEN_LABEL and "open" in src.OPEN_LABEL


def test_audit_titles_hit_their_groups():
    for sid, (title, groups) in AUDIT_TITLES.items():
        assert src.adverse_groups(title) == groups, sid


def test_adverse_wording():
    assert src.adverse_groups("SPY falls 1% as yields rise") == []          # the ETF
    assert src.adverse_groups("Former spy chief joins board") == ["allegation"]
    assert src.adverse_groups("Shareholders sued Acme over disclosures") == ["legal"]
    assert src.adverse_groups("DOJ opens antitrust probe") == ["investigation"]
    assert src.adverse_groups("CEO steps down after board review") == ["governance"]
    assert src.adverse_groups("Nvidia beats on revenue") == []
    assert src.adverse_groups(None) == []


def test_risk_words():
    assert src.risk_words("US Chip Curbs Can't Shake China's Appetite", RISK57, "NVDA") == ["china"]
    # prefix rule: "regulators" and "regulatory" share their first 6 characters
    assert src.risk_words("Regulators weigh hyperscaler rules", RISK57, "NVDA") == \
        ["regulators", "hyperscaler"]
    # the ticker, the company name and stopwords never count
    assert src.risk_words("Nvidia NVDA could slow", "Nvidia NVDA could stall", "NVDA",
                          frozenset({"nvidia"})) == []
    assert src.risk_words("anything", None, "NVDA") is None
    assert src.risk_words("anything", "  ", "NVDA") is None


def test_mask_dollars():
    assert src.mask_dollars("Record $57.0B in revenue, $4.5 trillion cap, $120 PT") == \
        "Record $X in revenue, $X cap, $X PT"
    assert src.mask_dollars("Buying 5,000 chips") == "Buying 5,000 chips"
    assert src.mask_dollars(None) is None


def test_company_names(tmp_path):
    p = tmp_path / "w.md"
    p.write_text("| TICKER | Company |\n|---|---|\n| NVDA | NVIDIA Corp | Core |\n")
    assert src.company_names(str(p)) == {"NVDA": frozenset({"nvidia", "corp"})}
    assert src.company_names(str(tmp_path / "missing.md")) == {}


# ── rows ────────────────────────────────────────────────────────────────────

def test_build_rows_keeps_the_raw_fields():
    out = src.build_rows([_raw(), _raw(sid=2, sig="SCREENED_OUT", tone="neg", nf="false",
                                       wph={"would_pass": True, "override_skipped": "cooldown"},
                                       vid=None, title="Nvidia export curbs widen")],
                         {"NVDA": frozenset({"nvidia"})})
    a, b = out
    assert a["et"] == "2026-11-26 07:24" and a["verdict_et"] == "2026-11-26 07:27"
    assert a["new_fact"] is True and a["verdict_action"] == "HOLD" and a["plan_id"] == 57
    assert a["adverse_groups"] == ["credit"] and a["risk_words"] == []
    assert a["would_pass_held_negative"] is None and a["override_skipped"] is None
    assert b["new_fact"] is False and b["would_pass_held_negative"] is True
    assert b["override_skipped"] == "cooldown" and b["verdict_id"] is None
    assert b["adverse_groups"] == ["export_regulatory"]


def test_build_rows_tolerates_odd_values():
    r = src.build_rows([_raw(nf="maybe", wph={"would_pass": "yes"}, prisk=None, title=None)], {})[0]
    assert r["new_fact"] is None and r["would_pass_held_negative"] is None
    assert r["risk_words"] is None and r["adverse_groups"] == []


# ── report and export ───────────────────────────────────────────────────────

def _rows():
    return src.build_rows([
        _raw(sid=1),
        _raw(sid=2, override="held_negative_or_rating", tone="neg", vfinal="HOLD",
             title="Nvidia slumps on export halt"),
        _raw(sid=3, sig="SCREENED_OUT", tone="neutral", nf="false",
             wph={"would_pass": False}, vid=None, title="Did Jensen Huang's Alleged Spy act"),
        _raw(sid=4, sig="SCREENED_OUT", tone="neg", nf="false",
             wph={"would_pass": True, "override_skipped": "cooldown"}, vid=None,
             title="Nvidia falls in China, $5 billion at stake"),
    ], {})


def test_render_counts_only_and_no_titles():
    rows = _rows()
    text = src.render(rows, "live", 14, TS, "abc1234")
    assert "Rows: 4 (SCREENED_IN 2, SCREENED_OUT 2)" in text
    assert "| neutral | 1 | 1 | 2 |" in text            # title_tone by outcome
    assert "| held_negative_or_rating | 1 | 1 |" in text  # override by verdict action
    assert f"sha {src.ADVERSE_TERMS_SHA}" in text and f"sha {src.RISK_MATCH_SHA}" in text
    assert src.OPEN_LABEL in text
    for r in rows:
        assert r["title"] not in text
    assert "$" not in text
    assert "%" not in text                               # no rate is printed


def test_export_masks_and_drops_the_url():
    ex = src.export_rows(_rows())
    assert ex[3]["title"] == "Nvidia falls in China, $X at stake"
    assert all("url" not in r for r in ex)
    assert ex[0]["primary_risk"] == RISK57
    assert src.export_rows([]) == []


def test_sql_shape():
    q = src._SQL
    assert "provenance->>'held' = 'true'" in q and "SCREENED_IN', 'SCREENED_OUT'" in q
    assert "left(sv.provenance->'trigger'->>'url', 300) = s.decision->>'url'" in q
    assert "tp.created_at <= s.timestamp" in q and "tp.closed_at > s.timestamp" in q
    assert "make_interval(hours => %(window_h)s)" in q
    for w in ("insert", "update", "delete"):
        assert w not in q.lower().split()


# ── main ────────────────────────────────────────────────────────────────────

def test_main_refuses_other_ports(capsys):
    with patch.dict(os.environ, {"SUPABASE_PORT": "5432"}):
        assert src.main([]) == 2
    assert "6543" in capsys.readouterr().err


def test_main_reads_once_read_only_and_rolls_back(tmp_path):
    cur = MagicMock()
    cur.fetchall.return_value = [_raw(), _raw(sid=2, sig="SCREENED_OUT", vid=None, nf="false")]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    fake_db = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=MagicMock())
    out, ex = tmp_path / "r.md", tmp_path / "r.json"
    with patch.dict(os.environ, {"SUPABASE_PORT": "6543"}), \
         patch.dict("sys.modules", {"db": fake_db}):
        assert src.main(["--out", str(out), "--export", str(ex),
                         "--watchlist", str(tmp_path / "none.md")]) == 0
    sql = [c.args[0] for c in cur.execute.call_args_list]
    assert sql[0] == "SET TRANSACTION READ ONLY" and sql[2] == src._SQL
    assert cur.execute.call_args_list[2].args[1]["window_h"] == src.VERDICT_WINDOW_H
    conn.rollback.assert_called_once()
    fake_db._put_conn.assert_called_once_with(conn)
    assert "Rows: 2" in out.read_text()
    data = json.loads(ex.read_text())
    assert data["adverse_terms_sha"] == src.ADVERSE_TERMS_SHA and len(data["rows"]) == 2
    assert data["label_status"] == src.OPEN_LABEL
    assert Counter(r["outcome"] for r in data["rows"]) == {"SCREENED_IN": 1, "SCREENED_OUT": 1}
