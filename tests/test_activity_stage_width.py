"""Every literal activity stage name fits activity_feed.stage (audit
2026-11-10 16:53 #11: three 21-character names failed varchar(20) and went
to the spill file)."""
import pathlib
import re

import db

ROOT = pathlib.Path(__file__).resolve().parents[1]
_CALL = re.compile(r"log_activity(?:_direct|_queued)?\(\s*\"([a-z_]+)\"", re.S)


def _stage_width() -> int:
    m = re.search(r"\(\d+, \"ALTER TABLE activity_feed ALTER COLUMN stage TYPE VARCHAR\((\d+)\)",
                  pathlib.Path(db.__file__).read_text())
    return int(m.group(1))


def _stages() -> dict:
    out = {}
    for path in [*ROOT.glob("*.py"), *ROOT.glob("agents/*.py"), *ROOT.glob("ingestion/*.py")]:
        for name in _CALL.findall(path.read_text()):
            out.setdefault(name, path.name)
    return out


def test_the_scan_finds_the_known_long_names():
    names = _stages()
    for n in ("stale_record_at_entry", "candidate_scan_funnel", "quote_fallback_shadow"):
        assert n in names


def test_every_stage_fits_the_column():
    width = _stage_width()
    assert width >= 40
    too_long = {n: f for n, f in _stages().items() if len(n) > width}
    assert not too_long, too_long


def test_create_table_matches_the_migration():
    src = pathlib.Path(db.__file__).read_text()
    block = src[src.index("CREATE TABLE IF NOT EXISTS activity_feed"):]
    assert re.search(r"stage\s+VARCHAR\((\d+)\)", block[:400]).group(1) == str(_stage_width())
