"""scripts/alert_triage.py: cursor, grouping, new-type detection."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import alert_triage  # noqa: E402

import alert_policy as ap


def _write(logs, *texts, status="sent", source="tasks.py:x"):
    for t in texts:
        sev, typ = ap.classify(t)
        ap.record(source, t, sev, typ, status, outbox_path=logs / alert_triage.OUTBOX_NAME)


def _run(logs, *extra, capsys):
    rc = alert_triage.main(["--logs-dir", str(logs), *extra])
    return rc, capsys.readouterr().out


def test_groups_repeats_and_flags_new_types(tmp_path, capsys):
    _write(tmp_path, "🚨 UNPROTECTED POSITION — AAPL\n1", "🚨 UNPROTECTED POSITION — AAPL\n2",
           "⚠️ <b>Analyst falling back to stub</b> — AAPL\n• x")
    _write(tmp_path, "⚠️ <b>Analyst falling back to stub</b> — AAPL\n• y", status="held")
    rc, out = _run(tmp_path, capsys=capsys)
    assert rc == 2                                    # a new critical type
    assert "4 new alert row(s), 2 type(s), 1 held" in out
    assert "×2" in out and "New alert types" in out and "critical:" in out


def test_second_run_reads_only_new_rows_and_known_types_are_not_new(tmp_path, capsys):
    _write(tmp_path, "🚨 SHORT POSITION — AAPL")
    _run(tmp_path, capsys=capsys)
    rc, out = _run(tmp_path, capsys=capsys)
    assert rc == 0 and "no new alerts" in out
    _write(tmp_path, "🚨 SHORT POSITION — AAPL")
    rc, out = _run(tmp_path, capsys=capsys)
    assert rc == 0 and "1 new alert row(s)" in out and "NEW" not in out


def test_no_advance_leaves_the_cursor(tmp_path, capsys):
    _write(tmp_path, "🚨 SHORT POSITION — AAPL")
    _run(tmp_path, "--no-advance", capsys=capsys)
    assert not (tmp_path / alert_triage.CURSOR_NAME).exists()
    rc, out = _run(tmp_path, capsys=capsys)
    assert "1 new alert row(s)" in out


def test_rotation_and_partial_lines(tmp_path, capsys):
    outbox = tmp_path / alert_triage.OUTBOX_NAME
    _write(tmp_path, "🚨 SHORT POSITION — AAPL")
    _run(tmp_path, capsys=capsys)
    outbox.unlink()                                  # rotated
    _write(tmp_path, "⚠️ <b>Analyst falling back to stub</b> — AAPL")
    with open(outbox, "a") as f:
        f.write('{"partial": ')                      # being written
    rc, out = _run(tmp_path, capsys=capsys)
    assert "1 new alert row(s)" in out and rc == 0   # a new low type is not exit 2
    with open(outbox, "a") as f:
        f.write('"x"}\n')
    _, out = _run(tmp_path, capsys=capsys)
    assert "1 new alert row(s)" in out


def test_a_file_recreated_on_the_same_inode_is_read_from_the_start(tmp_path, capsys):
    # Some filesystems hand a freed inode straight back, so a rotated outbox
    # can keep the old inode number and be longer than the saved offset.
    outbox = tmp_path / alert_triage.OUTBOX_NAME
    _write(tmp_path, "🚨 SHORT POSITION — AAPL")
    _run(tmp_path, capsys=capsys)
    ino = outbox.stat().st_ino
    with open(outbox, "r+") as f:                    # same inode, new content
        f.truncate(0)
    _write(tmp_path, "⚠️ <b>Analyst falling back to stub</b> — AAPL, a longer first row than before")
    _write(tmp_path, "🚨 SHORT POSITION — MSFT")
    assert outbox.stat().st_ino == ino
    _, out = _run(tmp_path, capsys=capsys)
    assert "2 new alert row(s)" in out


def test_missing_outbox_and_corrupt_cursor(tmp_path, capsys):
    (tmp_path / alert_triage.CURSOR_NAME).write_text("{bad")
    rc, out = _run(tmp_path, capsys=capsys)
    assert rc == 0 and "no new alerts" in out


def test_writes_nothing_but_the_cursor(tmp_path, capsys):
    _write(tmp_path, "🚨 SHORT POSITION — AAPL")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    _run(tmp_path, capsys=capsys)
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert set(after) - set(before) == {alert_triage.CURSOR_NAME}
    assert after[alert_triage.OUTBOX_NAME] == before[alert_triage.OUTBOX_NAME]
    cur = json.loads(after[alert_triage.CURSOR_NAME])
    assert cur["offset"] == len(before[alert_triage.OUTBOX_NAME])


def test_rows_written_before_a_rotation_are_not_lost(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ap, "OUTBOX_ROTATE_BYTES", 10 ** 9)
    _write(tmp_path, "🚨 SHORT POSITION — AAPL")
    _run(tmp_path, capsys=capsys)
    _write(tmp_path, "🚨 SHORT POSITION — MSFT")            # after the last read
    monkeypatch.setattr(ap, "OUTBOX_ROTATE_BYTES", 1)
    _write(tmp_path, "🚨 SHORT POSITION — NVDA")            # rotates, then lands in the new file
    assert (tmp_path / (alert_triage.OUTBOX_NAME + ".1")).exists()
    _, out = _run(tmp_path, capsys=capsys)
    assert "2 new alert row(s)" in out
