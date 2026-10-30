"""Backups keep 14 daily and 8 weekly (Sunday) copies (Luke D16)."""
import importlib.util
from datetime import date, timedelta
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "backup_db", Path(__file__).resolve().parent.parent / "scripts" / "backup_db.py")
backup_db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup_db)


def test_rotation_keeps_14_daily_and_8_sundays(tmp_path):
    start = date(2026, 6, 1)
    for i in range(120):
        d = start + timedelta(days=i)
        (tmp_path / f"helios-{d:%Y%m%d}-1915.tar.gpg").write_text("x")
    backup_db._rotate(tmp_path)
    left = sorted(p.name for p in tmp_path.iterdir())
    newest = [f"helios-{start + timedelta(days=i):%Y%m%d}-1915.tar.gpg" for i in range(106, 120)]
    assert set(newest) <= set(left)
    sundays = [n for n in left if n not in newest]
    assert len(sundays) <= 8
    assert all(date(int(n[7:11]), int(n[11:13]), int(n[13:15])).weekday() == 6 for n in sundays)
