"""helios3d.panel.timing on fixture log lines and fixture DB rows (no DB, no logs dir)."""
from datetime import datetime

import pytest

from helios3d.panel import timing

FIXTURE_LOG = """\
[2026-10-27 05:00:00,000: INFO/ForkPoolWorker-1] [NVDA] Analyzing with qwen2.5:72b-instruct-q3_K_M...
[2026-10-27 05:00:30,000: INFO/MainProcess] Task tasks.deep_analyze_event[abc] received
[2026-10-27 05:01:40,000: INFO/ForkPoolWorker-1] [NVDA] → WATCH | conviction=6/10 | UDR=1.5 | pass_to_risk=False
[2026-10-27 05:02:00,000: INFO/ForkPoolWorker-1] [MSFT] Analyzing with qwen2.5:72b-instruct-q3_K_M...
[2026-10-27 05:04:00,000: INFO/ForkPoolWorker-1] [MSFT] → OPEN | conviction=7/10 | UDR=2.5 | pass_to_risk=True
[2026-10-27 05:05:00,000: INFO/ForkPoolWorker-1] [AMD] Analyzing with qwen2.5:72b-instruct-q3_K_M...
[2026-10-27 05:11:40,000: INFO/ForkPoolWorker-1] [AMD] Stub recommendation (timed out after 400s)
[2026-10-27 05:12:00,000: INFO/ForkPoolWorker-1] [META] Analyzing with qwen2.5:72b-instruct-q3_K_M...
[2026-10-27 05:13:20,000: INFO/ForkPoolWorker-1] [META] → HOLD | conviction=6/10 | UDR=2.0 | pass_to_risk=False
[2026-10-27 10:00:00,500: INFO/ForkPoolWorker-1] [GOOG] Analyzing with qwen2.5:72b-instruct-q3_K_M...
[2026-10-27 10:01:30,500: INFO/ForkPoolWorker-1] [GOOG] → WATCH | conviction=5/10 | UDR=1.0 | pass_to_risk=False
[2026-10-27 11:00:00,000: INFO/ForkPoolWorker-2] [TSLA] Analyzing with qwen2.5:7b...
[2026-10-27 11:00:20,000: INFO/ForkPoolWorker-2] [TSLA] → WATCH | conviction=5/10 | UDR=1.0 | pass_to_risk=False
[2026-10-27 12:00:00,000: INFO/ForkPoolWorker-1] [AMZN] Analyzing with qwen2.5:72b-instruct-q3_K_M...
[2026-10-27 12:02:00,000: INFO/ForkPoolWorker-1] [NFLX] → WATCH | conviction=5/10 | UDR=1.0 | pass_to_risk=False
not a log line
"""


def _calls():
    return list(timing.parse_log_lines(FIXTURE_LOG.splitlines(True)))


def test_pairs_start_and_verdict_per_worker():
    calls = _calls()
    got = [(c.start.strftime("%H:%M"), round(c.seconds, 1), c.model) for c in calls]
    assert got == [
        ("05:00", 100.0, "qwen2.5:72b-instruct-q3_K_M"),
        ("05:02", 120.0, "qwen2.5:72b-instruct-q3_K_M"),
        ("05:12", 80.0, "qwen2.5:72b-instruct-q3_K_M"),   # AMD stub dropped
        ("10:00", 90.0, "qwen2.5:72b-instruct-q3_K_M"),
        ("11:00", 20.0, "qwen2.5:7b"),
    ]                                                      # AMZN/NFLX mismatch dropped
    assert all(c.source == "logs" for c in calls)


def test_model_and_date_filter():
    calls = timing.filter_calls(_calls(), "2026-10-27", "2026-10-27", "72b")
    assert len(calls) == 4
    assert timing.filter_calls(_calls(), "2026-10-28", "2026-11-02", "72b") == []


def test_percentile_and_summary():
    assert timing.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == pytest.approx(9.1)
    s = timing.summarize([100.0, 120.0, 80.0, 90.0])
    assert s["n"] == 4 and s["median"] == 95.0 and s["p90"] == pytest.approx(114.0)
    assert timing.summarize([])["n"] == 0


def test_capacity_matches_phase0_table():
    # Phase 0 report, 10-26 basis: 97.9 s median, 125.0 s p90.
    assert timing.capacity(97.9, timing.PREOPEN_SECONDS) == {"calls": 165, 6: 27, 7: 23, 8: 20}
    assert timing.capacity(125.0, timing.PREOPEN_SECONDS) == {"calls": 129, 6: 21, 7: 18, 8: 16}
    assert timing.capacity(97.9, timing.HOUR_SECONDS) == {"calls": 36, 6: 6, 7: 5, 8: 4}
    assert timing.capacity(float("nan"), 3600)["calls"] == 0
    assert timing.PREOPEN_SECONDS == 16200


def test_preopen_and_hour_buckets():
    calls = timing.filter_calls(_calls(), "2026-10-27", "2026-10-27", "72b")
    assert [timing.in_preopen(c) for c in calls] == [True, True, True, False]
    assert timing.in_preopen(timing.Call(datetime(2026, 10, 27, 9, 29), 1, "m", "logs"))
    assert not timing.in_preopen(timing.Call(datetime(2026, 10, 27, 9, 30), 1, "m", "logs"))
    assert timing.by_start_hour(calls) == {5: [100.0, 120.0, 80.0], 10: [90.0]}


def test_db_rows_shift_to_call_start():
    rows = [(datetime(2026, 11, 2, 9, 0, 0), "qwen2.5:72b-instruct-q3_K_M", 120_000_000_000),
            (datetime(2026, 11, 2, 9, 5, 0), "qwen2.5:72b-instruct-q3_K_M", None),
            (datetime(2026, 11, 2, 9, 10, 0), None, 85_500_000_000)]
    calls = timing.calls_from_db_rows(rows)
    assert [(c.start, c.seconds, c.model) for c in calls] == [
        (datetime(2026, 11, 2, 8, 58, 0), 120.0, "qwen2.5:72b-instruct-q3_K_M"),
        (datetime(2026, 11, 2, 9, 8, 34, 500000), 85.5, "?"),
    ]


def test_report_and_cli_on_fixture_logs(tmp_path, capsys):
    (tmp_path / "celery_llm_worker.log").write_text(FIXTURE_LOG)
    rc = timing.main(["--source", "logs", "--logs-dir", str(tmp_path),
                      "--start", "2026-10-27", "--end", "2026-10-27"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "all                        4      95.0    114.0" in out
    assert "pre-open 05:00-09:30       3     100.0    116.0" in out
    assert "05:00" in out and "10:00" in out and "11:00" not in out


def test_db_source_failure_is_reported_not_raised(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("no db in tests")
    monkeypatch.setattr(timing, "read_db_calls", boom)
    assert timing.main(["--source", "db"]) == 0
    assert "db: unavailable (RuntimeError" in capsys.readouterr().out


def test_default_paths_come_from_the_checkout_or_env(monkeypatch):
    """No hard-coded home-directory paths."""
    import importlib
    import pathlib
    from helios3d.panel import timing
    src = pathlib.Path(timing.__file__).read_text()
    assert "/home/" not in src
    root = pathlib.Path(timing.__file__).resolve().parents[2]
    monkeypatch.delenv("HELIOS3D_REPO", raising=False)
    monkeypatch.delenv("HELIOS3D_LOGS_DIR", raising=False)
    t = importlib.reload(timing)
    assert pathlib.Path(t.DEFAULT_REPO) == root and pathlib.Path(t.DEFAULT_LOGS_DIR) == root / "logs"
    monkeypatch.setenv("HELIOS3D_REPO", "/srv/x")
    monkeypatch.setenv("HELIOS3D_LOGS_DIR", "/srv/logs")
    t = importlib.reload(timing)
    assert (t.DEFAULT_REPO, t.DEFAULT_LOGS_DIR) == ("/srv/x", "/srv/logs")
    monkeypatch.delenv("HELIOS3D_REPO")
    monkeypatch.delenv("HELIOS3D_LOGS_DIR")
    importlib.reload(timing)
