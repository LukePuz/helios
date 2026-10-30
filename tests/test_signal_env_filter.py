"""Paper verdicts must not reach live decisions (professor M35, 2026-10-28)."""
from unittest.mock import MagicMock, patch

import db


def _sql(fn, *args):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        try:
            fn(*args)
        except Exception:
            pass
    return [c.args for c in cur.execute.call_args_list if "FROM signals" in c.args[0]]


def test_latest_signal_filters_by_env():
    (sql, params), = _sql(db.get_latest_signal_for, "NVDA")
    assert "trading_env=%s" in sql and db.TRADING_ENV in params


def test_position_context_verdicts_filter_by_env():
    calls = _sql(db.get_position_context, "NVDA")
    assert calls and all("trading_env=%s" in sql for sql, _ in calls)


def test_every_non_verdict_signal_kind_is_excluded_from_latest_verdict():
    """A routing/shadow/error row read as the latest verdict makes a holding
    look like that row's confidence (CATALYST_GATE_ERROR is written at 0.0,
    so it read as conviction 0, a swap candidate). Every literal signal kind
    written in the code must be an analyst action or in the exclusion list."""
    import pathlib
    import re
    root = pathlib.Path(db.__file__).parent
    pat = re.compile(r'(?:log_signal|_log_route_exit)\(\s*[\w\[\]"\'.]+,\s*"([A-Z_]{3,})"')
    kinds = set()
    for f in [*root.glob("*.py"), *root.glob("agents/*.py"), *root.glob("ingestion/*.py")]:
        kinds |= set(pat.findall(f.read_text()))
    verdicts = {"OPEN", "ADD", "HOLD", "TRIM", "EXIT", "WATCH", "PASS"}
    assert "CATALYST_GATE_ERROR" in kinds          # the pattern actually finds them
    assert kinds - verdicts <= set(db._NON_ANALYST_SIGNALS), kinds - verdicts - set(db._NON_ANALYST_SIGNALS)
