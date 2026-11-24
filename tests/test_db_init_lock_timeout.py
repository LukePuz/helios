"""init_db sets a lock timeout inside each migration (audit 2026-11-08 23:22 #1)."""
from unittest.mock import MagicMock, patch

import db


def test_each_new_migration_runs_under_a_lock_timeout():
    cur = MagicMock()
    # The last two plain migrations are new; CREATE INDEX CONCURRENTLY ones
    # run outside a transaction (tests/test_signals_index_migration.py).
    plain = [(v, s) for v, s in db._MIGRATIONS if not db._CONCURRENT_INDEX_RE.match(s)]
    cur.fetchall.return_value = [(v,) for v, _ in db._MIGRATIONS if (v, _) not in plain[-2:]]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        db.init_db()
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    sets = [i for i, q in enumerate(sqls) if q.startswith("SET LOCAL lock_timeout")]
    assert len(sets) == 2
    for i in sets:
        assert "30s" in sqls[i] and sqls[i + 1] in {s for _, s in plain[-2:]}
