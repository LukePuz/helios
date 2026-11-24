"""scripts/rescale_stale_head.py: the dry run finds old-basis steps in stored
closes; --apply rescales one ticker's rows before the step in one transaction
and refuses when the step no longer matches the factor. Synthetic data only."""
import importlib.util
import os
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest


def _script():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "rescale_stale_head.py")
    spec = importlib.util.spec_from_file_location("rescale_stale_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()


def _rows(step_at=100, ratio=0.1277, vol_step=6.0, n=250):
    d, rows = date(2025, 7, 1), []
    for i in range(n):
        c = 200.0 * (1 + 0.01 * ((i * 7) % 5 - 2) / 2)
        v = 1e6
        if i >= step_at:
            c *= ratio
            v *= vol_step
        rows.append({"date": d + timedelta(days=i), "close": c, "volume": v})
    return rows


def test_parse_factor():
    assert S.parse_factor("1/8") == 0.125 and S.parse_factor("0.5") == 0.5 and S.parse_factor("2") == 2.0
    with pytest.raises(ValueError):
        S.parse_factor("0")


def test_candidates_sndk_like():
    c = S.candidates(_rows())
    assert len(c) == 1
    c = c[0]
    assert c["step_date"] == (date(2025, 7, 1) + timedelta(days=100)).isoformat()
    assert c["nearest"] == "1/8" and c["rows_before"] == 100      # x0.1277 is 2.2% from 1/8
    assert 5.5 < c["volume_ratio"] < 6.5 and c["spike_ratio"] == 1.0


def test_candidates_two_for_one_and_clean():
    c = S.candidates(_rows(ratio=0.5, vol_step=2.0))
    assert len(c) == 1 and c[0]["nearest"] == "1/2"
    assert S.candidates(_rows(ratio=1.0, vol_step=1.0)) == []


def _conn(rows, rowcount=0):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [(r["date"], r["close"], r["volume"]) for r in rows]
    cur.rowcount = rowcount
    return conn, cur


def test_apply_rescales_before_the_step():
    rows = _rows()
    conn, cur = _conn(rows, rowcount=100)
    res = S.apply(conn, "SNDK", 0.125, None)
    assert res["rows"] == 100 and res["before"] == S.candidates(rows)[0]["step_date"]
    sql, params = cur.execute.call_args.args
    assert sql.startswith("UPDATE price_history") and params[:5] == (0.125,) * 5
    assert params[5:] == ("SNDK", res["before"])
    conn.commit.assert_called_once()


def test_apply_refuses_a_wrong_or_repeated_factor():
    conn, cur = _conn(_rows())
    with pytest.raises(ValueError, match="no stored step"):
        S.apply(conn, "SNDK", 0.5, None)                    # x0.1277 is not ~1/2
    conn.commit.assert_not_called()
    conn, cur = _conn(_rows(ratio=1.0, vol_step=1.0))        # already rescaled: no step left
    with pytest.raises(ValueError, match="0 candidate steps"):
        S.apply(conn, "SNDK", 0.125, None)
    conn.commit.assert_not_called()


def test_apply_needs_before_with_two_steps():
    rows = _rows()
    rows[200]["close"] *= 0.5
    for r in rows[201:]:
        r["close"] *= 0.5
    conn, _ = _conn(rows)
    with pytest.raises(ValueError, match="pass --before"):
        S.apply(conn, "X", 0.5, None)


def test_dry_run_is_read_only():
    conn, cur = _conn(_rows())
    found = S.dry_run(conn, ["SNDK"])
    assert len(found) == 1 and found[0][0] == "SNDK"
    assert cur.execute.call_args_list[0].args[0] == "SET TRANSACTION READ ONLY"
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()


def test_main_apply_argument_checks(monkeypatch, capsys):
    import db
    monkeypatch.setattr(db, "_get_conn", lambda: MagicMock())
    monkeypatch.setattr(db, "_put_conn", lambda c: None)
    assert S.main(["--apply", "--ticker", "SNDK"]) == 2
    assert "needs exactly one --ticker and --factor" in capsys.readouterr().out
