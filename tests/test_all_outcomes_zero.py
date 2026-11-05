"""get_all_outcomes keeps a 0.0 outcome (breakeven) as 0.0, not None."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import db


def test_zero_values_survive():
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    now = datetime.now(timezone.utc)
    cur.fetchall.return_value = [
        (1, "NVDA", "OPEN", 7, "Accelerator", 100.0, now, 0.0, None, 101.5, 100.0, now, 0.0, False, 0.0),
    ]
    with patch("db._get_conn", return_value=conn), patch("db._put_conn"):
        (row,) = db.get_all_outcomes(10)
    assert row["outcome_pct"] == 0.0 and row["alpha_pct"] == 0.0
    assert row["review_30d_price"] == 0.0 and row["review_60d_price"] is None
    assert row["review_90d_price"] == 101.5 and row["exit_price"] == 100.0
