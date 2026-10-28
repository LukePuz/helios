"""Back up and (optionally) repair rows the test suite wrote to production.

Professor finding C9 / OPEN-QUESTIONS Q29. Before 9eb81fd the tests could reach
the live DB and wrote:
  - fake NVDA EXIT outcome rows (approval_id=1, entry $100) in trade_outcomes
  - real outcome rows closed at exit_price=100 by close_trade_outcome
  - fake "Auto-executed TRIM 29 shares @ ~$100.00" rows in activity_feed
  - approval 10's status/decided_at (the original values can't be recovered)

Default run is read-only: it saves the affected tables to CSV under
~/helios-db-backups/<timestamp>/ and prints exactly what --apply would change.
--apply makes the changes in one transaction. Whether to apply is Luke's call.

    venv/bin/python scripts/repair_test_artifacts.py            # backup + dry run
    venv/bin/python scripts/repair_test_artifacts.py --apply    # backup + repair
"""
import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db  # noqa: E402

FAKE_OUTCOMES = ("FROM trade_outcomes WHERE action='EXIT' AND ticker='NVDA' "
                 "AND approval_id=1 AND entry_price=100")
CLOSED_BY_TEST = ("FROM trade_outcomes WHERE exit_price=100 "
                  "AND NOT (action='EXIT' AND ticker='NVDA' AND approval_id=1 AND entry_price=100)")
FAKE_FEED = ("FROM activity_feed WHERE stage='execution' AND ticker='NVDA' "
             "AND message LIKE 'Auto-executed TRIM 29 shares @ ~$100.00%%'")
REOPEN_COLUMNS = ("exit_price", "exit_date", "outcome_pct", "was_correct",
                  "exit_benchmark_price", "benchmark_return_pct", "alpha_pct")


def backup(cur, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir.parent, 0o700)
    for table in ("trade_outcomes", "pending_approvals", "activity_feed"):
        cur.execute(f"SELECT * FROM {table}")
        cols = [d[0] for d in cur.description]
        with open(out_dir / f"{table}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(cur.fetchall())
        print(f"backed up {table} -> {out_dir / (table + '.csv')}")


def plan(cur) -> None:
    cur.execute(f"SELECT id, entry_date::date, trading_env, exit_date IS NOT NULL {FAKE_OUTCOMES} ORDER BY id")
    rows = cur.fetchall()
    print(f"\nDELETE {len(rows)} fake outcome rows (id, entry date, env, closed):")
    for r in rows:
        print("  ", r)

    cur.execute(f"SELECT id, ticker, action, entry_date::date, exit_date, trading_env {CLOSED_BY_TEST} ORDER BY id")
    rows = cur.fetchall()
    print(f"\nREOPEN {len(rows)} real rows closed at $100 by a test "
          f"(clear {', '.join(REOPEN_COLUMNS)}):")
    for r in rows:
        print("  ", r)
    print("   Check first: if the position was really sold since, the row should be closed "
          "at that sale's price instead of reopened.")

    cur.execute(f"SELECT count(*), min(ts), max(ts) {FAKE_FEED}")
    print("\nDELETE fake activity_feed rows (count, first, last):", cur.fetchone())

    cur.execute("SELECT id, ticker, action, status, decided_at FROM pending_approvals WHERE id=10")
    print("\nNO CHANGE to approval 10 (originals unrecoverable; flag it in analysis):", cur.fetchone())


def apply(cur) -> None:
    cur.execute(f"DELETE {FAKE_OUTCOMES}")
    print(f"deleted {cur.rowcount} fake outcome rows")
    sets = ", ".join(f"{c}=NULL" for c in REOPEN_COLUMNS)
    cur.execute(f"UPDATE trade_outcomes SET {sets} WHERE id IN (SELECT id {CLOSED_BY_TEST})")
    print(f"reopened {cur.rowcount} real outcome rows")
    cur.execute(f"DELETE {FAKE_FEED}")
    print(f"deleted {cur.rowcount} fake activity_feed rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="make the changes (default: dry run)")
    args = ap.parse_args()

    out_dir = Path.home() / "helios-db-backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            backup(cur, out_dir)
            plan(cur)
            if args.apply:
                print("\nAPPLYING")
                apply(cur)
                conn.commit()
            else:
                conn.rollback()
                print("\nDry run only. Nothing changed. Re-run with --apply to repair.")
    except Exception:
        conn.rollback()
        raise
    finally:
        db._put_conn(conn)


if __name__ == "__main__":
    main()
