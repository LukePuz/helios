"""Apply exactly one plain migration from db._MIGRATIONS now, ahead of the next
stack start, the way init_db does: its SQL and its schema_version row in one
transaction, so init_db later finds it recorded and skips it. No other
migration runs. The SQL runs under SET LOCAL lock_timeout (default 5s,
shorter than init_db's 30s) and statement_timeout (default 60s); a timeout
rolls back both the change and the row, and the exit status is 1.

    env -C ~/Desktop/trading_bot venv/bin/python scripts/apply_migration.py --id 92 --dry-run
    env -C ~/Desktop/trading_bot venv/bin/python scripts/apply_migration.py --id 92

--dry-run only reads (READ ONLY transaction, rolled back): whether the
version is recorded, whether an ADD COLUMN's column already exists, and the
locks other sessions hold on the table. CREATE INDEX CONCURRENTLY migrations
are refused here; scripts/apply_concurrent_indexes.py builds those.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True, help="the migration version in db._MIGRATIONS")
    ap.add_argument("--dry-run", action="store_true", help="read only, change nothing")
    ap.add_argument("--lock-timeout", default="5s")
    ap.add_argument("--statement-timeout", default="60s")
    args = ap.parse_args(argv)
    import time
    import db
    t = time.perf_counter()
    try:
        out = db.apply_one_migration(args.id, dry_run=args.dry_run, lock_timeout=args.lock_timeout,
                                     statement_timeout=args.statement_timeout)
    except ValueError as e:
        print(f"error: {e}")
        return 2
    print(f"migration {out['version']}: {out['status']}: {out['sql']}")
    for key in ("recorded", "column_exists", "other_locks"):
        if key in out:
            print(f"  {key}: {out[key]}")
    print(f"done in {time.perf_counter() - t:.1f} s")
    return 0 if out["status"] in ("applied", "already applied", "pending") else 1


if __name__ == "__main__":
    sys.exit(main())
