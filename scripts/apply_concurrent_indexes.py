"""Build the pending CREATE INDEX CONCURRENTLY migrations (db._MIGRATIONS) now,
in a quiet window, instead of at the next stack start. Only those
migrations: no ALTER or other pending migration runs. Each is built
concurrently (writes go on), under db._MIGRATION_LOCK_TIMEOUT, and recorded
in schema_version, so init_db later skips it. A failed build leaves no
invalid index behind and no version row.

    env -C ~/Desktop/trading_bot venv/bin/python scripts/apply_concurrent_indexes.py --dry-run
    env -C ~/Desktop/trading_bot venv/bin/python scripts/apply_concurrent_indexes.py

--dry-run only reads (schema_version, when it exists; no CREATE TABLE) and
says when a real run would be refused. A real run needs the session port
(SUPABASE_PORT 5432, the default), not the 6543 transaction pooler, holds one
session and the concurrent-index advisory lock while it runs, and skips a
build another session is already doing. Exit status 1 when a build failed
or was skipped.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list pending builds, change nothing")
    args = ap.parse_args()
    import time
    import db
    t = time.perf_counter()
    out = db.apply_concurrent_index_migrations(dry_run=args.dry_run)
    sql = dict(db._MIGRATIONS)
    for version, status in out:
        print(f"migration {version}: {status}: {sql[version]}")
    if not out:
        print("no concurrent index migrations defined")
    print(f"done in {time.perf_counter() - t:.1f} s")
    return 1 if any(s == "failed" for _, s in out) else 0


if __name__ == "__main__":
    sys.exit(main())
