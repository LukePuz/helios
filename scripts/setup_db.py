"""
Run once (or any time) to create/update all tables in Supabase.
Usage: python scripts/setup_db.py

This used to hand-maintain its own copy of the schema (CREATE TABLE for only
3 of the ~14 tables the app actually uses) which had drifted badly out of
sync with db.py's real schema — running it on a fresh database would have
created an incomplete schema and then main.py would fail on every table
db.py expects but this script never created. It now delegates entirely to
db.init_db(), the single source of truth for schema (base tables +
versioned migrations tracked in schema_version).
"""

import logging
import sys
from pathlib import Path

# Run from scripts/; the modules it imports live at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


def main() -> None:
    import db

    print("Initializing database schema (see db.py for the authoritative schema)...")
    try:
        db.init_db()
    except Exception as e:
        print(f"ERROR: database setup failed: {e}")
        sys.exit(1)

    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' ORDER BY table_name"
            )
            tables = [row[0] for row in cur.fetchall()]
    finally:
        db._put_conn(conn)

    print(f"\n{len(tables)} tables present:")
    for t in tables:
        print(f"  {t}")
    print("\nDone.")


if __name__ == "__main__":
    main()
