"""
Run once to create all tables in Supabase.
Usage: python setup_db.py
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10),
    side        VARCHAR(4),
    quantity    DECIMAL,
    price       DECIMAL,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    reason      TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10),
    signal      VARCHAR(10),
    confidence  DECIMAL,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    reasoning   TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id             SERIAL PRIMARY KEY,
    timestamp      TIMESTAMPTZ DEFAULT NOW(),
    total_value    DECIMAL,
    cash           DECIMAL,
    positions_json JSONB
);
"""

_VERIFY = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('trades', 'signals', 'portfolio_snapshots')
ORDER BY table_name;
"""


def main() -> None:
    host = os.getenv("SUPABASE_HOST")
    port = int(os.getenv("SUPABASE_PORT", "5432"))
    dbname = os.getenv("SUPABASE_DB", "postgres")
    user = os.getenv("SUPABASE_USER", "postgres")
    password = os.getenv("SUPABASE_PASSWORD")

    if not host or not password:
        print("ERROR: SUPABASE_HOST and SUPABASE_PASSWORD must be set in .env")
        sys.exit(1)

    print(f"Connecting to {host}:{port}/{dbname} ...")
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname,
        user=user, password=password, sslmode="require",
    )

    try:
        with conn.cursor() as cur:
            cur.execute(_SQL)
        conn.commit()
        print("Migration executed.\n")

        with conn.cursor() as cur:
            cur.execute(_VERIFY)
            found = [row[0] for row in cur.fetchall()]

        expected = {"trades", "signals", "portfolio_snapshots"}
        for table in sorted(expected):
            status = "OK" if table in found else "MISSING"
            print(f"  {table:<25} {status}")

        if expected == set(found):
            print("\nAll three tables created successfully.")
        else:
            missing = expected - set(found)
            print(f"\nERROR: missing tables: {missing}")
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
