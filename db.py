import logging
import os
from pathlib import Path

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

_pool: psycopg2.pool.ThreadedConnectionPool | None = None

_CREATE_TABLES = """
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


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            host=os.getenv("SUPABASE_HOST"),
            port=int(os.getenv("SUPABASE_PORT", "5432")),
            dbname=os.getenv("SUPABASE_DB", "postgres"),
            user=os.getenv("SUPABASE_USER", "postgres"),
            password=os.getenv("SUPABASE_PASSWORD"),
            sslmode="require",
        )
    return _pool


def _get_conn():
    return _get_pool().getconn()


def _put_conn(conn) -> None:
    _get_pool().putconn(conn)


def init_db() -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLES)
        conn.commit()
        logging.info("Database tables ready.")
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)


def log_trade(symbol: str, side: str, quantity: int, price: float, reason: str) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO trades (symbol, side, quantity, price, reason) VALUES (%s,%s,%s,%s,%s)",
                (symbol, side, quantity, price, reason),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_trade failed: {e}")
    finally:
        _put_conn(conn)


def log_signal(symbol: str, signal: str, confidence: float, reasoning: str) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO signals (symbol, signal, confidence, reasoning) VALUES (%s,%s,%s,%s)",
                (symbol, signal, confidence, reasoning),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_signal failed: {e}")
    finally:
        _put_conn(conn)


def log_snapshot(total_value: float, cash: float, positions: dict) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO portfolio_snapshots (total_value, cash, positions_json) VALUES (%s,%s,%s)",
                (total_value, cash, psycopg2.extras.Json(positions)),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_snapshot failed: {e}")
    finally:
        _put_conn(conn)


def get_recent_trades(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, side, quantity, price, timestamp, reason "
                "FROM trades ORDER BY timestamp DESC LIMIT %s",
                (limit,),
            )
            return [
                {"symbol": r[0], "side": r[1], "quantity": r[2],
                 "price": float(r[3]), "timestamp": r[4], "reason": r[5]}
                for r in cur.fetchall()
            ]
    except Exception as e:
        logging.error(f"get_recent_trades failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_recent_signals(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, signal, confidence, timestamp, reasoning "
                "FROM signals ORDER BY timestamp DESC LIMIT %s",
                (limit,),
            )
            return [
                {"symbol": r[0], "signal": r[1], "confidence": float(r[2] or 0),
                 "timestamp": r[3], "reasoning": r[4]}
                for r in cur.fetchall()
            ]
    except Exception as e:
        logging.error(f"get_recent_signals failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_portfolio_history(limit: int = 288) -> list[dict]:
    """Returns up to `limit` snapshots in chronological order (288 = 24h at 5-min ticks)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT timestamp, total_value, cash FROM portfolio_snapshots "
                "ORDER BY timestamp DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {"timestamp": r[0], "total_value": float(r[1]), "cash": float(r[2])}
            for r in reversed(rows)
        ]
    except Exception as e:
        logging.error(f"get_portfolio_history failed: {e}")
        return []
    finally:
        _put_conn(conn)
