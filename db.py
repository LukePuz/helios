import logging
import os
from datetime import datetime, timedelta, timezone
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

CREATE TABLE IF NOT EXISTS market_regime_log (
    id                   SERIAL PRIMARY KEY,
    timestamp            TIMESTAMPTZ DEFAULT NOW(),
    regime               VARCHAR(20),
    score                DECIMAL,
    confidence           DECIMAL,
    signals_json         JSONB,
    portfolio_impact_json JSONB,
    reasoning            TEXT
);

CREATE TABLE IF NOT EXISTS pending_approvals (
    id                  SERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ DEFAULT NOW(),
    ticker              VARCHAR(10),
    action              VARCHAR(10),
    conviction          INTEGER,
    thesis_bucket       VARCHAR(20),
    position_size_pct   DECIMAL,
    stop_loss_price     DECIMAL,
    entry_strategy      TEXT,
    reasoning           TEXT,
    analyst_json        JSONB,
    risk_json           JSONB,
    status              VARCHAR(20) DEFAULT 'PENDING',
    decided_at          TIMESTAMPTZ,
    telegram_message_id BIGINT,
    expires_at          TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS trade_outcomes (
    id              SERIAL PRIMARY KEY,
    approval_id     INTEGER REFERENCES pending_approvals(id),
    ticker          VARCHAR(10),
    action          VARCHAR(10),
    conviction      INTEGER,
    thesis_bucket   VARCHAR(20),
    entry_price     DECIMAL,
    entry_date      TIMESTAMPTZ DEFAULT NOW(),
    review_30d_price DECIMAL,
    review_60d_price DECIMAL,
    review_90d_price DECIMAL,
    exit_price      DECIMAL,
    exit_date       TIMESTAMPTZ,
    outcome_pct     DECIMAL,
    was_correct     BOOLEAN,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    id           SERIAL PRIMARY KEY,
    ticker       VARCHAR(10),
    earnings_date DATE,
    confirmed    BOOLEAN DEFAULT FALSE,
    refreshed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, earnings_date)
);

CREATE TABLE IF NOT EXISTS position_stops (
    id                  SERIAL PRIMARY KEY,
    ticker              VARCHAR(10) UNIQUE,
    entry_price         DECIMAL,
    stop_price          DECIMAL,
    stop_method         TEXT,
    trailing_active     BOOLEAN DEFAULT FALSE,
    trailing_high       DECIMAL,
    trail_distance_pct  DECIMAL,
    thesis_bucket       VARCHAR(20),
    conviction          INTEGER,
    opened_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scanner_candidates (
    id                      SERIAL PRIMARY KEY,
    timestamp               TIMESTAMPTZ DEFAULT NOW(),
    ticker                  VARCHAR(10),
    scan_type               VARCHAR(20),
    composite_score         INTEGER,
    technical_score         INTEGER,
    momentum_score          INTEGER,
    catalyst_score          INTEGER,
    thesis_alignment_score  INTEGER,
    priority                VARCHAR(10),
    catalyst_detail         TEXT,
    passed_to_analyst       BOOLEAN,
    regime_at_scan          VARCHAR(20)
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


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------

def log_regime(
    regime: str,
    score: float,
    confidence: float,
    signals: dict,
    portfolio_impact: dict,
    reasoning: str,
) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO market_regime_log "
                "(regime, score, confidence, signals_json, portfolio_impact_json, reasoning) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    regime, score, confidence,
                    psycopg2.extras.Json(signals),
                    psycopg2.extras.Json(portfolio_impact),
                    reasoning,
                ),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_regime failed: {e}")
    finally:
        _put_conn(conn)


def get_latest_regime() -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT regime, score, confidence, signals_json, portfolio_impact_json, reasoning, timestamp "
                "FROM market_regime_log ORDER BY timestamp DESC LIMIT 1"
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "regime": row[0],
            "score": float(row[1]),
            "confidence": float(row[2]),
            "signals_json": row[3],
            "portfolio_impact_json": row[4],
            "reasoning": row[5],
            "timestamp": row[6],
        }
    except Exception as e:
        logging.error(f"get_latest_regime failed: {e}")
        return None
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# Pending approvals
# ---------------------------------------------------------------------------

def log_pending_approval(
    ticker: str,
    action: str,
    conviction: int,
    bucket: str,
    size_pct: float,
    stop_price: float,
    entry_strategy: str,
    reasoning: str,
    analyst_json: dict,
    risk_json: dict,
) -> int:
    """Insert a pending approval and return its row id."""
    from config import APPROVAL_EXPIRY_HOURS
    expires_at = datetime.now(timezone.utc) + timedelta(hours=APPROVAL_EXPIRY_HOURS)
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO pending_approvals "
                "(ticker, action, conviction, thesis_bucket, position_size_pct, stop_loss_price, "
                "entry_strategy, reasoning, analyst_json, risk_json, expires_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (
                    ticker, action, conviction, bucket, size_pct, stop_price,
                    entry_strategy, reasoning,
                    psycopg2.extras.Json(analyst_json),
                    psycopg2.extras.Json(risk_json),
                    expires_at,
                ),
            )
            row_id: int = cur.fetchone()[0]
        conn.commit()
        return row_id
    except Exception as e:
        conn.rollback()
        logging.error(f"log_pending_approval failed: {e}")
        return -1
    finally:
        _put_conn(conn)


def update_approval_status(
    approval_id: int,
    status: str,
    telegram_message_id: int | None = None,
) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE pending_approvals SET status=%s, decided_at=NOW(), telegram_message_id=%s "
                "WHERE id=%s",
                (status, telegram_message_id, approval_id),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"update_approval_status failed: {e}")
    finally:
        _put_conn(conn)


def set_approval_telegram_id(approval_id: int, telegram_message_id: int) -> None:
    """Store the Telegram message ID without changing status."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE pending_approvals SET telegram_message_id=%s WHERE id=%s",
                (telegram_message_id, approval_id),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"set_approval_telegram_id failed: {e}")
    finally:
        _put_conn(conn)


def get_pending_approvals(status: str | None = "PENDING") -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if status is None:
                cur.execute(
                    "SELECT id, ticker, action, conviction, thesis_bucket, position_size_pct, "
                    "stop_loss_price, entry_strategy, reasoning, analyst_json, risk_json, "
                    "status, timestamp, decided_at, expires_at "
                    "FROM pending_approvals ORDER BY timestamp DESC",
                )
            else:
                cur.execute(
                    "SELECT id, ticker, action, conviction, thesis_bucket, position_size_pct, "
                    "stop_loss_price, entry_strategy, reasoning, analyst_json, risk_json, "
                    "status, timestamp, decided_at, expires_at "
                    "FROM pending_approvals WHERE status=%s ORDER BY timestamp DESC",
                    (status,),
                )
            rows = cur.fetchall()
        return [
            {
                "id": r[0], "ticker": r[1], "action": r[2], "conviction": r[3],
                "thesis_bucket": r[4], "position_size_pct": float(r[5] or 0),
                "stop_loss_price": float(r[6] or 0) if r[6] else None,
                "entry_strategy": r[7], "reasoning": r[8],
                "analyst_json": r[9], "risk_json": r[10],
                "status": r[11], "timestamp": r[12], "decided_at": r[13],
                "expires_at": r[14],
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_pending_approvals failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_approval_by_id(approval_id: int) -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, ticker, action, conviction, thesis_bucket, position_size_pct, "
                "stop_loss_price, entry_strategy, reasoning, analyst_json, risk_json, status, expires_at "
                "FROM pending_approvals WHERE id=%s",
                (approval_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0], "ticker": row[1], "action": row[2], "conviction": row[3],
            "thesis_bucket": row[4], "position_size_pct": float(row[5] or 0),
            "stop_loss_price": float(row[6] or 0) if row[6] else None,
            "entry_strategy": row[7], "reasoning": row[8],
            "analyst_json": row[9], "risk_json": row[10],
            "status": row[11], "expires_at": row[12],
        }
    except Exception as e:
        logging.error(f"get_approval_by_id failed: {e}")
        return None
    finally:
        _put_conn(conn)


def expire_stale_approvals() -> int:
    """Mark PENDING approvals past their expires_at as EXPIRED. Returns count updated."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE pending_approvals SET status='EXPIRED', decided_at=NOW() "
                "WHERE status='PENDING' AND expires_at IS NOT NULL AND expires_at < NOW()"
            )
            count = cur.rowcount
        conn.commit()
        return count
    except Exception as e:
        conn.rollback()
        logging.error(f"expire_stale_approvals failed: {e}")
        return 0
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# Position stops
# ---------------------------------------------------------------------------

def upsert_position_stop(
    ticker: str,
    entry_price: float,
    stop_price: float,
    stop_method: str,
    bucket: str,
    conviction: int,
    trail_distance_pct: float | None = None,
) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO position_stops
                    (ticker, entry_price, stop_price, stop_method, thesis_bucket,
                     conviction, trail_distance_pct, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
                ON CONFLICT (ticker) DO UPDATE SET
                    entry_price       = EXCLUDED.entry_price,
                    stop_price        = EXCLUDED.stop_price,
                    stop_method       = EXCLUDED.stop_method,
                    thesis_bucket     = EXCLUDED.thesis_bucket,
                    conviction        = EXCLUDED.conviction,
                    trail_distance_pct= EXCLUDED.trail_distance_pct,
                    trailing_active   = FALSE,
                    trailing_high     = NULL,
                    updated_at        = NOW()
                """,
                (ticker, entry_price, stop_price, stop_method, bucket, conviction, trail_distance_pct),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"upsert_position_stop failed: {e}")
    finally:
        _put_conn(conn)


def update_trailing_stop(ticker: str, new_stop: float, trailing_high: float) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE position_stops SET stop_price=%s, trailing_high=%s, "
                "trailing_active=TRUE, updated_at=NOW() WHERE ticker=%s",
                (new_stop, trailing_high, ticker),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"update_trailing_stop failed: {e}")
    finally:
        _put_conn(conn)


def get_position_stop(ticker: str) -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, entry_price, stop_price, stop_method, trailing_active, "
                "trailing_high, trail_distance_pct, thesis_bucket, conviction, opened_at "
                "FROM position_stops WHERE ticker=%s",
                (ticker,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "ticker": row[0], "entry_price": float(row[1]), "stop_price": float(row[2]),
            "stop_method": row[3], "trailing_active": row[4],
            "trailing_high": float(row[5]) if row[5] else None,
            "trail_distance_pct": float(row[6]) if row[6] else None,
            "thesis_bucket": row[7], "conviction": row[8], "opened_at": row[9],
        }
    except Exception as e:
        logging.error(f"get_position_stop failed: {e}")
        return None
    finally:
        _put_conn(conn)


def delete_position_stop(ticker: str) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM position_stops WHERE ticker=%s", (ticker,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"delete_position_stop failed: {e}")
    finally:
        _put_conn(conn)


def get_all_position_stops() -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, entry_price, stop_price, stop_method, trailing_active, "
                "trailing_high, trail_distance_pct, thesis_bucket, conviction "
                "FROM position_stops ORDER BY ticker"
            )
            rows = cur.fetchall()
        return [
            {
                "ticker": r[0], "entry_price": float(r[1]), "stop_price": float(r[2]),
                "stop_method": r[3], "trailing_active": r[4],
                "trailing_high": float(r[5]) if r[5] else None,
                "trail_distance_pct": float(r[6]) if r[6] else None,
                "thesis_bucket": r[7], "conviction": r[8],
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_all_position_stops failed: {e}")
        return []
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# Scanner candidates
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Trade outcomes (feedback loop)
# ---------------------------------------------------------------------------

def log_trade_outcome(
    approval_id: int,
    ticker: str,
    action: str,
    conviction: int,
    bucket: str,
    entry_price: float,
) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO trade_outcomes "
                "(approval_id, ticker, action, conviction, thesis_bucket, entry_price) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (approval_id, ticker, action, conviction, bucket, entry_price),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_trade_outcome failed: {e}")
    finally:
        _put_conn(conn)


def update_outcome_price(approval_id: int, days: int, price: float) -> None:
    col_map = {30: "review_30d_price", 60: "review_60d_price", 90: "review_90d_price"}
    col = col_map.get(days)
    if not col:
        return
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE trade_outcomes SET {col}=%s WHERE approval_id=%s",
                (price, approval_id),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"update_outcome_price failed: {e}")
    finally:
        _put_conn(conn)


def get_open_outcomes() -> list[dict]:
    """Returns trade outcomes where exit_date is NULL (position still open)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT approval_id, ticker, action, conviction, thesis_bucket, "
                "entry_price, entry_date, review_30d_price, review_60d_price, review_90d_price "
                "FROM trade_outcomes WHERE exit_date IS NULL ORDER BY entry_date"
            )
            rows = cur.fetchall()
        return [
            {
                "approval_id": r[0], "ticker": r[1], "action": r[2],
                "conviction": r[3], "thesis_bucket": r[4],
                "entry_price": float(r[5]), "entry_date": r[6],
                "review_30d_price": float(r[7]) if r[7] else None,
                "review_60d_price": float(r[8]) if r[8] else None,
                "review_90d_price": float(r[9]) if r[9] else None,
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_open_outcomes failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_all_outcomes(limit: int = 100) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT approval_id, ticker, action, conviction, thesis_bucket, "
                "entry_price, entry_date, review_30d_price, review_60d_price, review_90d_price, "
                "exit_price, exit_date, outcome_pct, was_correct "
                "FROM trade_outcomes ORDER BY entry_date DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "approval_id": r[0], "ticker": r[1], "action": r[2],
                "conviction": r[3], "thesis_bucket": r[4],
                "entry_price": float(r[5]) if r[5] else None,
                "entry_date": r[6],
                "review_30d_price": float(r[7]) if r[7] else None,
                "review_60d_price": float(r[8]) if r[8] else None,
                "review_90d_price": float(r[9]) if r[9] else None,
                "exit_price": float(r[10]) if r[10] else None,
                "exit_date": r[11],
                "outcome_pct": float(r[12]) if r[12] else None,
                "was_correct": r[13],
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_all_outcomes failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_outcomes_summary() -> dict:
    """Aggregate win rate and avg outcome grouped by bucket and conviction band."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thesis_bucket, "
                "  COUNT(*) AS total, "
                "  COUNT(*) FILTER (WHERE was_correct = TRUE) AS wins, "
                "  AVG(outcome_pct) AS avg_pct "
                "FROM trade_outcomes "
                "WHERE outcome_pct IS NOT NULL "
                "GROUP BY thesis_bucket"
            )
            rows = cur.fetchall()
        return {
            r[0]: {
                "total": r[1],
                "wins": r[2],
                "win_rate": round(r[2] / r[1], 3) if r[1] else 0,
                "avg_outcome_pct": round(float(r[3]), 4) if r[3] else None,
            }
            for r in rows
        }
    except Exception as e:
        logging.error(f"get_outcomes_summary failed: {e}")
        return {}
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# Earnings calendar
# ---------------------------------------------------------------------------

def upsert_earnings_date(ticker: str, earnings_date, confirmed: bool = False) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO earnings_calendar (ticker, earnings_date, confirmed, refreshed_at) "
                "VALUES (%s,%s,%s,NOW()) "
                "ON CONFLICT (ticker, earnings_date) DO UPDATE SET "
                "confirmed=EXCLUDED.confirmed, refreshed_at=NOW()",
                (ticker, earnings_date, confirmed),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"upsert_earnings_date failed: {e}")
    finally:
        _put_conn(conn)


def get_next_earnings(ticker: str):
    """Returns the next upcoming earnings date for ticker, or None."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT earnings_date FROM earnings_calendar "
                "WHERE ticker=%s AND earnings_date >= CURRENT_DATE "
                "ORDER BY earnings_date ASC LIMIT 1",
                (ticker,),
            )
            row = cur.fetchone()
        return row[0] if row else None
    except Exception as e:
        logging.error(f"get_next_earnings failed: {e}")
        return None
    finally:
        _put_conn(conn)


def is_near_earnings(ticker: str, days: int = 5) -> bool:
    """Returns True if ticker has an earnings date within `days` calendar days."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM earnings_calendar "
                "WHERE ticker=%s AND earnings_date >= CURRENT_DATE "
                "AND earnings_date <= CURRENT_DATE + INTERVAL '%s days' LIMIT 1",
                (ticker, days),
            )
            return cur.fetchone() is not None
    except Exception as e:
        logging.error(f"is_near_earnings failed: {e}")
        return False
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# Scanner candidates
# ---------------------------------------------------------------------------

def log_scanner_candidate(
    ticker: str,
    scan_type: str,
    composite_score: int,
    technical_score: int,
    momentum_score: int,
    catalyst_score: int,
    thesis_alignment_score: int,
    priority: str,
    catalyst_detail: str,
    passed_to_analyst: bool,
    regime_at_scan: str,
) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO scanner_candidates "
                "(ticker, scan_type, composite_score, technical_score, momentum_score, "
                "catalyst_score, thesis_alignment_score, priority, catalyst_detail, "
                "passed_to_analyst, regime_at_scan) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    ticker, scan_type, composite_score, technical_score, momentum_score,
                    catalyst_score, thesis_alignment_score, priority, catalyst_detail,
                    passed_to_analyst, regime_at_scan,
                ),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_scanner_candidate failed: {e}")
    finally:
        _put_conn(conn)
