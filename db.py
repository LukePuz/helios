import json
import logging
import os
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Scopes every position_stops read/write to the account this process is
# actually connected to -- added 2026-08-14 after switching a live-connected
# process onto a DB whose position_stops table only ever held paper-account
# rows. reconcile_stop_orders read those 41 paper stops, found none of them
# on the (empty) live account, concluded all 41 "disappeared with no
# matching SELL execution," alerted on every one, and deleted all 41
# records -- while the actual paper positions sat completely untouched at
# the broker the whole time. The table had no notion of which account a
# stop belonged to at all.
from config import TRADING_ENV

_pool: psycopg2.pool.ThreadedConnectionPool | None = None

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS trades (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10),
    trading_env VARCHAR(10) NOT NULL DEFAULT 'paper',
    side        VARCHAR(4),
    quantity    DECIMAL,
    price       DECIMAL,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    reason      TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10),
    signal      VARCHAR(30),
    confidence  DECIMAL,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    reasoning   TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id             SERIAL PRIMARY KEY,
    timestamp      TIMESTAMPTZ DEFAULT NOW(),
    trading_env    VARCHAR(10) NOT NULL DEFAULT 'paper',
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
    trading_env         VARCHAR(10) NOT NULL DEFAULT 'paper',
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
    trading_env     VARCHAR(10) NOT NULL DEFAULT 'paper',
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
    ticker              VARCHAR(10),
    trading_env         VARCHAR(10) NOT NULL DEFAULT 'paper',
    entry_price         DECIMAL,
    stop_price          DECIMAL,
    stop_method         TEXT,
    trailing_active     BOOLEAN DEFAULT FALSE,
    trailing_high       DECIMAL,
    trail_distance_pct  DECIMAL,
    thesis_bucket       VARCHAR(20),
    conviction          INTEGER,
    opened_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (ticker, trading_env)
);

CREATE TABLE IF NOT EXISTS activity_feed (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ DEFAULT NOW(),
    stage       VARCHAR(20),
    ticker      VARCHAR(10),
    message     TEXT,
    sentiment   VARCHAR(10) DEFAULT 'neutral',
    extra_json  JSONB
);

CREATE INDEX IF NOT EXISTS activity_feed_ts_idx ON activity_feed (ts DESC);

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

CREATE TABLE IF NOT EXISTS system_state (
    key         VARCHAR(200) PRIMARY KEY,
    value       TEXT,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS seen_items (
    id           SERIAL PRIMARY KEY,
    source       VARCHAR(30) NOT NULL,
    external_id  TEXT NOT NULL,
    first_seen   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source, external_id)
);

CREATE TABLE IF NOT EXISTS ingestion_health (
    source           VARCHAR(30) PRIMARY KEY,
    last_success_at  TIMESTAMPTZ,
    last_item_count  INTEGER,
    last_error       TEXT,
    last_error_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker              VARCHAR(10) PRIMARY KEY,
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    revenue_growth_yoy  DECIMAL,
    gross_margin        DECIMAL,
    fcf_margin          DECIMAL,
    price_to_sales      DECIMAL,
    ev_to_ebitda        DECIMAL,
    forward_pe          DECIMAL,
    market_cap          DECIMAL,
    raw_json            JSONB
);

CREATE TABLE IF NOT EXISTS price_history (
    ticker      VARCHAR(10),
    date        DATE,
    open        DECIMAL,
    high        DECIMAL,
    low         DECIMAL,
    close       DECIMAL,
    volume      BIGINT,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS price_history_ticker_date_idx ON price_history (ticker, date DESC);

CREATE TABLE IF NOT EXISTS orders (
    id                SERIAL PRIMARY KEY,
    broker_order_id   BIGINT,
    trading_env       VARCHAR(10) NOT NULL DEFAULT 'paper',
    ticker            VARCHAR(10),
    side              VARCHAR(4),
    order_type        VARCHAR(20),
    qty               DECIMAL,
    limit_price       DECIMAL,
    stop_price        DECIMAL,
    decision_price    DECIMAL,
    fill_price        DECIMAL,
    filled_qty        DECIMAL,
    slippage_pct      DECIMAL,
    status            VARCHAR(20) DEFAULT 'submitted',
    approval_id       INTEGER REFERENCES pending_approvals(id),
    parent_order_id   BIGINT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS orders_ticker_idx ON orders (ticker, created_at DESC);

CREATE TABLE IF NOT EXISTS chat_history (
    id          SERIAL PRIMARY KEY,
    user_id     VARCHAR(30) NOT NULL,
    role        VARCHAR(10) NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chat_history_user_idx ON chat_history (user_id, timestamp DESC);
"""


_POOL_HEALTH_CHECK_INTERVAL_S = 60
_last_pool_health_check: float = 0.0


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool, _last_pool_health_check
    if _pool is not None:
        # Health check was previously run on EVERY _get_conn() call — an
        # extra network round-trip to Supabase per query, and the SELECT 1
        # opened a transaction that was never committed/rolled back before
        # putconn, so every checkout left a connection "idle in transaction"
        # (Supabase eventually reaps these, but under concurrent workers it
        # could exhaust the pool first). Now throttled to once/minute, and
        # rolled back explicitly.
        now = _time.monotonic()
        if now - _last_pool_health_check < _POOL_HEALTH_CHECK_INTERVAL_S:
            return _pool
        try:
            conn = _pool.getconn()
            conn.cursor().execute("SELECT 1")
            conn.rollback()
            _pool.putconn(conn)
            _last_pool_health_check = now
            return _pool
        except Exception:
            logging.warning("DB pool health check failed — recreating pool.")
            try:
                _pool.closeall()
            except Exception:
                pass
            _pool = None

    maxconn = int(os.getenv("DB_MAXCONN", "5"))
    for attempt in range(1, 4):
        try:
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=maxconn,
                host=os.getenv("SUPABASE_HOST"),
                port=int(os.getenv("SUPABASE_PORT", "5432")),
                dbname=os.getenv("SUPABASE_DB", "postgres"),
                user=os.getenv("SUPABASE_USER", "postgres"),
                password=os.getenv("SUPABASE_PASSWORD"),
                sslmode="require",
            )
            logging.info(f"DB connection pool created (maxconn={maxconn}).")
            _last_pool_health_check = _time.monotonic()
            return _pool
        except Exception as e:
            logging.warning(f"DB pool creation attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                _time.sleep(2 ** attempt)  # 2s, 4s
    raise RuntimeError("Could not connect to PostgreSQL after 3 attempts.")


def _get_conn():
    return _get_pool().getconn()


def _put_conn(conn) -> None:
    _get_pool().putconn(conn)


# ---------------------------------------------------------------------------
# Versioned migrations
#
# Each entry is (version, sql). Applied in order, tracked in schema_version
# so re-running init_db() (every process start) is a no-op after the first
# time instead of re-executing every ALTER on every boot. All statements
# remain idempotent (IF NOT EXISTS) as a second line of defense, but the
# version table means `python setup_db.py` / init_db() logs exactly which
# migrations are new versus already-applied, instead of that being invisible.
# ---------------------------------------------------------------------------
_MIGRATIONS: list[tuple[int, str]] = [
    (1, "ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;"),
    (2, "ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS decided_at TIMESTAMPTZ;"),
    (3, "ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS telegram_message_id BIGINT;"),
    (4, "ALTER TABLE position_stops ADD COLUMN IF NOT EXISTS broker_order_id BIGINT;"),
    (5, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS entry_benchmark_price DECIMAL;"),
    (6, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS exit_benchmark_price DECIMAL;"),
    (7, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS benchmark_return_pct DECIMAL;"),
    (8, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS alpha_pct DECIMAL;"),
    (9, "ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS benchmark_json JSONB;"),
    # signal VARCHAR(10) silently rejected every log_signal() call using a
    # custom signal type longer than 10 chars — CATALYST_BLOCKED (16),
    # COOLDOWN_SKIPPED (17), SCREENED_OUT (12), SELL_NO_POSITION (17) have
    # all been failing (caught and logged, not crashing the caller, but
    # never actually written) since each was introduced. Found 2026-07-16
    # while testing the screener split.
    (10, "ALTER TABLE signals ALTER COLUMN signal TYPE VARCHAR(30);"),
    # Decomposes the single 1-10 conviction into four independently-
    # calibratable components (see agents/analyst.py's sub-score rubric and
    # get_subscore_calibration below) — added 2026-07-17 so "conviction-8
    # calls underperform" can become "catalyst_quality-5 calls underperform,
    # valuation calls are fine" instead of one blended, less actionable number.
    (11, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS thesis_strength INTEGER;"),
    (12, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS catalyst_quality INTEGER;"),
    (13, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS valuation_risk_reward INTEGER;"),
    (14, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS data_completeness INTEGER;"),
    # Chase risk (agents/chase_risk.py) — deterministic "good thesis, bad
    # entry price" sub-score from Ichimoku Kijun-sen distance, added
    # 2026-07-25, informational only (doesn't affect sizing/gating yet).
    (15, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS chase_risk INTEGER;"),
    (16, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS chase_risk_pct_from_kijun DECIMAL;"),
    # Price action / volume (agents/price_action.py) — deterministic swing-
    # structure + 200DMA-slope + OBV-volume-confirmation sub-score, added
    # 2026-07-26. Already wired into the scanner's composite_score (a real
    # behavior change to screening); this column just makes it trackable/
    # calibratable against real outcomes the same way the other sub-scores are.
    (17, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS price_action INTEGER;"),
    (18, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS price_action_structure VARCHAR(20);"),
    # Sector-specific regime (agents/market_regime.py's assess_sector_regime,
    # added 2026-07-26 after finding the overall SPY-based regime stayed
    # "Neutral" throughout the real June 2026 tech/semis rotation -- SOXX
    # ran +29% vs SPY over 5 weeks then reversed -10% relative in 4 sessions
    # while SPY itself barely moved. regime_type distinguishes rows in the
    # same table rather than standing up a parallel one; NULL/'overall' on
    # every existing row is the pre-existing behavior.
    (19, "ALTER TABLE market_regime_log ADD COLUMN IF NOT EXISTS regime_type VARCHAR(20) DEFAULT 'overall';"),
    (20, "ALTER TABLE market_regime_log ADD COLUMN IF NOT EXISTS sector_etf VARCHAR(10);"),
    # New table, not a repurposed one -- scanner_candidates' schema
    # (catalyst_score, thesis_alignment_score computed from an existing
    # watchlist tier/role entry) doesn't apply to tickers that aren't on
    # the watchlist yet, which is exactly what agents/candidate_scanner.py
    # screens. Added 2026-07-26.
    (21, """
        CREATE TABLE IF NOT EXISTS watchlist_candidates (
            id                SERIAL PRIMARY KEY,
            ticker            VARCHAR(10) NOT NULL,
            gics_sector       VARCHAR(50),
            movement_score    INTEGER,
            thesis_match_score INTEGER,
            composite_score   INTEGER,
            matched_thesis    VARCHAR(50),
            notes             TEXT,
            scanned_at        TIMESTAMPTZ DEFAULT NOW()
        );
    """),
    (22, "CREATE INDEX IF NOT EXISTS idx_watchlist_candidates_scanned_at ON watchlist_candidates (scanned_at DESC);"),
    # Durable news headlines -- added 2026-07-26 for the public research
    # site's News section. Every other public-site table (market_regime_log,
    # trade_outcomes, ...) is written by trading_bot and read directly by
    # helios-research's backend (same Supabase DB); this follows the same
    # pattern rather than the site hitting yfinance/Google RSS live.
    # ChromaDB's news collection can't serve "most recent N headlines"
    # cheaply (no real timestamp-ordered query), so this is a plain table,
    # not a vector_store addition.
    (23, """
        CREATE TABLE IF NOT EXISTS news_log (
            id            SERIAL PRIMARY KEY,
            ticker        VARCHAR(10),
            title         TEXT NOT NULL,
            url           TEXT,
            published_at  TIMESTAMPTZ DEFAULT NOW()
        );
    """),
    (24, "CREATE INDEX IF NOT EXISTS idx_news_log_ticker_published ON news_log (ticker, published_at DESC);"),
    (25, "CREATE INDEX IF NOT EXISTS idx_news_log_published_at ON news_log (published_at DESC);"),
    # Every headline gets a photo -- the article's own thumbnail
    # (yfinance carries one ~90-100% of the time) or, failing that, a
    # company-related fallback image (ingestion/company_photos.py).
    (26, "ALTER TABLE news_log ADD COLUMN IF NOT EXISTS image_url TEXT;"),
    # Incident 2026-07-30: tasks._reconcile_alert_due builds throttle keys
    # like "reconcile_alert_ingestion_<all stale source names joined>" --
    # with several sources stale at once this routinely exceeded 50 chars,
    # so every set_state() call silently failed (caught and logged, not
    # raised) and the throttle could never actually persist. The ingestion-
    # health alert re-fired on every hourly check instead of respecting its
    # cooldown -- the exact "same alert every cycle" bug this table's
    # throttle mechanism exists to prevent, just via a different path.
    (27, "ALTER TABLE system_state ALTER COLUMN key TYPE VARCHAR(200);"),
    # position_stops had no notion of which broker account a stop belonged
    # to -- switching a live-connected process onto the same DB that held
    # 41 paper-account stop rows made reconcile_stop_orders conclude all 41
    # positions had "disappeared with no matching SELL execution" (none of
    # them existed on the empty live account) and delete every record,
    # while the real paper positions sat untouched at the broker the whole
    # time. Existing rows default to 'paper' -- true for every row that
    # could exist before this migration, since live wasn't reachable until
    # tonight. DROP the old single-column UNIQUE (its default Postgres name
    # for an inline `ticker VARCHAR(10) UNIQUE`) before adding the
    # composite one, or the same ticker in both envs would still collide.
    (28, "ALTER TABLE position_stops ADD COLUMN IF NOT EXISTS trading_env VARCHAR(10) NOT NULL DEFAULT 'paper'; "
         "ALTER TABLE position_stops DROP CONSTRAINT IF EXISTS position_stops_ticker_key; "
         "ALTER TABLE position_stops DROP CONSTRAINT IF EXISTS position_stops_ticker_trading_env_key; "
         "ALTER TABLE position_stops ADD CONSTRAINT position_stops_ticker_trading_env_key UNIQUE (ticker, trading_env);"),
    # Same class of gap as position_stops (28), audited 2026-08-15 ahead of
    # enabling full unattended live auto-execution -- found by going
    # looking this time, not by another incident. pending_approvals is the
    # actual execution queue (has_active_approval/get_last_executed_approval
    # gate duplicate signals and the rebuy cooldown off it); trade_outcomes
    # feeds agents/analyst.py's real-time conviction calibration
    # (get_subscore_calibration) -- paper's much larger, differently-sized
    # trade history would otherwise silently bias live's own decisions;
    # orders drives fill/slippage reconciliation, keyed in part by
    # broker_order_id, which is NOT globally unique -- paper and live can
    # each independently produce the same order id for two completely
    # unrelated orders. trades/portfolio_snapshots are lower-severity
    # (dashboard/day-P&L display, no decision-making reads) but scoped too
    # for consistency, now that this pass is already being done.
    (29, "ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS trading_env VARCHAR(10) NOT NULL DEFAULT 'paper'; "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS trading_env VARCHAR(10) NOT NULL DEFAULT 'paper'; "
         "ALTER TABLE orders ADD COLUMN IF NOT EXISTS trading_env VARCHAR(10) NOT NULL DEFAULT 'paper'; "
         "ALTER TABLE trades ADD COLUMN IF NOT EXISTS trading_env VARCHAR(10) NOT NULL DEFAULT 'paper'; "
         "ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS trading_env VARCHAR(10) NOT NULL DEFAULT 'paper';"),
    # Regime-adaptive profit-taking (2026-08-29, see config.py's
    # PROFIT_TARGET_PCT_BY_STYLE) -- tracks whether the partial-sell-at-
    # target has already fired for a position, so check_profit_targets()
    # doesn't re-sell a portion every time it runs (every 30 min once
    # wired into the schedule) once the target's already been taken once.
    (30, "ALTER TABLE position_stops ADD COLUMN IF NOT EXISTS profit_target_taken BOOLEAN DEFAULT FALSE;"),
    # Catalyst/expected-return tracking (2026-08-30, user request): promotes
    # what used to live only transiently in pending_approvals.analyst_json
    # (catalyst description, expected return, cited sources) into first-class
    # columns on trade_outcomes -- the table that actually tracks a position
    # long-term (review_30d/60d/90d prices, outcome_pct) -- so a thesis can
    # be checked against what it actually predicted, not just an outcome
    # percentage with no record of what was supposed to happen or why.
    # catalyst_materialized is deliberately nullable with no default: it's
    # populated later by a comparison job once the catalyst_date has passed,
    # not known at entry time.
    (31, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS catalyst_event TEXT; "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS catalyst_date VARCHAR(20); "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS catalyst_is_priced_in BOOLEAN; "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS catalyst_materialized BOOLEAN; "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS expected_return_pct DECIMAL; "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS expected_downside_pct DECIMAL; "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS thesis_summary TEXT; "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS sources_json JSONB; "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS source_count INTEGER;"),
    # Expected-return RATE tracking (2026-08-30, user request): expected_return_pct
    # alone doesn't say whether a trade actually cleared the >=1%/week bar --
    # a 20% target 6 months out and a 3% target tomorrow read identically
    # without the implied weekly rate alongside them. Both are derived at
    # analysis time from catalyst_date/expected_return_pct (see
    # agents/analyst.py's compute_weeks_to_catalyst/expected_return_per_week)
    # and persisted here so the realized outcome can later be compared
    # against what rate was actually predicted, not just the raw pct.
    (32, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS weeks_to_catalyst DECIMAL; "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS expected_return_per_week DECIMAL;"),
]

_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ DEFAULT NOW()
);
"""


def init_db() -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLES)
            cur.execute(_SCHEMA_VERSION_TABLE)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT version FROM schema_version")
            applied = {row[0] for row in cur.fetchall()}

        newly_applied = []
        for version, sql in _MIGRATIONS:
            if version in applied:
                continue
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute("INSERT INTO schema_version (version) VALUES (%s)", (version,))
            conn.commit()
            newly_applied.append(version)

        if newly_applied:
            logging.info(f"Applied migrations: {newly_applied}")
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
                "INSERT INTO trades (symbol, trading_env, side, quantity, price, reason) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (symbol, TRADING_ENV, side, quantity, price, reason),
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


def log_snapshot(total_value: float, cash: float, positions: dict,
                 benchmark: dict | None = None) -> None:
    """
    benchmark: optional {"SPY": price, "QQQ": price} snapshot taken at the
    same moment, so the equity curve can be rebased against a real benchmark
    instead of only a flat $-start-value line.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO portfolio_snapshots (total_value, trading_env, cash, positions_json, benchmark_json) "
                "VALUES (%s,%s,%s,%s,%s)",
                (total_value, TRADING_ENV, cash, psycopg2.extras.Json(positions),
                 psycopg2.extras.Json(benchmark) if benchmark else None),
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
                "FROM trades WHERE trading_env=%s ORDER BY timestamp DESC LIMIT %s",
                (TRADING_ENV, limit),
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


def append_chat_message(user_id: str, role: str, content: str) -> None:
    """Persists one turn of a chat_assistant conversation. Keyed by Telegram
    numeric user_id (not display name, which can collide/change) so DB-backed
    history survives process restarts and is shared between the full
    pipeline's telegram_bot.py and the overnight standby_bot.py — a user
    chatting through an evening handoff shouldn't lose context."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (user_id, role, content) VALUES (%s,%s,%s)",
                (str(user_id), role, content),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"append_chat_message failed: {e}")
    finally:
        _put_conn(conn)


def get_chat_history(user_id: str, limit: int = 12, max_age_hours: float = 4) -> list[dict]:
    """Returns the most recent turns for this user, oldest first, ready to
    prepend to a fresh messages list. Bounded by both count and age — an
    unrelated question hours/days later shouldn't drag in a stale, no-longer-
    relevant conversation (max_age_hours matches REBUY_COOLDOWN_HOURS'
    convention elsewhere in this codebase for "how long is context fresh")."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM chat_history "
                "WHERE user_id = %s AND timestamp > NOW() - (%s || ' hours')::INTERVAL "
                "ORDER BY timestamp DESC LIMIT %s",
                (str(user_id), max_age_hours, limit),
            )
            rows = cur.fetchall()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as e:
        logging.error(f"get_chat_history failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_latest_signal_for(ticker: str) -> dict | None:
    """Most recent analysis result for a ticker, regardless of verdict —
    every call to agents.analyst.analyze() logs here (see analyst.py's
    db.log_signal call), including routine HOLDs on positions already
    held, not just actionable OPEN/ADD/TRIM/EXIT ones. This is the
    freshest conviction reading available for a held position between
    its entry and whenever position_stops.conviction next gets touched
    (only on a fresh ADD) — used by RiskManager.find_swap_candidate to
    judge whether a held position still deserves its size, not just what
    conviction it was entered at."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal, confidence, timestamp FROM signals "
                "WHERE symbol=%s ORDER BY timestamp DESC LIMIT 1",
                (ticker,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"signal": row[0], "confidence": float(row[1] or 0), "timestamp": row[2]}
    except Exception as e:
        logging.error(f"get_latest_signal_for({ticker}) failed: {e}")
        return None
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
                "SELECT timestamp, total_value, cash, benchmark_json FROM portfolio_snapshots "
                "WHERE trading_env=%s ORDER BY timestamp DESC LIMIT %s",
                (TRADING_ENV, limit),
            )
            rows = cur.fetchall()
        return [
            {"timestamp": r[0], "total_value": float(r[1]), "cash": float(r[2]),
             "benchmark": r[3] or {}}
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
    regime_type: str = "overall",
    sector_etf: str | None = None,
) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO market_regime_log "
                "(regime, score, confidence, signals_json, portfolio_impact_json, reasoning, "
                " regime_type, sector_etf) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    regime, score, confidence,
                    psycopg2.extras.Json(signals),
                    psycopg2.extras.Json(portfolio_impact),
                    reasoning, regime_type, sector_etf,
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

def has_active_approval(ticker: str) -> bool:
    """Return True if this ticker already has a PENDING or PROCESSING approval."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pending_approvals "
                "WHERE ticker=%s AND trading_env=%s AND status IN ('PENDING','PROCESSING') LIMIT 1",
                (ticker, TRADING_ENV),
            )
            return cur.fetchone() is not None
    except Exception as e:
        logging.error(f"has_active_approval failed: {e}")
        return False
    finally:
        _put_conn(conn)


def get_last_executed_approval(ticker: str, hours: int) -> dict | None:
    """Most recent decided OPEN/ADD for this ticker within the lookback
    window, or None. Used to cool down repeat buys — auto-execute completes
    a decision within seconds, so has_active_approval's PENDING/PROCESSING
    check alone doesn't stop the next independent event from re-buying the
    same ticker minutes later.

    Originally matched status='APPROVED' only -- but a market-closed
    deferral (execution.execute_approved's market-hours gate, 2026-07-28)
    leaves an approval at status='FAILED' with a retry scheduled for the
    next open, and an APPROVED-only filter couldn't see that as "already
    decided." Incident 2026-07-29: the same Teradyne earnings release,
    covered by six different outlets, independently triggered six full
    analyses pre-market; each deferred (never reached APPROVED), so this
    cooldown never blocked the next one, and all six retried and executed
    within the same 20-second window at the open -- 6x the intended
    position, bought chasing the top of the day's gap. FAILED now counts
    too, so the first analysis's deferral alone is enough to cool down a
    same-or-lower-conviction duplicate. EXPIRED/REJECTED are deliberately
    excluded -- a stale-and-discarded or human-vetoed recommendation
    shouldn't suppress a fresh, independent read on the same ticker."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT conviction, decided_at FROM pending_approvals "
                "WHERE ticker=%s AND trading_env=%s AND action IN ('OPEN','ADD') "
                "AND status IN ('APPROVED','FAILED') "
                "AND decided_at > NOW() - (%s || ' hours')::interval "
                "ORDER BY decided_at DESC LIMIT 1",
                (ticker, TRADING_ENV, hours),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"conviction": row[0], "decided_at": row[1]}
    except Exception as e:
        logging.error(f"get_last_executed_approval failed: {e}")
        return None
    finally:
        _put_conn(conn)


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
                "(ticker, trading_env, action, conviction, thesis_bucket, position_size_pct, stop_loss_price, "
                "entry_strategy, reasoning, analyst_json, risk_json, expires_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (
                    ticker, TRADING_ENV, action, conviction, bucket, size_pct, stop_price,
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


def claim_approval_for_execution(approval_id: int) -> bool:
    """
    Atomically transition status PENDING → PROCESSING.
    Returns True only if THIS call won the race (exactly one row updated).
    Any concurrent duplicate callback will get False and skip execution.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE pending_approvals SET status='PROCESSING', decided_at=NOW() "
                "WHERE id=%s AND status='PENDING' RETURNING id",
                (approval_id,),
            )
            claimed = cur.fetchone() is not None
        conn.commit()
        return claimed
    except Exception as e:
        conn.rollback()
        logging.error(f"claim_approval_for_execution failed: {e}")
        return False
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


# ---------------------------------------------------------------------------
# Activity feed
# ---------------------------------------------------------------------------

def log_activity(
    stage:     str,
    ticker:    str,
    message:   str,
    sentiment: str = "neutral",
    extra:     dict | None = None,
) -> None:
    """Insert one row into the live activity feed."""
    import json as _json
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO activity_feed (stage, ticker, message, sentiment, extra_json)
                   VALUES (%s, %s, %s, %s, %s)""",
                (stage, ticker or "", message, sentiment,
                 _json.dumps(extra) if extra else None),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_activity failed: {e}")
    finally:
        _put_conn(conn)


def get_activity_feed(limit: int = 120) -> list[dict]:
    """Return the most recent activity feed rows, newest first."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, ts, stage, ticker, message, sentiment, extra_json "
                "FROM activity_feed ORDER BY ts DESC LIMIT %s",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logging.error(f"get_activity_feed failed: {e}")
        return []
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
                    "FROM pending_approvals WHERE trading_env=%s ORDER BY timestamp DESC",
                    (TRADING_ENV,),
                )
            else:
                cur.execute(
                    "SELECT id, ticker, action, conviction, thesis_bucket, position_size_pct, "
                    "stop_loss_price, entry_strategy, reasoning, analyst_json, risk_json, "
                    "status, timestamp, decided_at, expires_at "
                    "FROM pending_approvals WHERE status=%s AND trading_env=%s ORDER BY timestamp DESC",
                    (status, TRADING_ENV),
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
                    (ticker, trading_env, entry_price, stop_price, stop_method, thesis_bucket,
                     conviction, trail_distance_pct, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                ON CONFLICT (ticker, trading_env) DO UPDATE SET
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
                (ticker, TRADING_ENV, entry_price, stop_price, stop_method, bucket, conviction, trail_distance_pct),
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
                "trailing_active=TRUE, updated_at=NOW() WHERE ticker=%s AND trading_env=%s",
                (new_stop, trailing_high, ticker, TRADING_ENV),
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
                "FROM position_stops WHERE ticker=%s AND trading_env=%s",
                (ticker, TRADING_ENV),
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
            cur.execute("DELETE FROM position_stops WHERE ticker=%s AND trading_env=%s", (ticker, TRADING_ENV))
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
                "trailing_high, trail_distance_pct, thesis_bucket, conviction, broker_order_id, "
                "opened_at, profit_target_taken "
                "FROM position_stops WHERE trading_env=%s ORDER BY ticker",
                (TRADING_ENV,),
            )
            rows = cur.fetchall()
        return [
            {
                "ticker": r[0], "entry_price": float(r[1]), "stop_price": float(r[2]),
                "stop_method": r[3], "trailing_active": r[4],
                "trailing_high": float(r[5]) if r[5] else None,
                "trail_distance_pct": float(r[6]) if r[6] else None,
                "thesis_bucket": r[7], "conviction": r[8],
                "broker_order_id": int(r[9]) if r[9] else None,
                "opened_at": r[10], "profit_target_taken": bool(r[11]),
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_all_position_stops failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_open_position_symbols() -> set[str]:
    """Returns the set of tickers that have active stop records (proxy for open positions)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ticker FROM position_stops WHERE trading_env=%s", (TRADING_ENV,))
            return {row[0] for row in cur.fetchall()}
    except Exception as e:
        logging.error(f"get_open_position_symbols failed: {e}")
        return set()
    finally:
        _put_conn(conn)


def set_stop_broker_order_id(ticker: str, broker_order_id: int | None) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE position_stops SET broker_order_id=%s, updated_at=NOW() WHERE ticker=%s AND trading_env=%s",
                (broker_order_id, ticker, TRADING_ENV),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"set_stop_broker_order_id failed: {e}")
    finally:
        _put_conn(conn)


def mark_profit_target_taken(ticker: str, new_stop: float) -> None:
    """Called once by risk_manager.check_profit_targets() the moment a
    position's profit target is crossed and its partial-sell fires --
    records that the target's been taken (so it isn't re-sold on the next
    check) and moves the recorded stop to breakeven in the same write, so
    a crash between the partial sell and the stop update can't leave the
    remainder both unmarked AND unprotected."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE position_stops SET profit_target_taken=TRUE, stop_price=%s, updated_at=NOW() "
                "WHERE ticker=%s AND trading_env=%s",
                (new_stop, ticker, TRADING_ENV),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"mark_profit_target_taken failed: {e}")
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# System state (persisted key-value store shared by all processes)
# ---------------------------------------------------------------------------

def get_state(key: str, default: str | None = None) -> str | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM system_state WHERE key=%s", (key,))
            row = cur.fetchone()
        return row[0] if row else default
    except Exception as e:
        logging.error(f"get_state({key}) failed: {e}")
        return default
    finally:
        _put_conn(conn)


def set_state(key: str, value: str) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO system_state (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()
                """,
                (key, value),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"set_state({key}) failed: {e}")
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# Persistent dedupe (survives restarts — replaces in-memory _seen_* sets)
# ---------------------------------------------------------------------------

def is_seen(source: str, external_id: str) -> bool:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM seen_items WHERE source=%s AND external_id=%s",
                (source, external_id),
            )
            return cur.fetchone() is not None
    except Exception as e:
        logging.error(f"is_seen({source}, {external_id}) failed: {e}")
        return False  # fail open: worst case is a duplicate event, not a missed one
    finally:
        _put_conn(conn)


def mark_seen(source: str, external_id: str) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO seen_items (source, external_id) VALUES (%s, %s) "
                "ON CONFLICT (source, external_id) DO NOTHING",
                (source, external_id),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"mark_seen({source}, {external_id}) failed: {e}")
    finally:
        _put_conn(conn)


def prune_seen_items(days: int = 90, source: str | None = None) -> int:
    """seen_items grows unbounded otherwise — a row per distinct
    news/filing/insider/options item ever ingested, indefinitely. Pass
    source to prune only that source with its own horizon (e.g. the
    options_flow_dispatch dedup keys only need ~7 days). Returns the number
    of rows deleted."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if source:
                cur.execute(
                    "DELETE FROM seen_items WHERE source=%s "
                    "AND first_seen < NOW() - (%s || ' days')::interval",
                    (source, days),
                )
            else:
                cur.execute(
                    "DELETE FROM seen_items WHERE first_seen < NOW() - (%s || ' days')::interval",
                    (days,),
                )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    except Exception as e:
        conn.rollback()
        logging.error(f"prune_seen_items failed: {e}")
        return 0
    finally:
        _put_conn(conn)


def prune_chat_history(days: int = 30) -> int:
    """chat_history is only ever read within a max_age_hours=4 window (see
    get_chat_history) — nothing older than that is functionally used, so it
    just accumulates otherwise. Returns the number of rows deleted."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_history WHERE timestamp < NOW() - (%s || ' days')::interval",
                (days,),
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    except Exception as e:
        conn.rollback()
        logging.error(f"prune_chat_history failed: {e}")
        return 0
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# Ingestion health (heartbeat per source — detects "dead" vs "quiet")
# ---------------------------------------------------------------------------

def record_ingestion_success(source: str, item_count: int) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_health (source, last_success_at, last_item_count)
                VALUES (%s, NOW(), %s)
                ON CONFLICT (source) DO UPDATE SET
                    last_success_at = NOW(),
                    last_item_count = EXCLUDED.last_item_count
                """,
                (source, item_count),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"record_ingestion_success({source}) failed: {e}")
    finally:
        _put_conn(conn)


def record_ingestion_error(source: str, error: str) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_health (source, last_error, last_error_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (source) DO UPDATE SET
                    last_error    = EXCLUDED.last_error,
                    last_error_at = NOW()
                """,
                (source, error[:2000]),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"record_ingestion_error({source}) failed: {e}")
    finally:
        _put_conn(conn)


def get_ingestion_health() -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source, last_success_at, last_item_count, last_error, last_error_at "
                "FROM ingestion_health ORDER BY source"
            )
            rows = cur.fetchall()
        return [
            {
                "source": r[0], "last_success_at": r[1], "last_item_count": r[2],
                "last_error": r[3], "last_error_at": r[4],
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_ingestion_health failed: {e}")
        return []
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# Fundamentals (free, via yfinance)
# ---------------------------------------------------------------------------

def upsert_fundamentals(ticker: str, metrics: dict) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fundamentals
                    (ticker, updated_at, revenue_growth_yoy, gross_margin, fcf_margin,
                     price_to_sales, ev_to_ebitda, forward_pe, market_cap, raw_json)
                VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    updated_at         = NOW(),
                    revenue_growth_yoy = EXCLUDED.revenue_growth_yoy,
                    gross_margin       = EXCLUDED.gross_margin,
                    fcf_margin         = EXCLUDED.fcf_margin,
                    price_to_sales     = EXCLUDED.price_to_sales,
                    ev_to_ebitda       = EXCLUDED.ev_to_ebitda,
                    forward_pe         = EXCLUDED.forward_pe,
                    market_cap         = EXCLUDED.market_cap,
                    raw_json           = EXCLUDED.raw_json
                """,
                (
                    ticker,
                    metrics.get("revenue_growth_yoy"),
                    metrics.get("gross_margin"),
                    metrics.get("fcf_margin"),
                    metrics.get("price_to_sales"),
                    metrics.get("ev_to_ebitda"),
                    metrics.get("forward_pe"),
                    metrics.get("market_cap"),
                    json.dumps(metrics.get("raw", {})),
                ),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"upsert_fundamentals({ticker}) failed: {e}")
    finally:
        _put_conn(conn)


def get_fundamentals(ticker: str) -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, updated_at, revenue_growth_yoy, gross_margin, fcf_margin, "
                "price_to_sales, ev_to_ebitda, forward_pe, market_cap "
                "FROM fundamentals WHERE ticker=%s",
                (ticker,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "ticker": row[0], "updated_at": row[1],
            "revenue_growth_yoy": float(row[2]) if row[2] is not None else None,
            "gross_margin":       float(row[3]) if row[3] is not None else None,
            "fcf_margin":         float(row[4]) if row[4] is not None else None,
            "price_to_sales":     float(row[5]) if row[5] is not None else None,
            "ev_to_ebitda":       float(row[6]) if row[6] is not None else None,
            "forward_pe":         float(row[7]) if row[7] is not None else None,
            "market_cap":         float(row[8]) if row[8] is not None else None,
        }
    except Exception as e:
        logging.error(f"get_fundamentals({ticker}) failed: {e}")
        return None
    finally:
        _put_conn(conn)


def upsert_price_bars(ticker: str, bars: list[dict]) -> int:
    """
    bars: list of {date (YYYY-MM-DD str or date), open, high, low, close, volume}.
    Returns count upserted.
    """
    if not bars:
        return 0
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            for b in bars:
                cur.execute(
                    """
                    INSERT INTO price_history (ticker, date, open, high, low, close, volume)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                        close=EXCLUDED.close, volume=EXCLUDED.volume
                    """,
                    (ticker, b["date"], b["open"], b["high"], b["low"], b["close"], b["volume"]),
                )
        conn.commit()
        return len(bars)
    except Exception as e:
        conn.rollback()
        logging.error(f"upsert_price_bars({ticker}) failed: {e}")
        return 0
    finally:
        _put_conn(conn)


def get_price_history(ticker: str, days: int = 260) -> list[dict]:
    """Returns up to `days` most recent daily bars for ticker, oldest first."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT date, open, high, low, close, volume FROM price_history "
                "WHERE ticker=%s ORDER BY date DESC LIMIT %s",
                (ticker, days),
            )
            rows = cur.fetchall()
        rows.reverse()
        return [
            {"date": str(r[0]), "open": float(r[1]), "high": float(r[2]),
             "low": float(r[3]), "close": float(r[4]), "volume": int(r[5])}
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_price_history({ticker}) failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_price_history_bulk(tickers: list[str], days: int = 260) -> dict[str, list[dict]]:
    """
    Same as get_price_history but for many tickers in one round trip.
    `days` is the number of most-recent TRADING bars per ticker (matching
    get_price_history's semantics) — NOT a calendar-date cutoff. An earlier
    version used `date >= CURRENT_DATE - days`, which under-counts by ~30%+
    once weekends/holidays are factored in (e.g. 251 cached trading days
    span ~350 calendar days, so a 210-calendar-day cutoff returned only
    ~150 bars — silently starving the 200-bar breadth/correlation checks).

    Returns full OHLCV per bar (not just close) — added 2026-07-26 for
    agents/candidate_scanner.py, which feeds these bars into agents.scanner's
    technical/momentum scoring, and that in turn calls agents.price_action's
    swing-structure/OBV analysis needing high/low/volume. Those calls were
    already wrapped in try/except (silently degrading to a weaker fallback
    on a KeyError) so this would never have crashed -- it would just have
    quietly scored every single candidate on degraded signals forever.
    Harmless for the existing breadth caller, which only ever reads close.
    """
    if not tickers:
        return {}
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, date, open, high, low, close, volume FROM (
                    SELECT ticker, date, open, high, low, close, volume,
                           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                    FROM price_history
                    WHERE ticker = ANY(%s)
                ) ranked
                WHERE rn <= %s
                ORDER BY ticker, date ASC
                """,
                (tickers, days),
            )
            rows = cur.fetchall()
        result: dict[str, list[dict]] = {}
        for ticker, d, o, h, l, c, v in rows:
            result.setdefault(ticker, []).append({
                "date": str(d), "open": float(o), "high": float(h),
                "low": float(l), "close": float(c), "volume": int(v),
            })
        return result
    except Exception as e:
        logging.error(f"get_price_history_bulk failed: {e}")
        return {}
    finally:
        _put_conn(conn)


def log_order(
    ticker: str,
    side: str,
    order_type: str,
    qty: float,
    broker_order_id: int | None = None,
    limit_price: float | None = None,
    stop_price: float | None = None,
    decision_price: float | None = None,
    approval_id: int | None = None,
    parent_order_id: int | None = None,
    status: str = "submitted",
) -> int:
    """Logs a newly-placed order. Returns the local orders.id (-1 on failure)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO orders
                    (broker_order_id, trading_env, ticker, side, order_type, qty, limit_price,
                     stop_price, decision_price, approval_id, parent_order_id, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (broker_order_id, TRADING_ENV, ticker, side, order_type, qty, limit_price,
                 stop_price, decision_price, approval_id, parent_order_id, status),
            )
            order_id = cur.fetchone()[0]
        conn.commit()
        return order_id
    except Exception as e:
        conn.rollback()
        logging.error(f"log_order failed: {e}")
        return -1
    finally:
        _put_conn(conn)


def compute_slippage_pct(decision_price: float | None, fill_price: float | None,
                         side: str) -> float | None:
    """
    Pure function, no DB access — kept separate from update_order_fill so it
    can be unit-tested without a database connection.
    Positive slippage = paid more than the decision-time price (BUY) or
    received less than it (SELL) — i.e. slippage is always "cost" signed.
    """
    if not decision_price or not fill_price:
        return None
    raw = (fill_price - decision_price) / decision_price
    return raw if side == "BUY" else -raw


def update_order_fill(order_id: int, fill_price: float, filled_qty: float, status: str) -> None:
    """Records the fill and computes slippage_pct vs decision_price if one was logged."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT decision_price, side FROM orders WHERE id=%s", (order_id,))
            row = cur.fetchone()
            slippage_pct = None
            if row:
                decision_price = float(row[0]) if row[0] else None
                slippage_pct = compute_slippage_pct(decision_price, fill_price, row[1])

            cur.execute(
                "UPDATE orders SET fill_price=%s, filled_qty=%s, status=%s, "
                "slippage_pct=%s, updated_at=NOW() WHERE id=%s",
                (fill_price, filled_qty, status, slippage_pct, order_id),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"update_order_fill failed: {e}")
    finally:
        _put_conn(conn)


def get_unconfirmed_orders(max_age_hours: int = 24) -> list[dict]:
    """Orders logged with status='unconfirmed' (no fill seen within
    place_market_order's wait window) in the last max_age_hours. Feeds
    tasks.reconcile_unconfirmed_orders -- see get_fill_for_order's docstring
    in ibkr.py for the incident (CRWD 2026-07-28) that motivated checking
    these against the broker's own execution record instead of assuming
    "unconfirmed" means "never filled".

    Scoped to trading_env -- broker_order_id is NOT globally unique across
    accounts (paper and live can each independently produce the same id
    for two unrelated orders), so an unscoped read here could look up a
    stale paper order's fill status against whichever broker is currently
    connected and get a nonsensical or wrong-order match."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, ticker, side, qty, broker_order_id, decision_price, approval_id "
                "FROM orders WHERE status='unconfirmed' AND broker_order_id IS NOT NULL "
                "AND trading_env=%s "
                "AND created_at > NOW() - INTERVAL '%s hours' ORDER BY created_at",
                (TRADING_ENV, max_age_hours),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0], "ticker": r[1], "side": r[2],
                "qty": float(r[3]) if r[3] is not None else None,
                "broker_order_id": r[4], "decision_price": float(r[5]) if r[5] is not None else None,
                "approval_id": r[6],
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_unconfirmed_orders failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_recent_orders(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, broker_order_id, ticker, side, order_type, qty, limit_price, "
                "stop_price, decision_price, fill_price, filled_qty, slippage_pct, status, "
                "created_at FROM orders WHERE trading_env=%s ORDER BY created_at DESC LIMIT %s",
                (TRADING_ENV, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0], "broker_order_id": r[1], "ticker": r[2], "side": r[3],
                "order_type": r[4], "qty": float(r[5]) if r[5] is not None else None,
                "limit_price": float(r[6]) if r[6] is not None else None,
                "stop_price": float(r[7]) if r[7] is not None else None,
                "decision_price": float(r[8]) if r[8] is not None else None,
                "fill_price": float(r[9]) if r[9] is not None else None,
                "filled_qty": float(r[10]) if r[10] is not None else None,
                "slippage_pct": float(r[11]) if r[11] is not None else None,
                "status": r[12], "created_at": r[13],
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_recent_orders failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_slippage_summary() -> dict:
    """Average slippage_pct and count, for the weekly review / dashboard."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), AVG(slippage_pct) FROM orders "
                "WHERE slippage_pct IS NOT NULL AND trading_env=%s",
                (TRADING_ENV,),
            )
            count, avg = cur.fetchone()
        return {"count": count or 0, "avg_slippage_pct": round(float(avg), 4) if avg else None}
    except Exception as e:
        logging.error(f"get_slippage_summary failed: {e}")
        return {"count": 0, "avg_slippage_pct": None}
    finally:
        _put_conn(conn)


def get_fundamentals_all() -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, revenue_growth_yoy, gross_margin, fcf_margin, "
                "price_to_sales, ev_to_ebitda, forward_pe, market_cap FROM fundamentals"
            )
            rows = cur.fetchall()
        return [
            {
                "ticker": r[0],
                "revenue_growth_yoy": float(r[1]) if r[1] is not None else None,
                "gross_margin":       float(r[2]) if r[2] is not None else None,
                "fcf_margin":         float(r[3]) if r[3] is not None else None,
                "price_to_sales":     float(r[4]) if r[4] is not None else None,
                "ev_to_ebitda":       float(r[5]) if r[5] is not None else None,
                "forward_pe":         float(r[6]) if r[6] is not None else None,
                "market_cap":         float(r[7]) if r[7] is not None else None,
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_fundamentals_all failed: {e}")
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
    thesis_strength: int | None = None,
    catalyst_quality: int | None = None,
    valuation_risk_reward: int | None = None,
    data_completeness: int | None = None,
    chase_risk: int | None = None,
    chase_risk_pct_from_kijun: float | None = None,
    price_action: int | None = None,
    price_action_structure: str | None = None,
    catalyst_event: str | None = None,
    catalyst_date: str | None = None,
    catalyst_is_priced_in: bool | None = None,
    expected_return_pct: float | None = None,
    expected_downside_pct: float | None = None,
    thesis_summary: str | None = None,
    sources: list | None = None,
    source_count: int | None = None,
    weeks_to_catalyst: float | None = None,
    expected_return_per_week: float | None = None,
) -> None:
    # Stamp the SPY price at entry so we can compute alpha (return vs. benchmark)
    # when the trade eventually closes.
    entry_benchmark_price = None
    try:
        import yfinance as yf
        spy_price = yf.Ticker("SPY").fast_info.last_price
        if spy_price:
            entry_benchmark_price = float(spy_price)
    except Exception as e:
        logging.debug(f"log_trade_outcome: SPY benchmark price fetch failed: {e}")

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO trade_outcomes "
                "(approval_id, trading_env, ticker, action, conviction, thesis_bucket, entry_price, "
                " entry_benchmark_price, thesis_strength, catalyst_quality, "
                " valuation_risk_reward, data_completeness, chase_risk, "
                " chase_risk_pct_from_kijun, price_action, price_action_structure, "
                " catalyst_event, catalyst_date, catalyst_is_priced_in, "
                " expected_return_pct, expected_downside_pct, thesis_summary, "
                " sources_json, source_count, weeks_to_catalyst, expected_return_per_week) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (approval_id, TRADING_ENV, ticker, action, conviction, bucket, entry_price,
                 entry_benchmark_price, thesis_strength, catalyst_quality,
                 valuation_risk_reward, data_completeness, chase_risk,
                 chase_risk_pct_from_kijun, price_action, price_action_structure,
                 catalyst_event, catalyst_date, catalyst_is_priced_in,
                 expected_return_pct, expected_downside_pct, thesis_summary,
                 psycopg2.extras.Json(sources) if sources is not None else None, source_count,
                 weeks_to_catalyst, expected_return_per_week),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_trade_outcome failed: {e}")
    finally:
        _put_conn(conn)


def get_tickers_missing_catalyst_data(tickers: list[str]) -> list[str]:
    """Backfill target list (2026-08-30, user request): a currently-held
    ticker "is missing catalyst data" if its MOST RECENT trade_outcomes
    row (by entry_date) has no catalyst_event -- covers both "bought
    before this tracking existed" and "has a trade_outcomes row, but from
    before this feature shipped." A ticker with zero trade_outcomes rows
    at all also counts as missing (nothing to check against)."""
    if not tickers:
        return []
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT ON (ticker) ticker, catalyst_event "
                "FROM trade_outcomes WHERE trading_env=%s AND ticker = ANY(%s) "
                "ORDER BY ticker, entry_date DESC",
                (TRADING_ENV, tickers),
            )
            latest = {row[0]: row[1] for row in cur.fetchall()}
        return [t for t in tickers if not latest.get(t)]
    except Exception as e:
        logging.error(f"get_tickers_missing_catalyst_data failed: {e}")
        return list(tickers)  # fail toward re-checking, not silently skipping
    finally:
        _put_conn(conn)


def backfill_catalyst_outcome(
    ticker: str,
    conviction: int,
    bucket: str,
    entry_price: float,
    entry_date,
    catalyst_event: str | None,
    catalyst_date: str | None,
    catalyst_is_priced_in: bool | None,
    expected_return_pct: float | None,
    expected_downside_pct: float | None,
    thesis_summary: str | None,
    sources: list | None,
    source_count: int | None,
    weeks_to_catalyst: float | None = None,
    expected_return_per_week: float | None = None,
) -> None:
    """Writes a fresh catalyst/expected-return snapshot for an ALREADY-HELD
    position (2026-08-30, user request) -- distinct from log_trade_outcome,
    which fires at real trade execution. If a trade_outcomes row already
    exists for this ticker (bought before this feature shipped), updates
    its MOST RECENT row's catalyst columns in place rather than creating a
    confusing duplicate with a fabricated entry price; otherwise inserts a
    new row using position_stops' real entry_price/opened_at as the best
    available entry data, with approval_id left NULL (nullable FK -- no
    real approval exists for a position that predates this tracking)."""
    conn = _get_conn()
    sources_json = psycopg2.extras.Json(sources) if sources is not None else None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM trade_outcomes WHERE trading_env=%s AND ticker=%s "
                "ORDER BY entry_date DESC LIMIT 1",
                (TRADING_ENV, ticker),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE trade_outcomes SET catalyst_event=%s, catalyst_date=%s, "
                    "catalyst_is_priced_in=%s, expected_return_pct=%s, expected_downside_pct=%s, "
                    "thesis_summary=%s, sources_json=%s, source_count=%s, "
                    "weeks_to_catalyst=%s, expected_return_per_week=%s WHERE id=%s",
                    (catalyst_event, catalyst_date, catalyst_is_priced_in,
                     expected_return_pct, expected_downside_pct, thesis_summary,
                     sources_json, source_count, weeks_to_catalyst, expected_return_per_week, row[0]),
                )
            else:
                cur.execute(
                    "INSERT INTO trade_outcomes "
                    "(approval_id, trading_env, ticker, action, conviction, thesis_bucket, "
                    " entry_price, entry_date, catalyst_event, catalyst_date, catalyst_is_priced_in, "
                    " expected_return_pct, expected_downside_pct, thesis_summary, sources_json, source_count, "
                    " weeks_to_catalyst, expected_return_per_week) "
                    "VALUES (NULL,%s,%s,'BACKFILL',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (TRADING_ENV, ticker, conviction, bucket, entry_price, entry_date,
                     catalyst_event, catalyst_date, catalyst_is_priced_in,
                     expected_return_pct, expected_downside_pct, thesis_summary,
                     sources_json, source_count, weeks_to_catalyst, expected_return_per_week),
                )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"backfill_catalyst_outcome({ticker}) failed: {e}")
    finally:
        _put_conn(conn)


def compute_outcome_metrics(
    entry_price: float,
    exit_price: float,
    entry_benchmark_price: float | None,
    exit_benchmark_price: float | None,
) -> dict:
    """
    Pure function, no DB access — kept separate from close_trade_outcome so
    the outcome/alpha math can be unit-tested without a database connection.
    """
    outcome_pct = (exit_price - entry_price) / entry_price if entry_price else None
    was_correct = (outcome_pct is not None) and (outcome_pct > 0)

    benchmark_return_pct = None
    alpha_pct = None
    if entry_benchmark_price and exit_benchmark_price:
        benchmark_return_pct = (exit_benchmark_price - entry_benchmark_price) / entry_benchmark_price
        if outcome_pct is not None:
            alpha_pct = outcome_pct - benchmark_return_pct

    return {
        "outcome_pct": outcome_pct,
        "was_correct": was_correct,
        "benchmark_return_pct": benchmark_return_pct,
        "alpha_pct": alpha_pct,
    }


def close_trade_outcome(ticker: str, exit_price: float, exit_date=None) -> bool:
    """
    Closes the most recent still-open (exit_date IS NULL) trade_outcomes row
    for this ticker: computes outcome_pct, was_correct, and alpha vs. SPY over
    the same holding window. Called after a confirmed EXIT fill.
    Returns True if a row was closed.
    """
    from datetime import datetime, timezone as _tz

    exit_date = exit_date or datetime.now(_tz.utc)

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, entry_price, entry_benchmark_price FROM trade_outcomes "
                "WHERE ticker=%s AND trading_env=%s AND exit_date IS NULL "
                "ORDER BY entry_date DESC LIMIT 1",
                (ticker, TRADING_ENV),
            )
            row = cur.fetchone()
            if not row:
                logging.warning(f"close_trade_outcome: no open outcome row for {ticker}")
                return False

            outcome_id, entry_price, entry_benchmark_price = row
            entry_price = float(entry_price)
            entry_benchmark_price = float(entry_benchmark_price) if entry_benchmark_price else None

            exit_benchmark_price = None
            try:
                import yfinance as yf
                spy_price = yf.Ticker("SPY").fast_info.last_price
                if spy_price:
                    exit_benchmark_price = float(spy_price)
            except Exception as e:
                logging.debug(f"close_trade_outcome: SPY exit price fetch failed: {e}")

            metrics = compute_outcome_metrics(
                entry_price, exit_price, entry_benchmark_price, exit_benchmark_price
            )
            outcome_pct          = metrics["outcome_pct"]
            was_correct          = metrics["was_correct"]
            benchmark_return_pct = metrics["benchmark_return_pct"]
            alpha_pct            = metrics["alpha_pct"]

            cur.execute(
                "UPDATE trade_outcomes SET exit_price=%s, exit_date=%s, outcome_pct=%s, "
                "was_correct=%s, exit_benchmark_price=%s, benchmark_return_pct=%s, alpha_pct=%s "
                "WHERE id=%s",
                (exit_price, exit_date, outcome_pct, was_correct, exit_benchmark_price,
                 benchmark_return_pct, alpha_pct, outcome_id),
            )
        conn.commit()
        logging.info(
            f"[{ticker}] Outcome closed: {outcome_pct:+.1%} "
            f"(alpha vs SPY: {alpha_pct:+.1%})" if outcome_pct is not None and alpha_pct is not None
            else f"[{ticker}] Outcome closed: outcome_pct={outcome_pct}"
        )
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"close_trade_outcome failed: {e}")
        return False
    finally:
        _put_conn(conn)


def get_conviction_calibration() -> list[dict]:
    """
    Realized win-rate and average alpha grouped by the conviction the analyst
    stated at entry. Used to show the model its own track record.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT conviction, COUNT(*) AS total, "
                "  COUNT(*) FILTER (WHERE was_correct = TRUE) AS wins, "
                "  AVG(outcome_pct) AS avg_pct, AVG(alpha_pct) AS avg_alpha "
                "FROM trade_outcomes "
                "WHERE outcome_pct IS NOT NULL AND trading_env=%s "
                "GROUP BY conviction ORDER BY conviction DESC",
                (TRADING_ENV,),
            )
            rows = cur.fetchall()
        return [
            {
                "conviction": r[0], "total": r[1], "wins": r[2],
                "win_rate": round(r[2] / r[1], 3) if r[1] else 0,
                "avg_outcome_pct": round(float(r[3]), 4) if r[3] else None,
                "avg_alpha_pct": round(float(r[4]), 4) if r[4] else None,
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_conviction_calibration failed: {e}")
        return []
    finally:
        _put_conn(conn)


_SUBSCORE_COLUMNS = (
    "thesis_strength", "catalyst_quality", "valuation_risk_reward", "data_completeness",
    "chase_risk", "price_action",
)


def get_subscore_calibration(dimension: str) -> list[dict]:
    """
    Same shape as get_conviction_calibration, but grouped by one of the four
    1-5 sub-scores (see agents/analyst.py) instead of the blended 1-10
    conviction. This is what actually makes the sub-scores useful: instead of
    only knowing "conviction-8 calls win 55% of the time", the model can see
    "catalyst_quality-5 calls win 70%, but valuation_risk_reward-5 calls only
    win 45%" — a specific, actionable signal a single blended number hides.
    """
    if dimension not in _SUBSCORE_COLUMNS:
        raise ValueError(f"Unknown sub-score dimension: {dimension}")
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            # Scoped to the current trading_env deliberately -- feeds
            # agents/analyst.py's real-time conviction calibration, so
            # paper's much larger and differently-sized trade history must
            # not silently bias live's own decisions (or vice versa). Live
            # simply has no calibration signal until it accumulates its own
            # trades; analyst.py already handles an empty result gracefully
            # (checks `if r.get("total")` before using a bucket).
            cur.execute(
                f"SELECT {dimension}, COUNT(*) AS total, "
                f"  COUNT(*) FILTER (WHERE was_correct = TRUE) AS wins, "
                f"  AVG(outcome_pct) AS avg_pct, AVG(alpha_pct) AS avg_alpha "
                f"FROM trade_outcomes "
                f"WHERE outcome_pct IS NOT NULL AND {dimension} IS NOT NULL AND trading_env=%s "
                f"GROUP BY {dimension} ORDER BY {dimension} DESC",
                (TRADING_ENV,),
            )
            rows = cur.fetchall()
        return [
            {
                "score": r[0], "total": r[1], "wins": r[2],
                "win_rate": round(r[2] / r[1], 3) if r[1] else 0,
                "avg_outcome_pct": round(float(r[3]), 4) if r[3] else None,
                "avg_alpha_pct": round(float(r[4]), 4) if r[4] else None,
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_subscore_calibration({dimension}) failed: {e}")
        return []
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
                "FROM trade_outcomes WHERE exit_date IS NULL AND trading_env=%s ORDER BY entry_date",
                (TRADING_ENV,),
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
                "exit_price, exit_date, outcome_pct, was_correct, alpha_pct "
                "FROM trade_outcomes WHERE trading_env=%s ORDER BY entry_date DESC LIMIT %s",
                (TRADING_ENV, limit),
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
                "alpha_pct": float(r[14]) if r[14] is not None else None,
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
                "WHERE outcome_pct IS NOT NULL AND trading_env=%s "
                "GROUP BY thesis_bucket",
                (TRADING_ENV,),
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


def get_bucket_attribution() -> list[dict]:
    """
    Approximate realized dollar P&L contribution per bucket: joins closed
    trade_outcomes to the pending_approvals row that sized them and reads
    the dollar amount actually invested (risk_json->>'initial_dollar') at
    entry, then applies outcome_pct. This is an approximation — it doesn't
    account for later adds/trims to the same position — but it's the first
    time this system can answer "is the Moonshot bucket paying for its risk?"
    with real numbers instead of a guess.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    o.thesis_bucket,
                    COUNT(*) AS closed_trades,
                    SUM((pa.risk_json->>'initial_dollar')::DECIMAL * o.outcome_pct) AS dollar_pnl,
                    SUM((pa.risk_json->>'initial_dollar')::DECIMAL) AS dollar_deployed
                FROM trade_outcomes o
                JOIN pending_approvals pa ON pa.id = o.approval_id
                WHERE o.outcome_pct IS NOT NULL
                  AND pa.risk_json ? 'initial_dollar'
                  AND o.trading_env = %s
                GROUP BY o.thesis_bucket
                ORDER BY dollar_pnl DESC NULLS LAST
                """,
                (TRADING_ENV,),
            )
            rows = cur.fetchall()
        return [
            {
                "thesis_bucket": r[0], "closed_trades": r[1],
                "dollar_pnl": round(float(r[2]), 2) if r[2] is not None else 0.0,
                "dollar_deployed": round(float(r[3]), 2) if r[3] is not None else 0.0,
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_bucket_attribution failed: {e}")
        return []
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


def get_earnings_in_range(start, end) -> list[dict]:
    """All watchlist tickers with an earnings_calendar row in [start, end]
    (inclusive), for sector-level earnings-cluster detection (2026-08-30,
    user request) -- ingestion.sector_calendar.refresh_sector_calendar()
    groups these by GICS sector to flag "3+ of a sector's names report
    within the same week" as its own sector-scope catalyst event, distinct
    from any single ticker's own earnings date."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, earnings_date, confirmed FROM earnings_calendar "
                "WHERE earnings_date >= %s AND earnings_date <= %s "
                "ORDER BY earnings_date ASC",
                (start, end),
            )
            rows = cur.fetchall()
        return [{"ticker": r[0], "earnings_date": r[1], "confirmed": r[2]} for r in rows]
    except Exception as e:
        logging.error(f"get_earnings_in_range failed: {e}")
        return []
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


# ---------------------------------------------------------------------------
# Watchlist candidates (new-ticker discovery — agents/candidate_scanner.py)
# ---------------------------------------------------------------------------

def log_watchlist_candidate(
    ticker: str,
    gics_sector: str | None,
    movement_score: int,
    thesis_match_score: int,
    composite_score: int,
    matched_thesis: str | None,
    notes: str,
) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO watchlist_candidates "
                "(ticker, gics_sector, movement_score, thesis_match_score, "
                "composite_score, matched_thesis, notes) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (ticker, gics_sector, movement_score, thesis_match_score,
                 composite_score, matched_thesis, notes),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_watchlist_candidate failed: {e}")
    finally:
        _put_conn(conn)


def get_recent_watchlist_candidates(days: int = 7, limit: int = 25) -> list[dict]:
    """Latest scan's top candidates, deduped to each ticker's most recent
    row (a ticker can appear across multiple weekly scans)."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT DISTINCT ON (ticker) ticker, gics_sector, movement_score, "
                "thesis_match_score, composite_score, matched_thesis, notes, scanned_at "
                "FROM watchlist_candidates "
                "WHERE scanned_at > NOW() - (%s || ' days')::interval "
                "ORDER BY ticker, scanned_at DESC",
                (days,),
            )
            rows = cur.fetchall()
        rows.sort(key=lambda r: r["composite_score"], reverse=True)
        return rows[:limit]
    except Exception as e:
        logging.error(f"get_recent_watchlist_candidates failed: {e}")
        return []
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# News headlines (public research site's News section)
# ---------------------------------------------------------------------------

def log_news_item(ticker: str, title: str, url: str, image_url: str | None = None) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO news_log (ticker, title, url, image_url) VALUES (%s,%s,%s,%s)",
                (ticker, title, url, image_url),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_news_item failed: {e}")
    finally:
        _put_conn(conn)


def get_recent_news(limit: int = 30) -> list[dict]:
    """Latest headlines across the whole watchlist, newest first."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT ticker, title, url, image_url, published_at FROM news_log "
                "ORDER BY published_at DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()
    except Exception as e:
        logging.error(f"get_recent_news failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_recent_news_for_ticker(ticker: str, limit: int = 15) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT ticker, title, url, image_url, published_at FROM news_log "
                "WHERE ticker = %s ORDER BY published_at DESC LIMIT %s",
                (ticker.upper(), limit),
            )
            return cur.fetchall()
    except Exception as e:
        logging.error(f"get_recent_news_for_ticker({ticker}) failed: {e}")
        return []
    finally:
        _put_conn(conn)
