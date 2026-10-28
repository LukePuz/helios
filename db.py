import json
import logging
import os
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import psycopg2.errors
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
# version table means `python scripts/setup_db.py` / init_db() logs exactly which
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

    # ------------------------------------------------------------------
    # Swing-trading strategy redesign (2026-09-13, user request): replaces
    # the long-term conviction buckets (Compounder/Accelerator/Moonshot)
    # with setup-defined swing categories (Breakout/Catalyst/Continuation),
    # backed by a per-ticker technical+valuation trade ladder and original/
    # latest thesis tracking. setup_category is added ALONGSIDE the
    # existing thesis_bucket column rather than replacing it in place --
    # thesis_bucket stays populated during the transition (shadow mode,
    # see agents/setup_gates.py) so nothing reading it breaks until the
    # SETUP_GATES_ENABLED cutover is actually flipped.
    # ------------------------------------------------------------------
    (33, """
        CREATE TABLE IF NOT EXISTS trade_ladder (
            ticker              VARCHAR(10) PRIMARY KEY,
            computed_at         TIMESTAMPTZ DEFAULT NOW(),
            current_price       DECIMAL,
            atr                 DECIMAL,
            trend_stage         VARCHAR(20),
            base_detected       BOOLEAN,
            base_ceiling        DECIMAL,
            base_floor          DECIMAL,
            base_contractions   INTEGER,
            base_volume_dryup   BOOLEAN,
            gap_catalyst_active BOOLEAN,
            gap_day_high        DECIMAL,
            gap_day_low         DECIMAL,
            peer_implied_fair_value DECIMAL,
            street_consensus_target DECIMAL,
            fair_value_blended      DECIMAL,
            valuation_pct_of_fair   DECIMAL,
            valuation_tier          VARCHAR(20),
            valuation_outlier       BOOLEAN DEFAULT FALSE,
            rungs               JSONB
        );
    """),
    (34, "ALTER TABLE position_stops ADD COLUMN IF NOT EXISTS setup_category VARCHAR(20); "
         "ALTER TABLE position_stops ADD COLUMN IF NOT EXISTS valuation_tier_at_entry VARCHAR(20); "
         "ALTER TABLE position_stops ADD COLUMN IF NOT EXISTS original_thesis JSONB; "
         "ALTER TABLE position_stops ADD COLUMN IF NOT EXISTS latest_thesis JSONB; "
         "ALTER TABLE position_stops ADD COLUMN IF NOT EXISTS thesis_updated_at TIMESTAMPTZ;"),
    (35, "ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS setup_category VARCHAR(20); "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS setup_category VARCHAR(20); "
         "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS valuation_tier_at_entry VARCHAR(20);"),
    (36, """
        CREATE TABLE IF NOT EXISTS thesis_history (
            id           SERIAL PRIMARY KEY,
            ticker       VARCHAR(10),
            trading_env  VARCHAR(10) NOT NULL DEFAULT 'paper',
            recorded_at  TIMESTAMPTZ DEFAULT NOW(),
            thesis_json  JSONB
        );
    """),
    (37, "CREATE INDEX IF NOT EXISTS idx_thesis_history_ticker_recorded ON thesis_history (ticker, recorded_at DESC);"),
    # sharesOutstanding/targetMeanPrice already flow through yfinance's
    # .info dict (targetMeanPrice already lands in fundamentals.raw_json
    # today but nothing reads it) -- explicit columns make both queryable
    # without unpacking JSON, needed for the peer-implied fair-value calc
    # (agents/trade_ladder.py) and the Street-consensus anchor.
    (38, "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS shares_outstanding DECIMAL; "
         "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS target_mean_price DECIMAL;"),

    # Multi-pronged fundamentals-based valuation ensemble (2026-09-13, user
    # request) -- Street consensus alone was found to skew 60% of the
    # watchlist into "Deep Value" (sell-side targets are structurally
    # optimistic almost everywhere), so it's demoted to a displayed
    # reference range and replaced as the primary valuation driver by four
    # independent fundamentals-based methods (DCF, comps, Rule-of-40
    # multiple, PEG). These new fundamentals columns feed those calcs.
    (39, "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS free_cash_flow DECIMAL; "
         "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS net_cash DECIMAL; "
         "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS beta DECIMAL; "
         "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS earnings_growth DECIMAL; "
         "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS forward_eps DECIMAL; "
         "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS target_low_price DECIMAL; "
         "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS target_high_price DECIMAL;"),
    (40, "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS dcf_fair_value DECIMAL; "
         "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS comps_fair_value DECIMAL; "
         "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS rule40_fair_value DECIMAL; "
         "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS peg_fair_value DECIMAL; "
         "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS valuation_methods_used JSONB; "
         "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS valuation_confidence VARCHAR(10); "
         "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS street_target_low DECIMAL; "
         "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS street_target_high DECIMAL;"),

    # Reverse DCF / "Expectations Investing" (2026-09-13, user request) --
    # PEG and Rule-of-40 demoted out of the blended fair_value_blended
    # median (they're growth-quality CONTEXT, same tier as Street
    # consensus, not price-target inputs) and replaced as a primary
    # decision input by this: instead of assuming a growth rate to get a
    # price, hold price fixed and solve for the growth rate the market is
    # implicitly requiring -- then compare that ask against the ticker's
    # own actual growth. A genuinely different axis (plausibility, not
    # price), so it's NOT part of the dollar-value median either -- see
    # agents/valuation.py's compute_reverse_dcf docstring.
    (41, "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS implied_growth_rate DECIMAL; "
         "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS growth_gap_vs_actual DECIMAL; "
         "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS growth_assumption_flag VARCHAR(20);"),

    # Residual Income (Ohlson model) + Justified Multiple (2026-09-13, user
    # request) -- both CFA-standard, price-producing methods, added to
    # reach a 4-method blend (DCF, Comps, Residual Income, Justified
    # Multiple) after DCF+Comps alone proved too thin (median of 2 = plain
    # average, letting DCF's hypergrowth-undervaluation bias dominate).
    # book_value_per_share/return_on_equity/net_profit_margin are what
    # both new methods need; not previously promoted out of raw_json.
    (42, "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS book_value_per_share DECIMAL; "
         "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS return_on_equity DECIMAL; "
         "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS net_profit_margin DECIMAL;"),
    (43, "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS residual_income_fair_value DECIMAL; "
         "ALTER TABLE trade_ladder ADD COLUMN IF NOT EXISTS justified_multiple_fair_value DECIMAL;"),

    # Fundamentals history (2026-09-13, user request: "approaches that use
    # more of the company's own pricing history and past results"). The
    # `fundamentals` table has only ever stored the LATEST snapshot per
    # ticker (nightly upsert overwrites it) -- no method here can compare
    # a company against its OWN historical multiple/margin trajectory
    # without a real time series, which doesn't exist yet and can't be
    # backfilled. This starts logging one nightly snapshot per ticker
    # going forward (append-only, same pattern as thesis_history) so a
    # genuine "reversion to own historical average multiple" method
    # becomes possible once enough real history accumulates -- it produces
    # nothing useful today, same "can't backfill what wasn't recorded"
    # situation as thesis_history's own backfill-from-approvals earlier.
    (44, """
        CREATE TABLE IF NOT EXISTS fundamentals_history (
            id                  SERIAL PRIMARY KEY,
            ticker              VARCHAR(10) NOT NULL,
            recorded_at         TIMESTAMPTZ DEFAULT NOW(),
            price_to_sales      DECIMAL,
            ev_to_ebitda        DECIMAL,
            forward_pe          DECIMAL,
            revenue_growth_yoy  DECIMAL,
            fcf_margin          DECIMAL,
            net_profit_margin   DECIMAL,
            return_on_equity    DECIMAL
        );
    """),
    (45, "CREATE INDEX IF NOT EXISTS idx_fundamentals_history_ticker_recorded ON fundamentals_history (ticker, recorded_at DESC);"),

    # Broadened Comps universe (2026-09-13) -- the original sector-scoped
    # peer median was drawn only from the ~75-ticker watchlist, which
    # produced fragile/thin peer groups (LUNR's 3-name "Space Data
    # Centers" group spanning a 43x multiple range) and left most of the
    # watchlist at "Insufficient Data." yfinance's own sector/industry
    # classification works for ANY ticker (not just S&P 500 members --
    # confirmed 2026-09-13: RGTI/QBTS/LUNR all have real industry labels
    # despite being too small for index membership), so `industry` is
    # added to `fundamentals` for the watchlist, and a companion
    # sp500_fundamentals table gives a genuinely large peer pool (~500
    # names) to draw from, both grouped by the SAME field.
    (46, "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS sector VARCHAR(50); "
         "ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS industry VARCHAR(100);"),
    (47, """
        CREATE TABLE IF NOT EXISTS sp500_fundamentals (
            ticker              VARCHAR(10) PRIMARY KEY,
            updated_at          TIMESTAMPTZ DEFAULT NOW(),
            sector              VARCHAR(50),
            industry            VARCHAR(100),
            price_to_sales      DECIMAL,
            ev_to_ebitda        DECIMAL,
            revenue_growth_yoy  DECIMAL,
            gross_margin        DECIMAL,
            fcf_margin          DECIMAL
        );
    """),
    (48, "CREATE INDEX IF NOT EXISTS idx_sp500_fundamentals_industry ON sp500_fundamentals (industry);"),

    # Fixes a real gap found 2026-09-15: agents/scanner.py's shadow setup-
    # gate classification (agents/setup_gates.py, wired in 2026-09-13) was
    # only ever added to the in-memory candidate dict, never persisted --
    # "what did the new Breakout/Catalyst/Continuation system say during
    # today's actual scan" was unanswerable after the fact, even though
    # the classification itself ran. These two columns close that gap
    # going forward (can't backfill what wasn't recorded, same situation
    # as thesis_history's own backfill-from-approvals).
    (49, "ALTER TABLE scanner_candidates ADD COLUMN IF NOT EXISTS setup_category VARCHAR(20); "
         "ALTER TABLE scanner_candidates ADD COLUMN IF NOT EXISTS setup_triggered BOOLEAN;"),

    # 2026-09-17, user request: migration 49 only ever persisted the BEST-
    # match category and whether it triggered -- the full per-gate
    # criteria breakdown (which of the ~4-6 checks each of Breakout/
    # Catalyst/Continuation passed or failed) was computed every run but
    # never stored, so "nothing qualified" and "here's exactly how close
    # it got" were indistinguishable after the fact. This closes that --
    # can't backfill the two days already run before this column existed.
    (50, "ALTER TABLE scanner_candidates ADD COLUMN IF NOT EXISTS setup_gates_detail JSONB;"),
    # signals is the only record of every analyst verdict (WATCH/HOLD and
    # rejected calls never reach pending_approvals), but it had no account,
    # model or provenance (professor M19, 2026-10-28). Old rows stay NULL.
    (51, "ALTER TABLE signals ADD COLUMN IF NOT EXISTS trading_env VARCHAR(10);"),
    (52, "ALTER TABLE signals ADD COLUMN IF NOT EXISTS model VARCHAR(80);"),
    (53, "ALTER TABLE signals ADD COLUMN IF NOT EXISTS provenance JSONB;"),
    # Rows whose catalyst fields were filled after entry by
    # backfill_catalyst_outcome (professor C6 residual, 2026-10-28).
    (54, "ALTER TABLE trade_outcomes ADD COLUMN IF NOT EXISTS catalyst_backfilled BOOLEAN DEFAULT FALSE;"),
    # Per-verdict decision record: the model's raw action/conviction, each
    # gate that changed it, and the full model JSON (roadmap 0e #8).
    (55, "ALTER TABLE signals ADD COLUMN IF NOT EXISTS decision JSONB;"),
    # Full rendered analyst prompt per verdict (professor M7), so a decision
    # can be replayed. Kept out of `decision` because the brief contains
    # portfolio values: research exports must not copy this column.
    (56, "ALTER TABLE signals ADD COLUMN IF NOT EXISTS prompt TEXT;"),
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


def log_signal(symbol: str, signal: str, confidence: float, reasoning: str,
               model: str | None = None, provenance: dict | None = None,
               decision: dict | None = None, prompt: str | None = None) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO signals (symbol, signal, confidence, reasoning, trading_env, model, "
                "provenance, decision, prompt) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (symbol, signal, confidence, reasoning, TRADING_ENV, model,
                 psycopg2.extras.Json(provenance) if provenance is not None else None,
                 psycopg2.extras.Json(decision, dumps=lambda o: json.dumps(o, default=str))
                 if decision is not None else None,
                 prompt),
            )
        conn.commit()
    except psycopg2.errors.UndefinedColumn:
        # Migrations 51-56 not applied yet (they run in init_db at stack
        # start); keep the verdict rather than lose it.
        conn.rollback()
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


# Signal types crew.py logs for routing decisions, not analyst verdicts.
_NON_ANALYST_SIGNALS = ("SCREENED_OUT", "CATALYST_BLOCKED", "COOLDOWN_SKIPPED", "SELL_NO_POSITION",
                        "EARNINGS_BLOCKED", "ROUTE_APPROVAL_PENDING", "ROUTE_NO_PRICE",
                        "ROUTE_ZERO_SHARES", "ROUTE_CONCENTRATION_ZERO")


def get_latest_signal_for(ticker: str) -> dict | None:
    """Most recent analysis result for a ticker, regardless of verdict —
    every call to agents.analyst.analyze() logs here (see analyst.py's
    db.log_signal call), including routine HOLDs on positions already
    held, not just actionable OPEN/ADD/TRIM/EXIT ones. This is the
    freshest conviction reading available for a held position between
    its entry and whenever position_stops.conviction next gets touched
    (only on a fresh ADD) — used by RiskManager.find_swap_candidate to
    judge whether a held position still deserves its size, not just what
    conviction it was entered at.

    Skips the routing rows crew.py writes to the same table (SCREENED_OUT and
    CATALYST_BLOCKED at 0.0, COOLDOWN_SKIPPED, SELL_NO_POSITION). Before
    2026-10-27 one screened-out headline made a healthy holding read as
    conviction 0, which passes the swap test."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal, confidence, timestamp FROM signals "
                "WHERE symbol=%s AND signal NOT IN %s "
                "ORDER BY timestamp DESC LIMIT 1",
                (ticker, _NON_ANALYST_SIGNALS),
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


def has_newer_buy_approval(ticker: str, approval_id: int) -> bool:
    """True if a later OPEN/ADD approval exists for ticker (pending, in flight,
    executed or itself deferred). retry_deferred_entry drops the older one so
    two deferred buys can't both fire at the open."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pending_approvals "
                "WHERE ticker=%s AND trading_env=%s AND id>%s AND action IN ('OPEN','ADD') "
                "AND status IN ('PENDING','PROCESSING','APPROVED','FAILED') LIMIT 1",
                (ticker, TRADING_ENV, approval_id),
            )
            return cur.fetchone() is not None
    except Exception as e:
        logging.error(f"has_newer_buy_approval failed: {e}")
        return False
    finally:
        _put_conn(conn)


def get_approval_by_id(approval_id: int) -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, ticker, action, conviction, thesis_bucket, position_size_pct, "
                "stop_loss_price, entry_strategy, reasoning, analyst_json, risk_json, status, expires_at, "
                "timestamp FROM pending_approvals WHERE id=%s",
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
            # Before 2026-10-28 this was missing, so retry_deferred_entry's
            # 16 h age check always saw None and never fired.
            "timestamp": row[13],
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


def get_position_context(ticker: str) -> dict:
    """Thesis and recent verdicts for a held ticker, for the brief's POSITION
    CONTEXT block: {original_thesis, latest_thesis, thesis_updated_at,
    profit_target_taken, recent_verdicts: [{timestamp, signal, conviction}]}.
    Empty dict on error."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT original_thesis, latest_thesis, thesis_updated_at, profit_target_taken "
                "FROM position_stops WHERE ticker=%s AND trading_env=%s",
                (ticker, TRADING_ENV),
            )
            row = cur.fetchone()
            cur.execute(
                "SELECT timestamp, signal, confidence FROM signals "
                "WHERE symbol=%s AND signal NOT IN %s ORDER BY timestamp DESC LIMIT 3",
                (ticker, _NON_ANALYST_SIGNALS),
            )
            verdicts = cur.fetchall()
        out = {"recent_verdicts": [{"timestamp": v[0], "signal": v[1],
                                    "conviction": round(float(v[2] or 0) * 10)} for v in verdicts]}
        if row:
            out.update(original_thesis=row[0], latest_thesis=row[1], thesis_updated_at=row[2],
                       profit_target_taken=row[3])
        return out
    except Exception as e:
        conn.rollback()
        logging.error(f"get_position_context failed: {e}")
        return {}
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
                "opened_at, profit_target_taken, setup_category, valuation_tier_at_entry "
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
                "setup_category": r[12], "valuation_tier_at_entry": r[13],
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


def set_setup_category(ticker: str, setup_category: str, valuation_tier: str | None = None) -> None:
    """Stamps the live setup-gate classification onto an open position, alongside
    the legacy thesis_bucket (kept populated during the SETUP_GATES_ENABLED
    shadow period -- see agents/setup_gates.py module docstring)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE position_stops SET setup_category=%s, valuation_tier_at_entry=%s "
                "WHERE ticker=%s AND trading_env=%s",
                (setup_category, valuation_tier, ticker, TRADING_ENV),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"[{ticker}] set_setup_category failed: {e}")
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# Trade ladder (2026-09-13, swing-trading redesign) -- one row per ticker,
# latest snapshot only. See agents/trade_ladder.py for how `rungs` and the
# valuation fields are computed.
# ---------------------------------------------------------------------------

def upsert_trade_ladder(ticker: str, ladder: dict) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trade_ladder (
                    ticker, computed_at, current_price, atr, trend_stage,
                    base_detected, base_ceiling, base_floor, base_contractions, base_volume_dryup,
                    gap_catalyst_active, gap_day_high, gap_day_low,
                    peer_implied_fair_value, street_consensus_target, fair_value_blended,
                    valuation_pct_of_fair, valuation_tier, valuation_outlier, rungs,
                    dcf_fair_value, comps_fair_value, rule40_fair_value, peg_fair_value,
                    valuation_methods_used, valuation_confidence, street_target_low, street_target_high,
                    implied_growth_rate, growth_gap_vs_actual, growth_assumption_flag,
                    residual_income_fair_value, justified_multiple_fair_value
                ) VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                          %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    computed_at = NOW(),
                    current_price = EXCLUDED.current_price,
                    atr = EXCLUDED.atr,
                    trend_stage = EXCLUDED.trend_stage,
                    base_detected = EXCLUDED.base_detected,
                    base_ceiling = EXCLUDED.base_ceiling,
                    base_floor = EXCLUDED.base_floor,
                    base_contractions = EXCLUDED.base_contractions,
                    base_volume_dryup = EXCLUDED.base_volume_dryup,
                    gap_catalyst_active = EXCLUDED.gap_catalyst_active,
                    gap_day_high = EXCLUDED.gap_day_high,
                    gap_day_low = EXCLUDED.gap_day_low,
                    peer_implied_fair_value = EXCLUDED.peer_implied_fair_value,
                    street_consensus_target = EXCLUDED.street_consensus_target,
                    fair_value_blended = EXCLUDED.fair_value_blended,
                    valuation_pct_of_fair = EXCLUDED.valuation_pct_of_fair,
                    valuation_tier = EXCLUDED.valuation_tier,
                    valuation_outlier = EXCLUDED.valuation_outlier,
                    rungs = EXCLUDED.rungs,
                    dcf_fair_value = EXCLUDED.dcf_fair_value,
                    comps_fair_value = EXCLUDED.comps_fair_value,
                    rule40_fair_value = EXCLUDED.rule40_fair_value,
                    peg_fair_value = EXCLUDED.peg_fair_value,
                    valuation_methods_used = EXCLUDED.valuation_methods_used,
                    valuation_confidence = EXCLUDED.valuation_confidence,
                    street_target_low = EXCLUDED.street_target_low,
                    street_target_high = EXCLUDED.street_target_high,
                    implied_growth_rate = EXCLUDED.implied_growth_rate,
                    growth_gap_vs_actual = EXCLUDED.growth_gap_vs_actual,
                    growth_assumption_flag = EXCLUDED.growth_assumption_flag,
                    residual_income_fair_value = EXCLUDED.residual_income_fair_value,
                    justified_multiple_fair_value = EXCLUDED.justified_multiple_fair_value
                """,
                (
                    ticker, ladder.get("current_price"), ladder.get("atr"), ladder.get("trend_stage"),
                    ladder.get("base_detected"), ladder.get("base_ceiling"), ladder.get("base_floor"),
                    ladder.get("base_contractions"), ladder.get("base_volume_dryup"),
                    ladder.get("gap_catalyst_active"), ladder.get("gap_day_high"), ladder.get("gap_day_low"),
                    ladder.get("peer_implied_fair_value"), ladder.get("street_consensus_target"),
                    ladder.get("fair_value_blended"), ladder.get("valuation_pct_of_fair"),
                    ladder.get("valuation_tier"), ladder.get("valuation_outlier", False),
                    psycopg2.extras.Json(ladder.get("rungs", [])),
                    ladder.get("dcf_fair_value"), ladder.get("comps_fair_value"),
                    ladder.get("rule40_fair_value"), ladder.get("peg_fair_value"),
                    psycopg2.extras.Json(ladder.get("valuation_methods_used", [])),
                    ladder.get("valuation_confidence"),
                    ladder.get("street_target_low"), ladder.get("street_target_high"),
                    ladder.get("implied_growth_rate"), ladder.get("growth_gap_vs_actual"),
                    ladder.get("growth_assumption_flag"),
                    ladder.get("residual_income_fair_value"), ladder.get("justified_multiple_fair_value"),
                ),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"[{ticker}] upsert_trade_ladder failed: {e}")
    finally:
        _put_conn(conn)


_TRADE_LADDER_DECIMAL_FIELDS = (
    "current_price", "atr", "base_ceiling", "base_floor", "gap_day_high", "gap_day_low",
    "peer_implied_fair_value", "street_consensus_target", "fair_value_blended",
    "valuation_pct_of_fair", "dcf_fair_value", "comps_fair_value", "rule40_fair_value",
    "peg_fair_value", "street_target_low", "street_target_high", "implied_growth_rate",
    "growth_gap_vs_actual", "residual_income_fair_value", "justified_multiple_fair_value",
)


def _normalize_trade_ladder(row: dict) -> dict:
    """psycopg2's RealDictCursor returns DECIMAL columns as decimal.Decimal,
    not float -- found 2026-09-13 wiring agents/setup_gates.py: `ladder.
    get("atr") * 0.5` raised TypeError (Decimal * float unsupported)
    despite the same column reading fine as a value. Every OTHER accessor
    in this file explicitly casts to float; this one didn't because it
    reads via `SELECT *` for schema flexibility. Cast here once instead
    of at every call site. `rungs` (JSONB) is unaffected -- JSON numbers
    round-trip as native float/int, not Decimal."""
    for field in _TRADE_LADDER_DECIMAL_FIELDS:
        if row.get(field) is not None:
            row[field] = float(row[field])
    return row


def get_trade_ladder(ticker: str) -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM trade_ladder WHERE ticker=%s", (ticker,))
            row = cur.fetchone()
        return _normalize_trade_ladder(dict(row)) if row else None
    except Exception as e:
        logging.error(f"[{ticker}] get_trade_ladder failed: {e}")
        return None
    finally:
        _put_conn(conn)


def get_all_trade_ladders() -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM trade_ladder ORDER BY ticker")
            return [_normalize_trade_ladder(dict(r)) for r in cur.fetchall()]
    except Exception as e:
        logging.error(f"get_all_trade_ladders failed: {e}")
        return []
    finally:
        _put_conn(conn)


# ---------------------------------------------------------------------------
# Thesis tracking (2026-09-13) -- original_thesis is written once at entry
# (execution.py's fill-confirmation path), latest_thesis is overwritten on
# EVERY scheduled position review regardless of action (fixing the prior
# gap where a plain HOLD verdict's full reasoning was lost, only a 176-char
# activity_feed line survived). thesis_history keeps every snapshot, not
# just the latest, for drift analysis over a position's whole life.
# ---------------------------------------------------------------------------

def set_original_thesis(ticker: str, thesis: dict) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE position_stops SET original_thesis=%s, latest_thesis=%s, thesis_updated_at=NOW() "
                "WHERE ticker=%s AND trading_env=%s",
                (psycopg2.extras.Json(thesis), psycopg2.extras.Json(thesis), ticker, TRADING_ENV),
            )
        conn.commit()
        log_thesis_history(ticker, thesis)
    except Exception as e:
        conn.rollback()
        logging.error(f"[{ticker}] set_original_thesis failed: {e}")
    finally:
        _put_conn(conn)


def update_latest_thesis(ticker: str, thesis: dict) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE position_stops SET latest_thesis=%s, thesis_updated_at=NOW() "
                "WHERE ticker=%s AND trading_env=%s",
                (psycopg2.extras.Json(thesis), ticker, TRADING_ENV),
            )
        conn.commit()
        log_thesis_history(ticker, thesis)
    except Exception as e:
        conn.rollback()
        logging.error(f"[{ticker}] update_latest_thesis failed: {e}")
    finally:
        _put_conn(conn)


def get_thesis(ticker: str) -> dict | None:
    """Returns {"original": {...}, "latest": {...}, "updated_at": ...}, or
    None only if NEITHER exists yet. Found 2026-09-13 wiring
    review_held_position's latest_thesis capture: a scheduled review can
    legitimately run before an original was ever recorded (a position
    opened before thesis tracking existed, or one whose entry-side
    capture failed) -- checking only original_thesis for None treated a
    real, successfully-captured latest_thesis as if nothing were there at
    all. Callers should treat a None "original" as "no baseline to diff
    against" (db.compute_thesis_drift already can't run without one),
    not as the whole thesis being absent."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT original_thesis, latest_thesis, thesis_updated_at FROM position_stops "
                "WHERE ticker=%s AND trading_env=%s",
                (ticker, TRADING_ENV),
            )
            row = cur.fetchone()
        if not row or (row[0] is None and row[1] is None):
            return None
        return {"original": row[0], "latest": row[1], "updated_at": row[2]}
    except Exception as e:
        logging.error(f"[{ticker}] get_thesis failed: {e}")
        return None
    finally:
        _put_conn(conn)


def log_thesis_history(ticker: str, thesis: dict) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO thesis_history (ticker, trading_env, thesis_json) VALUES (%s, %s, %s)",
                (ticker, TRADING_ENV, psycopg2.extras.Json(thesis)),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"[{ticker}] log_thesis_history failed: {e}")
    finally:
        _put_conn(conn)


def get_thesis_history(ticker: str, limit: int = 30) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT recorded_at, thesis_json FROM thesis_history "
                "WHERE ticker=%s AND trading_env=%s ORDER BY recorded_at DESC LIMIT %s",
                (ticker, TRADING_ENV, limit),
            )
            return [{"recorded_at": r[0], "thesis": r[1]} for r in cur.fetchall()]
    except Exception as e:
        logging.error(f"[{ticker}] get_thesis_history failed: {e}")
        return []
    finally:
        _put_conn(conn)


# Keyword heuristic (2026-09-15, user request) -- a broad macro event
# (FOMC, CPI, jobs report) is not a company-specific catalyst even when
# it's the most proximate dated event in a situation brief; found reading
# 4 real held positions that LUNR and NVDA's latest thesis both listed
# "FOMC meeting" as THE catalyst, which says nothing about either
# company's own thesis actually playing out. Deliberately a simple
# substring check, not NLP -- transparent and auditable over a fancier
# classifier for a binary "is this actually about the company" signal.
_MACRO_CATALYST_KEYWORDS = (
    "fomc", "cpi", "federal reserve", "fed meeting", "jobs report",
    "nonfarm payroll", "nfp", "pce", "gdp report", "rate decision",
)


def compute_no_near_term_catalyst(catalyst: dict | None, weeks_to_catalyst: float | None) -> bool:
    """True if a position has no genuine, dated, COMPANY-SPECIFIC catalyst
    to justify continued holding -- either no computable timeframe at all
    (weeks_to_catalyst is None), or the stated catalyst is actually a
    broad macro event masquerading as one. Used by thesis capture (agents/
    crew.py, execution.py) to make this a first-class, queryable flag
    instead of something only visible by reading the reasoning prose."""
    if weeks_to_catalyst is None:
        return True
    event = ((catalyst or {}).get("event") or "").lower()
    return any(kw in event for kw in _MACRO_CATALYST_KEYWORDS)


def compute_thesis_drift(original: dict, latest: dict) -> dict:
    """Pure, deterministic diff between a position's original and latest
    thesis snapshot -- no LLM call, same 'compute the hard number in code'
    convention as chase_risk/price_action. `material_change` is the alert
    throttle: most daily reviews won't trip it, so send_thesis_update_alert
    doesn't turn into daily noise."""
    orig_conviction = original.get("conviction")
    latest_conviction = latest.get("conviction")
    conviction_delta = (
        latest_conviction - orig_conviction
        if orig_conviction is not None and latest_conviction is not None else None
    )

    catalyst_status = latest.get("catalyst_status", "pending")
    catalyst_overdue = False
    try:
        from agents.analyst import compute_weeks_to_catalyst
        weeks = compute_weeks_to_catalyst((original.get("catalyst") or {}).get("date"))
        # weeks_to_catalyst is computed relative to "now" -- a negative
        # result would mean the date's already passed, but the helper
        # floors at 1 day out, so overdue is instead detected by comparing
        # the ORIGINAL weeks-out figure against how much wall-clock time
        # has elapsed since the thesis was first recorded.
        orig_weeks = original.get("weeks_to_catalyst")
        as_of = original.get("as_of")
        if orig_weeks is not None and as_of:
            from datetime import datetime, timezone
            recorded = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            elapsed_weeks = (datetime.now(timezone.utc) - recorded).total_seconds() / (7 * 86400)
            catalyst_overdue = elapsed_weeks > orig_weeks and catalyst_status == "pending"
    except Exception:
        pass

    new_risks = list(set(latest.get("key_risks") or []) - set(original.get("key_risks") or []))

    # Found 2026-09-15 reviewing 4 real held positions: QBTS's ORIGINAL
    # thesis was anchored on a specific, named, dated catalyst (a Nasdaq
    # Verafin partnership); its LATEST thesis had quietly replaced that
    # with vague "ongoing" technical milestones -- a real degradation
    # conviction_delta alone doesn't necessarily catch (QBTS's own
    # conviction only dropped 7->5, "material" by the >=2 rule already,
    # but a position could lose its whole dated catalyst while conviction
    # drops by just 1). Distinct from "no_near_term_catalyst persisting" --
    # that would fire every single review forever and defeat the alert
    # throttle; this only fires on an actual CHANGE from having one.
    orig_no_catalyst = original.get("no_near_term_catalyst")
    latest_no_catalyst = latest.get("no_near_term_catalyst")
    catalyst_degraded = bool(orig_no_catalyst is False and latest_no_catalyst is True)

    material_change = bool(
        (conviction_delta is not None and abs(conviction_delta) >= 2)
        or catalyst_status in ("invalidated", "confirmed_negative")
        or (latest.get("action") not in (None, "OPEN", "ADD", "HOLD") and original.get("action") in ("OPEN", "ADD"))
        or catalyst_degraded
    )

    return {
        "conviction_delta": conviction_delta,
        "catalyst_status": catalyst_status,
        "catalyst_overdue": catalyst_overdue,
        "new_risks": new_risks,
        "catalyst_degraded": catalyst_degraded,
        "material_change": material_change,
    }


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
                     price_to_sales, ev_to_ebitda, forward_pe, market_cap, raw_json,
                     shares_outstanding, target_mean_price, free_cash_flow, net_cash,
                     beta, earnings_growth, forward_eps, target_low_price, target_high_price,
                     book_value_per_share, return_on_equity, net_profit_margin, sector, industry)
                VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    updated_at         = NOW(),
                    revenue_growth_yoy = EXCLUDED.revenue_growth_yoy,
                    gross_margin       = EXCLUDED.gross_margin,
                    fcf_margin         = EXCLUDED.fcf_margin,
                    price_to_sales     = EXCLUDED.price_to_sales,
                    ev_to_ebitda       = EXCLUDED.ev_to_ebitda,
                    forward_pe         = EXCLUDED.forward_pe,
                    market_cap         = EXCLUDED.market_cap,
                    raw_json           = EXCLUDED.raw_json,
                    shares_outstanding = EXCLUDED.shares_outstanding,
                    target_mean_price  = EXCLUDED.target_mean_price,
                    free_cash_flow     = EXCLUDED.free_cash_flow,
                    net_cash           = EXCLUDED.net_cash,
                    beta               = EXCLUDED.beta,
                    earnings_growth    = EXCLUDED.earnings_growth,
                    forward_eps        = EXCLUDED.forward_eps,
                    target_low_price   = EXCLUDED.target_low_price,
                    target_high_price  = EXCLUDED.target_high_price,
                    book_value_per_share = EXCLUDED.book_value_per_share,
                    return_on_equity     = EXCLUDED.return_on_equity,
                    net_profit_margin    = EXCLUDED.net_profit_margin,
                    sector               = EXCLUDED.sector,
                    industry             = EXCLUDED.industry
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
                    metrics.get("shares_outstanding"),
                    metrics.get("target_mean_price"),
                    metrics.get("free_cash_flow"),
                    metrics.get("net_cash"),
                    metrics.get("beta"),
                    metrics.get("earnings_growth"),
                    metrics.get("forward_eps"),
                    metrics.get("target_low_price"),
                    metrics.get("target_high_price"),
                    metrics.get("book_value_per_share"),
                    metrics.get("return_on_equity"),
                    metrics.get("net_profit_margin"),
                    metrics.get("sector"),
                    metrics.get("industry"),
                ),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"upsert_fundamentals({ticker}) failed: {e}")
    finally:
        _put_conn(conn)


def log_fundamentals_history(ticker: str, metrics: dict) -> None:
    """Append-only nightly snapshot -- see migration 44's comment. Called
    alongside upsert_fundamentals (which only keeps the latest row) so a
    real time series starts accumulating now; failures here are logged,
    not raised, since losing one night's history snapshot should never
    block the (more important) latest-snapshot upsert."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO fundamentals_history "
                "(ticker, price_to_sales, ev_to_ebitda, forward_pe, revenue_growth_yoy, "
                "fcf_margin, net_profit_margin, return_on_equity) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    ticker, metrics.get("price_to_sales"), metrics.get("ev_to_ebitda"),
                    metrics.get("forward_pe"), metrics.get("revenue_growth_yoy"),
                    metrics.get("fcf_margin"), metrics.get("net_profit_margin"),
                    metrics.get("return_on_equity"),
                ),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"log_fundamentals_history({ticker}) failed: {e}")
    finally:
        _put_conn(conn)


def get_fundamentals_history(ticker: str, limit: int = 400) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM fundamentals_history WHERE ticker=%s ORDER BY recorded_at DESC LIMIT %s",
                (ticker, limit),
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logging.error(f"get_fundamentals_history({ticker}) failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_fundamentals(ticker: str) -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, updated_at, revenue_growth_yoy, gross_margin, fcf_margin, "
                "price_to_sales, ev_to_ebitda, forward_pe, market_cap, "
                "shares_outstanding, target_mean_price, free_cash_flow, net_cash, "
                "beta, earnings_growth, forward_eps, target_low_price, target_high_price, "
                "book_value_per_share, return_on_equity, net_profit_margin, sector, industry "
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
            "shares_outstanding": float(row[9]) if row[9] is not None else None,
            "target_mean_price":  float(row[10]) if row[10] is not None else None,
            "free_cash_flow":     float(row[11]) if row[11] is not None else None,
            "net_cash":           float(row[12]) if row[12] is not None else None,
            "beta":               float(row[13]) if row[13] is not None else None,
            "earnings_growth":    float(row[14]) if row[14] is not None else None,
            "forward_eps":        float(row[15]) if row[15] is not None else None,
            "target_low_price":   float(row[16]) if row[16] is not None else None,
            "target_high_price":  float(row[17]) if row[17] is not None else None,
            "book_value_per_share": float(row[18]) if row[18] is not None else None,
            "return_on_equity":     float(row[19]) if row[19] is not None else None,
            "net_profit_margin":    float(row[20]) if row[20] is not None else None,
            "sector":               row[21],
            "industry":             row[22],
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
                "price_to_sales, ev_to_ebitda, forward_pe, market_cap, "
                "shares_outstanding, target_mean_price, free_cash_flow, net_cash, "
                "beta, earnings_growth, forward_eps, target_low_price, target_high_price, "
                "book_value_per_share, return_on_equity, net_profit_margin, sector, industry "
                "FROM fundamentals"
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
                "shares_outstanding": float(r[8]) if r[8] is not None else None,
                "target_mean_price":  float(r[9]) if r[9] is not None else None,
                "free_cash_flow":     float(r[10]) if r[10] is not None else None,
                "net_cash":           float(r[11]) if r[11] is not None else None,
                "beta":               float(r[12]) if r[12] is not None else None,
                "earnings_growth":    float(r[13]) if r[13] is not None else None,
                "forward_eps":        float(r[14]) if r[14] is not None else None,
                "target_low_price":   float(r[15]) if r[15] is not None else None,
                "target_high_price":  float(r[16]) if r[16] is not None else None,
                "book_value_per_share": float(r[17]) if r[17] is not None else None,
                "return_on_equity":     float(r[18]) if r[18] is not None else None,
                "net_profit_margin":    float(r[19]) if r[19] is not None else None,
                "sector":               r[20],
                "industry":             r[21],
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_fundamentals_all failed: {e}")
        return []
    finally:
        _put_conn(conn)


def upsert_sp500_fundamentals(ticker: str, metrics: dict) -> None:
    """Lightweight companion to upsert_fundamentals -- only the fields the
    broadened Comps peer search needs (see ingestion.fundamentals.
    refresh_sp500_fundamentals), not the full valuation-ensemble depth
    the watchlist's own fundamentals table carries."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sp500_fundamentals
                    (ticker, updated_at, sector, industry, price_to_sales,
                     ev_to_ebitda, revenue_growth_yoy, gross_margin, fcf_margin)
                VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    updated_at         = NOW(),
                    sector             = EXCLUDED.sector,
                    industry           = EXCLUDED.industry,
                    price_to_sales     = EXCLUDED.price_to_sales,
                    ev_to_ebitda       = EXCLUDED.ev_to_ebitda,
                    revenue_growth_yoy = EXCLUDED.revenue_growth_yoy,
                    gross_margin       = EXCLUDED.gross_margin,
                    fcf_margin         = EXCLUDED.fcf_margin
                """,
                (
                    ticker, metrics.get("sector"), metrics.get("industry"),
                    metrics.get("price_to_sales"), metrics.get("ev_to_ebitda"),
                    metrics.get("revenue_growth_yoy"), metrics.get("gross_margin"),
                    metrics.get("fcf_margin"),
                ),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"upsert_sp500_fundamentals({ticker}) failed: {e}")
    finally:
        _put_conn(conn)


_SP500_FUNDAMENTALS_DECIMAL_FIELDS = (
    "price_to_sales", "ev_to_ebitda", "revenue_growth_yoy", "gross_margin", "fcf_margin",
)


def _normalize_sp500_fundamentals(row: dict) -> dict:
    """Same Decimal-vs-float fix as _normalize_trade_ladder -- found
    2026-09-13 in the same session: get_industry_peer_median mixes these
    rows with get_fundamentals_all's (already float-cast) rows in one
    sorted list, and Decimal/float arithmetic between them raises
    TypeError even though both print identically."""
    for field in _SP500_FUNDAMENTALS_DECIMAL_FIELDS:
        if row.get(field) is not None:
            row[field] = float(row[field])
    return row


def get_sp500_fundamentals_all() -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM sp500_fundamentals")
            return [_normalize_sp500_fundamentals(dict(r)) for r in cur.fetchall()]
    except Exception as e:
        logging.error(f"get_sp500_fundamentals_all failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_sp500_fundamentals_by_industry(industry: str) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM sp500_fundamentals WHERE industry=%s", (industry,))
            return [_normalize_sp500_fundamentals(dict(r)) for r in cur.fetchall()]
    except Exception as e:
        logging.error(f"get_sp500_fundamentals_by_industry({industry}) failed: {e}")
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
            # One outcome row per approval: 07-29 logged two rows seconds apart
            # for approvals 62, 67 and 71 (SEF data QA, 2026-10-27).
            cur.execute(
                "SELECT 1 FROM trade_outcomes WHERE approval_id=%s AND trading_env=%s LIMIT 1",
                (approval_id, TRADING_ENV),
            )
            if cur.fetchone():
                logging.warning(f"log_trade_outcome: approval {approval_id} already has an "
                                f"outcome row — not adding another")
                conn.rollback()
                return
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
    exists for this ticker (bought before this feature shipped), fills
    its MOST RECENT row's catalyst columns only if they were never set at
    entry (marked in notes), rather than creating a
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
                # Only rows with no entry-time snapshot get filled, and they're
                # marked in notes. Overwriting a real entry-time snapshot with
                # today's view was look-ahead (professor C6, 2026-10-28).
                cur.execute(
                    "UPDATE trade_outcomes SET catalyst_event=%s, catalyst_date=%s, "
                    "catalyst_is_priced_in=%s, expected_return_pct=%s, expected_downside_pct=%s, "
                    "thesis_summary=%s, sources_json=%s, source_count=%s, "
                    "weeks_to_catalyst=%s, expected_return_per_week=%s, catalyst_backfilled=TRUE, "
                    "notes=CONCAT_WS('; ', notes, %s) "
                    "WHERE id=%s AND catalyst_event IS NULL AND expected_return_pct IS NULL "
                    "AND thesis_summary IS NULL",
                    (catalyst_event, catalyst_date, catalyst_is_priced_in,
                     expected_return_pct, expected_downside_pct, thesis_summary,
                     sources_json, source_count, weeks_to_catalyst, expected_return_per_week,
                     f"catalyst fields backfilled {datetime.now(timezone.utc).date()}", row[0]),
                )
                if cur.rowcount == 0:
                    logging.info(f"backfill_catalyst_outcome({ticker}): row {row[0]} already has an "
                                 f"entry-time snapshot — left unchanged")
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
    Closes every still-open (exit_date IS NULL) trade_outcomes row for this
    ticker: computes outcome_pct, was_correct, and alpha vs. SPY over each
    row's holding window. Called after a confirmed full exit.
    Returns True if any row was closed.
    """
    from datetime import datetime, timezone as _tz

    exit_date = exit_date or datetime.now(_tz.utc)

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            # Every open row for the ticker: the position is gone, so an
            # OPEN followed by ADDs must close all of its rows, not just the
            # newest (professor C5, 2026-10-28).
            cur.execute(
                "SELECT id, entry_price, entry_benchmark_price FROM trade_outcomes "
                "WHERE ticker=%s AND trading_env=%s AND exit_date IS NULL "
                "ORDER BY entry_date DESC",
                (ticker, TRADING_ENV),
            )
            open_rows = cur.fetchall()
            if not open_rows:
                logging.warning(f"close_trade_outcome: no open outcome row for {ticker}")
                return False

            exit_benchmark_price = None
            try:
                import yfinance as yf
                spy_price = yf.Ticker("SPY").fast_info.last_price
                if spy_price:
                    exit_benchmark_price = float(spy_price)
            except Exception as e:
                logging.debug(f"close_trade_outcome: SPY exit price fetch failed: {e}")

            for outcome_id, entry_price, entry_benchmark_price in open_rows:
                entry_price = float(entry_price)
                entry_benchmark_price = float(entry_benchmark_price) if entry_benchmark_price else None
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


# Rows the test suite wrote into production before 9eb81fd (professor C9):
# fake NVDA EXITs (approval 1, entry $100) and real NVDA rows closed at the
# fake $100. Kept out of every outcome aggregate that feeds the analyst's
# track record or the stats. A no-op once scripts/repair_test_artifacts.py
# --apply has run; remove it then.
_NOT_TEST_ARTIFACT = (
    "NOT (action='EXIT' AND ticker='NVDA' AND COALESCE(approval_id, 0)=1 AND entry_price=100) "
    "AND NOT (ticker='NVDA' AND COALESCE(exit_price, -1)=100)"
)


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
                f"WHERE outcome_pct IS NOT NULL AND trading_env=%s AND {_NOT_TEST_ARTIFACT} "
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
                f"AND {_NOT_TEST_ARTIFACT} "
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
                f"WHERE outcome_pct IS NOT NULL AND trading_env=%s AND {_NOT_TEST_ARTIFACT} "
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
                  AND (NOT (o.action='EXIT' AND o.ticker='NVDA' AND COALESCE(o.approval_id, 0)=1 AND o.entry_price=100) AND NOT (o.ticker='NVDA' AND COALESCE(o.exit_price, -1)=100))
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


def get_category_attribution() -> list[dict]:
    """Same approximation as get_bucket_attribution, grouped by the new
    setup_category (Breakout/Catalyst/Continuation) instead of the legacy
    thesis_bucket -- this is the number that actually tells you whether the
    setup-category reframe (2026-09-13) is working, once enough trades have
    closed under it to be meaningful."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    o.setup_category,
                    COUNT(*) AS closed_trades,
                    SUM(CASE WHEN o.was_correct THEN 1 ELSE 0 END) AS wins,
                    AVG(o.alpha_pct) AS avg_alpha_pct,
                    SUM((pa.risk_json->>'initial_dollar')::DECIMAL * o.outcome_pct) AS dollar_pnl,
                    SUM((pa.risk_json->>'initial_dollar')::DECIMAL) AS dollar_deployed
                FROM trade_outcomes o
                JOIN pending_approvals pa ON pa.id = o.approval_id
                WHERE o.outcome_pct IS NOT NULL
                  AND pa.risk_json ? 'initial_dollar'
                  AND o.trading_env = %s
                  AND (NOT (o.action='EXIT' AND o.ticker='NVDA' AND COALESCE(o.approval_id, 0)=1 AND o.entry_price=100) AND NOT (o.ticker='NVDA' AND COALESCE(o.exit_price, -1)=100))
                  AND o.setup_category IS NOT NULL
                GROUP BY o.setup_category
                ORDER BY dollar_pnl DESC NULLS LAST
                """,
                (TRADING_ENV,),
            )
            rows = cur.fetchall()
        return [
            {
                "setup_category": r[0], "closed_trades": r[1], "wins": r[2],
                "win_rate": round(r[2] / r[1], 3) if r[1] else None,
                "avg_alpha_pct": float(r[3]) if r[3] is not None else None,
                "dollar_pnl": round(float(r[4]), 2) if r[4] is not None else 0.0,
                "dollar_deployed": round(float(r[5]), 2) if r[5] is not None else 0.0,
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"get_category_attribution failed: {e}")
        return []
    finally:
        _put_conn(conn)


def get_portfolio_heat() -> dict:
    """Total remaining dollar risk across every open position, as a % of
    equity -- NOT position count, NOT capital deployed. A position whose
    trailing stop has moved to/above breakeven contributes zero heat (it
    can no longer lose money, only give back profit), so this differs
    materially from a simple 'positions held' count. Single source of
    truth for the dashboard/digest/Telegram, which must never each
    independently recompute this and risk showing three different numbers.
    Equity is passed in by the caller (risk_manager already knows it from
    the broker) rather than fetched here, keeping this a pure DB+math
    function with no broker dependency of its own.
    """
    stops = get_all_position_stops()
    by_category: dict[str, float] = {}
    total_heat_dollars = 0.0
    for s in stops:
        entry, stop = s.get("entry_price"), s.get("stop_price")
        if entry is None or stop is None:
            continue
        # qty isn't stored on position_stops -- caller must multiply by
        # live share count if per-position dollar heat (not just the
        # per-share risk) is needed; get_portfolio_heat_pct(equity, positions)
        # below does that using broker-supplied quantities.
        per_share_risk = max(0.0, float(entry) - float(stop))
        cat = s.get("setup_category") or s.get("thesis_bucket") or "Unclassified"
        by_category[cat] = by_category.get(cat, 0.0) + per_share_risk
    return {"per_share_risk_by_category": by_category, "stops": stops}


def get_portfolio_heat_pct(equity: float, positions: dict[str, dict]) -> dict:
    """
    positions: {ticker: {"qty": float, ...}} from the broker (ibkr.py's
    get_all_positions), joined here against position_stops' entry/stop to
    get real dollar heat per position, not just per-share.
    Returns {"heat_pct": float, "heat_dollars": float,
             "by_category": {category: heat_dollars}, "positions": [...]}.
    """
    stops = get_all_position_stops()
    by_category: dict[str, float] = {}
    detail = []
    total = 0.0
    for s in stops:
        ticker = s["ticker"]
        pos = positions.get(ticker)
        entry, stop = s.get("entry_price"), s.get("stop_price")
        if not pos or entry is None or stop is None:
            continue
        qty = abs(float(pos.get("qty", 0)))
        per_share_risk = max(0.0, float(entry) - float(stop))
        dollar_risk = per_share_risk * qty
        cat = s.get("setup_category") or s.get("thesis_bucket") or "Unclassified"
        by_category[cat] = by_category.get(cat, 0.0) + dollar_risk
        total += dollar_risk
        detail.append({"ticker": ticker, "category": cat, "heat_dollars": round(dollar_risk, 2)})
    return {
        "heat_pct": round(total / equity, 4) if equity else 0.0,
        "heat_dollars": round(total, 2),
        "by_category": {k: round(v, 2) for k, v in by_category.items()},
        "positions": detail,
    }


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


def get_earnings_around(tickers: list[str], days_ahead: int = 4, days_back: int = 4) -> dict:
    """{ticker: earnings_date} for tickers with an earnings_calendar row in
    [today - days_back, today + days_ahead], nearest date per ticker. Feeds
    the pre/post-earnings position review (crew.review_all_held_positions).
    4 calendar days covers T-2 and T+1 in trading days across a weekend."""
    if not tickers:
        return {}
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT ON (ticker) ticker, earnings_date FROM earnings_calendar "
                "WHERE ticker = ANY(%s) "
                "AND earnings_date BETWEEN CURRENT_DATE - %s AND CURRENT_DATE + %s "
                "ORDER BY ticker, ABS(earnings_date - CURRENT_DATE), earnings_date",
                (list(tickers), days_back, days_ahead),
            )
            return {r[0]: r[1] for r in cur.fetchall()}
    except Exception as e:
        conn.rollback()
        logging.error(f"get_earnings_around failed: {e}")
        return {}
    finally:
        _put_conn(conn)


def is_near_earnings(ticker: str, days: int = 5, strict: bool = False) -> bool | None:
    """Returns True if ticker has an earnings date within `days` calendar days.
    On a DB error returns False, or None with strict=True so the earnings
    blackout can fail closed (limitations audit 2026-10-28, finding 11)."""
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
        return None if strict else False
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
    setup_category: str | None = None,
    setup_triggered: bool | None = None,
    setup_gates_detail: dict | None = None,
) -> None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO scanner_candidates "
                "(ticker, scan_type, composite_score, technical_score, momentum_score, "
                "catalyst_score, thesis_alignment_score, priority, catalyst_detail, "
                "passed_to_analyst, regime_at_scan, setup_category, setup_triggered, setup_gates_detail) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    ticker, scan_type, composite_score, technical_score, momentum_score,
                    catalyst_score, thesis_alignment_score, priority, catalyst_detail,
                    passed_to_analyst, regime_at_scan, setup_category, setup_triggered,
                    psycopg2.extras.Json(setup_gates_detail) if setup_gates_detail is not None else None,
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
