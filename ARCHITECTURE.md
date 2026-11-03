# Helios architecture

This document describes how the Helios trading bot is built and how it runs, as the code in this repository stands. Every claim cites a file and line (`file.py:123`) or a function (`file.py:function`). Where a comment in the code disagrees with what the code does, this document follows the code and says so in "Known inconsistencies" at the end.

Helios is a single-machine system. It reads market and company events, has a local LLM judge them, sizes trades with fixed rules, and places orders with Interactive Brokers (IBKR). One Python entry point (`main.py`) starts a set of Celery workers, a Celery beat scheduler, a Streamlit dashboard, a Telegram bot and a small FastAPI catalyst service. State lives in PostgreSQL, Redis (Celery broker and locks) and an embedded ChromaDB vector store.

## Contents

1. Processes and systemd units
2. Celery queues and workers
3. Beat schedule (times in ET)
4. Data sources and failure modes
5. Database tables
6. Decision flow: event to order
7. LLM call sites
8. Stops and exits
9. Halts and kill switches
10. Logging and provenance
11. Known inconsistencies and open doubts

---

## 1. Processes and systemd units

### 1.1 Daily lifecycle

The machine runs a weekday schedule driven by systemd timers, all on `America/New_York`:

| Time (ET) | What happens | Source |
|---|---|---|
| Mon-Fri 05:00 | `helios-start.timer` starts `helios-local.service` (WakeSystem, Persistent) | `deploy/systemd/helios-start.timer:17,21,24-25` |
| 05:00-18:55, every 10 min | `helios-watchdog.timer` runs `deploy/watchdog.py --heal` | `deploy/systemd/helios-watchdog.timer:13` |
| 05:00-18:55, every 5 min, 7 days | `helios-gateway-watchdog.timer` runs `deploy/watchdog.py --gateway-only` | `deploy/systemd/helios-gateway-watchdog.timer:18` |
| Daily 19:00 | `helios-evening-stop` stops helios-local, ibc, xvfb, then starts `helios-standby` | `deploy/systemd/helios-evening-stop.timer:13`, `helios-evening-stop.service:14-17` |
| Daily 19:15 | user unit `helios-backup` runs `scripts/backup_db.py` | `deploy/systemd/user/helios-backup.timer:5-6` |

So the full stack is up roughly 05:00 to 19:00 ET on weekdays and down overnight and on weekends, when only the standby Telegram listener runs.

### 1.2 systemd units in `deploy/systemd/`

The unit files use `YOUR_USER` as a placeholder (`deploy/systemd/README.md:9-12`).

| Unit | Runs | Restart and ordering |
|---|---|---|
| `helios-local.service` (production) | Pre-start `-wait-for-ibkr.sh 180` (the leading `-` makes a timeout non-fatal), then `venv/bin/python main.py` (`:48-49`) | `Restart=on-failure`, 30 s (`:50-51`); StartLimit 10 per 1800 s (`:25-26`); `OnFailure=helios-notify-failure@%n` (`:17`); `Conflicts=helios-standby.service` (`:32`); deliberately no `Requires=` on ibc (`:3-10`); `KillMode=mixed`, `TimeoutStopSec=480` so main.py can warm-stop workers (`:57-58`) |
| `helios.service` | Generic `/opt/helios` template of the same thing (`:13`) | on-failure, 10 s. Not the production unit |
| `helios-standby.service` | `python standby_bot.py` (`:16`) | on-failure, 10 s; `Conflicts=helios-local.service` (`:10`) |
| `helios-evening-stop.service` | oneshot: stop helios-local, ibc, xvfb; start standby (`:14-17`) | runs as root |
| `helios-watchdog.service` | `deploy/watchdog.py --heal` (`:8`) | oneshot |
| `helios-gateway-watchdog.service` | `deploy/watchdog.py --gateway-only` (`:8`) | oneshot |
| `helios-notify-failure@.service` | `deploy/notify_failure.py %i` (`:8`): Telegram message naming the failed unit | oneshot, used by `OnFailure=` |
| `ibc.service` | IBC-managed TWS paper gateway on `DISPLAY=:99` (`:20,36`), `PrivateDevices=true` to keep TWS off the GPUs (`:35`) | `Restart=always`, 30 s, because TWS's nightly self-restart exits 0 (`:46-47`); `Requires=xvfb` (`:4-6`) |
| `ollama-cpu.service` | A second, CPU-only `ollama serve` on 127.0.0.1:11435, `NUM_PARALLEL=2` (`:13-31`) | `Restart=always`, 3 s |
| `disable-usb-wake.service` | Turns off ACPI wake for USB controllers (`disable-usb-wake.sh`) | oneshot before sleep |

Helper scripts: `wait-for-ibkr.sh` polls the IBKR API port with bash `/dev/tcp` every 3 s until a timeout (default 180 s) (`deploy/systemd/wait-for-ibkr.sh:50-62`). `install-auto-schedule.sh` installs xvfb/ibc, helios-local, evening-stop and the two start/stop timers only (`:12-23`). `ibkr-access.sh` is a manual helper for gateway status and VNC over SSH (`:30-41`).

### 1.3 `main.py`: the supervisor

`main.py:main` (`main.py:496`) is the entry point for both systemd units and `launch.sh`. In order it:

1. Installs log redaction and a rotating log handler (`main.py:23-44`).
2. Starts Redis if it is local and not running: systemctl, then `redis-server`, then Docker (`main.py:_try_start_redis`, `:47-109`).
3. Runs `db.init_db()` (`main.py:509-510`), which creates tables and applies migrations (section 5).
4. Checks the main Ollama server on port 11434, warning only (`main.py:_check_ollama`, `:281-295`).
5. Warms the market regime cache (`main.py:_init_regime`, `:298-305`).
6. Starts the Telegram poller and worker threads (`main.py:308-312`, `telegram_bot.py:598-608`).
7. Starts the Streamlit dashboard as a subprocess on port 8501 with `IB_CLIENT_ID=2` (`main.py:315-334`).
8. In a background thread, `_auto_start_services` (`main.py:466-493`) starts the orders worker first, runs a bounded startup reconcile, then starts all other services.

The startup reconcile waits up to 180 s for the orders worker to answer a Celery ping (`main.py:337-353`). It then adopts IBKR long positions that have no DB stop record, guessing and placing a stop through the orders worker (`main.py:_reconcile_positions`, `:163-228`; shorts are never adopted, `:119-121`), and cancels stale market BUY orders under the Redis lock `helios:task:reconcile_unconfirmed_orders` (`main.py:_reconcile_open_orders`, `:231-278`, `:381-413`). The whole reconcile is capped at 300 s; past that, services start anyway and an alert is sent (`main.py:416-463`).

After startup the main loop only sleeps (`main.py:566-568`). On SIGINT/SIGTERM it stops all services, Telegram polling and the dashboard (`main.py:535-556`). main.py itself connects to IBKR with clientId 1 in paper (`config.py:79`).

### 1.4 `process_manager.py`: child processes

`process_manager.SERVICES` (`process_manager.py:45-140`) defines seven children, tracked by PID and create time in `.processes.json` (`process_manager.py:12`, `:325-326`):

| Service | Command | IBKR clientId |
|---|---|---|
| `celery_worker` (ops) | `celery -A tasks worker -Q ops -n ops@%h` (`:51-52`) | 10, 11, 12 (base 10 plus fork index) |
| `celery_ingestion_worker` | `-Q ingestion` (`:68-69`) | none, never touches IBKR |
| `celery_llm_worker` | `-Q llm_p0,llm_p1,llm_p2 --prefetch-multiplier=1` (`:85-87`) | 20 |
| `celery_screen_worker` | `-Q screen` (`:96-97`) | none |
| `celery_orders_worker` | `-Q orders` (`:126-127`) | 30 |
| `celery_beat` | `celery -A tasks beat` (`:132`) | none |
| `catalyst_service` | `uvicorn catalyst_service.main:app --host 127.0.0.1 --port 8001` (`:137`) | none |

ClientIds are assigned in `process_manager.py:280-309`; `ibkr.py:_resolve_client_id` (`ibkr.py:61`) adds the prefork child index so the ops worker's three forks do not collide. Each worker gets `DB_MAXCONN=2`.

process_manager does not respawn dead children by itself. Recovery comes from the dashboard Start buttons, the watchdog restarting helios-local, or systemd restarting main.py. Liveness uses psutil with a create-time check against PID reuse (`process_manager.py:_alive`, `:205-225`). Stopping sends SIGTERM to each Celery parent for a warm shutdown, with 420 s grace for the LLM worker and 60 s for others (`:332-368`); `stop_all` stops beat first and the orders worker last (`:415-441`). Logs over 20 MB are rotated at start (`:143-165`).

### 1.5 Other processes

- **`launch.sh`**: manual launcher. Activates the venv, starts Redis and `ollama serve` if needed, warns if port 7497 is closed, runs `python main.py` (`launch.sh:8-73`).
- **`standby_bot.py`**: overnight and weekend Telegram listener started by evening-stop. It deliberately imports no IBKR, Celery or dashboard code (`standby_bot.py:18-22`). It answers allowlisted mentions and DMs through `chat_assistant.respond` using cached data (`:89-115`), and if helios-local is not active it runs `sudo -n systemctl start helios-local.service` (`:59-76`, `:117-123`); `Conflicts=` then stops standby. It has no trading commands.
- **`dashboard.py`**: Streamlit app on port 8501 (`dashboard.py:11,23`). Twelve tabs (`:287-289`) covering workflow, activity feed, portfolio, regime, pending approvals (read-only), outcomes, performance, macro, technicals, signal log, manual trade and logs. Its only write actions are service Start/Stop buttons (`:218-254`) and the Manual Trade form, a DAY market order behind a confirmation checkbox via `execution.execute_manual` (`:1149-1163`). It has no halt or flatten button.
- **`telegram_bot.py`**: runs as two daemon threads inside main.py, polling every 2 s (`telegram_bot.py:38`). Only messages that mention the bot are queued (`:556-561`). Both callbacks and mentions fail closed when `TELEGRAM_ALLOWED_USER_IDS` is empty (`:173-181`, `:394-396`). It handles Approve/Reject buttons with a 6 h expiry and an atomic claim, then `execution.execute_approved` (`:211-244`); `/halt`, `/resume`, `/rebase` (section 9); and `!urgent buy|sell [qty] TICKER` through `execution.execute_manual` (`:300-370`). Other text goes to `chat_assistant.respond`.
- **`catalyst_service/`**: FastAPI "Catalyst Calendar Service" backed by SQLite `catalyst_calendar.db` (`catalyst_service/storage.py:17`), with `POST /catalysts`, `GET /catalysts`, `GET /catalysts/conflicts` (`catalyst_service/main.py:13-82`). Callers use HTTP only when `CATALYST_SERVICE_URL` is set and otherwise read SQLite directly (`catalyst_service/client.py:22-33`). Writers import storage directly (`ingestion/earnings_calendar.py:72`, `ingestion/macro_calendar.py:151`, `ingestion/sector_calendar.py:54`). The gate used in the decision flow is `catalyst_service/gates.py:catalyst_gate_check`.

### 1.6 Watchdog

`deploy/watchdog.py` is standalone and posts directly to the Telegram API (`deploy/watchdog.py:84-95`).

- `--gateway-only`: if a gateway unit is active but the IBKR port does not accept a TCP connection, it alerts that login is probably stuck on 2FA (`:202-232`).
- Full check (`:430-464`): skipped while helios-local is in a transitional state or while `logs/.planned_restart` is under 600 s old (`:180-199`). It checks that helios-local is active, that `celery_worker`, `celery_llm_worker` and `celery_beat` are alive (`:45`, `:125-134`), and that the newest `portfolio_snapshots` row is under 15 minutes old (`:49`, `:235-270`). The orders, screen and ingestion workers are not in the critical list.
- Alerts are throttled to one per 30 minutes per key, stored in `system_state` (`:98-122`).
- `--heal` restarts helios-local (`sudo -n systemctl reset-failed` then `start`) only on total failure: unit inactive or all three critical processes dead (`:342-396`). Limits: 15-minute cooldown and at most 4 restarts per rolling 24 h, recorded in `logs/.watchdog_heal_history.json` (`:62-81`, `:273-320`).

---

## 2. Celery queues and workers

### 2.1 App configuration

`tasks.py:34-70`: broker and result backend are both `REDIS_URL` (default `redis://localhost:6379/0`, `tasks.py:32`). JSON serialization, `timezone="America/New_York"`, `enable_utc=True`, `task_track_started=True`, `worker_prefetch_multiplier=1`, `task_default_queue="ops"`. Broker transport options set `queue_order_strategy="sorted"` (a worker on several queues drains them in name order) and `visibility_timeout` of 6 days so ETA tasks scheduled over a weekend are not redelivered (`tasks.py:66-69`).

There is no `task_routes` table. Routing is done per call with `apply_async(queue=...)` or per beat entry with `options.queue`. Anything without an explicit queue goes to `ops`.

### 2.2 Queues

| Queue | Worker, pool and concurrency | What goes there | Source |
|---|---|---|---|
| `ops` | `celery_worker`, prefork `-c 3` | default: snapshots, stop checks, profit targets, drawdown, reconciles, expiry, health checks, daily scan, digest, deferred-entry retries | `tasks.py:48`, `process_manager.py:25` |
| `ingestion` | `celery_ingestion_worker`, prefork `-c 4` | all external pollers and nightly refreshes | `tasks.py:98`, `process_manager.py:40` |
| `screen` | `celery_screen_worker`, prefork `-c 3` | `tasks.screen_event` (7B screen), expires after 2 h | `tasks.py:110-111`, `:876-878`, `process_manager.py:35` |
| `llm_p0` | `celery_llm_worker`, prefork `-c 1` | 72B analysis of events on held tickers, regime task, position review, plan-level reviews | `tasks.py:73`, `:902-906` |
| `llm_p1` | same worker | 72B analysis of non-held events and scanner candidates, weekly self-review | `tasks.py:74`, `:2350-2353` |
| `llm_p2` | same worker | chat-requested research (`research_ticker_on_demand`), never routed to execution | `tasks.py:79`, `:973-981` |
| `orders` | `celery_orders_worker`, prefork `-c 1` | the only process allowed to place or cancel stop orders | `tasks.py:153`, `:215-250`, `process_manager.py:29` |

Because one LLM worker drains `llm_p0`, `llm_p1`, `llm_p2` with sorted ordering, held-ticker analyses always run before new ideas, and chat research runs last (`tasks.py:49-51`, `:72-79`). LLM events expire after 4 h (`tasks.py:114`).

### 2.3 The orders funnel

IBKR ignores a cancel or modify from a clientId that did not place the order, so every stop change goes through one process with clientId 30 (`tasks.py:116-139`). Callers use `tasks.manage_stop_place`, `manage_cancel_stop_orders_for`, `manage_place_trailing_stop` and `manage_cancel_order` (`tasks.py:329-390`). Inside the orders worker these call `ibkr.py` directly; elsewhere they dispatch to the `orders` queue and wait via `tasks._orders_call` (`tasks.py:259-289`). Each request carries a deadline; the worker refuses requests that arrive late (`OrdersCallExpired`), and a missing reply after an extra 120 s wait raises `OrdersCallUnknown` (`tasks.py:164-178`, `:256`). A Redis "started" marker distinguishes "never ran" from "lost reply" (`tasks.py:_mark_started`).

### 2.4 Locks

- **Per-ticker stop lock**: `stop_lock.ticker_lock` / `@locked_by_ticker` (section 8.5).
- **Single-run task locks**: `reconcile_stop_orders` takes `helios:task:<name>` with `blocking_timeout=0`, so overlapping runs skip (`tasks.py:4032-4033`).
- **Manual-actions and reprotect queue locks**: `helios:manual_actions` and per-key `helios:<key>` Redis locks (`tasks.py:1602`, `:1680`).
- **Event dedupe**: pollers mark items in `seen_items` after the screen task is enqueued (`tasks.py:_dispatch_event`, `:856-888`).

---

## 3. Beat schedule (times in ET)

Celery's `timezone` is `America/New_York` with `enable_utc=True` (`tasks.py:40-41`). Celery interprets `crontab()` fields in the configured timezone, so the crontab hours below are already Eastern time and need no conversion. Interval entries (seconds) are not tied to the clock: they fire whenever beat is running, which in practice is 05:00-19:00 ET on weekdays (section 1.1). Comments saying "during market hours" are not enforced by beat; a few tasks check market hours themselves (noted below). The schedule is at `tasks.py:401-735`; `poll-congressional` is removed when `QUIVER_ENABLED` is false (`tasks.py:736-738`, default false at `config.py:343`).

### 3.1 Clock-based entries

| Entry | Task | When (ET) | Queue | Purpose |
|---|---|---|---|---|
| `refresh-sp500-history-catchup` | `refresh_sp500_history_catchup` | Mon-Fri 05:15 | ingestion | Refresh S&P 500 bars only for tickers over 7 days stale |
| `refresh-earnings` | `refresh_earnings_calendar` | Mon-Fri 08:00 | ingestion | Earnings dates into catalyst DB and `earnings_calendar` |
| `refresh-macro-calendar` | `refresh_macro_calendar` | Mon 08:05 | ingestion | FOMC, CPI, jobs, election dates |
| `alert-catalyst-conflicts` | `alert_catalyst_conflicts` | Mon-Fri 08:10 | ops | Telegram on conflicting catalyst entries |
| `refresh-sector-calendar` | `refresh_sector_calendar` | Mon 08:15 | ingestion | Sector earnings clusters |
| `ipo-watch` | `check_ipo_watch` | Mon-Fri 08:20 | ingestion | Checks whether watched private companies have listed |
| `candidate-scan` | `run_candidate_scan` | Mon 09:00 | ops | Weekly watchlist-candidate discovery |
| `poll-arxiv` | `poll_arxiv` | daily 09:00 | ingestion | arXiv papers |
| `assess-regime` | `assess_regime_daily` | Mon-Fri 09:30 | llm_p0 | Market regime packet (sizing multiplier) |
| `daily-scan` | `run_daily_scan` | Mon-Fri 09:35 | ops | Watchlist scan; enqueues `analyze_candidates` on llm_p1 |
| `assess-gics-sector-map` | `assess_gics_sector_map` | Mon-Fri 09:45 | ops | Price/volume read of 11 sector ETFs |
| `position-review` | `review_positions` | Mon-Fri 10:15 | llm_p0 | Re-analyse held positions for TRIM/EXIT/HOLD |
| `flag-orphan-outcomes` | `flag_orphan_outcomes` | Mon-Fri 15:45 | ingestion | Flags open outcome rows with no position |
| `daily-digest` | `send_daily_digest` | Mon-Fri 16:15 | ops | End-of-day Telegram digest |
| `review-outcomes` | `review_trade_outcomes` | Mon-Fri 16:30 | ops | 30/60/90-day outcome price reviews |
| `refresh-price-history` | `refresh_price_history` | Mon-Fri 16:45 | ingestion | Watchlist daily bars cache |
| `refresh-fundamentals` | `refresh_fundamentals` | Mon-Fri 17:00 | ingestion | Fundamentals table |
| `refresh-trade-ladder` | `refresh_trade_ladder` | Mon-Fri 17:15 | ingestion | Nightly price-level ladder (`agents/trade_ladder.py`) |
| `prune-seen-items` | `prune_seen_items` | Fri 17:50 | ops | Prune dedupe table |
| `weekly-self-review` | `weekly_self_review` | Fri 18:00 | llm_p1 | LLM-written weekly performance review |
| `prune-vector-store` | `prune_vector_store` | Sun 03:00 (runs via Monday catch-up) | ops | Prune ChromaDB |
| `refresh-sp500-history` | `refresh_sp500_history` | Sat 09:00 (runs via Monday catch-up) | ingestion | S&P 500 universe bars |
| `refresh-sp500-fundamentals` | `refresh_sp500_fundamentals` | Sat 09:30 (runs via Monday catch-up) | ingestion | Peer universe fundamentals |

### 3.2 Interval entries

| Entry | Task | Every | Queue | Notes |
|---|---|---|---|---|
| `portfolio-snapshot` | `take_portfolio_snapshot` | 5 min | ops | Positions, cash, SPY/QQQ benchmark (`tasks.py:3641`) |
| `reconcile-unconfirmed-orders` | `reconcile_unconfirmed_orders` | 5 min | ops | Expires after 3600 s on purpose (`tasks.py:571-598`) |
| `poll-options` | `poll_options_flow` | 5 min | ingestion | Returns early outside market hours (`tasks.py:2963-2966`) |
| `poll-edgar` | `poll_edgar` | 10 min | ingestion | SEC filings |
| `reconcile-stops` | `reconcile_stop_orders` | 15 min | ops | Broker vs DB stop reconciliation |
| `watch-plan-levels` | `watch_plan_levels` | 15 min | ingestion | Log-only shadow plan watcher |
| `poll-news` | `poll_news` | 15 min | ingestion | yfinance and Google News RSS |
| `check-stops` | `check_trailing_stops` | 30 min | ops | Stop breach and trailing logic |
| `check-profit-targets` | `check_profit_targets` | 30 min | ops | Partial profit-taking |
| `check-drawdown` | `check_portfolio_drawdown` | 30 min | ops | Drawdown circuit breaker |
| `expire-approvals` | `auto_expire_approvals` | 30 min | ops | Expire pending approvals |
| `check-ollama-gpu` | `check_ollama_gpu_health` | 30 min | ops | Probes the 72B model |
| `poll-insider` | `poll_insider` | 30 min | ingestion | SEC Form 4 |
| `poll-analyst-ratings` | `poll_analyst_ratings` | 30 min | ingestion | yfinance rating changes |
| `poll-macro` | `poll_macro` | 60 min | ingestion | FRED series |
| `check-vector-store` | `check_vector_store_health` | 60 min | ops | ChromaDB integrity check in a subprocess |
| `check-ingestion-health` | `check_ingestion_health` | 60 min | ops | Stale-source alerts |
| `poll-congressional` | `poll_congressional` | 6 h | ingestion | Only if `QUIVER_ENABLED` |

Each entry's `expires` equals its interval except where noted, so a backed-up task is dropped rather than run late (`tasks.py:390-400`).

---

## 4. Data sources and failure modes

### 4.1 Sources

| Source | Module | Used for | When it is down |
|---|---|---|---|
| IBKR TWS/Gateway via `ib_async` | `ibkr.py` | Orders, positions, account values, open/completed orders, executions, historical bars and latest prices | `get_ib` retries 3 times with 3 s and 6 s backoff, then raises `ConnectionError` (`ibkr.py:89-147`). Account-wide reads time out at 30 s and drop the session (`ibkr.py:_bounded_req`, `:158-181`). A clientId collision falls back to `100 + pid % 800` (`ibkr.py:107-115`). Execution refuses and alerts (`execution.py:execute_approved`, IBKR connection check). Analysis skips when positions cannot be read (`agents/crew.py:deep_analyze`, `:126-130`) |
| IBKR historical bars | `market_data.py` | Central price authority with staleness metadata; NYSE hours and holidays 2026-2027 | `get_priced_quote` returns `None`; a 1-min bar older than 20 min is stale (`market_data.py:19`, `:95`). Calendar ends after 2027 (`:31`) |
| Alpaca data API (IEX feed) | `agents/scanner.py:_fetch_bars`, `agents/market_regime.py:_fetch_bars` | Daily bars for the scanner and regime scoring | Returns `[]` if `ALPACA_API_KEY` is unset or the call fails (`agents/scanner.py:31-51`, `agents/market_regime.py:539-573`); scanner disqualifies the ticker with "No price data available" (`agents/scanner.py:443-446`) |
| Polygon REST | `ingestion/polygon_stream.py` (`POLYGON_API_KEY`) | Snapshots and indicators for briefs, dashboard and price fallback | Optional; callers wrap it in `try` (`context_engine.py:402`, `agents/crew.py:1371`) |
| yfinance | `ingestion/news.py`, `analyst_ratings.py`, `fundamentals.py`, `price_history.py`, `options_free.py`, `earnings_calendar.py`, `company_photos.py` | News, ratings, fundamentals, daily bars cache, free options proxy, earnings dates, SPY/QQQ benchmark | Pollers record failures in `ingestion_health`; `check_ingestion_health` alerts when a source exceeds its expected gap (`tasks.py:3371-3409`) |
| SEC EDGAR | `ingestion/edgar.py`, `ingestion/insider.py` | 8-K and other filings, Form 4 | 10-15 s HTTP timeouts; isolated on the ingestion queue so a slow SEC cannot starve ops (`tasks.py:81-98`) |
| Unusual Whales | `ingestion/options_flow.py` (`UNUSUAL_WHALES_API_KEY`) | Options flow | Falls back to `ingestion/options_free.py` when the key starts with `your_` (`tasks.py:2969-2973`) |
| FRED | `ingestion/macro.py` (`FRED_API_KEY`) | Macro series | Hourly poll; brief shows what is cached |
| arXiv | `ingestion/arxiv_watcher.py` | Research papers | Daily |
| QuiverQuant | `ingestion/congressional.py` (`QUIVER_API_KEY`) | Congressional trades | Disabled unless `QUIVER_ENABLED` |
| Fed, BLS pages | `ingestion/macro_calendar.py` | Macro calendar | Weekly idempotent upsert |
| Wikipedia | `ingestion/sp500_universe.py` | S&P 500 constituent list | Weekly |
| Ollama (local) | `agents/analyst.py`, `agents/screener.py`, `chat_assistant.py` | LLM inference (section 7) | Analyst returns a stub (WATCH, no trade); screener uses a keyword fallback; chat returns nothing (section 7) |
| ChromaDB (embedded) | `vector_store.py`, path `.chromadb` (`vector_store.py:25`) | Event memory for briefs; embeddings with `all-MiniLM-L6-v2` on CPU (`:26`, `:106`) | Writes are best-effort (`agents/crew.py:59-72`); hourly integrity check rebuilds corrupted collections (`tasks.py:4524`) |
| Knowledge files | `knowledge/*.md` via `knowledge_loader.py` | Watchlist with role/tier/sector, theses, risk and regime rules | Read from disk |

### 4.2 Price fallback chain

`agents/crew.py:_get_price` (`:1360-1404`) tries, in order: the extended-hours session quote from IBKR (`market_data.get_session_quote`, only outside regular hours), a Polygon snapshot, the IBKR regular-hours quote (`market_data.get_latest_quotes`), then yfinance `info` with a sanity range of 0.01 to 50000. If all fail, routing exits with `ROUTE_NO_PRICE`.

### 4.3 Regime fallback

`agents/market_regime.py:get_cached_regime` (`:1119-1155`) returns the in-memory packet if under `REGIME_CACHE_TTL_SECONDS` (1800 s, `config.py:458`), else the latest `market_regime_log` row if fresh, else runs `assess_regime()` again. Regime scoring is computed from price and volume signals, not from an LLM (no LLM call exists in `agents/market_regime.py`).

### 4.4 What "down" means for trading

| Failure | Effect |
|---|---|
| IBKR unreachable | No analysis (positions unreadable), no execution, snapshots stop; watchdog flags stale snapshots after 15 min. Existing resting stops stay at the broker |
| Redis down | Celery stops entirely. Stop changes run unlocked (`stop_lock.py:12-15`) but the buy room check fails closed (`execution.py:_buy_room_at_execution`, `:450-460`) |
| PostgreSQL down | Halt flag reads fail closed, blocking buys (`risk_manager.py:is_halted`, `:256-266`) |
| 72B Ollama down | Analyses become stubs; two stubs in 20 min trigger an Ollama restart (section 7.3) |
| One ingestion source down | Only that source stops producing events; hourly alert from `check_ingestion_health` |

---

## 5. Database tables

### 5.1 Engine and connection

PostgreSQL through `psycopg2.pool.ThreadedConnectionPool` (`db.py:9-12`, `:294-307`). Connection settings come from `SUPABASE_HOST`, `SUPABASE_PORT`, `SUPABASE_DB`, `SUPABASE_USER`, `SUPABASE_PASSWORD` (`db.py:297-301`); `DB_MAXCONN` defaults to 5 (`db.py:291`). The pool is health-checked at most once a minute (`db.py:_get_pool`). Base tables are in `_CREATE_TABLES` (`db.py:30-246`); later changes are versioned migrations in `_MIGRATIONS` (`db.py:369-815`), tracked in `schema_version` (`db.py:819`) and applied by `init_db` (`db.py:826`). Many tables carry a `trading_env` column so paper and live rows never mix (`db.py:16-26`).

### 5.2 Tables by group

**Decisions and approvals**

| Table | Line | Purpose | Key columns |
|---|---|---|---|
| `signals` | `db.py:42` | Every screen and analyst verdict, plus gate exits (`SCREENED_OUT`, `CATALYST_BLOCKED`, `ROUTE_*`) | `symbol`, `signal`, `confidence`, `reasoning`, `model` (m52), `provenance` JSONB (m53), `decision` JSONB (m55), `prompt` (m56), `trading_env` |
| `pending_approvals` | `db.py:71` | One row per routed trade, auto-executed or awaiting a Telegram tap | `ticker`, `action`, `conviction`, `thesis_bucket`, `position_size_pct`, `stop_loss_price`, `analyst_json`, `risk_json`, `status`, `status_reason` (m57), `expires_at`, `telegram_message_id` |
| `scanner_candidates` | `db.py:149` | Daily scan scores | `composite_score` and four sub-scores, `priority`, `passed_to_analyst`, `regime_at_scan`, `setup_gates_detail` (m50), `regime_detail` (m61) |
| `trade_plans` | m63 (`db.py:771`) | Shadow trade plans (log-only) | `ticker`, `status`, `entry_price`, `invalidation_price`, `target_price`, `catalyst_date`, `horizon_end`, `revision` |
| `thesis_history` | m36 (`db.py:581`) | Thesis snapshots per ticker | `ticker`, `recorded_at`, `thesis_json` |
| `market_regime_log` | `db.py:60` | Regime packets (overall and sector) | `regime`, `score`, `confidence`, `signals_json`, `regime_type` (m19), `sector_etf` (m20) |

**Orders, positions and outcomes**

| Table | Line | Purpose | Key columns |
|---|---|---|---|
| `orders` | `db.py:213` | Every broker order, including `unconfirmed` ones | `broker_order_id`, `perm_id` (m70), `ticker`, `side`, `order_type`, `qty`, `decision_price`, `fill_price`, `filled_qty`, `slippage_pct`, `status`, `intent` (m59), `approval_id`, `filled_at` (m71) |
| `trades` | `db.py:31` | Confirmed fills | `symbol`, `side`, `quantity`, `price`, `reason`, `price_source` (m65) |
| `position_stops` | `db.py:120` | The DB's view of each position's stop, unique per ticker and env | `entry_price`, `stop_price`, `stop_method`, `trailing_active`, `trailing_high`, `trail_amount`/`trail_percent` (m58), `broker_order_id` (m4), `profit_target_taken` (m30), `setup_category`, `original_thesis`, `latest_thesis` (m34) |
| `trade_outcomes` | `db.py:91` | Entry to exit record for research | `approval_id`, `entry_price`, 30/60/90-day review prices, `exit_price`, `outcome_pct`, `was_correct`, `alpha_pct` (m8), sub-scores (m11-18), catalyst fields (m31-32), `planned_risk_per_share`, `fill_risk_per_share` (m73-74), `orphaned_at` (m67) |
| `portfolio_snapshots` | `db.py:51` | 5-minute account snapshots | `total_value`, `cash`, `positions_json`, `benchmark_json` (m9) |

**Market and reference data**

| Table | Line | Purpose |
|---|---|---|
| `price_history` | `db.py:200` | Daily OHLCV cache keyed by ticker and date |
| `fundamentals` | `db.py:187` | Latest fundamentals per ticker, extended in m38, m39, m42, m46 |
| `fundamentals_history` | m44 (`db.py:661`) | Dated fundamentals snapshots |
| `sp500_fundamentals` | m47 (`db.py:690`) | Peer universe for comparisons |
| `earnings_calendar` | `db.py:111` | Next earnings dates |
| `earnings_calendar_history` | m62 (`db.py:765`) | Dated record of earnings-date changes |
| `trade_ladder` | m33 (`db.py:549`) | Nightly levels, valuation estimates and `rungs` JSONB per ticker |
| `watchlist_candidates` | m21 (`db.py:422`) | Weekly candidate-scan output |
| `news_log` | m23 (`db.py:444`) | News headlines per ticker |

**Ingestion bookkeeping**

| Table | Line | Purpose |
|---|---|---|
| `seen_items` | `db.py:171` | Dedupe of external items by `(source, external_id)`, plus `ticker`, `form` (m64) |
| `ingestion_health` | `db.py:179` | `last_success_at`, `last_item_count`, `last_error` per source |

**State, provenance and UI**

| Table | Line | Purpose |
|---|---|---|
| `system_state` | `db.py:165` | Key/value store: halt flags, high-water marks, throttles, streak counters, manual-action list |
| `activity_feed` | `db.py:137` | Dashboard feed and halt-event log (`stage`, `ticker`, `message`, `sentiment`, `extra_json`) |
| `chat_history` | `db.py:236` | Telegram chat memory per user |
| `schema_version` | `db.py:819` | Applied migration numbers |

Outside PostgreSQL: the catalyst calendar lives in SQLite `catalyst_calendar.db` (`catalyst_service/storage.py:17`), and the vector store in `.chromadb` (`vector_store.py:25`).

---

## 6. Decision flow: event to order

### 6.1 Entry paths into analysis

1. **Events**: each poller builds an event dict and calls `tasks._dispatch_event` (`tasks.py:856`), which enqueues `screen_event` on the `screen` queue and then marks the item seen.
2. **Daily scan**: `run_daily_scan` (`tasks.py:2332`) runs `agents/scanner.py:run_scan`, keeps HIGH and MEDIUM candidates and enqueues `analyze_candidates` on `llm_p1` with at most 5 per run (`tasks.py:2342-2353`, `agents/crew.py:process_candidates`, `:706-750`).
3. **Position review**: `review_positions` (`tasks.py:2447`) calls `agents/crew.py:review_all_held_positions` (`:469`) with up to 10 positions per run, putting stale positions (`risk_manager.py:flag_stale_positions`) first.
4. **Deferred retries**: `retry_deferred_entry` (`tasks.py:1007`) re-runs `execute_approved` for an approval deferred by the market-hours, gap or stale-price gates. Approvals older than 16 h are expired and sent back through a fresh analysis instead (`tasks.py:1195`).
5. **Manual**: Telegram `!urgent`, Telegram Approve buttons and the dashboard Manual Trade form go straight to `execution.py`.

### 6.2 Gates in order (event path)

Each gate either stops the flow or downgrades the action. Downgrades are recorded in the `gates` list stored in `signals.decision`.

**Scanner (scan path only)**

1. Strong Risk-Off regime suppresses the whole scan (`agents/scanner.py:run_scan`, `:405-414`).
2. No bars: ticker disqualified (`agents/scanner.py:443-446`).
3. Earnings blackout, when `EARNINGS_ENTRY_BLACKOUT_DAYS` is non-zero (`agents/scanner.py:451-458`; default 0, `config.py:331`).
4. Priority thresholds: composite score 75 for HIGH and 50 for MEDIUM (`config.py:371-372`), adjusted by regime (`agents/scanner.py:_regime_threshold`).

**Screen (`tasks.screen_event`, `agents/crew.py:screen_event_only`)**

5. Event stored in ChromaDB, best-effort (`agents/crew.py:59-72`).
6. 7B relevance screen (`agents/screener.py:screen`). Not relevant: `SCREENED_OUT` signal and stop (`agents/crew.py:76-85`). If Ollama is unavailable or returns bad JSON, a keyword fallback decides (`agents/screener.py:345-357`). Relevant events go to `llm_p0` if the ticker is held, else `llm_p1` (`tasks.py:901-906`).

**Analysis (`tasks.deep_analyze_event`, `agents/crew.py:deep_analyze`)**

7. Broker positions must be readable, else `SKIP` (`agents/crew.py:126-130`).
8. Brief assembled by `context_engine.assemble_brief` with a live price (`agents/crew.py:135-136`).
9. 72B analysis (`agents/analyst.py:analyze`, `:1219`). No reply or unparseable JSON returns a stub with no trade (`agents/analyst.py:1264-1276`).

**Analyst post-processing (`agents/analyst.py:analyze`, `:1278-1605`)**, applied to the model's output:

10. ADD or HOLD on an unheld ticker becomes OPEN or WATCH (`:1286-1291`).
11. Bucket comes from the watchlist role, not the model, unless `SETUP_GATES_ENABLED` (default false, `config.py:515`), in which case the live setup category is used (`:1344-1366`).
12. Upside/downside ratio under 2.0 on OPEN/ADD: WATCH (`:1368-1374`).
13. Fewer than 2 distinct cited sources on OPEN/ADD: WATCH (`:1376-1384`).
14. Accelerator/Moonshot OPEN/ADD whose expected return per week to the catalyst is below `MIN_EXPECTED_RETURN_PCT_PER_WEEK` (1.0, `config.py:274`): WATCH (`:1389-1401`). On HOLD it is only flagged.
15. Accelerator OPEN/ADD failing the trend template: WATCH (`:1414-1424`).
16. Moonshot OPEN with conviction under 7: WATCH (`:1426-1430`).
17. OPEN in a Neutral, Risk-Off or Strong Risk-Off regime with conviction under 7: WATCH (`:1432-1437`).
18. Missing fundamentals cap conviction at 6 on OPEN/ADD (ETFs exempt only if `ETF_FUNDAMENTALS_CAP_EXEMPT`, default false, `config.py:520`) (`:1439-1462`).
19. `pass_to_risk` is forced false unless the action is OPEN/ADD/TRIM/EXIT and conviction is at least 5 (`:1586-1590`).
20. TRIM/EXIT at conviction 4 or less goes through `_low_conviction_sell`, which decides whether it passes (`agents/analyst.py:1025`, `:1040`, `:1596-1604`).

The verdict, gates, provenance and redacted prompt are written to `signals` (`agents/analyst.py:1647-1670`). Only `pass_to_risk=True` continues (`agents/crew.py:183-184`).

**Routing (`agents/crew.py:_route_to_approval`, `:877`)**

21. An active pending approval for the ticker already exists: stop (`:897-903`).
22. Rebuy cooldown: OPEN/ADD within `REBUY_COOLDOWN_HOURS` (4, `config.py:285`) of an executed buy at equal or higher conviction: stop (`:909-929`).
23. TRIM/EXIT: positions must be readable (else alert and stop) and a position must exist; sell quantity is the whole position (EXIT) or `TRIM_FRACTION` 0.33 of it (`config.py:323`). Sells skip all buy gates below (`:931-984`).
24. Catalyst gate: `catalyst_gate_check(ticker, days_ahead=45)` blocks if an event within 45 days has `halt_auto_execution` set or is conflicting or rumored (`:986-1004`, `catalyst_service/gates.py:6-24`). An error in the gate fails open with a `CATALYST_GATE_ERROR` signal (`:1005-1013`).
25. Earnings blackout, fail closed if the calendar is unreadable; off when the setting is 0 (`:1015-1040`).
26. Price must be available (`:1048-1052`).
27. `RiskManager.size_position` (section 6.3). Zero shares: if the cause is cash or heat, try a swap out of a weaker Accelerator/Moonshot holding during market hours only, then re-size (`:1055-1155`); otherwise stop with `ROUTE_ZERO_SHARES`.
28. Concentration (`risk_manager.py:check_concentration`, `:860`), fail closed: shrinks the order to fit theme (35%), AI-infrastructure (60%) and total moonshot (10%) caps (`config.py:104-106`), or stops if no room (`agents/crew.py:1162-1195`).
29. Correlation: a correlated holding (threshold 0.7, `risk_manager.py:982`) cuts size by 25% (`agents/crew.py:1197-1201`).
30. Approval row written to `pending_approvals` (`agents/crew.py:1208-1219`).
31. If `AUTO_EXECUTE` is false, a Telegram Approve/Reject request is sent and the flow waits for a human (`:1227-1249`). `AUTO_EXECUTE` is true in paper, and in live only if `LIVE_TRADING_ENABLED` is also true (`config.py:64-65`).
32. Otherwise the approval is atomically claimed and `execution.execute_approved(trigger="auto")` runs (`agents/crew.py:1253-1260`).

**Execution (`execution.py:execute_approved`, `:603`)**

33. IBKR must connect (`execution.py:631-645`).
34. `shares_initial` must be positive.
35. OPEN/ADD: `is_halted()` re-checked; a halt or an unreadable halt state blocks the buy (`:658-696`).
36. Market closed: buy deferred to the next open via `retry_deferred_entry`; a sell is skipped and the stop kept (`:707-773`).
37. Stale price on a BUY: retry up to `STALE_PRICE_MAX_RETRIES` (4) every 3 minutes, then skip (`config.py:317-318`, `execution.py:782-880`).
38. Price drift over 8% since sizing aborts a BUY; a SELL proceeds with a warning (`execution.py:892-915`).
39. Opening-gap confirmation for BUYs: a gap of 5% or more must not have retraced 50%; pending gaps are re-checked every 5 minutes up to 3 times; the check fails closed (`config.py:298-306`, `execution.py:917-1008`).
40. Buy room re-check at send time under a Redis lock, subtracting buys of the last 5 minutes that IBKR cash does not yet show; fails closed (`execution.py:_buy_room_at_execution`, `:450`).
41. For sells: resting stops must be readable, then they are cancelled before the sell, and restored if the sell does not fill (`execution.py:1020-1070`, `:_restore_stop_after_failed_sell`).
42. Order placement: a native bracket order if `USE_BRACKET_ORDERS` (default false, `config.py:490`), otherwise `ibkr.place_market_order`, a DAY market order with a 30 s fill wait (`ibkr.py:293`). A pending order when the market is closed is cancelled inside `place_market_order` (`ibkr.py:300-311`). No fill within the window records the order as `unconfirmed` for later reconciliation (`execution.py:1140-1161`).
43. On fill: trade logged, `position_stops` upserted, and the protective stop placed through `tasks.manage_stop_place` (`execution.py:1176-1310`). If placement fails it is queued for retry, and a manual-action item is opened if needed (`execution.py:_queue_stop_for_retry`, `tasks._queue_reprotect`, `tasks.open_manual_action`).

### 6.3 Sizing (`risk_manager.py:size_position`, `:320`)

In order:

1. Halt check (`is_halted`); halted or unreadable returns 0 shares.
2. Account tier conviction floor: standard tier uses `MIN_CONVICTION_TO_QUEUE` 5; reduced and micro tiers use 6 and 7 (`config.py:369`, `:401`). Tiers depend on account size (`config.py:395-396`).
3. If `SETUP_GATES_ENABLED`, a separate risk-first path (`_size_position_setup_gates`, `:636`) with per-category risk 1.5%/1.0%/1.0% and a 7% portfolio heat cap (`config.py:536-577`). Off by default.
4. Allocation band from `BUCKET_SIZE_TABLE` by bucket and 2-point conviction band (`config.py:131-135`), scaled for reduced/micro tiers, times the regime multiplier (Strong Risk-On 1.2, Risk-On 1.0, Neutral 0.75, Risk-Off 0.5, Strong Risk-Off 0.25, `config.py:138-144`).
5. Hard caps per bucket: Compounder 12%, Accelerator 7%, Moonshot 2% at standard tier (`config.py:99-101`), larger at smaller tiers (`config.py:407`, `:421-422`).
6. Max concurrent positions (reduced and micro tiers: 8) and single-share price caps (`config.py:400`, `:428`).
7. Cash cap: at most 20% of current cash per trade (`config.py:113`), with an allowance to buy one share above allocation up to 50% of cash (`config.py:335-339`).
8. Stop from `calculate_stop` (`:804`): Moonshot fixed 20% below entry (`config.py:159`); others entry minus the midpoint ATR multiple (Compounder 2.0-3.0, Accelerator 1.5-2.0, so 2.5x and 1.75x) of 14-day ATR from IBKR daily bars (`config.py:147-150`, `risk_manager.py:112`). If ATR is missing, a flat 8% (Compounder) or 5% stop with a Telegram warning.
9. Risk cap: shares limited so a stop-out loses at most `RISK_PCT_PER_TRADE` 2% of account value (`config.py:172`, `risk_manager.py:541-559`).

---

## 7. LLM call sites

All models run on local Ollama. There are no cloud LLM calls in the trading path. Model names come from `config.py:583-586` and can be overridden by environment variables: `SCREENER_MODEL` qwen2.5:7b, `ANALYST_MODEL` qwen2.5:72b, `CHAT_MODEL` qwen2.5:3b. On the box the environment overrides the analyst default: every analyst call logged since 2026-10-19 used `qwen2.5:72b-instruct-q3_K_M` (llm worker log, `[T] Analyzing with <model>` lines). Two Ollama servers: the main one at `OLLAMA_HOST` (default `localhost:11434`) and a CPU-only one at `OLLAMA_CPU_HOST` (default `localhost:11435`, `deploy/systemd/ollama-cpu.service`).

| Call site | Model | Server | Timeout | Temperature and options | Purpose |
|---|---|---|---|---|---|
| `agents/screener.py:_call_ollama` / `_post_generate` (`:107-152`) | qwen2.5:7b | CPU server first, main server only on connection refused | 60 s | 0.1, `num_predict` 256, `num_gpu` 0, keep-alive 30 m | Event relevance screen |
| `agents/analyst.py:_call_ollama` (`:773-834`) | qwen2.5:72b | main | 400 s | 0.2, `num_predict` 1600 (`:117`), `num_ctx` 32768 (`:19`), `repeat_penalty` 1.1, logprobs top 5 | Trade verdict (action, conviction, bucket, catalyst, valuation, sources, UDR) |
| `tasks.py:weekly_self_review` (`:4755-4765`) | qwen2.5:72b | main | 400 s | 0.3, `num_predict` 500, `num_ctx` 8192 | Weekly narrative review; falls back to raw stats |
| `tasks.py:check_ollama_gpu_health` (`:4483-4494`) | qwen2.5:72b | main | 300 s | `num_predict` 5 | Health probe ("Reply with the single word OK.") |
| `chat_assistant.py:_call_ollama_chat` (`:419-461`) | qwen2.5:3b | CPU server first, then main | 90 s | `num_gpu` 0, keep-alive 30 m, tools enabled; temperature not set (Ollama default) | Telegram chat assistant, also used by standby |

`/api/generate` is used for the screener, analyst, self-review and probe; `/api/chat` for the chat assistant.

### 7.1 Analyst prompt

`agents/analyst.py:_build_prompt` (`:532`) assembles the prompt from the situation brief (`context_engine.assemble_brief`, capped near 24,000 characters, `agents/analyst.py:20`), knowledge files, a track record and the regime packet. A token-budget guard trims the brief to fit `num_ctx` minus an output reserve (`:705-711`). Account dollar figures are redacted from the stored prompt (`agents/analyst.py:redact_account_figures`, `:427`).

### 7.2 Screener fallback

When the 7B call fails or returns bad JSON, `agents/screener.py:_keyword_fallback` decides relevance and the provenance records `model: keyword_fallback` (`agents/screener.py:347-372`).

### 7.3 LLM failure handling

`agents/analyst.py:_stub_recommendation` (`:1064`) returns WATCH with `pass_to_risk=False`. It sends a Telegram alert at most once per hour (state key `stub_recommendation_alert_at`), and after 2 stubs within 20 minutes calls `tasks.restart_ollama_and_alert` (`agents/analyst.py:1088-1135`, `tasks.py:4413`), which is throttled through the `ollama_restart_at` state key.

---

## 8. Stops and exits

### 8.1 Initial stop

Computed at sizing (section 6.3, step 8), stored in `pending_approvals.stop_loss_price` and `position_stops`, and placed as a resting STP sell at the broker right after the fill through the orders worker (`execution.py:1274-1310`). In `execution.py:_anchored_stop` (`:552`) the stop is shifted to keep the planned distance from the actual fill.

### 8.2 Stop monitor (`check_trailing_stops`, every 30 min)

`tasks.py:check_trailing_stops` (`:2776`) calls `risk_manager.py:RiskManager.check_trailing_stops` (`:1278`):

- **Breach**: if price is at or below the recorded stop, an EXIT alert is raised (`risk_manager.py:1313-1321`). The task checks whether a protective stop is resting at the broker; if none exists and `AUTO_EXECUTE_PROTECTIVE_SELLS` is true (`config.py:357`), it market-sells the whole position (`tasks.py:_execute_backstop_sell`, `:2590`). A sell with no fill schedules `retry_backstop_sell` (`tasks.py:2562`, `:2761`).
- **Regular hours only**: stops are only moved on regular-session prices (`risk_manager.py:1293-1302`).
- **Trailing activation**: Compounder after a 30% gain, Accelerator after 20% (`config.py:175-178`). The fixed stop is cancelled and replaced by a native IBKR TRAIL order with a trail amount of the bucket's ATR multiple (fallback: the midpoint of 15-20% or 10-15% as a percentage) (`config.py:179-182`, `risk_manager.py:1336-1385`). The trail is never armed below the current stop. If TRAIL placement fails, the fixed stop is restored or a manual action opened (`risk_manager.py:1386-1450`).
- **Ladder ratchet**: only with `SETUP_GATES_ENABLED`, the stop is ratcheted up to the highest nightly ladder rung below price, never down (`risk_manager.py:_ratchet_ladder_stop`, `:1507`; ladder built by `agents/trade_ladder.py` at 17:15).
- Each cycle also reviews open manual-action items and re-alerts hourly until the position is covered or flat (`tasks.py:open_manual_action`, `:1699`).

### 8.3 Profit taking (`check_profit_targets`, every 30 min)

`risk_manager.py:check_profit_targets` (`:1750`), for Accelerator and Moonshot only (Compounder is buy-and-hold by design). The target depends on the ticker's trade style:

| Style | Accelerator target | Moonshot target | Partial sell |
|---|---|---|---|
| Trending/Expanding | 20-30% | 40-60% | 25% |
| Choppy/Distribution | 10-15% | 20-30% | 45% |

(`config.py:214-236`). On the first crossing it sells that fraction, resizes the stop to the remaining shares and moves it to breakeven, and sets `profit_target_taken` (`risk_manager.py:_partial_sell_and_resize_stop`, `:1619`).

### 8.4 Model-driven exits and staleness

- TRIM/EXIT verdicts from event analysis or the daily position review are sold through `execute_approved` (section 6.2, steps 23 and 41). A sell held because resting stops could not be read is retried once by `retry_held_sell` (`tasks.py:1124`).
- `flag_stale_positions` (`risk_manager.py:1984`) flags positions that moved less than 5% within their window (Trending: Accelerator 28 days, Moonshot 21; Choppy: 14 and 10, `config.py:246-249`). They are reviewed first; nothing is sold automatically.
- `watch_plan_levels` (every 15 min) logs plan-level hits only. A hit queues a review only if `PLAN_LEVEL_REVIEW_ENABLED` (default false, `config.py:526`).

### 8.5 `stop_lock.py`

Every stop change is cancel-then-place and runs from several processes (ops forks, LLM worker, Telegram thread). `stop_lock.ticker_lock` serialises them per ticker with a Redis lock `helios:stop_lock:<TICKER>` (`stop_lock.py:70-90`):

- TTL 90 s, renewed every 30 s by a background thread while held, up to a 900 s cap (`stop_lock.py:29-32`, `_keep_renewed` `:118`).
- Waiters block up to 120 s, then get `TickerBusy`; the `@locked_by_ticker` decorator logs and returns a fallback (`stop_lock.py:149-165`).
- Re-entrant per thread (`stop_lock.py:71-80`).
- `assert_held` raises `LockLost` if renewal lapsed, so the holder stops sending orders (`stop_lock.py:48-58`).
- If Redis is unreachable, the change runs unlocked: protecting a position takes priority over serialising (`stop_lock.py:12-15`, `:89-90`).

### 8.6 Reconciliation

| Mechanism | Schedule | What it does |
|---|---|---|
| Startup reconcile (`main.py:_bounded_startup_reconcile`) | each start | Adopts unrecorded long positions with a stop, cancels stale market BUYs |
| `reconcile_unconfirmed_orders` (`tasks.py:1361`) | 5 min | Checks every `unconfirmed` order of the last 30 days against IBKR's execution records (`ibkr.get_fill_for_order`); records real fills and re-protects with a stop |
| `reconcile_stop_orders` (`tasks.py:4013`, body `:4053`) | 15 min, single-run lock | Cancels orphan stops with no position; deletes stale DB stop records; fixes over- or under-covered stops after two consecutive sightings (`_fix_overstop`, `:3911`; never places a stop at or above the last price); for a position with no broker stop, the first sighting is recorded, a later cycle calls `_fix_overstop_safe` to add one, and after 45 minutes unprotected a throttled "UNPROTECTED POSITION" Telegram alert is sent (`tasks.py:4344-4395`) |
| `flag_orphan_outcomes` | 15:45 weekdays | Marks `trade_outcomes` rows with no live position |

Routine reconcile actions are log-only and throttled per issue (`tasks.py:_reconcile_alert_due`, `:3989`; `RECONCILE_ALERT_THROTTLE_HOURS` 4, `config.py:364`). The unprotected-position alert and stop fixes go to Telegram.

---

## 9. Halts and kill switches

### 9.1 Halt flag

The halt is the `system_state` key `trading_halted_<env>` with `halt_reason_<env>`. It blocks new buys only. Sells, stop placement, trailing, profit taking and reconciliation all continue.

It is checked in:

- `risk_manager.py:is_halted` (`:240`), called by `size_position` and again in `execution.py:execute_approved` before any OPEN/ADD (`:658-696`).
- Telegram `!urgent buy` (`telegram_bot.py:321-323`).

`is_halted` fails closed: if the flag cannot be read, or the account value reads as zero or under half the high-water mark, that buy is blocked without latching a halt (`risk_manager.py:256-285`).

### 9.2 Drawdown halts

Two layers, both against a per-env high-water mark (`high_water_mark_<env>`, seeded from `START_VALUE`):

| Layer | Threshold | Behaviour | Source |
|---|---|---|---|
| Per-buy check in `is_halted` | `DRAWDOWN_HALT_PCT` 10% from HWM | Latches the halt immediately and logs a halt event; also ratchets HWM upward | `config.py:96`, `risk_manager.py:287-312` |
| `check_portfolio_drawdown` task, every 30 min | `MAX_DRAWDOWN_PCT` 15% from HWM | Latches the halt after 2 consecutive breaching reads and sends `alerts.send_halt_alert`; maintains HWM | `config.py:476`, `tasks.py:3779`, `:3870-3895` |
| Implausible reads | value under half the HWM on 3 consecutive checks | Latches a halt with reason prefix "Auto: account value under half the HWM" and stores a rebase candidate | `tasks.py:3775`, `:3815-3850`, `config.py:478` |

Because the per-buy check fires at 10%, it is in practice the tighter limit; the 15% task acts as a backstop that also halts when no buys are being attempted.

### 9.3 Manual controls

| Control | Effect | Source |
|---|---|---|
| Telegram `/halt` or "halt trading" | Sets the halt flag and reason, logs a halt event | `telegram_bot.py:409-424` |
| Telegram `/resume` | Clears the halt and the implausible-read counter | `telegram_bot.py:474-491` |
| Telegram `/rebase` | Applies the pending HWM candidate; lifts the halt only if it was the implausible-read halt | `telegram_bot.py:425-473` |
| `AUTO_EXECUTE` (`TRADING_ENV`, `LIVE_TRADING_ENABLED`) | In live without `LIVE_TRADING_ENABLED=true`, every trade needs a Telegram tap | `config.py:53-65` |
| Approval expiry | Pending approvals expire after 6 h | `config.py:471`, `tasks.py:3686` |
| Dashboard Stop All | Stops all child processes (not a trading halt) | `dashboard.py:218-228` |
| systemd / evening stop | Stack down overnight; standby has no trading code | section 1 |

There is no flatten-all command in Telegram or the dashboard.

### 9.4 Standby

From 19:00 ET until the next 05:00 weekday start, `helios-standby.service` runs `standby_bot.py` instead of the full stack. Only resting broker orders (stops and trails) protect positions overnight. Standby can wake the full stack on request.

### 9.5 Other guards

- Market-hours guard for buys and sells (section 6.2, step 36), using the NYSE calendar in `market_data.py:is_market_hours` (`:47`).
- Swaps are blocked outside regular hours (`agents/crew.py:1090-1101`).
- LLM circuit breaker: stubs never trade; repeated stubs restart Ollama (section 7.3).
- Shadow features that log but do not trade: setup gates (`SETUP_GATES_ENABLED` false), trade plans (`agents/plans.py:observe`), heat shadow, swap shadow, rule shadows (`agents/analyst.py:_rule_shadows`).

---

## 10. Logging and provenance

### 10.1 Log files

| Log | Written by | Rotation |
|---|---|---|
| `helios.log` | main.py process (Telegram threads included) | `RotatingFileHandler` 10 MB x 5 (`main.py:29-44`) |
| `logs/<service>.log` (one per Celery worker, beat, catalyst service) | process_manager children | renamed to `.1` at start once over 20 MB, older `.1` moved to `logs/archive/` (`process_manager.py:143-165`) |
| `helios-systemd.log` | systemd stdout/stderr of helios-local | `deploy/logrotate-helios.conf`: weekly, keep 8, compress, delaycompress, copytruncate |
| `logs/watchdog.log`, `logs/standby_bot.log` | watchdog and standby | no rotation found in the repo |

### 10.2 Log redaction

`log_redaction.install()` replaces the `LogRecord` factory and the formatter's exception and stack formatting so every logger, including third-party ones, is redacted (`log_redaction.py:35-48`). Patterns: Telegram bot tokens, IBKR account IDs, and `api_key=` / `token=` query parameters (`log_redaction.py:12-19`). It is installed in `main.py:23-25`, `alerts.py:20-21`, `telegram_bot.py:23-25` and `standby_bot.py:33`, and outgoing Telegram text is also passed through `redact` (`alerts.py:40`, `:68`; `telegram_bot.py:88`, `:136`).

### 10.3 Activity feed

`feed.emit(stage, ticker, message, sentiment, extra)` writes to `activity_feed` from a daemon thread and never raises (`feed.py:25-50`). Stages include screener, analyst, risk, telegram, execution, system and the pollers (`feed.py:12-23`). The dashboard's Activity tab reads it with `db.get_activity_feed` (`db.py:1516`). Halts, resumes and rebases are written there too with stage `halt` by `db.log_halt_event` (`db.py:1502-1513`); fields carry ratios only, not account values.

### 10.4 Decision provenance

Every analyst verdict row in `signals` carries:

- `model` (`ANALYST_MODEL`) and `provenance` from `agents/analyst.py:_provenance` (`:326-368`):
  - `code_sha` and `head_sha` (git commit, with `-dirty` if tracked files changed), pinned at worker start (`tasks.py:_pin_code_stamp`, `:143-151`)
  - `model_digest`: the first 12 hex characters of the Ollama model digest, not just the tag
  - `prompt_template_sha`: SHA-256 of the `_build_prompt` source
  - `brief_template_sha`: SHA-256 of the `context_engine` source
  - `prompt_sha`: SHA-256 of the full rendered prompt
  - `config`: the env-overridable treatment switches (`_treatment_config`, `:312-323`)
  - `llm_call` token statistics and conviction log-probabilities (`:773-834`)
- `decision`: model action and conviction versus final action and conviction after gates, the list of gates applied, the raw model output and the shadow checks (`agents/analyst.py:1656-1668`)
- `prompt`: the full prompt with account figures redacted (`agents/analyst.py:1669`)

Screener rows carry the same kind of provenance with the 7B digest, a template hash and a prompt hash (`agents/screener.py:362-376`). The analyst result, including provenance, is also stored in `pending_approvals.analyst_json`, and sizing in `risk_json` (`agents/crew.py:1208-1219`).

### 10.5 Alerts

`alerts.py` sends to one Telegram chat (`TELEGRAM_CHAT_ID`) with up to 3 attempts (`alerts.py:_send`, `:33`). It provides approval requests with buttons, trade alerts, stop-triggered, halt, digest and catalyst-conflict messages (`alerts.py:128-342`). The watchdog and `notify_failure.py` send directly to the Telegram API without importing the app.

### 10.6 AI usage log

Separate from the trading system, `research-log/` records AI coding sessions for ISEF Rule 8 and Form 2A: curated entries in `research-log/prompts/YYYY-MM.md`, verbatim prompts and transcripts captured by hooks in `.claude/settings.json` and kept outside git, and SHA-256 hashes of archived transcripts committed in `research-log/transcripts-manifest.csv` (`research-log/README.md`).

---

## 11. Known inconsistencies and open doubts

These are places where comments and code disagree, or where behaviour may surprise a reader. None were changed.

1. **Regime task "LLM-heavy"**: the beat comment routes `assess-regime` to `llm_p0` as LLM-heavy (`tasks.py:402-403`), but `agents/market_regime.py` makes no LLM call. It still occupies the single LLM worker slot.
2. **Alpaca dependency**: the scanner and regime fetch daily bars from Alpaca's IEX feed and return nothing without `ALPACA_API_KEY` (`agents/scanner.py:31-51`, `agents/market_regime.py:539-573`), while `ibkr.py` says it replaced Alpaca (`ibkr.py:5`). If the key is absent, every ticker is disqualified in the scan and the regime has no SPY bars. I did not check `.env` (by rule), so whether the key is set is unknown.
3. **Options fallback condition**: `poll_options_flow` switches to the free proxy only when the key starts with `your_` (`tasks.py:2969`). An empty key takes the Unusual Whales path, which returns early inside `ingestion/options_flow.py:43`.
4. **Live clientIds**: `config.py:69` says live uses clientId 40 "distinct from every paper clientId", but `process_manager.py:280-309` sets 10, 20 and 30 for workers in both environments. Only main.py gets 40 in live.
5. **Two drawdown thresholds**: the per-buy check halts at 10% (`DRAWDOWN_HALT_PCT`) while the task comment and config describe 15% (`MAX_DRAWDOWN_PCT`) as the circuit breaker. Both exist; 10% fires first when a buy is attempted.
6. **Market-hours comments on interval tasks**: beat comments say "during market hours" for EDGAR, snapshots and others, but interval schedules run whenever beat is up (05:00-19:00 ET).
7. **`take_portfolio_snapshot` docstring** says it logs "to Supabase"; the connection uses `SUPABASE_*` variables but could point at any PostgreSQL server.
8. **Systemd docs**: `ibc-live.service` and `xvfb.service` are referenced (`deploy/watchdog.py:204`, `helios-local.service:3-10`) but not in the repo; README says xvfb uses display `:1` while `ibc.service` uses `:99`; the watchdog timer comments still mention suspending the machine, which evening-stop no longer does; `standby_bot.py:9-11` says starting helios-local cascades to ibc, but helios-local has no `Requires=` on ibc.
9. **Partial installer**: `install-auto-schedule.sh` does not install the watchdog, standby, notify-failure, ollama-cpu or backup units; how they were installed is not documented.
10. **Main Ollama server**: there is a unit for the CPU server (11435) but none for the main server on 11434 in `deploy/systemd/`.
11. **Paper-only remnants**: `launch.sh:54` probes only port 7497, and the dashboard labels itself "Paper trading only" (`dashboard.py:188`, `:1120`).
12. **Catalyst gate fails open** on error (`agents/crew.py:1005-1013`), unlike the earnings and concentration gates, which fail closed. The code comment notes this is a deliberate pending decision.
