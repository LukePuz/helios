# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-27 05:00:15 ET (helios-local ActiveEnterTimestamp; pulled 2026-11-26 19:09 ET while stopped; Fri 11-27 is a 13:00 early close) |
| Local deploy tag | `deploy-2026-11-27` (local commit `8ec6fa4`: `3b8b7c1` plus batch-1127: queued flat-exit remainder and fill lookups matched on permId where known; a failed positions confirm read is named in the refusal and the STOP REFUSED alert; market-order fill time waits up to 2 s for late execution records and is stamped complete or not; after an add on a stale position record, a still-working (or unread) old stop is kept and only the new shares get their own stop, with the record basis noted) |
| Settings at this load | No .env change. Migrations 93 (`orders.fill_time_complete` BOOLEAN) and 94 (`trade_outcomes.exit_fill_complete` BOOLEAN), applied 2026-11-26 19:09 ET with scripts/apply_migration.py while the stack was stopped. New constant in config.py, not read from .env: `FILL_TIME_WAIT_S = 2.0` (0 = off): after a market order is done, wait up to 2 s while the execution records are short of the filled count. Price, quantity and status unchanged. Entries, sizing, routing and analyst input unchanged; template shas unchanged. |
| Previous load | 2026-11-26 05:00-19:00 (`deploy-2026-11-26`, local commit `3b8b7c1`) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
