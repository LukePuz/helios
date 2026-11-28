# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | From the 2026-11-30 05:00 ET start (pulled 2026-11-27 19:19 ET while stopped) |
| Local deploy tag | `deploy-2026-11-30` (local commit `1ac1104`: `8ec6fa4` plus batch-1128: unrouted-verdict forward record, its registry stage; per-source poll health record and digest line; reconcile trim fresh check in shadow, its registry stage; the screen and ingestion workers on the transaction pooler with one DB pool per process, and the db-port.conf drop-in removed from the repo) |
| Settings at this load | No .env change. No migration. New constants in config.py, not read from .env: `UNROUTED_VERDICT_FORWARD_SHADOW_ENABLED = True` (record only: forward closes 5/10/20 sessions after each name's first non-routed verdict of the day); `SOURCE_HEALTH_RECORD_ENABLED = True`, `SOURCE_HEALTH_DEAD_SESSIONS = 2` (record and report only: per-source poll health, one digest line naming sources dead 2+ sessions); `RECONCILE_TRIM_FRESH_CHECK_MODE = "shadow"` (record only: reconcile's over-cover trim runs as before; trims that sent a cancel are checked against one fresh positions read after the cycle's order work, and a disagreeing or failed read writes a `reconcile_trim_fresh_check` row; enforce, add-only on those reads, is a later decision). The celery screen and ingestion workers connect through the Supabase transaction pooler (port 6543, set by process_manager); main.py and the other workers stay on the session pooler (5432); one DB pool per process (creation under a lock). Entries, sizing, routing and analyst input unchanged; no data, prompt or decision change; template shas unchanged. |
| Previous load | 2026-11-27 05:00:15-19:00:22 ET (`deploy-2026-11-27`, local commit `8ec6fa4`; 13:00 early close) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
