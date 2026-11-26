# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | From the 2026-11-26 05:00 ET start (pulled 2026-11-25 19:16 ET while stopped; Thu is a market holiday, first session Fri 11-27, early close 13:00) |
| Local deploy tag | `deploy-2026-11-26` (local commit `3b8b7c1`: `b210343` plus batch-1126: re-protect after a late or queued sell decided by one pure plan (`_reprotect_plan`) from per-stop cancel statuses and fresh position reads, part-filled cancels kept as Filled, one coverage re-check per ticker, a failed second look stops the whole remainder, queued-exit drop alert and manual action, throttled backstop alerts, Celery reruns leave retry budgets unspent, would-be sizing on gate-blocked buys, event-cluster label in story de-dup, material-event terms and per-ticker deal record, position weight readout) |
| Settings at this load | No .env change. No migration. New constants in config.py, not read from .env: `BLOCKED_SIZING_SHADOW_ENABLED = True` (record only), `STORY_EVENT_CLUSTER_JACCARD = 0.5` (record only). Safety fixes with no switch: the re-protect never sizes a new stop for shares a stop of ours sold during the cancel, and never counts a stop still cancelling as cover; a Celery rerun of reconcile leaves queued retry budgets unspent. Entries, sizing of new buys, routing and analyst input unchanged; template shas unchanged. |
| Previous load | 2026-11-25 05:00 to 2026-11-25 19:00 (`deploy-2026-11-25`, local commit `b210343`: `d6fd0b6` plus batch-1125) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
