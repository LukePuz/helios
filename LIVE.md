# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-20 05:00 ET (pulled 2026-11-19 21:14, 21:31 and 21:56 ET while stopped) |
| Local deploy tag | `deploy-2026-11-20` (local commit `f56e902`: `6d7cccd` plus the record-only decision fields, the no-quote BUY deferral (merge `d662d32`), the stop-out re-entry shadow record (`6ec4d85`) and the entry-quality shadow record; pulled 21:14, 21:31 and 21:56) |
| Settings at this load | No .env change and no new settings. Record-only fields are added to existing rows: previous verdict on buy verdicts, exit state and held shares on ADD routes, shown versus entry conviction, the event-spanning implied move beside the nearest-expiry one, plan invalidation loosening, the Yahoo fallback counterfactual on stale-quote BUY checks; unheld plan level hits say "no price". New record-only kind `ENTRY_QUALITY_SHADOW` (`ENTRY_QUALITY_SHADOW_ENABLED`, a constant): one row per OPEN/ADD verdict past the rebuy cooldown with the trigger's new_fact and trend vs the 50/200-day averages and drawdown in ATR from cached bars. New record-only kind `STOPOUT_REENTRY_SHADOW` (`STOPOUT_REENTRY_SHADOW_ENABLED`, a constant, not read from .env): one row per OPEN/ADD past the rebuy cooldown within 10 days of a broker exit. One behaviour change: a BUY with no pre-order IBKR quote, or whose quote check raises, is deferred like a stale-quote BUY instead of going ahead. Analyst input unchanged. |
| Previous load | 2026-11-17 05:00 to 2026-11-19 19:00 (`deploy-2026-11-17`, local commit `6d7cccd`: the regime `shadow_200` record and the trim/exit rule in shadow, `TRIM_EXIT_ENABLED=true`, `TRIM_EXIT_LIVE` false) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
