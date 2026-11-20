# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | Loads at the 2026-11-20 05:00 ET start (pulled 2026-11-19 21:14 ET while stopped); until then the 2026-11-17 load below ran |
| Local deploy tag | `deploy-2026-11-20` (local commit `d662d32`: `6d7cccd` plus the record-only decision fields and the no-quote BUY deferral) |
| Settings at this load | No .env change and no new settings. Record-only fields are added to existing rows: previous verdict on buy verdicts, exit state and held shares on ADD routes, shown versus entry conviction, the event-spanning implied move beside the nearest-expiry one, plan invalidation loosening, the Yahoo fallback counterfactual on stale-quote BUY checks; unheld plan level hits say "no price". One behaviour change: a BUY with no pre-order IBKR quote, or whose quote check raises, is deferred like a stale-quote BUY instead of going ahead. Analyst input unchanged. |
| Previous load | 2026-11-17 05:00 to 2026-11-19 19:00 (`deploy-2026-11-17`, local commit `6d7cccd`: the regime `shadow_200` record and the trim/exit rule in shadow, `TRIM_EXIT_ENABLED=true`, `TRIM_EXIT_LIVE` false) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
