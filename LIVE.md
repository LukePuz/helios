# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-25 05:00:21 ET (helios-local ActiveEnterTimestamp; boot check 05:12 clean: 'Database tables ready.' 05:00:22, no migrations, all five queue banners as expected, 0 old-tree workers, errscan since 05:00 empty, batch_verify 0 deploy FAIL; its one FAIL is entry_funnel, the open trade blocker) (pulled 2026-11-24 19:13 ET while stopped; no migration) |
| Local deploy tag | `deploy-2026-11-25` (local commit `b210343`: `d6fd0b6` plus batch-1125: confirmed stop cancels (cancel_order, parallel and orphan cancels from trade.log), an ADD's whole stop when the old stop's cancel is unconfirmed or the old stop filled, fresh position reads before a sale's stop record is deleted, a fail-closed market-hours gate on buys, one halt alert per blocked buy, held-alert order, scan tier funnel and entry-gate reclaim records, entry gate readout). Before it, `deploy-2026-11-24` (`d6fd0b6`) |
| Settings at this load | No .env change. No migration. New constants in config.py, not read from .env: `SCAN_TIER_FUNNEL_RECORD_ENABLED = True`, `ENTRY_GATE_RECLAIM_SHADOW_ENABLED = True` (record only), `ENTRY_GATE_RECLAIM_SHADOW_SESSIONS = 20`. Safety fixes with no switch: EXIT sizing takes the larger of the cache and the pushed position count; a sale's stop record is deleted only on a flat pushed read; a cancel counts only when the broker confirms it; an ADD whose old stop filled during the cancel stops only the new shares and sends CHECK POSITION; a market-hours check error places no buy. Entries, sizing of new buys, routing and analyst input unchanged; the six template shas unchanged. helios-local's installed unit still has the 90 s default start timeout (roadmap 04dc, Luke action). |
| Previous load | 2026-11-24 05:00 to 2026-11-24 19:00 (`deploy-2026-11-24`, local commit `d6fd0b6`: `fcad43c` plus batch-1124) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
