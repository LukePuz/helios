# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-24 05:00:07 ET (helios-local ActiveEnterTimestamp; boot check 05:12 clean: no migrations at start, batch_verify 0 FAIL, all queues up) (pulled 2026-11-23 19:16 ET while stopped; migration 92 applied 19:16:47 and indexes 90-91 built 19:17 while stopped) |
| Local deploy tag | `deploy-2026-11-24` (local commit `d6fd0b6`: `fcad43c` plus batch-1124: record-only shadows and readouts, ADD and profit-target stop quantity from the broker count, stop-path timing fixes). Before it, `deploy-2026-11-23b` (`fcad43c`) |
| Settings at this load | No .env change. Migration 92 (`orders.held_before_qty`) pre-applied; indexes 90-91 on signals pre-built. New constants in config.py, not read from .env. Live: `ADD_STOP_QTY_FROM_BROKER = True` (an ADD's whole-position stop sized from the broker's position count; an unusable count keeps the old sizing and alerts), `PT_STOP_QTY_FROM_BROKER = True` (the profit-target remainder stop sized the same way, never above held − sale). Record only: `STOP_GUARD_MODE`, `LATE_FILL_EXPECT_MODE`, `LATE_FILL_HELD_BEFORE_MODE`, `ADD_LOSER_GATE_MODE`, `HWM_GUARD_MODE`, `RISK_SUBCHECK_FAIL_MODE` = "shadow"; `BROKER_SELL_WATCH_ENABLED`, `ENTRY_BASIS_RECORD_ENABLED`, `GAP_RISK_RECORD_ENABLED`, `FEED_HEALTH_ENABLED` and the batch's shadow switches = True. Off: `PRICE_CACHE_RESCALE_ENABLED`, `CANCEL_PARALLEL_ENABLED`, `VERDICT_VOTE_SHADOW_ENABLED` = False. Timing only, no level/size/type change: alerts and record writes moved after the stop re-place. `brief_template_full_sha` changes to `a994de527dcd` (market_data.py, a brief module, gained record-only code; brief text unchanged); the other five shas unchanged. Entries, sells, sizing of new buys and analyst input unchanged. helios-local's installed unit still has the 90 s default start timeout (roadmap 04dc, Luke action). |
| Previous load | 2026-11-23 05:00 to 2026-11-23 19:00 (`deploy-2026-11-23b`, local commit `fcad43c`: `ee766bd` plus the per-buy drawdown halt alert) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
