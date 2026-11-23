# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-23 05:00:09 ET (helios-local start on `fcad43c`; boot check 05:12: IBKR connected, stop reconcile in sync, no errors). Pulled 2026-11-21 13:57, 16:35 and 20:15 ET while stopped (weekend) |
| Local deploy tag | `deploy-2026-11-23b` (local commit `fcad43c`: `ee766bd` plus the per-buy drawdown halt alert; pulled 2026-11-21 20:15 ET while stopped). Before it, `deploy-2026-11-23` (`ee766bd`: `f56e902` plus the live entry gates (`345a9d6`) and the market condition on the trend gate; pulled 13:57 and 16:35) |
| Settings at this load | No .env change. New constants in config.py, not read from .env: `STOPOUT_REENTRY_GATE_DAYS = 10`, `ENTRY_TREND_GATE_ENABLED = True`, `ENTRY_TREND_GATE_MAX_BAR_AGE_DAYS = 5`. Behaviour change: an OPEN/ADD that passed the rebuy cooldown is skipped, with one `ENTRY_GATE_BLOCKED` row, when the ticker's newest exit the bot did not place (a stop fill) is under 10 days old, or its last completed close is below both the 50- and 200-day averages (cached bars at most 5 days old) while SPY closes below its own 200-day average (`ENTRY_TREND_GATE_MARKET_TICKER = "SPY"`). Failed reads and stale bars do not block; conviction does not override. Sells, stops, sizing and analyst input unchanged. Also (`fcad43c`): when the per-buy drawdown check latches the halt (`DRAWDOWN_HALT_PCT`, unchanged), one Telegram alert is sent, in percentages only; before, only the 30-min task alerted, at `MAX_DRAWDOWN_PCT`. No threshold changed. |
| Previous load | 2026-11-20 05:00 to 2026-11-20 19:00 (`deploy-2026-11-20`, local commit `f56e902`: record-only decision fields, the no-quote BUY deferral, the stop-out re-entry and entry-quality shadow records) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
