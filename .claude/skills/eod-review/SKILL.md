---
name: eod-review
description: End-of-day full system review for Helios — every agent's decisions, execution/broker reconciliation, pipeline health, trade-volume-vs-market correlation, and watchlist ideas — then fixes what it finds (code fixes, DB reconciliation, live protective orders) and deploys (test, commit, push, restart affected services) before reporting. Use when the user asks for an end-of-day review, daily wrap-up, or runs /eod-review.
---

Run a full end-of-day review of the Helios trading system for today. Use
real data — DB queries, live IBKR state, and log greps — not summaries
from memory. Work through every section below; don't skip ahead to
conclusions before pulling the evidence.

## 1. Per-agent decision audit
For every signal/decision produced today, pull it from the DB rather than
describing the pipeline in the abstract:
- `signals` table: every SCREENED_OUT, COOLDOWN_SKIPPED, SELL_NO_POSITION
  logged today — what got filtered and why.
- `pending_approvals`: every approval created today (ticker, action,
  conviction, thesis_bucket, status, reasoning). Group by status
  (APPROVED/FAILED/EXPIRED/REJECTED) and flag anything that looks
  contradictory (e.g. high conviction but a weak/thin reasoning field,
  or a chase_risk/drift flag that got overridden).
- `trade_outcomes`: today's sub-scores (thesis_strength, catalyst_quality,
  valuation_risk_reward, chase_risk, price_action) for each decision —
  do the sub-scores actually support the conviction assigned, or does
  anything look like the analyst talked itself into a trade?
- Screener (7B) vs analyst (72B) agreement rate: how many events the
  screener passed did the analyst actually act on vs HOLD/WATCH?
- Any ticker with more than one approval today — check timestamps and
  reasoning to confirm they're genuinely independent signals, not the
  same event re-analyzed (the TER failure mode from 2026-07-29).

## 2. Execution & broker reconciliation
Cross-check our own records against the live broker directly — most of
the real incidents this system has had were DB/broker divergence that
nothing caught on its own:
- For every fill in `trades`/`orders` today, confirm a matching execution
  exists at IBKR (`ibkr.get_fill_for_order` / a fresh `ib.reqExecutions()`
  call via a distinct `IB_CLIENT_ID`).
- For every open position, confirm a live protective stop exists at the
  broker covering the *actual* held quantity (not just ticker presence —
  check quantity match, per the AMKR/AMD incidents). Flag any naked
  position immediately, at the top of the report, before anything else.
- Check `orders` for any row still `status='unconfirmed'` older than an
  hour — `reconcile_unconfirmed_orders` should have caught these; if one
  is still sitting there, that task itself may be failing.
- Spot-check `position_stops` vs live broker stop orders for quantity
  drift.

## 3. Pipeline health
- Celery: `redis-cli llen ops/llm_p0/llm_p1/llm_p2/screen` — any queue
  still backed up after close? Grep all four service logs
  (`logs/celery_worker.log`, `celery_llm_worker.log`,
  `celery_screen_worker.log`, `celery_beat.log`) for ERROR-level lines
  and WorkerLostError/SIGTERM entries not explained by a known restart.
- Ingestion: did every source (edgar, insider, news, congressional,
  macro, options_flow, sp500_history, analyst_ratings) report a healthy
  heartbeat today, or is `check_ingestion_health` flagging anything stale?
- Ollama/GPU: any "unresponsive" / stub-fallback entries in the logs?
- Any repeated log line (same message firing more than ~3x) is itself a
  finding — that shape has twice now meant a throttle or dedup mechanism
  silently broke (the ingestion-alert key-length bug, the ETA-redelivery
  bug). Don't dismiss repetition as "just noisy."

## 4. What actually hit the Telegram group
Reconstruct what was actually sent (grep for the alert text patterns
next to their log.error/log.warning calls, since alerts.py doesn't store
sent messages) and categorize: legitimate one-off alerts, vs. anything
that fired more than once for the same underlying issue. Duplicate
alerts are a bug signal, not just annoyance — trace each one to a cause.

## 5. Portfolio & P&L
- Day P&L and current drawdown from `high_water_mark` (system_state),
  compared to SPY and QQQ from `portfolio_snapshots.benchmark_json` —
  is today's move explained by the market, or concentrated in specific
  positions?
- Biggest winners/losers by unrealized P&L from the latest snapshot —
  for the largest mover, pull its `activity_feed` history and check
  whether the entry thesis is holding up or was a chase/misread.
- Any position that's unusually large relative to the rest of the book,
  or came from a scaled-in/multi-tranche entry — verify total share count
  matches what was actually intended.

## 6. Trade volume vs. market conditions
This is a standing check, not a one-off — build it fresh each day so
patterns show up over time:
- Count today's new-entry decisions (OPEN/ADD approvals created, split
  into executed vs. deferred/failed/rejected) against today's SPY and
  QQQ daily return, and against `market_regime_log`'s overall regime row
  for today (`regime_type='overall'`) — is `regime_multiplier`/
  `accelerator_sizing` actually showing up as reduced trade volume on
  weak/Neutral-or-worse days, or did the system trade just as much
  regardless of regime?
- Separately pull `market_regime_log` rows with `regime_type != 'overall'`
  for tech/semis-relevant `sector_etf` values (QQQ, SOXX, XLK) — the
  overall SPY regime has previously stayed "Neutral" through a real
  tech/semis rotation (see the 2026-07-26 finding that motivated tracking
  sector regime separately), so check AI/tech conditions on their own,
  not just the blended market read.
- Compute today's screen/reject rate (SCREENED_OUT + COOLDOWN_SKIPPED +
  REJECTED, divided by total signals considered) and compare it to a
  trailing 5-10 day baseline split by regime (does the reject rate rise
  on down days the way it should, or does everything still pass at the
  same rate regardless of conditions — this is the exact concern raised
  2026-07-27 about a concentrated buying day during a market selloff).
- Flag explicitly: more trades on a down/weak-tech day than the trailing
  baseline is worth a closer look, not an automatic red flag — say
  whether today's volume looks proportionate or not, with the numbers.

## 7. Code health
- `git status` — anything uncommitted?
- `venv/bin/python -m pytest tests/` — full suite green?
- Any TODO/follow-up mentioned in a recent commit message that never got
  addressed?

## 8. Watchlist / ticker universe ideas
- Check `watchlist_candidates` (agents/candidate_scanner.py's output) for
  anything that scored well today but isn't in `config.SYMBOLS` or
  `knowledge/watchlist.md` yet.
- Look for a *theme*, not just individual tickers: if today's news/edgar/
  options-flow activity clustered around a sector or sub-industry not
  represented by any existing `knowledge/*.md` thesis file (currently:
  ai_infrastructure, ai_software, ai_data, power_infrastructure, robotics,
  quantum, space_data_centers, financials), that's worth naming as a
  candidate new realm, with the evidence for why it's showing up now.
- Don't recommend tickers/sectors based on general knowledge alone — tie
  every suggestion back to something observed in today's actual pipeline
  activity (a catalyst, a repeated candidate-scan hit, a correlated-name
  pattern in the correlation-adjustment logs).

## 9. Findings
List every concrete issue found above with evidence (exact numbers,
timestamps, ticker, dollar amounts) — not "some alerts seemed noisy" but
"X fired N times between T1 and T2 because Y." Rank by severity: live
unprotected exposure first, then data-integrity gaps, then noise/cost
issues.

## 10. Workflow improvement ideas
Separately from bug fixes, note any *pattern* visible across today's
decisions that suggests a missing safeguard or a new pipeline step
(e.g. the 2026-07-29/30 TER/AMD/gap-confirmation/ETA-redelivery fixes
all came from patterns like this). Ground each idea in what actually
happened today, not a generic best-practice list.

## 11. Action plan — then execute it
Compile a prioritized punch list, then work through it — this skill fixes
what it finds, it doesn't just report it. Distinguish two kinds of items:

**Bug fixes and reconciliation — execute automatically, no need to stop
and ask:**
- Code fixes for a root-caused bug (a missing retry, a stale-data guard, a
  throttle that broke, a reconciliation gap). Follow the pattern already
  established in this codebase: identify the specific incident/evidence,
  write the fix, add or update a regression test that encodes the incident
  (not just the happy path), run the full suite, and only proceed once
  it's green.
- DB corrections that close a data-integrity gap found in Section 1/2/9
  (e.g. reconciling a stuck `unconfirmed` order once there's solid
  evidence it filled) — use existing `db.py` functions where they exist.
- Placing or adjusting a live protective order to close a real risk gap
  found in Section 2 (e.g. a naked or under-covered position) — verify
  against the broker afterward that the gap is actually closed.
- Restarting Celery workers/beat if a fix touches `tasks.py`, `config.py`,
  or anything else read at process start — a fix sitting in committed code
  that the running processes haven't picked up isn't actually deployed.

If a specific write gets blocked by the Claude Code auto-mode permission
classifier, don't try to work around it — note it plainly in the final
report as still-pending and move on to the rest of the plan.

**Discretionary trading/investment decisions — surface, don't execute:**
Anything that isn't a bug fix but an actual judgment call about the
portfolio (exit a position, resize it, override a thesis, hold through a
known event like earnings) is not this skill's call to make
autonomously. Flag it clearly with the evidence (Section 5's largest
movers, an activity_feed thesis check) and let the user decide — this is
the one category still worth stopping for.

## 12. Deploy
Once the action-plan fixes are implemented and tested:
- If the working checkout and the live deployment are separate clones of
  the same repo (check `git remote -v` / compare `git rev-parse HEAD` in
  each), export the diff as a patch (`git diff -- <files> > patch`) and
  apply it to the other clone (`git apply`) rather than hand-editing twice
  — then diff the changed files between the two to confirm byte-identical
  before trusting either copy.
- Run the full test suite once more against whichever checkout actually
  has a runnable venv.
- `git add` the changed files (not `-A`), commit with a message that
  explains the root cause and the fix (mirroring this session's commit
  style — what broke, why, what changes), and push to `origin`.
- If any restarted Celery workers are needed (see Section 11), do that
  restart and verify the new config/code is actually live (e.g. check
  `app.conf` for a config change, or grep fresh log output for the new
  code path) rather than assuming a restart alone means it worked.
- Report what was fixed, what got deployed, what's still pending (blocked
  writes, discretionary decisions left for the user), and the commit(s)
  pushed — don't let a successful deploy go unmentioned in the final
  summary.
