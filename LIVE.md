# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-10 19:08 ET (pulled while stopped; runs from the 2026-11-11 05:00 boot, which applies database migration 89) |
| Local deploy tag | `deploy-2026-11-10` (local commit `a7f301c`: `9fbe88f` plus yfinance 1.8.0, the ADD whole-position stop fix, activity stage width, and log-only last-exit, catalyst-rule and ADD-context fields) |
| Previous load | 2026-11-09 (`deploy-2026-11-09b`, local commit `9fbe88f`) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
