# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-09 16:19 ET (loaded and restarted after the close; database schema 88); `9fbe88f` pulled 19:08 ET while stopped, running from the 2026-11-10 05:00 boot |
| Local deploy tag | `deploy-2026-11-09b` (local commit `9fbe88f`: `6b56ff1` plus log-only new-fact tone fields and a news-feed outage alert) |
| Previous load | 2026-11-07 (`deploy-2026-11-07-hotfix`, local commit `2f3b3eb`) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
