# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-09 05:00 ET boot (loaded 2026-11-07 04:59 ET, live clone reflog) |
| Local deploy tag | `deploy-2026-11-07-hotfix` (local commit `2f3b3eb`: `f715d71` plus the dashboard loopback bind) |
| Previous load | 2026-11-06 (`deploy-2026-11-06`, local commit `f715d71`) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
