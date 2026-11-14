# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-13 17:28 ET (drained restart after the close) |
| Local deploy tag | `deploy-2026-11-13b` (local commit `b0ddb02`: `7ba0857` plus the ADD whole-position stop keeping the old stop unless the whole position is known) |
| Previous load | 2026-11-13 16:11 (`deploy-2026-11-13`, local commit `7ba0857`) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
