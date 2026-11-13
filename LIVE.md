# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-12 16:11 ET (drained restart after the close) |
| Local deploy tag | `deploy-2026-11-12` (local commit `92ea06f`: `a7f301c` plus booking of untracked partial exits of still-held longs) |
| Previous load | 2026-11-10 (`deploy-2026-11-10`, local commit `a7f301c`) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
