# Live version

The bot runs from a local deploy branch, not from this branch. Each load is tagged locally, and the tags aren't
pushed here. So this branch can be ahead of, or differ from, the code that is running.

| | |
|---|---|
| Live since | 2026-11-16 05:00 ET (pulled 2026-11-14 while stopped; first loaded at the Monday start) |
| Local deploy tag | `deploy-2026-11-14` (local commit `2b0d52c`: `b0ddb02` plus an OPEN on an already-held ticker taking the ADD stop path) |
| Previous load | 2026-11-13 17:28 (`deploy-2026-11-13b`, local commit `b0ddb02`) |

Commits on this branch that aren't in the deploy commit above aren't live yet (for example the held-move shadow);
they go live with a later load listed here. Updated at each deploy.
