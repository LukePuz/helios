# AI usage log

Records of every Claude Code session on this repository, kept for the ISEF
Rule 8 citation requirement and the Form 2A AI Prompt Log.

| What | Where | In git? | Written by |
|---|---|---|---|
| Curated log: original prompt, AI-refined prompt, what was done, files, commit | `prompts/YYYY-MM.md` | yes | Claude, per `.claude/CLAUDE.md` |
| Every prompt, verbatim, with timestamp and session id | `raw-prompts.jsonl` | no (see below) | `UserPromptSubmit` hook |
| Full session transcripts (prompts, responses, tool calls) | `~/helios-ai-log/transcripts/` | no | `SessionEnd` / `SessionStart` hooks |
| SHA-256 of every archived transcript | `transcripts-manifest.csv` | yes | same hooks |

Capture code: `.claude/hooks/ai_log.py`, wired up in `.claude/settings.json`.

## Commit hashes before 2026-10-31

The repository history was rewritten on 2026-10-31, before it went public:
personal details were removed and commit messages were reworded. Every commit
got a new hash, so hashes cited in log entries (and code comments) written
before that date are pre-rewrite hashes. To find the commit today, look up the
old full hash in `history-rewrite-2026-10-31-commit-map.txt`, which lists old
full hash and new full hash on each line.

## One change, several hashes

A change built on one branch is often cherry-picked onto a deploy branch, so
the same diff runs under a different hash from the one a log entry cites.
`patch-id-map.csv` groups commits by `git patch-id --stable` (same diff, same
id), one row per commit, with the full 40-character hash in `commit`.
`in_live` = yes means the commit was an ancestor of the live checkout's HEAD
when the map was generated. It does not say which hash was running at any
earlier time: a later deploy can bring in either hash of a pair. The
generation time and the full live HEAD hash used are in the sidecar
`patch-id-map.meta.txt` (a sidecar rather than a comment line, so CSV readers
see a plain header row). Regenerate both with `python scripts/patch_id_map.py`.

Limits:
- A cherry-pick that needed conflict resolution or was amended gets a
  different patch id and isn't paired, so a missing row doesn't prove a
  commit was never re-hashed.
- A group with no "yes" is common: no hash of that change was in the live
  history at generation time (not deployed yet, or deployed under a hash that
  isn't paired).
- Hashes from before the 2026-10-31 history rewrite are in no ref, so they
  never appear here. A log entry citing one maps to its new hash through
  `history-rewrite-2026-10-31-commit-map.txt`.
- Squash commits on GitHub main bundle many changes and never pair with dev
  commits; trace those through `dev-squash-map.txt`.

## "Professor"

In these logs and in code comments, "professor" means an AI reviewer agent
running on the same machine that audits the code and the data. It is not a
person. Labels such as "professor M112" or "professor C13" refer to numbered
findings from that agent.

## Why transcripts stay out of git

A transcript holds everything shown in a session: command output, config
contents, possibly keys or account data. The repository is going public, so
the transcripts stay in a local archive and only their hashes are committed.
The manifest is what makes them evidence. Each row is committed with a
timestamp, so the matching archived file can later be shown to have existed,
unchanged, by that date. To verify any row:

```
sha256sum ~/helios-ai-log/<archive_file>    # must equal the row's sha256
```

Archive files are content-addressed (`<session>.<sha12>.jsonl`) and never
overwritten. A resumed session gets a new snapshot and a new row, and earlier
rows keep matching their own files.

`raw-prompts.jsonl` is gitignored for the same reason. Prompts can include
pasted terminal output. The verbatim prompts are also inside every archived
transcript, so they are covered by the manifest hashes either way.

## Backfill

Sessions from before this log existed were archived with:

```
python3 .claude/hooks/ai_log.py backfill
```

Those manifest rows have `event = backfill`. All work before 2027 is
documented prior work.

## Limits

- Each machine archives its own sessions. Sessions on the Windows PC go to
  that machine's `~/helios-ai-log/` and add their own manifest rows.
- Claude Code deletes old local transcripts after `cleanupPeriodDays` (30 by
  default). The archive copies are unaffected, but a session that never
  triggered any hook before cleanup would be lost. The `SessionStart` sweep
  makes that window small.
