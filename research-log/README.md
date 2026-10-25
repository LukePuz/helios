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
