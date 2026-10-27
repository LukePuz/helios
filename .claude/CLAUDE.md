# AI prompt protocol (ISEF / Regeneron STS)

Helios is a pre-registered study headed for ISEF and Regeneron STS. ISEF Rule 8
permits AI-written code only with explicit citation of what the AI generated,
and Form 2A requires an AI Prompt Log of prompts and responses. This file is
committed so the protocol applies on every machine that works on this repo.
Personal notes go in the root `CLAUDE.md`, which is gitignored.

Raw prompts and full transcripts are captured automatically by the hooks in
`.claude/settings.json` (see `research-log/README.md`). The steps below cover
the part a hook can't do: refining each request and writing the curated log.

## 1. Refine every new request before acting

For each new request from Luke, write a refined prompt that makes his intent
precise, show it in one short block, then act on the refined version:

```
Refined prompt (AI-refined, Claude):
Goal: ...
Scope / files: ...
Constraints: ...
Out of scope (do not touch): ...
Acceptance check: ...
Open questions for Luke: ... (or "none")
```

The refinement clarifies. It never decides research questions. Never fill in
any of these. List each one under "Open questions for Luke" and wait for
his answer before acting on that part:

- hypotheses, or what counts as support for one
- metrics, scoring, or evaluation criteria
- the sample, universe, date range, or inclusion/exclusion rules
- analysis or statistical choices
- anything that would change or contradict the OSF pre-registration

Short follow-ups inside an ongoing request ("yes", "go", "do the second one")
continue the current request. They don't need a new refinement. The hooks
still log them verbatim.

## 2. Log each request in the curated log

The log covers work that **edits the codebase**, because that's what ISEF
Rule 8 citation is for (Luke, 2026-10-27). Health checks, status answers, data
collection or organization, notes, drafts and graphs don't get an entry; their
own run reports cover them. Headless runs (the agent-bus status worker, the
research steward) run with hooks off and no saved transcript, so the automatic
capture skips them too.

When a request that edits the codebase is finished, add one entry to
`research-log/prompts/YYYY-MM.md` (the month the request started). Create the
file if it doesn't exist. Use this format:

```
## YYYY-MM-DD HH:MM TZ: <short title>

**Luke (original, verbatim):**
> <his exact words; quote multi-line prompts line by line>

**AI-refined (Claude):**
> <the refined prompt block shown to him>

**Done:** <1-3 lines on what was actually done>

**Files changed:** `<path>` (AI-generated), `<path>` (AI-edited: <which part>),
or "none"

**Commit:** `<short hash>`, or "none (no code change)"
```

Rules for the entry:

- Always include the original next to the refined version, labeled as above.
  The refined text is AI-written, and the log must show who wrote what.
- If the request text was drafted by another AI (e.g. Jarvis on the PC) and
  Luke pasted it, don't label it "Luke (original)". Name who drafted it and
  that Luke pasted it. Ask him if it's unclear.
- Name AI-generated code down to the file, or the function when only part of
  a file was AI-written. That is the citation Rule 8 requires.
- Commit the work first, then write the entry with that commit's hash, then
  commit the log entry separately (`research-log: <title>`). A file can't
  contain the hash of the commit it lives in.
- Stage `research-log/transcripts-manifest.csv` in every `research-log:`
  commit. The session hooks append rows to it, and committing them is what
  timestamps the hashes.
- Never put secrets, API keys, account numbers, or balances in the log. If the
  original prompt contained one, replace it with `[REDACTED: <kind>]` and say
  so in the entry. The verbatim original stays in the hook capture.
