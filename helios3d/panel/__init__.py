"""Many perspectives (Phase 2, shadow only): separate fundamentals,
technical/flow, macro/regime and news/sentiment analysts, each returning a
score, a confidence and "what would change my mind"; a PM layer that writes
the thesis and records agreement versus disagreement; and a red-team pass
that argues against every would-be buy. Runs in its own shadow worker that
yields to the live LLM lane.

Modules: config (settings, compute budget), roles (data slices and
prompts), schemas (output schemas and parsers), llm (injectable client),
gate (live-lane gate and kill file), runner (tails h3d.decision_record,
writes h3d.panel_output and h3d.pm_verdict), timing (72B call budgets).
CLI: python -m helios3d.panel run (dry-run by default).
"""
