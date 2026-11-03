"""CLI: python -m helios3d.panel run [--apply] [--limit N] [--from-signal-id ID] [--cursor NAME]
       python -m helios3d.panel gate

run: tails h3d.decision_record and runs the shadow panel (helios3d.panel.runner).
Dry-run by default: reads records and features, renders prompts, prints
what would run, calls no model and writes nothing. --apply calls the model
at HELIOS3D_OLLAMA_URL and writes h3d.panel_output, h3d.pm_verdict and
h3d.panel_cursor; it refuses to start while HELIOS3D_PANEL_BUDGET is off.

gate: prints one live-lane gate check (kill file and the live LLM queues,
read only) and exits 0 when clear, 3 when not.

timing (python -m helios3d.panel.timing) measures 72B call times from logs.
"""
import argparse
import json
import logging
import sys

from helios3d.panel.config import SettingsError, from_env


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m helios3d.panel",
                                description="Helios 3D shadow analyst panel.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the panel over new decision records")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="read and render only (default)")
    g.add_argument("--apply", action="store_true", help="call the model and write h3d rows")
    r.add_argument("--limit", type=int, default=0,
                   help="stop after N panels (dry-run: N records); 0 = no limit")
    r.add_argument("--from-signal-id", type=int, default=None,
                   help="start after this signal id instead of the cursor (no lookback)")
    r.add_argument("--cursor", default="panel", help="cursor name in h3d.panel_cursor")
    r.add_argument("--rerun-config", action="store_true",
                   help="re-panel decisions panelled under another config, after "
                        "--from-signal-id (default 0)")
    r.add_argument("--retry-failed", action="store_true",
                   help="read every decision with no successful pm_verdict under any "
                        "config (write-failure rows, skip markers and never-panelled ones), "
                        "after --from-signal-id (default 0)")
    sub.add_parser("gate", help="one live-lane gate check, read only")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        settings = from_env()
        if args.cmd == "gate":
            from helios3d.panel.gate import LiveLaneGate
            res = LiveLaneGate(settings).check()
            print(json.dumps({"ok": res.ok, "reason": res.reason, **res.detail}, sort_keys=True))
            return 0 if res.ok else 3
        from helios3d.panel import runner
        summary = runner.run(apply=bool(args.apply), limit=args.limit,
                             from_signal_id=args.from_signal_id, cursor_name=args.cursor,
                             settings=settings, rerun_config=args.rerun_config,
                             retry_failed=args.retry_failed)
    except SettingsError as e:
        print(f"settings: {e}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
