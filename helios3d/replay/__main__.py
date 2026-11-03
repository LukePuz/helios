"""CLI: python -m helios3d.replay --start YYYY-MM-DD --end YYYY-MM-DD

Replays every decision in the range (ET calendar days, inclusive) in LLM
stub mode (the stored model output) and writes a per-decision diff report
as JSON (default logs/helios3d/replay_<start>_<end>.json). Read-only: it
writes no database row. --sizing also reruns sizing from the nearest
portfolio snapshot (reads account values; the report holds share counts and
reason codes only).
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

from helios3d.replay import harness


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m helios3d.replay",
                                description="Replay recorded decisions from stored model output.")
    p.add_argument("--start", required=True, type=date.fromisoformat, help="first ET day (YYYY-MM-DD)")
    p.add_argument("--end", required=True, type=date.fromisoformat, help="last ET day (YYYY-MM-DD)")
    p.add_argument("--out", type=Path, default=None, help="report path (JSON)")
    p.add_argument("--llm", choices=[harness.LLM_MODE], default=harness.LLM_MODE,
                   help="model leg: stub = the stored model output (the only mode)")
    p.add_argument("--sizing", action="store_true",
                   help="rerun sizing from portfolio_snapshots (reads account values)")
    args = p.parse_args(argv)
    if args.end < args.start:
        p.error("--end is before --start")
    from helios3d.schema.conn import connection
    with connection() as conn:
        report = harness.run(args.start, args.end, harness.DbSource(conn), sizing=args.sizing)
    out = args.out or harness.default_out(args.start, args.end)
    sha = harness.write_report(report, out)
    print(json.dumps({"out": str(out), "report_sha256_16": sha, **report["summary"]},
                     sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
