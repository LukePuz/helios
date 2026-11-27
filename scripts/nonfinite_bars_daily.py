"""Daily non-finite bar line (eng-candidates-2026-11-26 sweep #9). A record,
not a gate: it always exits 0, and nothing in the bot reads it.

It runs batch_verify's nonfinite_bars check outside deploy days, by import:
the same SQL (batch_verify.nonfinite_bars_sql: a NaN or +/-Infinity open,
high, low or close within each ticker's last NONFINITE_BAR_ROWS (260) rows;
NULL is not counted), the same read (batch_verify.fetch_nonfinite_bars) and
the same WARN text (batch_verify.check_nonfinite_bars). No check is defined
here. scripts/_bar_qa.py is the Python-side form of the same rule for bars
the readouts already hold.

It prints one line:

  nonfinite_bars <YYYY-MM-DD HH:MM> ET <OK|WARN|SKIP>: <text>

where the WARN text is batch_verify's, e.g. "2 non-finite OHLC bar(s) in 1
ticker(s) within each ticker's last 260 rows: SNDK 2 (2026-11-19,
2026-11-20)". SKIP is a guard below; a failed read is batch_verify's WARN
"price_history read failed (<error class>)". No price is printed.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \\
        <path>/scripts/nonfinite_bars_daily.py [--out FILE]

--out appends the line to FILE as well as printing it. db is imported from
the working directory's clone (run it with env -C on the clone whose DB
settings it should use). Read-only: SUPABASE_PORT must be 6543 (the
transaction pooler); batch_verify's read is one SET TRANSACTION READ ONLY
transaction with a statement timeout, rolled back. It does not read between
04:50 and 05:12 ET (the morning restart window).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
REQUIRED_PORT = "6543"
QUIET = (time(4, 50), time(5, 12))     # ET: no DB reads in this window
BATCH_VERIFY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_verify.py")


def in_quiet_window(now: datetime) -> bool:
    t = now.astimezone(NY).time()
    return QUIET[0] <= t < QUIET[1]


def _import_db():
    """db from the working directory's clone, checked."""
    cwd = os.path.abspath(os.getcwd())
    sys.path[:] = [cwd] + [p for p in sys.path if os.path.abspath(p or ".") != cwd]
    import db
    if os.path.dirname(os.path.abspath(db.__file__)) != cwd:
        raise RuntimeError(f"db imported from {db.__file__}, not the working directory {cwd}")
    return db


def _load_batch_verify():
    """batch_verify from this script's directory. It puts its own clone first
    on sys.path at import; sys.path is restored, and db is already loaded
    from the working directory, so its lazy `import db` reuses that one."""
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("_batch_verify_nonfinite_daily",
                                                      BATCH_VERIFY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = saved


def line(now: datetime, status: str, text: str) -> str:
    return f"nonfinite_bars {now.astimezone(NY):%Y-%m-%d %H:%M} ET {status}: {text}"


def run(now: datetime) -> str:
    """The one line. Every guard and failure is a SKIP line, never a raise."""
    if os.getenv("SUPABASE_PORT", "") != REQUIRED_PORT:
        return line(now, "SKIP", f"SUPABASE_PORT must be {REQUIRED_PORT} (the transaction "
                                 "pooler); not reading")
    if in_quiet_window(now):
        return line(now, "SKIP", "04:50-05:12 ET is the restart window; not reading")
    try:
        _import_db()
    except Exception as e:
        return line(now, "SKIP", f"db import failed ({type(e).__name__})")
    try:
        bv = _load_batch_verify()
        chk = bv.check_nonfinite_bars(bv.fetch_nonfinite_bars(bv.NONFINITE_BAR_ROWS),
                                      bv.NONFINITE_BAR_ROWS)
    except Exception as e:
        return line(now, "SKIP", f"batch_verify check failed ({type(e).__name__})")
    return line(now, chk.get("status", "?"), chk.get("why", ""))


def main(argv=None, now: datetime | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="also append the line to this file")
    a = ap.parse_args(argv)
    text = run(now or datetime.now(timezone.utc))
    print(text)
    if a.out:
        try:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError as e:
            print(f"could not append to {a.out} ({type(e).__name__})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
