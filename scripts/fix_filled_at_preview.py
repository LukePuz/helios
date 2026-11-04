"""Preview the SQL that would correct fill times stored a TWS-zone offset late.

TWS can send an execution time as a UTC wall clock with no zone; ib_async
labels it with the TWS zone (the host's local zone when ib.TimezoneTWS is
unset), so a 20:07Z fill was stored as 01:07Z next day. Fixed in
ibkr._exec_time; this covers rows written before the fix.

    env -C ~/Desktop/trading_bot venv/bin/python scripts/fix_filled_at_preview.py [--tz America/New_York]

Prints SQL only. It connects to nothing and has no apply mode: review the
SELECTs' output, then run the UPDATEs by hand if they're right.

Rule, per row: fixed = (t AT TIME ZONE tz) AT TIME ZONE 'UTC' reads the
stored instant's tz wall clock back as UTC, which subtracts the zone's offset
at that time (5 h standard, 4 h daylight). A row is a candidate when t minus
its recording time is within 0.1 h of that offset (4.9-5.1 h or 3.9-4.1 h):
  orders.filled_at          vs orders.created_at
  trade_outcomes.exit_fill_at vs trade_outcomes.exit_date (set when the close was recorded)
A late close recorded hours after its fill falls outside the band and is
not proposed; those need a look by hand.
"""
from __future__ import annotations

import argparse
import re

BAND = "interval '6 minutes'"   # 0.1 h either side of the offset

TARGETS = (
    # table, fill-time column, recording-time column
    ("orders", "filled_at", "created_at"),
    ("trade_outcomes", "exit_fill_at", "exit_date"),
)


def _fixed(col: str, tz: str) -> str:
    return f"(({col} AT TIME ZONE '{tz}') AT TIME ZONE 'UTC')"


def _where(col: str, rec: str, tz: str) -> str:
    shift = f"({col} - {_fixed(col, tz)})"
    return (f"{col} IS NOT NULL AND {rec} IS NOT NULL\n"
            f"  AND {shift} IN (interval '4 hours', interval '5 hours')\n"
            f"  AND ({col} - {rec}) BETWEEN {shift} - {BAND} AND {shift} + {BAND}")


def build_sql(tz: str) -> str:
    out = [f"-- TWS zone: {tz}. Preview only; nothing here has been run.", ""]
    for table, col, rec in TARGETS:
        out += [f"-- {table}.{col}: candidates",
                f"SELECT id, trading_env, {rec}, {col} AS stored, {_fixed(col, tz)} AS corrected,\n"
                f"       {col} - {rec} AS stored_lag\n"
                f"FROM {table}\nWHERE {_where(col, rec, tz)}\nORDER BY id;", ""]
    out += ["-- The correction, to run by hand after checking the rows above:", "BEGIN;"]
    for table, col, rec in TARGETS:
        out += [f"UPDATE {table} SET {col} = {_fixed(col, tz)}\nWHERE {_where(col, rec, tz)};"]
    out += ["-- COMMIT;   -- or ROLLBACK;", ""]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tz", default="America/New_York",
                    help="zone ib_async labelled the times with (TimezoneTWS, else the host zone)")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z_]+(/[A-Za-z_+-]+)*", args.tz):
        ap.error(f"not a zone name: {args.tz!r}")
    print(build_sql(args.tz))


if __name__ == "__main__":
    main()
