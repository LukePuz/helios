"""Report on, and optionally replay, rows in logs/db_write_spill.jsonl.

db._spill writes one JSON line per failed DB write (writer, kind, uuid, env,
at, error, record). This script reads them and, for each line, checks whether
the row already exists in the DB: by the replay ledger (uuids this script
already wrote) or by a natural key within --window-minutes of the line's time.

Default run is read-only: it prints counts per writer and one status line per
record (line number, writer, id, status). It never prints record values.

--apply inserts the missing rows of the replayable kinds (trade, signal,
regime, activity) with their original time, one transaction per row, then
adds the line's id to the ledger (<spill>.replayed) and writes one
activity_feed row (stage spill_replay) naming the kind, the id and the new
row id. Re-running is a no-op for anything already in the ledger or the DB.
Orders and approvals are reported only: an inserted 'submitted' order or a
PENDING approval could be picked up by reconciliation or execution.

Never run automatically. Whether and when to apply is a human decision.

    venv/bin/python scripts/replay_spill.py              # dry run
    venv/bin/python scripts/replay_spill.py --apply      # write missing rows
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db  # noqa: E402

KIND_BY_WRITER = {"log_trade": "trade", "log_signal": "signal", "log_order": "order",
                  "log_regime": "regime", "log_pending_approval": "approval",
                  "log_activity": "activity"}
REPLAYABLE = {"trade", "signal", "regime", "activity"}

# (table, time column, [(column, record key)], env column or None)
NATURAL_KEYS = {
    "trade":    ("trades", "timestamp",
                 [("symbol", "symbol"), ("side", "side"), ("quantity", "quantity"),
                  ("reason", "reason")], "trading_env"),
    "signal":   ("signals", "timestamp",
                 [("symbol", "symbol"), ("signal", "signal"), ("reasoning", "reasoning")],
                 "trading_env"),
    "regime":   ("market_regime_log", "timestamp",
                 [("regime", "regime"), ("score", "score"), ("regime_type", "regime_type"),
                  ("sector_etf", "sector_etf")], None),
    "activity": ("activity_feed", "ts",
                 [("stage", "stage"), ("ticker", "ticker"), ("message", "message")], None),
    "order":    ("orders", "created_at",
                 [("ticker", "ticker"), ("side", "side"), ("qty", "qty"),
                  ("broker_order_id", "broker_order_id")], "trading_env"),
    "approval": ("pending_approvals", "timestamp",
                 [("ticker", "ticker"), ("action", "action")], "trading_env"),
}


def load_spill(path: Path) -> list[tuple[int, dict | None]]:
    """(line number, parsed line or None when it isn't valid JSON)."""
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append((n, json.loads(raw) | {"_raw_sha": hashlib.sha256(raw.encode()).hexdigest()}))
            except (ValueError, TypeError):
                out.append((n, None))
    return out


def record_id(line: dict) -> str:
    """The line's uuid, or a hash of the raw line for lines spilled before
    uuids were added (stable across runs, so the ledger still works)."""
    return line.get("uuid") or "sha256:" + line["_raw_sha"][:32]


def kind_of(line: dict) -> str | None:
    return line.get("kind") or KIND_BY_WRITER.get(line.get("writer"))


def env_of(line: dict) -> str | None:
    rec = line.get("record") or {}
    return rec.get("trading_env") or line.get("env")


def load_ledger(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def exists_in_db(cur, line: dict, window_minutes: int) -> bool | None:
    """True if a row with the line's natural key sits within the window of
    its time; None when the kind has no key or the line lacks a time."""
    kind = kind_of(line)
    if kind not in NATURAL_KEYS or not line.get("at"):
        return None
    table, tcol, cols, env_col = NATURAL_KEYS[kind]
    rec = line.get("record") or {}
    if kind == "activity":
        rec = {**rec, "ticker": rec.get("ticker") or ""}
    where, params = [], []
    for col, key in cols:
        where.append(f"{col} IS NOT DISTINCT FROM %s")
        params.append(rec.get(key))
    env = env_of(line)
    if env_col and env:
        where.append(f"{env_col} = %s")
        params.append(env)
    where.append(f"{tcol} BETWEEN %s::timestamptz - make_interval(mins => %s) "
                 f"AND %s::timestamptz + make_interval(mins => %s)")
    params += [line["at"], window_minutes, line["at"], window_minutes]
    cur.execute(f"SELECT 1 FROM {table} WHERE {' AND '.join(where)} LIMIT 1", params)
    return cur.fetchone() is not None


def _json(v):
    return db.psycopg2.extras.Json(v, dumps=lambda o: json.dumps(o, default=str)) if v is not None else None


def insert_row(cur, line: dict) -> int:
    """Insert one replayable line with its original time; returns the new id."""
    kind, rec, at, env = kind_of(line), line.get("record") or {}, line["at"], env_of(line)
    if kind == "trade":
        cur.execute("INSERT INTO trades (symbol, trading_env, side, quantity, price, reason, "
                    "price_source, timestamp) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::timestamptz) RETURNING id",
                    (rec.get("symbol"), env, rec.get("side"), rec.get("quantity"), rec.get("price"),
                     rec.get("reason"), rec.get("price_source"), at))
    elif kind == "signal":
        cur.execute("INSERT INTO signals (symbol, signal, confidence, reasoning, trading_env, model, "
                    "provenance, decision, timestamp) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::timestamptz) "
                    "RETURNING id",
                    (rec.get("symbol"), rec.get("signal"), rec.get("confidence"), rec.get("reasoning"),
                     env, rec.get("model"), _json(rec.get("provenance")), _json(rec.get("decision")), at))
    elif kind == "regime":
        cur.execute("INSERT INTO market_regime_log (regime, score, confidence, signals_json, "
                    "portfolio_impact_json, reasoning, regime_type, sector_etf, timestamp) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::timestamptz) RETURNING id",
                    (rec.get("regime"), rec.get("score"), rec.get("confidence"),
                     _json(rec.get("signals") or {}), _json(rec.get("portfolio_impact") or {}),
                     rec.get("reasoning"), rec.get("regime_type"), rec.get("sector_etf"), at))
    elif kind == "activity":
        cur.execute("INSERT INTO activity_feed (stage, ticker, message, sentiment, extra_json, ts) "
                    "VALUES (%s,%s,%s,%s,%s,%s::timestamptz) RETURNING id",
                    (rec.get("stage"), rec.get("ticker") or "", rec.get("message"),
                     rec.get("sentiment") or "neutral", _json(rec.get("extra")), at))
    else:
        raise ValueError(f"kind {kind!r} is not replayable")
    return cur.fetchone()[0]


def classify(lines, ledger: set[str], cur, window_minutes: int) -> list[dict]:
    """One status per line: in_ledger, exists, missing, missing_env,
    report_only, unparsable or unknown_kind."""
    out = []
    for n, line in lines:
        if line is None:
            out.append({"line": n, "writer": "?", "id": None, "status": "unparsable"})
            continue
        rid, kind = record_id(line), kind_of(line)
        row = {"line": n, "writer": line.get("writer") or "?", "id": rid, "kind": kind, "obj": line}
        if rid in ledger:
            row["status"] = "in_ledger"
        elif kind not in NATURAL_KEYS:
            row["status"] = "unknown_kind"
        elif exists_in_db(cur, line, window_minutes):
            row["status"] = "exists"
        elif kind not in REPLAYABLE:
            row["status"] = "report_only"
        elif NATURAL_KEYS[kind][3] and not env_of(line):
            row["status"] = "missing_env"     # can't tell paper from live: not replayed
        else:
            row["status"] = "missing"
        out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spill", type=Path, default=None, help="spill file (default: db._SPILL_PATH)")
    ap.add_argument("--apply", action="store_true", help="insert missing replayable rows")
    ap.add_argument("--window-minutes", type=int, default=30,
                    help="natural-key match window around each line's time (default 30)")
    args = ap.parse_args(argv)
    spill = args.spill or db._SPILL_PATH
    ledger_path = spill.with_name(spill.name + ".replayed")

    lines = load_spill(spill)
    if not lines:
        print(f"no spill lines in {spill}")
        return 0
    ledger = load_ledger(ledger_path)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            rows = classify(lines, ledger, cur, args.window_minutes)
        conn.rollback()

        per_writer: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            per_writer[r["writer"]][r["status"]] += 1
        print(f"{len(rows)} spill line(s) in {spill}")
        for writer in sorted(per_writer):
            c = per_writer[writer]
            print(f"  {writer}: {sum(c.values())} total, "
                  + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
        for r in rows:
            print(f"  line {r['line']}: {r['writer']} {r['id'] or '-'} {r['status']}")

        todo = [r for r in rows if r["status"] == "missing"]
        if not args.apply:
            print(f"dry run: {len(todo)} row(s) would be written with --apply")
            return 0
        written = 0
        for r in todo:
            try:
                with conn.cursor() as cur:
                    # Re-check inside the write: another run may have written it.
                    if exists_in_db(cur, r["obj"], args.window_minutes):
                        conn.rollback()
                        continue
                    new_id = insert_row(cur, r["obj"])
                    cur.execute("INSERT INTO activity_feed (stage, ticker, message, sentiment, extra_json) "
                                "VALUES (%s,%s,%s,%s,%s)",
                                ("spill_replay", "", f"replayed {r['kind']} row from the spill file",
                                 "neutral", json.dumps({"kind": r["kind"], "spill_id": r["id"],
                                                        "row_id": new_id, "spilled_at": r["obj"].get("at"),
                                                        "spill_line": r["line"]})))
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"  line {r['line']}: replay failed: {type(e).__name__}")
                continue
            with open(ledger_path, "a") as f:
                f.write(r["id"] + "\n")
            written += 1
            print(f"  line {r['line']}: written ({r['kind']})")
        print(f"applied: {written} of {len(todo)} row(s) written")
        return 0
    finally:
        db._put_conn(conn)


if __name__ == "__main__":
    sys.exit(main())
