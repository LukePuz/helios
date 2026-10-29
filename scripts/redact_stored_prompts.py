"""One-off backfill for audit 14:53 A1453-3: redact account dollar figures
from prompts already stored in signals.prompt. Uses the same redaction as
new rows (agents.analyst.redact_account_figures). Prints counts only.

    python scripts/redact_stored_prompts.py          # dry run
    python scripts/redact_stored_prompts.py --apply
"""
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

import db  # noqa: E402
from agents.analyst import redact_account_figures  # noqa: E402


def main(apply: bool) -> None:
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, prompt FROM signals WHERE prompt IS NOT NULL")
            rows = cur.fetchall()
        changed = [(rid, redact_account_figures(p)) for rid, p in rows
                   if redact_account_figures(p) != p]
        print(f"{len(rows)} stored prompts, {len(changed)} need redaction")
        if apply and changed:
            with conn.cursor() as cur:
                cur.executemany("UPDATE signals SET prompt=%s WHERE id=%s",
                                [(p, rid) for rid, p in changed])
            conn.commit()
            print(f"redacted {len(changed)}")
    finally:
        db._put_conn(conn)


if __name__ == "__main__":
    main("--apply" in sys.argv)
