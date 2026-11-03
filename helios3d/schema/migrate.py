"""Migration runner for the Helios 3D schema ``h3d``.

Separate from db.py's ``_MIGRATIONS`` list and its public.schema_version: this
runner creates ``h3d`` and ``h3d.schema_version`` and applies numbered
migrations from MIGRATIONS in order, one transaction each. Every statement is
idempotent (IF NOT EXISTS) and every object name is qualified with ``h3d.``;
each migration also runs under ``SET LOCAL search_path TO h3d`` so an
unqualified name could not land in public. tests/test_helios3d_schema.py
checks the qualification rule statically.

CLI (dry-run is the default and touches nothing but a read):
    python -m helios3d.schema.migrate            # list pending migrations
    python -m helios3d.schema.migrate --apply    # apply them
"""
import argparse
import logging
import sys

SCHEMA = "h3d"

BOOTSTRAP_SQL = (
    "CREATE SCHEMA IF NOT EXISTS h3d;",
    "CREATE TABLE IF NOT EXISTS h3d.schema_version ("
    " version INTEGER PRIMARY KEY,"
    " name TEXT NOT NULL,"
    " applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW());",
)

# (version, name, sql). Append only; never edit or renumber an applied entry.
MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "verdict_labels", """
CREATE TABLE IF NOT EXISTS h3d.verdict_labels (
    id              BIGSERIAL PRIMARY KEY,
    signal_id       INTEGER NOT NULL,
    ticker          VARCHAR(16) NOT NULL,
    verdict_at      TIMESTAMPTZ NOT NULL,
    action          VARCHAR(30) NOT NULL,
    conviction      NUMERIC,
    horizon_days    INTEGER NOT NULL,
    entry_ref_price NUMERIC,
    fwd_price       NUMERIC,
    fwd_return      NUMERIC,
    spy_fwd_return  NUMERIC,
    excess_return   NUMERIC,
    label_source    VARCHAR(60) NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT verdict_labels_signal_horizon_uq UNIQUE (signal_id, horizon_days)
);
CREATE INDEX IF NOT EXISTS verdict_labels_ticker_idx ON h3d.verdict_labels (ticker, verdict_at);
"""),
    # Provenance: the two sessions a label spans, the
    # session close time the entry used (13:00 on a half day), the code that
    # computed it and a hash of the four closes it used. Pairs that could not
    # be labelled are kept with their reason, and a label whose input closes
    # change on a re-run is copied to the history table before it is replaced.
    (2, "label_provenance", """
ALTER TABLE h3d.verdict_labels ADD COLUMN IF NOT EXISTS entry_date DATE,
    ADD COLUMN IF NOT EXISTS fwd_date DATE,
    ADD COLUMN IF NOT EXISTS entry_close_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS code_sha VARCHAR(40),
    ADD COLUMN IF NOT EXISTS price_snapshot_sha VARCHAR(64);
CREATE TABLE IF NOT EXISTS h3d.label_skips (
    id              BIGSERIAL PRIMARY KEY,
    signal_id       INTEGER NOT NULL,
    ticker          VARCHAR(16) NOT NULL,
    verdict_at      TIMESTAMPTZ NOT NULL,
    horizon_days    INTEGER NOT NULL,
    reason          VARCHAR(40) NOT NULL,
    entry_date      DATE,
    fwd_date        DATE,
    label_source    VARCHAR(60) NOT NULL,
    code_sha        VARCHAR(40),
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT label_skips_signal_horizon_uq UNIQUE (signal_id, horizon_days)
);
CREATE TABLE IF NOT EXISTS h3d.verdict_label_history (
    id                 BIGSERIAL PRIMARY KEY,
    signal_id          INTEGER NOT NULL,
    ticker             VARCHAR(16) NOT NULL,
    verdict_at         TIMESTAMPTZ NOT NULL,
    action             VARCHAR(30) NOT NULL,
    conviction         NUMERIC,
    horizon_days       INTEGER NOT NULL,
    entry_ref_price    NUMERIC,
    fwd_price          NUMERIC,
    fwd_return         NUMERIC,
    spy_fwd_return     NUMERIC,
    excess_return      NUMERIC,
    label_source       VARCHAR(60) NOT NULL,
    entry_date         DATE,
    fwd_date           DATE,
    entry_close_at     TIMESTAMPTZ,
    code_sha           VARCHAR(40),
    price_snapshot_sha VARCHAR(64),
    computed_at        TIMESTAMPTZ NOT NULL,
    superseded_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS verdict_label_history_signal_idx
    ON h3d.verdict_label_history (signal_id, horizon_days);
"""),
    # Append-only point-in-time copy of public.price_history, which is
    # upserted with no recorded_at. helios3d.features.capture inserts a row
    # only when a (ticker, date) bar is new or differs from its latest
    # capture; captured_at is when that value was seen. Nothing updates or
    # deletes rows here.
    (3, "price_history_asof", """
CREATE TABLE IF NOT EXISTS h3d.price_history_asof (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(16) NOT NULL,
    date            DATE NOT NULL,
    open            NUMERIC,
    high            NUMERIC,
    low             NUMERIC,
    close           NUMERIC,
    volume          BIGINT,
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    code_sha        VARCHAR(40)
);
CREATE INDEX IF NOT EXISTS price_history_asof_key_idx
    ON h3d.price_history_asof (ticker, date, captured_at DESC);
CREATE INDEX IF NOT EXISTS price_history_asof_captured_idx
    ON h3d.price_history_asof (captured_at);
"""),
    # One decision record per analyst verdict (helios3d.record.decision),
    # built by tailing public.signals. decision_id is a UUIDv5 of the
    # signal id, so a rebuild gives the same id. Links to the public tables
    # are plain columns (no FOREIGN KEY); `links` says how each was matched,
    # or why it was left NULL. Never holds the prompt text or account
    # figures: those stay in public.signals under signal_id.
    (4, "decision_record", """
CREATE TABLE IF NOT EXISTS h3d.decision_record (
    decision_id         UUID PRIMARY KEY,
    signal_id           INTEGER NOT NULL,
    as_of_ts            TIMESTAMPTZ NOT NULL,
    ticker              VARCHAR(16) NOT NULL,
    trading_env         VARCHAR(10),
    trigger             VARCHAR(40),
    action              VARCHAR(30) NOT NULL,
    conviction          NUMERIC,
    code_sha            VARCHAR(48),
    head_sha            VARCHAR(48),
    prompt_template_sha VARCHAR(64),
    brief_template_sha  VARCHAR(64),
    prompt_sha          VARCHAR(64),
    model               VARCHAR(80),
    model_digest        VARCHAR(80),
    regime_row_id       INTEGER,
    approval_id         INTEGER,
    order_ids           INTEGER[] NOT NULL DEFAULT '{}',
    trade_outcome_ids   INTEGER[] NOT NULL DEFAULT '{}',
    trade_plan_id       INTEGER,
    trade_plan_new_id   INTEGER,
    links               JSONB NOT NULL,
    inputs              JSONB NOT NULL,
    content_sha         VARCHAR(16) NOT NULL,
    built_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    builder_code_sha    VARCHAR(48),
    CONSTRAINT decision_record_signal_uq UNIQUE (signal_id)
);
CREATE INDEX IF NOT EXISTS decision_record_ticker_idx ON h3d.decision_record (ticker, as_of_ts);
CREATE INDEX IF NOT EXISTS decision_record_approval_idx ON h3d.decision_record (approval_id);
"""),
    # Shadow analyst panel (helios3d.panel). One panel_output row per role
    # call (four analysts and the red team), one pm_verdict row per decision,
    # each unique per (decision, panel_config_sha) so a re-run with the same
    # templates, models and call options writes nothing. decision_id links to
    # h3d.decision_record (plain column, no FOREIGN KEY, like the other
    # links). No prompt text and no account figures: prompt_sha and the
    # scrubbed data slice reproduce a prompt. panel_cursor holds the runner's
    # cursor (last signal_id handled) per cursor name.
    (5, "panel", """
CREATE TABLE IF NOT EXISTS h3d.panel_output (
    id                BIGSERIAL PRIMARY KEY,
    decision_id       UUID NOT NULL,
    signal_id         INTEGER NOT NULL,
    ticker            VARCHAR(16) NOT NULL,
    as_of_ts          TIMESTAMPTZ NOT NULL,
    role              VARCHAR(24) NOT NULL,
    model             VARCHAR(80),
    panel_config_sha  VARCHAR(16) NOT NULL,
    templates_sha     VARCHAR(16) NOT NULL,
    prompt_sha        VARCHAR(64),
    data_slice        JSONB NOT NULL,
    output_json       JSONB,
    score             NUMERIC,
    confidence        NUMERIC,
    parse_ok          BOOLEAN NOT NULL,
    error             VARCHAR(200),
    latency_s         NUMERIC,
    prompt_tokens     INTEGER,
    output_tokens     INTEGER,
    done_reason       VARCHAR(20),
    panel_code_sha    VARCHAR(48),
    budget            VARCHAR(16),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT panel_output_decision_role_cfg_uq UNIQUE (decision_id, role, panel_config_sha)
);
CREATE INDEX IF NOT EXISTS panel_output_ticker_idx ON h3d.panel_output (ticker, as_of_ts);
CREATE INDEX IF NOT EXISTS panel_output_signal_idx ON h3d.panel_output (signal_id);
CREATE TABLE IF NOT EXISTS h3d.pm_verdict (
    id                     BIGSERIAL PRIMARY KEY,
    decision_id            UUID NOT NULL,
    signal_id              INTEGER NOT NULL,
    ticker                 VARCHAR(16) NOT NULL,
    as_of_ts               TIMESTAMPTZ NOT NULL,
    live_action            VARCHAR(30),
    live_conviction        NUMERIC,
    model                  VARCHAR(80),
    panel_config_sha       VARCHAR(16) NOT NULL,
    templates_sha          VARCHAR(16) NOT NULL,
    prompt_sha             VARCHAR(64),
    thesis                 TEXT,
    agreement              VARCHAR(16),
    agreement_rationale    TEXT,
    agreement_facts        JSONB NOT NULL,
    verdict                VARCHAR(16),
    conviction             NUMERIC,
    plan                   JSONB,
    plan_issues            JSONB,
    red_team_ran           BOOLEAN NOT NULL DEFAULT FALSE,
    red_team_objection     VARCHAR(16),
    red_team_case_strength NUMERIC,
    analysts_ok            INTEGER NOT NULL,
    parse_ok               BOOLEAN NOT NULL,
    error                  VARCHAR(200),
    latency_s              NUMERIC,
    prompt_tokens          INTEGER,
    output_tokens          INTEGER,
    panel_code_sha         VARCHAR(48),
    budget                 VARCHAR(16),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pm_verdict_decision_cfg_uq UNIQUE (decision_id, panel_config_sha)
);
CREATE INDEX IF NOT EXISTS pm_verdict_ticker_idx ON h3d.pm_verdict (ticker, as_of_ts);
CREATE INDEX IF NOT EXISTS pm_verdict_signal_idx ON h3d.pm_verdict (signal_id);
CREATE TABLE IF NOT EXISTS h3d.panel_cursor (
    name            VARCHAR(40) PRIMARY KEY,
    last_signal_id  INTEGER NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""),
    # Panel skip markers: a decision whose rows and failure row could not be
    # written gets one row here, so the default read stops re-panelling it.
    # The panel runner's --retry-failed reads such decisions again and
    # deletes the marker when it writes them.
    (6, "panel_skip", """
CREATE TABLE IF NOT EXISTS h3d.panel_skip (
    decision_id       UUID PRIMARY KEY,
    signal_id         INTEGER NOT NULL,
    panel_config_sha  VARCHAR(16) NOT NULL,
    reason            VARCHAR(200),
    skipped_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS panel_skip_signal_idx ON h3d.panel_skip (signal_id);
"""),
]


def _check_unique_versions() -> None:
    versions = [v for v, _, _ in MIGRATIONS]
    if versions != sorted(set(versions)):
        raise RuntimeError(f"h3d migrations must be unique and ascending: {versions}")


def applied_versions(conn) -> set[int]:
    """Versions already in h3d.schema_version; empty if the schema is absent."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('h3d.schema_version')")
        if cur.fetchone()[0] is None:
            conn.rollback()
            return set()
        cur.execute("SELECT version FROM h3d.schema_version")
        out = {int(r[0]) for r in cur.fetchall()}
    conn.rollback()
    return out


def pending(conn) -> list[tuple[int, str, str]]:
    _check_unique_versions()
    done = applied_versions(conn)
    return [m for m in MIGRATIONS if m[0] not in done]


def apply(conn) -> list[int]:
    """Bootstrap h3d and apply every pending migration. Returns the versions
    applied this call (empty when up to date)."""
    _check_unique_versions()
    with conn.cursor() as cur:
        for stmt in BOOTSTRAP_SQL:
            cur.execute(stmt)
    conn.commit()
    done = applied_versions(conn)
    applied = []
    for version, name, sql in MIGRATIONS:
        if version in done:
            continue
        try:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL search_path TO h3d")
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO h3d.schema_version (version, name) VALUES (%s, %s) "
                    "ON CONFLICT (version) DO NOTHING",
                    (version, name),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        logging.info(f"h3d migration {version} ({name}) applied")
        applied.append(version)
    return applied


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True, help="list pending migrations (default)")
    g.add_argument("--apply", action="store_true", help="apply pending migrations")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from helios3d.schema.conn import connection
    with connection() as conn:
        if args.apply:
            done = apply(conn)
            print(f"applied: {done or 'none (up to date)'}")
        else:
            todo = pending(conn)
            print("pending: " + (", ".join(f"{v} {n}" for v, n, _ in todo) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
