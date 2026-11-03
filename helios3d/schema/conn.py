"""Database connections for Helios 3D.

By default, borrows a connection from db.py's pool (same credentials, read
from .env by db.py; this module never reads them). The live code never
imports this, so a helios3d process has its own pool instance. Writes go
only to h3d tables; public tables are read inside READ ONLY transactions
(read_only()).

When HELIOS3D_DATABASE_URL is set, connections use that DSN instead, so
helios3d can run as its own database role (one that owns h3d and has only
SELECT on public) and the boundary is a permission, not only a convention.
Creating that role and its grants is a database admin step
outside this code.
"""
import os
from contextlib import contextmanager

DSN_ENV = "HELIOS3D_DATABASE_URL"


@contextmanager
def connection():
    """A connection, rolled back (if left open) and closed or returned on exit:
    a dedicated one from HELIOS3D_DATABASE_URL when set, else db.py's pool."""
    dsn = os.getenv(DSN_ENV)
    if dsn:
        import db  # the conftest guard patches db.psycopg2.connect, so tests can't escape
        conn = db.psycopg2.connect(dsn, connect_timeout=10)
        try:
            yield conn
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            conn.close()
        return
    import db  # imported lazily: tests patch db before any connection exists
    conn = db._get_conn()
    try:
        yield conn
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)


@contextmanager
def read_only(conn):
    """A cursor inside a READ ONLY transaction, rolled back afterwards, so any
    write attempted through it fails at the server."""
    conn.rollback()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            yield cur
    finally:
        conn.rollback()
