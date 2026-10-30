"""Nightly encrypted backup of the study data (Luke D16, 10-29; audit A1853-5).

The database is hosted (Supabase), but nothing kept a copy we control of
it, the catalyst calendar or the vector store.

Writes ~/helios-db-backups/nightly/helios-YYYYMMDD-HHMM.tar.gpg containing:
  - postgres.dump        pg_dump -Fc of the bot's database
  - catalyst_calendar.db consistent SQLite copy
  - chromadb.tar         the .chromadb vector store
encrypted with gpg (AES256) using the passphrase in ~/.config/helios/backup.pass.
The dumps hold balances and unredacted history, so only the encrypted file is
kept and it never goes to Drive. Copies to BACKUP_USB_DIR too when that
directory exists (a mounted USB drive).

Keeps 14 daily and 8 weekly (Sunday) backups.

    venv/bin/python scripts/backup_db.py            # make a backup now
    venv/bin/python scripts/backup_db.py --check    # verify the newest one decrypts and lists
Run from ~/Desktop/trading_bot (config loads .env from there).
"""
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS_FILE = Path.home() / ".config" / "helios" / "backup.pass"
OUT_DIR = Path.home() / "helios-db-backups" / "nightly"
USB_DIR = Path(os.getenv("BACKUP_USB_DIR", f"/media/{os.getenv('USER', 'YOUR_USER')}/HELIOS-BACKUP"))
KEEP_DAILY, KEEP_WEEKLY = 14, 8


def _pg_env() -> dict:
    import config  # noqa: F401  (loads .env into os.environ)
    env = dict(os.environ)
    for pg, ours, default in (("PGHOST", "SUPABASE_HOST", None), ("PGPORT", "SUPABASE_PORT", "5432"),
                              ("PGDATABASE", "SUPABASE_DB", "postgres"),
                              ("PGUSER", "SUPABASE_USER", "postgres"),
                              ("PGPASSWORD", "SUPABASE_PASSWORD", None)):
        val = os.getenv(ours, default)
        if val:
            env[pg] = val
    env.setdefault("PGSSLMODE", "require")
    return env


def _gpg(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["gpg", "--batch", "--yes", "--pinentry-mode", "loopback",
                           "--passphrase-file", str(PASS_FILE), *args], check=True, **kw)


def backup() -> Path:
    if not PASS_FILE.exists():
        sys.exit(f"No passphrase file at {PASS_FILE}; create it first (see LUKE-TODO).")
    OUT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(["pg_dump", "-Fc", "-f", str(tmp / "postgres.dump")],
                       env=_pg_env(), check=True)
        cal = ROOT / "catalyst_calendar.db"
        if cal.exists():
            src, dst = sqlite3.connect(cal), sqlite3.connect(tmp / "catalyst_calendar.db")
            with dst:
                src.backup(dst)
            src.close()
            dst.close()
        chroma = ROOT / ".chromadb"
        if chroma.exists():
            with tarfile.open(tmp / "chromadb.tar", "w") as t:
                t.add(chroma, arcname=".chromadb")
        bundle = tmp / "bundle.tar"
        with tarfile.open(bundle, "w") as t:
            for f in sorted(tmp.iterdir()):
                if f != bundle:
                    t.add(f, arcname=f.name)
        out = OUT_DIR / f"helios-{stamp}.tar.gpg"
        _gpg(["--symmetric", "--cipher-algo", "AES256", "-o", str(out), str(bundle)])
    os.chmod(out, 0o600)
    if USB_DIR.is_dir():
        try:
            shutil.copy2(out, USB_DIR / out.name)
        except OSError as e:
            print(f"USB copy failed: {e}", file=sys.stderr)
    _rotate(OUT_DIR)
    if USB_DIR.is_dir():
        _rotate(USB_DIR)
    return out


def _rotate(folder: Path) -> None:
    files = sorted(folder.glob("helios-*.tar.gpg"), reverse=True)
    keep = set(files[:KEEP_DAILY])
    sundays = [f for f in files if datetime.strptime(f.name[7:15], "%Y%m%d").weekday() == 6]
    keep |= set(sundays[:KEEP_WEEKLY])
    for f in files:
        if f not in keep:
            f.unlink()


def check() -> None:
    newest = max(OUT_DIR.glob("helios-*.tar.gpg"), default=None)
    if newest is None:
        sys.exit("No backups yet.")
    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "bundle.tar"
        _gpg(["-o", str(plain), "--decrypt", str(newest)], capture_output=True)
        with tarfile.open(plain) as t:
            names = t.getnames()
    print(f"{newest.name}: decrypts; contains {', '.join(names)}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        print(f"backup written: {backup()}")
