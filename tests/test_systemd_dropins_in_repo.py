"""The repo tree carries every drop-in the install README names, so the
boot notice's unit-drift line can see a missing install (audit 2026-11-08
19:19 #8)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "deploy" / "systemd"


def test_named_drop_ins_exist():
    readme = (ROOT / "README.md").read_text()
    named = set(re.findall(r"([\w@-]+\.service\.d/[\w.-]+\.conf)", readme))
    retired = {"helios-local.service.d/db-port.conf"}   # named only as retired (roadmap 04e7)
    for rel in named - retired:
        assert (ROOT / rel).is_file(), rel


def test_no_unit_wide_port_drop_in():
    """main.py must stay on the session pooler (init_db's concurrent-index
    migrations); the workers that use 6543 get it from process_manager."""
    for conf in ROOT.glob("*.service.d/*.conf"):
        assert "SUPABASE_PORT" not in conf.read_text(), conf.name
