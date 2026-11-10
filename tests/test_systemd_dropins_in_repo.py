"""The repo tree carries every drop-in the install README names, so the
boot notice's unit-drift line can see a missing install (audit 2026-11-08
19:19 #8)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "deploy" / "systemd"


def test_named_drop_ins_exist():
    readme = (ROOT / "README.md").read_text()
    named = set(re.findall(r"([\w@-]+\.service\.d/[\w.-]+\.conf)", readme))
    assert "helios-local.service.d/db-port.conf" in named
    for rel in named:
        assert (ROOT / rel).is_file(), rel
