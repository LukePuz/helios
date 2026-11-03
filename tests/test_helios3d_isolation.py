"""helios3d is shadow-only: no live module may import it, and it may not
import the broker layer."""
import ast
import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIVE_FILES = ["main.py", "tasks.py", "execution.py", "risk_manager.py", "ibkr.py", "db.py", "alerts.py"]
SUBPACKAGES = ["schema", "record", "features", "panel", "portfolio", "feedback", "replay", "reports", "deploy"]


_NOT_LIVE = ("helios3d", "tests", "venv", ".venv", ".git", ".claude")


def _live_modules():
    # Every Python file in the repo outside helios3d/, tests/ and tooling dirs.
    files = [p for p in sorted(ROOT.rglob("*.py"))
             if not set(p.relative_to(ROOT).parts[:1]) & set(_NOT_LIVE)]
    assert all((ROOT / f) in files for f in LIVE_FILES)
    return files


def _imported_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
        elif isinstance(node, ast.Call):
            # importlib.import_module("helios3d...") / __import__("helios3d...")
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant):
                names.append(str(node.args[0].value))
    return names


def test_live_module_list_exists():
    files = _live_modules()
    missing = [str(f) for f in files if not f.exists()]
    assert not missing, missing
    assert len(files) > len(LIVE_FILES)  # agents/ contributed modules


@pytest.mark.parametrize("path", _live_modules(), ids=lambda p: str(p.relative_to(ROOT)))
def test_live_module_never_imports_helios3d(path):
    bad = [n for n in _imported_names(path) if n == "helios3d" or n.startswith("helios3d.")]
    assert not bad, f"{path.name} imports {bad}"
    assert "helios3d" not in path.read_text(encoding="utf-8"), f"{path.name} mentions helios3d"


def test_helios3d_never_imports_broker_layer():
    forbidden = {"ibkr", "execution", "ib_insync", "ib_async"}
    for path in sorted((ROOT / "helios3d").rglob("*.py")):
        roots = {n.split(".")[0] for n in _imported_names(path)}
        assert not roots & forbidden, f"{path} imports {roots & forbidden}"


@pytest.mark.parametrize("sub", SUBPACKAGES)
def test_subpackage_has_role_docstring(sub):
    assert (ROOT / "helios3d" / sub / "__init__.py").exists()
    mod = importlib.import_module(f"helios3d.{sub}")
    assert mod.__doc__ and len(mod.__doc__.split()) >= 20


def test_label_unit_is_commented_out():
    text = (ROOT / "helios3d" / "deploy" / "helios3d-label.service").read_text()
    live_lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    assert live_lines == []


def test_labels_not_in_celery_beat():
    assert "helios3d" not in (ROOT / "tasks.py").read_text(encoding="utf-8")
