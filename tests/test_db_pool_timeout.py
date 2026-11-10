"""The DB pool sets a connect timeout, like the direct connections (audit
2026-11-07 05:07 #5). conftest replaces db._get_pool so no test can open the
pool, so this reads the constructor call in the source."""
import ast
from pathlib import Path

import db


def test_pool_is_created_with_a_connect_timeout():
    tree = ast.parse(Path(db.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "ThreadedConnectionPool"]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw["connect_timeout"], ast.Constant) and kw["connect_timeout"].value == 10
