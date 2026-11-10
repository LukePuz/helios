"""The hash of every env-overridable setting in config.py (env_sha), shared by
agents.analyst (verdict and shadow provenance) and the shadow replay (the
replay report), which must not import the live analyst. Reads config.py's
source to find the names and the imported config module for the values;
only names are reported for the env vars, never their values. Log-only."""
from __future__ import annotations

import hashlib
import os

# Env-backed config names left out of the config hash: a per-process
# identity (each worker sets its own IB_CLIENT_ID) and an account figure.
EXCLUDED = ("IB_CLIENT_ID", "START_VALUE")


def _config_source() -> str | None:
    """config.py's source text read from disk without importing it."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("config")
        if spec is None or not spec.origin:
            return None
        with open(spec.origin, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def env_names(src: str | None = None) -> dict:
    """{config name: env var} for every module-level assignment in config.py
    whose value reads the environment (os.getenv, os.environ.get or
    os.environ[...]), found by parsing config.py, so a new switch is
    covered without being added to a hand-kept list (M182)."""
    import ast
    if src is None:
        src = _config_source()
    if src is None:
        return {}
    out: dict = {}

    def env_key(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                f = sub.func
                is_getenv = f.attr == "getenv" and isinstance(f.value, ast.Name)
                is_environ_get = (f.attr == "get" and isinstance(f.value, ast.Attribute)
                                  and f.value.attr == "environ")
                if (is_getenv or is_environ_get) and sub.args and isinstance(sub.args[0], ast.Constant):
                    return sub.args[0].value
            if (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Attribute)
                    and sub.value.attr == "environ" and isinstance(sub.slice, ast.Constant)):
                return sub.slice.value
        return None

    def visit(stmts):
        for st in stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None:
                key = env_key(st.value)
                if key:
                    targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                    for t in targets:
                        if isinstance(t, ast.Name):
                            out[t.id] = key
            for field in ("body", "orelse", "finalbody", "handlers"):
                block = getattr(st, field, None)
                if isinstance(block, list):
                    visit(block)
    visit(ast.parse(src).body)
    return out


def env_record(names: dict | None = None, excluded=EXCLUDED) -> dict:
    """{"env_sha", "n", "env_set", "excluded"}: a hash of each env-overridable
    config name (less `excluded`) and its value in the imported config, the
    number hashed, and which of their env vars are set in this process
    (names only). {"error": type} if it can't be computed; never raises."""
    try:
        import config
        if names is None:
            names = env_names()
        names = {k: v for k, v in names.items() if k not in excluded}
        body = "\n".join(f"{k}={getattr(config, k, '<unset>')!r}" for k in sorted(names))
        return {"env_sha": hashlib.sha256(body.encode()).hexdigest()[:12], "n": len(names),
                "env_set": sorted(e for e in set(names.values()) if e in os.environ),
                "excluded": list(excluded)}
    except Exception as e:
        return {"error": type(e).__name__}
