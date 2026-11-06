"""Installed library versions in the analyst verdict provenance and in
shadow_versions() (m4): read once from package metadata, missing as null."""
from importlib import metadata
from unittest.mock import patch

from agents import analyst


def _fresh(monkeypatch):
    monkeypatch.setattr(analyst, "_LIB_VERSIONS", {})


def test_lib_versions_reads_metadata_once_with_missing_as_null(monkeypatch):
    _fresh(monkeypatch)
    calls = []

    def version(name):
        calls.append(name)
        if name in ("ib_insync", "psycopg2"):
            raise metadata.PackageNotFoundError(name)
        return {"ib_async": "2.1.0", "yfinance": "1.5.1", "celery": "5.6.3",
                "kombu": "5.6.2", "psycopg2-binary": "2.9.12"}[name]
    with patch("importlib.metadata.version", side_effect=version):
        got = analyst._lib_versions()
        n = len(calls)
        assert analyst._lib_versions() == got
        assert len(calls) == n                      # cached
    assert got == {"ib_async": "2.1.0", "ib_insync": None, "yfinance": "1.5.1",
                   "celery": "5.6.3", "kombu": "5.6.2", "psycopg2": "2.9.12"}


def test_lib_versions_match_the_installed_packages(monkeypatch):
    _fresh(monkeypatch)
    got = analyst._lib_versions()
    import yfinance
    assert got["yfinance"] == yfinance.__version__
    assert got["celery"] == metadata.version("celery")
    assert set(got) == {"ib_async", "ib_insync", "yfinance", "celery", "kombu", "psycopg2"}


def test_verdict_provenance_and_shadow_versions_carry_them(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setattr(analyst, "_LIB_VERSIONS", {"yfinance": "9.9"})
    prov = analyst._provenance("prompt")
    assert prov["lib_versions"] == {"yfinance": "9.9"}
    assert analyst.shadow_versions()["lib_versions"] == {"yfinance": "9.9"}
