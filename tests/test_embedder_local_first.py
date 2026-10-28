"""The embedder loads from the local cache before trying the HF Hub
(limitations audit 2026-10-28 12:53, #8)."""
from unittest.mock import patch

import vector_store


def test_local_cache_first(monkeypatch):
    monkeypatch.setattr(vector_store, "_embedder", None)
    with patch("vector_store.SentenceTransformer", return_value="m") as st:
        assert vector_store._get_embedder() == "m"
    assert st.call_args.kwargs["local_files_only"] is True


def test_falls_back_to_hub(monkeypatch):
    monkeypatch.setattr(vector_store, "_embedder", None)
    with patch("vector_store.SentenceTransformer", side_effect=[OSError("not cached"), "m"]) as st:
        assert vector_store._get_embedder() == "m"
    assert "local_files_only" not in st.call_args.kwargs
