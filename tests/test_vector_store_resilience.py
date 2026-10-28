"""Vector store memory must survive a collection rebuild in another process
and must not return the triggering event as its own history (audit
2026-10-28 03:00, finding 7)."""
from unittest.mock import MagicMock

import vector_store


class _Col:
    def __init__(self, ids, fail=False):
        self.ids, self.fail = ids, fail

    def count(self):
        if self.fail:
            raise RuntimeError("Collection abc does not exist.")
        return len(self.ids)

    def query(self, query_embeddings, n_results, include, where=None):
        ids = self.ids[:n_results]
        return {"ids": [ids], "documents": [[f"doc {i}" for i in ids]],
                "metadatas": [[{} for _ in ids]], "distances": [[0.1 for _ in ids]]}


def _client(*cols):
    client = MagicMock()
    client.get_or_create_collection.side_effect = list(cols)
    return client


def test_stale_handle_is_dropped_and_reopened(monkeypatch):
    monkeypatch.setattr(vector_store, "_collections", {"news": _Col(["a"], fail=True)})
    monkeypatch.setattr(vector_store, "_get_client", lambda: _client(_Col(["a", "b"])))
    docs = vector_store.search("news", "q", n_results=2, query_embedding=[0.0])
    assert [d["text"] for d in docs] == ["doc a", "doc b"]


def test_triggering_doc_is_excluded_and_count_kept(monkeypatch):
    monkeypatch.setattr(vector_store, "_collections", {"news": _Col(["self", "x", "y"])})
    docs = vector_store.search("news", "q", n_results=2, query_embedding=[0.0],
                               exclude_ids={"self"})
    assert [d["text"] for d in docs] == ["doc x", "doc y"]


def test_one_broken_collection_does_not_empty_memory(monkeypatch):
    def fake_search(collection, *a, **k):
        if collection == "filings":
            raise RuntimeError("boom")
        return [{"text": collection, "metadata": {}, "score": 0.5}] if collection == "news" else []
    monkeypatch.setattr(vector_store, "embed", lambda q: [0.0])
    monkeypatch.setattr(vector_store, "search", fake_search)
    out = vector_store.search_all_collections("q")
    assert [r["collection"] for r in out] == ["news"]
