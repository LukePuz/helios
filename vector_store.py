"""
ChromaDB vector store — persistent memory for the trading system.
Every filing, paper, news article, and macro event gets embedded here.
The context engine uses semantic search to pull relevant history
into the model prompt before every reasoning call.

Uses sentence-transformers (runs on CPU, no GPU needed yet).
Model: all-MiniLM-L6-v2 (~90MB, fast, accurate for finance text)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent / ".chromadb"
_EMBED_MODEL = "all-MiniLM-L6-v2"

_client: chromadb.PersistentClient | None = None
_embedder: SentenceTransformer | None = None
_collections: dict[str, chromadb.Collection] = {}

COLLECTIONS = {
    "filings":      "SEC filings — 8-K, 10-Q, 10-K",
    "insider":      "Form 4 insider trades",
    "news":         "News articles and headlines",
    "arxiv":        "Academic research papers",
    "macro":        "Macro economic events and data",
    "congressional":"Congressional trading disclosures",
    "options":      "Options flow alerts",
}


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _DB_PATH.mkdir(exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(_DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        log.info(f"ChromaDB initialized at {_DB_PATH}")
    return _client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info(f"Loading embedding model {_EMBED_MODEL}...")
        _embedder = SentenceTransformer(_EMBED_MODEL)
        log.info("Embedding model ready.")
    return _embedder


def _get_collection(name: str) -> chromadb.Collection:
    if name not in _collections:
        _collections[name] = _get_client().get_or_create_collection(
            name=name,
            metadata={"description": COLLECTIONS.get(name, name)},
        )
    return _collections[name]


def embed(text: str) -> list[float]:
    """Converts text to embedding vector."""
    return _get_embedder().encode(text, convert_to_list=True)


def store(
    collection: str,
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Embeds and stores a document. Safe to call multiple times —
    upserts by doc_id so duplicates are overwritten, not duplicated.
    """
    if not text.strip():
        return
    col = _get_collection(collection)
    embedding = embed(text[:8192])  # cap at 8k chars for embedding
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }
    # ChromaDB metadata values must be str/int/float/bool
    meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in meta.items()}
    col.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text[:8192]],
        metadatas=[meta],
    )


def search(
    collection: str,
    query: str,
    n_results: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Semantic search — returns the most relevant documents for a query.
    Used by context_engine to find relevant history before a model call.
    """
    col = _get_collection(collection)
    count = col.count()
    if count == 0:
        return []

    n_results = min(n_results, count)
    query_embedding = embed(query)

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)

    docs = []
    for i, doc in enumerate(results["documents"][0]):
        docs.append({
            "text":     doc,
            "metadata": results["metadatas"][0][i],
            "score":    1 - results["distances"][0][i],  # convert distance to similarity
        })
    return docs


def search_all_collections(query: str, n_results: int = 3) -> list[dict]:
    """
    Searches every collection and returns the top results across all of them.
    Used for broad context assembly when the event type isn't known.
    """
    all_results = []
    for collection in COLLECTIONS:
        results = search(collection, query, n_results=n_results)
        for r in results:
            r["collection"] = collection
        all_results.extend(results)
    # Sort by relevance score descending
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:n_results * 2]


def store_filing(filing: dict) -> None:
    """Convenience wrapper for SEC filings."""
    store(
        collection="filings",
        doc_id=filing.get("url", filing.get("title", "")),
        text=filing.get("text", ""),
        metadata={
            "ticker": filing.get("ticker", ""),
            "form":   filing.get("form", ""),
            "date":   filing.get("timestamp", ""),
        },
    )


def store_news(article: dict) -> None:
    """Convenience wrapper for news articles."""
    store(
        collection="news",
        doc_id=article.get("url", article.get("title", "")),
        text=f"{article.get('headline', '')} {article.get('summary', '')}".strip(),
        metadata={
            "ticker":    ",".join(article.get("symbols", [])),
            "source":    article.get("source", ""),
            "timestamp": article.get("created_at", ""),
        },
    )


def store_arxiv(paper: dict) -> None:
    """Convenience wrapper for arXiv papers."""
    store(
        collection="arxiv",
        doc_id=paper.get("paper_id", paper.get("title", "")),
        text=paper.get("text", ""),
        metadata={
            "title":     paper.get("title", ""),
            "timestamp": paper.get("timestamp", ""),
        },
    )


def prune_old_documents(collection: str, max_age_days: int = 90) -> int:
    """
    Delete documents older than max_age_days from a collection.
    Uses the 'timestamp' metadata field set by store().
    Returns count of documents deleted.
    """
    from datetime import timedelta
    col = _get_collection(collection)
    if col.count() == 0:
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()

    try:
        # Fetch all document IDs and their timestamps
        all_docs = col.get(include=["metadatas"])
        ids_to_delete = []
        for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"]):
            ts = meta.get("timestamp") or meta.get("date") or ""
            if ts and ts < cutoff:
                ids_to_delete.append(doc_id)

        if ids_to_delete:
            col.delete(ids=ids_to_delete)
            log.info(f"Pruned {len(ids_to_delete)} docs from '{collection}' (>{max_age_days}d old)")

        return len(ids_to_delete)
    except Exception as e:
        log.warning(f"prune_old_documents failed for '{collection}': {e}")
        return 0


def collection_stats() -> dict[str, int]:
    """Returns document count per collection."""
    stats = {}
    for name in COLLECTIONS:
        try:
            stats[name] = _get_collection(name).count()
        except Exception:
            stats[name] = 0
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ChromaDB collection stats:")
    for name, count in collection_stats().items():
        print(f"  {name:<20} {count:>6} documents")

    # Quick test
    print("\nStoring test document...")
    store("news", "test-001", "NVIDIA announces record GPU sales driven by AI demand.", {"ticker": "NVDA"})
    print("Searching for 'NVIDIA AI chips'...")
    results = search("news", "NVIDIA AI chips", n_results=1)
    for r in results:
        print(f"  Score {r['score']:.3f}: {r['text'][:100]}")
