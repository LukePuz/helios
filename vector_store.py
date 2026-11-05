"""
ChromaDB vector store — persistent memory for the trading system.
Every filing, paper, news article, and macro event gets embedded here.
The context engine uses semantic search to pull relevant history
into the model prompt before every reasoning call.

Uses sentence-transformers (runs on CPU, no GPU needed yet).
Model: all-MiniLM-L6-v2 (~90MB, fast, accurate for finance text)
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent / ".chromadb"
_EMBED_MODEL = "all-MiniLM-L6-v2"
# ChromaDB's embedded PersistentClient is not safe for concurrent writes
# from multiple OS processes — celery_screen_worker runs 3 forked children
# that each open their own PersistentClient against the same .chromadb
# directory and all write to the "news" collection continuously (by far
# the highest-volume, highest-concurrency collection). That corrupted its
# on-disk index badly enough that even a plain col.count() segfaulted the
# whole process (found 2026-07-17 while testing chat-requested research,
# which shares this same read path — production's deep_analyze_event does
# too, so this was silently about to break every analysis, not just the
# new feature). Cross-process flock serializes the actual write so only
# one process ever touches the index at a time; reads/embedding aren't
# locked since only concurrent writes could corrupt the on-disk segments.
_WRITE_LOCK_PATH = _DB_PATH / ".write.lock"


@contextlib.contextmanager
def _write_lock():
    _DB_PATH.mkdir(exist_ok=True)
    with open(_WRITE_LOCK_PATH, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

_client: chromadb.PersistentClient | None = None
_embedder: SentenceTransformer | None = None
_collections: dict[str, chromadb.Collection] = {}
# Write generation (.write.gen next to .write.lock) that this process's client
# reflects: set when it reopens or writes under the lock. None means unknown,
# so the next write reopens.
_seen_generation: int | None = None
# Rebuild batch sizes: texts per encode() batch and records per upsert call.
_EMBED_BATCH = 64
_UPSERT_BATCH = 256

try:
    # Raised inside a Celery task at its soft time limit. A rebuild must let
    # it through: caught as a per-document failure, it would count the rest
    # of the collection as unrestorable and drop the snapshot.
    from celery.exceptions import SoftTimeLimitExceeded as _SoftTimeLimit
except ImportError:  # pragma: no cover - celery is a dependency
    class _SoftTimeLimit(Exception):
        pass

COLLECTIONS = {
    "filings":      "SEC filings — 8-K, 10-Q, 10-K",
    "insider":      "Form 4 insider trades",
    "news":         "News articles and headlines",
    "arxiv":        "Academic research papers",
    "macro":        "Macro economic events and data",
    "congressional":"Congressional trading disclosures",
    "options":      "Options flow alerts",
    # Added 2026-07-26 for agents/candidate_scanner.py: one document per
    # knowledge/*.md investment thesis file, seeded once (seed_thesis_
    # collection there), so a non-watchlist S&P 500 ticker's business
    # description can be semantically matched against our actual thesis
    # prose rather than a hardcoded keyword list.
    "thesis":       "Investment thesis descriptions per theme",
}


def _reopen_client() -> None:
    """Drops this process's Chroma client and collection handles, so the next
    _get_client() loads the store from disk. Writes reach it through
    _locked_write(), which calls it when another process wrote since this
    process's last write (see _reopen_if_stale).

    Chroma 1.x keeps a per-process in-memory copy of each collection's HNSW
    index and writes that copy back to disk. The lock stops two writes from
    running at once, but a process that wrote earlier still writes from its own
    old copy, which lacks whatever the other processes wrote since. Three
    long-lived writers (the screen worker's forks) re-upserting the same news
    urls leave the on-disk index out of step with the log, and a fresh
    process's col.count() then fails with "Failed to apply logs to the hnsw
    segment writer". That is what tasks.check_vector_store_health reported as
    'news' corrupted. Reproduced with chromadb 1.5.9 on a scratch store: three
    cached writers under the lock broke it; one writer with two cached readers,
    or three writers that reopen under the lock, did not. A reopen costs about
    20 ms. The old system is stopped, which frees its Rust bindings: only
    dropping it from Chroma's cache leaked about 5 MB per reopen."""
    global _client
    from chromadb.api.client import SharedSystemClient
    for system in list(SharedSystemClient._identifier_to_system.values()):
        try:
            system.stop()
        except Exception as e:
            log.debug(f"vector_store: stopping the old Chroma system failed: {e}")
    SharedSystemClient.clear_system_cache()
    _client = None
    _collections.clear()


def _read_generation() -> int | None:
    """The store's write generation from .write.gen: 0 when the file is
    missing, None when it is empty or can't be read (the caller then
    reopens)."""
    try:
        return int((_DB_PATH / ".write.gen").read_text().strip())
    except FileNotFoundError:
        return 0
    except Exception as e:
        log.debug(f"vector_store: could not read the write generation: {e}")
        return None


def _bump_generation() -> None:
    """Called under the write lock before a write: other processes see a new
    generation and reopen before their next write. The file is replaced
    whole (temp file, rename), so a kill never leaves it empty. When the
    current value can't be read, the new one is time.time_ns(), which no
    process can have seen: restarting the count at 1 would match a process
    whose last seen value was 1, and it would skip a reopen it needs."""
    global _seen_generation
    cur = _read_generation()
    gen = cur + 1 if cur is not None else time.time_ns()
    path = _DB_PATH / ".write.gen"
    tmp = path.with_name(".write.gen.tmp")
    tmp.write_text(str(gen))
    os.replace(tmp, path)
    _seen_generation = gen


def _reopen_if_stale() -> None:
    """Called under the write lock before a write. Reopens (_reopen_client)
    only when another process wrote since this process's last write or reopen,
    or the generation is unknown. Back-to-back writes from one process keep
    their client: no 20 ms reopen and no allocator growth from it."""
    global _seen_generation
    gen = _read_generation()
    if _client is None or gen is None or gen != _seen_generation:
        _reopen_client()
        _seen_generation = gen


def _forget_generation() -> None:
    """Makes this process's next write reopen. Registered for forks, so a
    Celery prefork child doesn't trust the client state it inherited, and
    called when a write raised partway."""
    global _seen_generation
    _seen_generation = None


os.register_at_fork(after_in_child=_forget_generation)


@contextlib.contextmanager
def _locked_write():
    """The write lock, a reopen when another process wrote since this one's
    last write, and a generation bump. The bump comes before the write, so a
    writer killed partway still makes every other process reopen. A write
    that raised leaves this process's own generation unknown, so its next
    write reopens."""
    with _write_lock():
        _reopen_if_stale()
        _bump_generation()
        try:
            yield
        except BaseException:
            _forget_generation()
            raise


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
        # Forced onto CPU explicitly -- same fix, same reason as
        # agents/screener.py's 7B model. Left to auto-select, this loads
        # onto cuda:0 (confirmed live 2026-07-26), and both GPUs already
        # sit at ~93% VRAM just holding the 72B analyst model. This module
        # gets called far more often than screener.py ever was (every
        # filing/news/arxiv/congressional store, plus 2026-07-26's new
        # candidate_scanner --  ~466 embeddings per weekly scan) --
        # contending with Ollama for the same near-full GPUs traces
        # directly to that day's real production incident: 72B analyst
        # calls that normally land at ~190s started repeatedly hitting the
        # 400s timeout and falling back to stub recommendations for hours.
        # MiniLM is tiny; CPU inference costs milliseconds, not the
        # multi-minute difference GPU contention was causing Ollama.
        # Local cache first: every worker process used to fetch metadata from
        # the HF Hub unauthenticated, so a Hub outage or rate limit became a
        # memory/embedding failure (limitations audit 12:53, #8).
        try:
            _embedder = SentenceTransformer(_EMBED_MODEL, device="cpu", local_files_only=True)
        except Exception as e:
            log.warning(f"Embedding model not in local cache ({e}); downloading from the Hub")
            _embedder = SentenceTransformer(_EMBED_MODEL, device="cpu")
        log.info("Embedding model ready (CPU).")
    return _embedder


def _get_collection(name: str) -> chromadb.Collection:
    if name not in _collections:
        _collections[name] = _get_client().get_or_create_collection(
            name=name,
            metadata={"description": COLLECTIONS.get(name, name)},
        )
    return _collections[name]


def _with_fresh_retry(name: str, fn):
    """Runs fn(collection), retrying once with a fresh handle. The health
    check (tasks.check_vector_store_health) rebuilds a corrupted collection
    from another process, which leaves this process's cached handle pointing
    at the deleted collection: every search then failed with "Collection ...
    does not exist" until the 19:00 restart (50 times on 10-15 and 10-26)."""
    try:
        return fn(_get_collection(name))
    except Exception as e:
        log.info(f"vector_store: '{name}' handle failed ({e}); reopening")
        _collections.pop(name, None)
        return fn(_get_collection(name))


def embed(text: str) -> list[float]:
    """
    Converts text to embedding vector.
    NOTE: `.encode(..., convert_to_list=True)` was silently broken by a
    sentence-transformers version bump — that kwarg no longer exists for
    this model, and .encode() raised on every call, meaning every
    vector_store.store() call (news, filings, arxiv, macro, congressional,
    self-reviews) was silently failing since store()'s callers all wrap it
    in a broad try/except. .encode() returns a numpy array; ChromaDB needs
    a plain list.
    """
    # No tqdm bar: the workers log stderr at WARNING (482 empty bars on 10-30).
    result = _get_embedder().encode(text, show_progress_bar=False)
    return result.tolist() if hasattr(result, "tolist") else list(result)


def embed_many(texts: list[str]) -> list[list[float]]:
    """embed() for a list of texts, encoded _EMBED_BATCH at a time. Used by
    rebuild_collection, which re-embeds a whole collection under the lock."""
    if not texts:
        return []
    result = _get_embedder().encode(texts, batch_size=_EMBED_BATCH, show_progress_bar=False)
    return [r.tolist() if hasattr(r, "tolist") else list(r) for r in result]


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
    # Cap is cosmetic beyond the model's own limit — all-MiniLM-L6-v2
    # truncates at ~256 tokens (~1k chars) regardless, so a long filing's
    # embedding is really only ever computed from its first ~1k chars, not
    # the full 8k this slice suggests. Kept at 8k anyway: it's a cheap
    # upper bound against pathological huge inputs and costs nothing since
    # the model truncates further internally.
    embedding = embed(text[:8192])  # CPU work — outside the write lock
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }
    # ChromaDB metadata values must be str/int/float/bool
    meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in meta.items()}
    with _locked_write():   # writes from the on-disk state, not this process's old copy
        _with_fresh_retry(collection, lambda col: col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text[:8192]],
            metadatas=[meta],
        ))


def search(
    collection: str,
    query: str,
    n_results: int = 5,
    where: dict | None = None,
    query_embedding: list[float] | None = None,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """
    Semantic search — returns the most relevant documents for a query.
    Used by context_engine to find relevant history before a model call.

    Pass query_embedding to reuse an already-computed embedding (see
    search_all_collections, which used to embed the identical query string
    once per collection — 7 CPU embeddings for one logical search).

    exclude_ids drops those doc ids from the results. The triggering event is
    stored just before its analysis (crew.screen_event), so without this it
    comes back as its own top "historical context" hit.
    """
    # A canary record left by a process killed mid-check is never a result.
    exclude_ids = (exclude_ids or set()) | {_CANARY_ID}
    if query_embedding is None:
        query_embedding = embed(query)

    def _query(col):
        count = col.count()
        if count == 0:
            return None
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results + len(exclude_ids), count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    results = _with_fresh_retry(collection, _query)
    if results is None:
        return []

    docs = []
    for i, doc in enumerate(results["documents"][0]):
        if results["ids"][0][i] in exclude_ids:
            continue
        if len(docs) >= n_results:
            break
        docs.append({
            "text":     doc,
            "metadata": results["metadatas"][0][i],
            "score":    1 - results["distances"][0][i],  # convert distance to similarity
        })
    return docs


def search_all_collections(query: str, n_results: int = 3,
                           exclude_ids: set[str] | None = None) -> list[dict]:
    """
    Searches every collection and returns the top results across all of them.
    Used for broad context assembly when the event type isn't known.
    A collection that fails is skipped, so one bad collection doesn't empty
    the whole memory section of the brief.
    """
    # Embed the query once, not once per collection (this used to run 7
    # identical CPU embeddings of the same query string per call).
    query_embedding = embed(query)
    all_results = []
    for collection in COLLECTIONS:
        try:
            results = search(collection, query, n_results=n_results,
                             query_embedding=query_embedding, exclude_ids=exclude_ids)
        except Exception as e:
            log.warning(f"vector_store: search in '{collection}' failed: {e}")
            continue
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


# Prune age per collection (tasks.prune_vector_store). arxiv is intentionally
# not pruned (papers are timeless reference material), nor is thesis.
PRUNE_MAX_AGE_DAYS = {
    "news":          90,
    "options":       60,
    "congressional": 180,
    "filings":       90,
    "insider":       365,
    "macro":         180,
}


def _age_cutoff(max_age_days: int) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()


def _past_max_age(meta: dict | None, cutoff: str) -> bool:
    """prune_old_documents' test: the 'timestamp' (else 'date') metadata,
    compared as an ISO string with the cutoff."""
    ts = (meta or {}).get("timestamp") or (meta or {}).get("date") or ""
    # Deliberate change from the old inline test: a timestamp that isn't a
    # string skips that record instead of raising, which used to abort the
    # whole prune of the collection.
    return isinstance(ts, str) and bool(ts) and ts < cutoff


def prune_old_documents(collection: str, max_age_days: int = 90) -> int:
    """
    Delete documents older than max_age_days from a collection.
    Uses the 'timestamp' metadata field set by store().
    Returns count of documents deleted.
    """
    cutoff = _age_cutoff(max_age_days)

    # A delete is a write: same lock and reopen rule as store().
    with _locked_write():
        if _snapshot_path(collection).exists():
            # An unfinished rebuild's snapshot would put pruned ids back on
            # resume. The health check resumes it within the hour; prune then.
            log.warning(f"prune_old_documents: '{collection}' has an unfinished rebuild; skipped")
            return 0
        col = _get_collection(collection)
        if col.count() == 0:
            return 0
        try:
            # Fetch all document IDs and their timestamps
            all_docs = col.get(include=["metadatas"])
            ids_to_delete = []
            for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"]):
                if _past_max_age(meta, cutoff):
                    ids_to_delete.append(doc_id)

            if ids_to_delete:
                col.delete(ids=ids_to_delete)
                log.info(f"Pruned {len(ids_to_delete)} docs from '{collection}' (>{max_age_days}d old)")

            return len(ids_to_delete)
        except Exception as e:
            log.warning(f"prune_old_documents failed for '{collection}': {e}")
            return 0


def _read_collection_docs(name: str) -> list[tuple[str, str, dict]]:
    """(id, document, metadata) for every record of a collection, read
    straight from chroma.sqlite3 over a read-only connection. A broken
    collection's HNSW index takes down any Chroma call that loads it (count,
    get), but its documents and metadata live in the sqlite metadata segment,
    which that breakage leaves intact."""
    import sqlite3
    con = sqlite3.connect(f"file:{_DB_PATH / 'chroma.sqlite3'}?mode=ro", uri=True, timeout=30)
    try:
        rows = con.execute(
            "SELECT e.embedding_id, m.key, m.string_value, m.int_value, m.float_value, m.bool_value "
            "FROM embeddings e JOIN segments s ON s.id = e.segment_id "
            "JOIN collections c ON c.id = s.collection "
            "LEFT JOIN embedding_metadata m ON m.id = e.id "
            "WHERE c.name = ? AND s.scope = 'METADATA'", (name,)).fetchall()
    finally:
        con.close()
    docs: dict[str, dict] = {}
    for doc_id, key, s, i, f, b in rows:
        d = docs.setdefault(doc_id, {"document": None, "metadata": {}})
        if key is None:
            continue
        value = (s if s is not None else i if i is not None else f if f is not None
                 else bool(b) if b is not None else None)
        if key == "chroma:document":
            d["document"] = value
        elif value is not None and not key.startswith("chroma:"):
            d["metadata"][key] = value
    without_text = [k for k, v in docs.items() if not v["document"] and k != _CANARY_ID]
    if without_text:
        log.warning(f"vector_store: '{name}' has {len(without_text)} records with no document "
                    f"text; they can't be re-embedded and are left out: {without_text[:5]}")
    # A canary record left by a process killed mid-check is not a document.
    return [(k, v["document"], v["metadata"]) for k, v in docs.items()
            if v["document"] and k != _CANARY_ID]


def _count_collection_rows(name: str) -> int:
    """Records in a collection's sqlite metadata segment, read-only, with or
    without a document. rebuild_collection compares it with what
    _read_collection_docs returned before it deletes anything."""
    import sqlite3
    con = sqlite3.connect(f"file:{_DB_PATH / 'chroma.sqlite3'}?mode=ro", uri=True, timeout=30)
    try:
        return con.execute(
            "SELECT COUNT(*) FROM embeddings e JOIN segments s ON s.id = e.segment_id "
            "JOIN collections c ON c.id = s.collection "
            "WHERE c.name = ? AND s.scope = 'METADATA' AND e.embedding_id != ?",
            (name, _CANARY_ID)).fetchone()[0]
    finally:
        con.close()


def _snapshot_path(name: str) -> Path:
    """A rebuild's snapshot of the collection's records, next to the store."""
    return _DB_PATH / f".rebuild-{name}.jsonl"


def _write_snapshot(name: str, docs: list[tuple[str, str, dict]]) -> None:
    """Writes the snapshot whole (temp file, fsync, rename), so a kill leaves
    either the old file or the complete new one."""
    path = _snapshot_path(name)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for doc_id, text, meta in docs:
            f.write(json.dumps({"id": doc_id, "document": text, "metadata": meta}) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_snapshot(name: str) -> list[tuple[str, str, dict]] | None:
    """The records of a rebuild that did not finish, or None if there is no
    snapshot. Raises if the file can't be read."""
    path = _snapshot_path(name)
    if not path.exists():
        return None
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                docs.append((r["id"], r["document"], r["metadata"] or {}))
    return docs


def pending_rebuilds() -> list[str]:
    """Collections with a snapshot left by a rebuild that did not finish."""
    if not _DB_PATH.exists():
        return []
    return sorted(p.name[len(".rebuild-"):-len(".jsonl")] for p in _DB_PATH.glob(".rebuild-*.jsonl")
                  if not p.name.endswith(".deadletter.jsonl"))


# Resumes of one snapshot that may fail in a row before they stop: each one
# deletes and re-embeds the collection, and the health check runs hourly.
_MAX_RESUME_FAILURES = 3


def _rebuild_state_path(name: str) -> Path:
    """Sidecar of a pending snapshot: when it was first written, and how many
    resumes of it failed in a row."""
    return _DB_PATH / f".rebuild-{name}.state.json"


def rebuild_state(name: str) -> dict[str, Any]:
    """{"created": epoch s, "failures": int, "last_error": str | None} for a
    pending snapshot; {} when there is no sidecar or it can't be read. A soft
    time limit landing mid-read is raised, not read as {}: that would reset
    the failure count and drop created."""
    try:
        return json.loads(_rebuild_state_path(name).read_text())
    except FileNotFoundError:
        return {}
    except _SoftTimeLimit:
        raise
    except Exception as e:
        log.warning(f"vector_store: '{name}' rebuild state unreadable: {e}")
        return {}


def _save_rebuild_state(name: str, state: dict[str, Any]) -> None:
    path = _rebuild_state_path(name)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


def snapshot_age_hours(name: str) -> float | None:
    """Hours since the pending snapshot of a collection was first written (a
    resume rewrites the file, so its own mtime is only the fallback); None
    when there is no snapshot, whatever the sidecar says (an orphan sidecar
    must not hold up prune)."""
    path = _snapshot_path(name)
    if not path.exists():
        return None
    try:
        created = rebuild_state(name).get("created") or path.stat().st_mtime
    except FileNotFoundError:
        return None
    return max(time.time() - float(created), 0.0) / 3600


def clear_orphan_rebuild_states() -> list[str]:
    """Removes, under the write lock, every .rebuild-<name>.state.json whose
    snapshot is gone (deleted by hand after a gave-up alert, or a kill
    between the two unlinks). Returns the collections cleared."""
    if not _DB_PATH.exists():
        return []
    cleared = []
    with _write_lock():
        for p in _DB_PATH.glob(".rebuild-*.state.json"):
            name = p.name[len(".rebuild-"):-len(".state.json")]
            if not _snapshot_path(name).exists():
                p.unlink(missing_ok=True)
                cleared.append(name)
                log.warning(f"vector_store: removed orphan rebuild state for '{name}' "
                            f"(no snapshot)")
    return sorted(cleared)


@contextlib.contextmanager
def _resume_attempt(name: str, report: dict[str, Any]):
    """Wraps one rebuild_collection run under the lock and keeps the
    sidecar's bookkeeping. rebuild_collection counts a resume as failed
    before it starts (the pre-count after its gave-up check), so a process
    killed anywhere in it
    (SIGKILL, out of memory, a segfault) has already been counted. At the
    end of a resume that kept its snapshot, by raising or by returning, this
    records last_error and the records still missing (last_missing, from
    report["remaining"]). When records landed in this attempt
    (report["landed"]) and it filled the collection in place (resume_mode
    incremental), the resume made progress and failures goes back to 0.
    Records that only left the snapshot (dead-lettered) are not progress,
    and neither is a full-mode attempt, which starts over from an empty
    collection. The count goes into report["resume_failures"]. A fresh
    rebuild that kept its snapshot records last_missing only. A run that ends
    with no snapshot removes the sidecar.

    Its own bookkeeping never raises, except that a soft time limit landing
    during it is re-raised once the bookkeeping has been finished (the new
    state is computed once, so a second pass writes the same thing)."""
    resuming = _snapshot_path(name).exists()
    error = None
    planned: dict[str, Any] = {}

    def settle():
        if not _snapshot_path(name).exists():
            _rebuild_state_path(name).unlink(missing_ok=True)
            return
        if report["method"] in ("already_resumed", "resume_gave_up", "skipped_count_failed"):
            return
        remaining = report.get("remaining")
        if not resuming:
            if remaining is not None and "state" not in planned:
                planned["state"] = {**rebuild_state(name), "last_missing": remaining}
            if "state" in planned:
                _save_rebuild_state(name, planned["state"])
            return
        if "state" not in planned:
            state = rebuild_state(name)          # this attempt is already counted in it
            landed = int(report.get("landed") or 0)
            if landed and report.get("resume_mode") == "incremental":
                log.info(f"rebuild_collection: '{name}' resume made progress ({landed} records "
                         f"landed); failure count reset")
                state["failures"] = 0
            if remaining is not None:
                state["last_missing"] = remaining
            state["last_error"] = str(error or report.get("error") or report["method"])[:500]
            state.setdefault("created", _snapshot_path(name).stat().st_mtime)
            planned["state"] = state
        _save_rebuild_state(name, planned["state"])
        report["resume_failures"] = int(planned["state"].get("failures") or 0)
        if report["resume_failures"]:
            log.error(f"rebuild_collection: resume of '{name}' failed "
                      f"({report['resume_failures']}/{_MAX_RESUME_FAILURES}): "
                      f"{planned['state']['last_error']}")

    try:
        yield resuming
    except BaseException as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        cut_off = None
        for _ in range(2):   # the soft limit fires once, so a second pass finishes
            try:
                settle()
                break
            except _SoftTimeLimit as e:
                cut_off = e
            except Exception as e:
                log.error(f"rebuild_collection: could not record the '{name}' resume outcome: {e}")
                break
        if cut_off is not None:
            raise cut_off


def _warm_embedder() -> None:
    """Loads the embedding model before rebuild_collection takes the write
    lock: on a cold cache _get_embedder() downloads from the Hub, and every
    writer would wait on that."""
    _get_embedder()


def _resume_todo(col, docs: list[tuple[str, str, dict]]) -> list[tuple[str, str, dict]]:
    """The records of a resume that the live collection lacks, or holds with
    a different text or metadata. Raises if the collection can't be read
    (count() first: that is the call a broken index fails)."""
    col.count()
    live: dict[str, tuple[str, dict]] = {}
    ids = [d[0] for d in docs]
    for lo in range(0, len(ids), _UPSERT_BATCH):
        got = col.get(ids=ids[lo:lo + _UPSERT_BATCH], include=["documents", "metadatas"])
        live.update((i, (d, m or {})) for i, d, m in
                    zip(got["ids"], got["documents"], got["metadatas"]))
    return [d for d in docs if live.get(d[0]) != (d[1], d[2] or {})]


def _deadletter_path(name: str) -> Path:
    """Records a resume gave up on: they failed to upsert on
    _MAX_RESUME_FAILURES resumes in a row, and were taken out of the
    snapshot so it can finish."""
    return _DB_PATH / f".rebuild-{name}.deadletter.jsonl"


def _fsync_dir(path: Path) -> None:
    """fsyncs a directory, so a file just created in it survives a crash."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _dead_letter(name: str, records: list[tuple[str, str, dict, str]]) -> None:
    """Appends (id, document, metadata, error) records to the dead-letter
    file, fsynced (and its directory, the first time), and logs them at
    WARNING."""
    now = datetime.now(timezone.utc).isoformat()
    path = _deadletter_path(name)
    created = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        for doc_id, text, meta, err in records:
            f.write(json.dumps({"id": doc_id, "document": text, "metadata": meta,
                                "error": err, "dead_lettered_at": now}) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if created:
        _fsync_dir(path.parent)
    log.warning(f"rebuild_collection: '{name}' {len(records)} documents failed on "
                f"{_MAX_RESUME_FAILURES} resumes in a row; moved to {_deadletter_path(name).name}: "
                f"{[r[0] for r in records][:5]}")


_CANARY_ID = "__vector_store_canary__"
_CANARY_TEXT = "vector store write check"


def _isolated_cap(n_docs: int) -> int:
    """Most records a run may set aside: 1% of the snapshot, at least 1 and
    at most 10, so a small collection can't lose most of itself."""
    return min(10, max(1, n_docs // 100))


def _canary_upsert(name: str, col) -> bool:
    """Whether the store takes writes now: embeds a fixed text, upserts it
    under a scratch id (_CANARY_ID), reads the document back, then deletes
    it and checks it is gone. A new id is used, not a re-upsert of a record
    already there: an unchanged record reads back the same whether or not the
    write happened. Its embedding is the text's scaled by 1000, far from every
    real (unit-length) one, so a search() in the few milliseconds it exists
    ranks it last; it is deleted before the lock is released, so nothing
    sees it afterwards. One left behind by a process killed mid-check is
    deleted first, and search() and _read_collection_docs skip the id. False
    when any step fails or the scratch record is still there."""
    ok = False
    try:
        col.delete(ids=[_CANARY_ID])     # a leftover from a killed check
        emb = [v * 1000.0 for v in embed_many([_CANARY_TEXT])[0]]
        col.upsert(ids=[_CANARY_ID], embeddings=[emb], documents=[_CANARY_TEXT],
                   metadatas=[{"canary": True}])
        ok = col.get(ids=[_CANARY_ID], include=["documents"])["documents"] == [_CANARY_TEXT]
    except _SoftTimeLimit:
        raise
    except Exception as e:
        log.warning(f"rebuild_collection: '{name}' canary write failed ({e}); nothing dead-lettered")
    finally:
        try:
            col.delete(ids=[_CANARY_ID])
            if col.get(ids=[_CANARY_ID], include=[])["ids"]:
                ok = False
                log.error(f"rebuild_collection: '{name}' canary record still present after delete")
        except _SoftTimeLimit:
            ok = False
            raise
        except Exception as e:
            ok = False
            log.error(f"rebuild_collection: '{name}' could not delete the canary record: {e}")
    return ok


def _set_aside_after_giving_up(name: str, state: dict[str, Any], report: dict[str, Any]) -> None:
    """One last look when resumes have just stopped: if the only records
    still missing are ones that already failed to upsert (id_failures), are
    few (_isolated_cap) and the store takes writes (_canary_upsert), they are
    dead-lettered and the snapshot is removed, so prune can run again. A cut
    or a crash can count a resume before a bad record has failed three
    times, which would otherwise leave prune paused for good. It runs once
    per snapshot (final_check_done is written before the live collection is
    read), so a crash in it is not repeated every hour."""
    failing = state.get("id_failures") or {}
    if not failing or state.get("final_check_done"):
        return
    try:
        _save_rebuild_state(name, {**state, "final_check_done": True})
    except _SoftTimeLimit:
        raise
    except Exception as e:
        log.error(f"rebuild_collection: '{name}' final check not recorded ({e}); skipped")
        return
    docs = _load_snapshot(name) or []
    _reopen_client()
    _bump_generation()
    try:
        col = _get_collection(name)
        ids = [d[0] for d in docs]
        present: set[str] = set()
        for lo in range(0, len(ids), _UPSERT_BATCH):
            present.update(col.get(ids=ids[lo:lo + _UPSERT_BATCH], include=[])["ids"])
        missing = set(ids) - present
        if not (missing and missing <= failing.keys() and len(missing) < len(docs)
                and len(missing) <= _isolated_cap(len(docs)) and _canary_upsert(name, col)):
            return
        err = f"failed to upsert on resumes; last error: {state.get('last_error')}"[:300]
        _dead_letter(name, [d + (err,) for d in docs if d[0] in missing])
        _snapshot_path(name).unlink(missing_ok=True)
        report.update(method="gave_up_set_aside", dead_lettered=len(missing), remaining=0,
                      restored=len(present), docs_found=len(docs), error=None)
    except _SoftTimeLimit:
        raise
    except Exception as e:
        _forget_generation()
        log.error(f"rebuild_collection: '{name}' final check after giving up failed: {e}")


def rebuild_collection(name: str, resume_only: bool = False) -> dict[str, Any]:
    """Recreates a broken collection and puts its documents back. Returns a
    report: collection, method, docs_before (records in sqlite),
    docs_without_text (records with no document text, which can't be
    re-embedded), docs_found, restored, docs_after (count() afterwards),
    error, started_at, finished_at, resume_failures, resume_mode
    (incremental or full, on a resume), upserted (records attempted), landed
    (records whose upsert succeeded in this run), remaining (snapshot records
    still missing at the end), dead_lettered.

    The ids, texts and metadata (timestamp included, so prune ages are kept)
    are read from sqlite and written to a snapshot file (_snapshot_path)
    before the collection is deleted. Each text is then re-embedded
    (embed_many, as store() embeds) and upserted _UPSERT_BATCH at a time. The
    snapshot is removed only once every id is back in the collection. A
    rebuild killed partway (the 60 s stop grace at the 19:00 stack-down)
    leaves the snapshot, and the next call (check_vector_store_health starts
    with pending_rebuilds()) merges it with whatever sqlite holds now (the
    newer copy of an id wins) and runs again: method snapshot_resume.

    A resume does not delete the collection when the earlier run had already
    recreated it (the sidecar's recreated flag): it upserts only the records
    the live collection lacks or holds with a different text or metadata
    (resume_mode incremental), so search() keeps what is there and a resume
    cut off by the soft limit makes progress each run. When the collection
    was never recreated (the run stopped before its delete, so the broken
    index may still be there) or can't be read, the resume deletes and
    rebuilds it in full (resume_mode full).

    A record whose upsert fails on _MAX_RESUME_FAILURES resumes in a row is
    moved to the dead-letter file (_dead_letter) and out of the snapshot, so
    the snapshot can finish and prune can run again. Only an isolated failure
    qualifies, and only when the store is shown to take writes in the same
    run: another upsert landed, or a canary re-upsert of a record already
    present succeeds. A resume restores every record in the snapshot,
    whatever its age; prune keeps its own schedule.

    Nothing is deleted when sqlite can't be read (skipped_read_failed) or
    yields no documents while the collection has records there
    (skipped_zero_docs); both are logged at ERROR and the store is left as it
    is. Runs under the write lock; nothing else writes until it is done.

    resume_only (the health check's resume of pending_rebuilds(), listed
    before the lock): when the snapshot is gone by the time the lock is held,
    another runner already finished it, and nothing is touched
    (already_resumed).

    Every resume is counted as failed in the snapshot's sidecar before it
    does anything (so a process killed partway is counted), and
    _resume_attempt clears or keeps that count when it ends. A resume that
    can't record the count does nothing (skipped_count_failed). After
    _MAX_RESUME_FAILURES in a row without progress, calls stop touching the
    collection (resume_gave_up) and the snapshot is kept; deleting
    .rebuild-<name>.state.json allows retries again."""
    report: dict[str, Any] = {
        "collection": name, "method": "rebuild", "docs_before": None,
        "docs_without_text": None, "docs_found": 0, "restored": 0, "docs_after": None,
        "error": None, "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
        "resume_failures": 0, "resume_mode": None, "upserted": 0, "landed": 0,
        "remaining": None, "dead_lettered": 0,
    }
    _warm_embedder()
    with _write_lock(), _resume_attempt(name, report) as resuming:
        if resume_only and not resuming:
            log.info(f"rebuild_collection: '{name}' snapshot already resumed by another run")
            report.update(method="already_resumed",
                          finished_at=datetime.now(timezone.utc).isoformat())
            return report
        if resuming:
            state = rebuild_state(name)
            report["resume_failures"] = int(state.get("failures") or 0)
            if report["resume_failures"] >= _MAX_RESUME_FAILURES:
                log.error(f"rebuild_collection: '{name}' resume failed {report['resume_failures']} "
                          f"times in a row; not retrying, snapshot kept")
                report.update(method="resume_gave_up", error=state.get("last_error"),
                              finished_at=datetime.now(timezone.utc).isoformat())
                _set_aside_after_giving_up(name, state, report)
                return report
            # Counted before anything can go wrong: a process killed from here
            # on (an upsert, an embed, the full rebuild's delete) still counts.
            try:
                _save_rebuild_state(name, {
                    **state, "failures": report["resume_failures"] + 1,
                    "last_error": "process died during the resume (killed, out of memory or crashed)"})
            except _SoftTimeLimit:
                raise
            except Exception as e:
                log.error(f"rebuild_collection: could not record the '{name}' resume attempt "
                          f"({e}); not resuming")
                report.update(method="skipped_count_failed",
                              error=f"could not record the resume attempt: {e}",
                              finished_at=datetime.now(timezone.utc).isoformat())
                return report
        try:
            pending = _load_snapshot(name)
            report["docs_before"] = _count_collection_rows(name)
            saved = _read_collection_docs(name)
            report["docs_without_text"] = max(report["docs_before"] - len(saved), 0)
        except Exception as e:
            log.error(f"rebuild_collection: could not read '{name}' records; nothing deleted: {e}")
            report.update(method="skipped_read_failed", error=str(e),
                          finished_at=datetime.now(timezone.utc).isoformat())
            return report
        if report["docs_before"] and not saved:
            log.error(f"rebuild_collection: '{name}' has {report['docs_before']} records in sqlite "
                      f"but none with a document were read; nothing deleted")
            report.update(method="skipped_zero_docs",
                          error=f"0 documents read from {report['docs_before']} sqlite records",
                          finished_at=datetime.now(timezone.utc).isoformat())
            return report
        merged: dict[str, tuple[str, dict]] = {}
        if pending is not None:
            report["method"] = "snapshot_resume"
            merged.update((doc_id, (text, meta)) for doc_id, text, meta in pending)
        merged.update((doc_id, (text, meta)) for doc_id, text, meta in saved)
        docs = [(doc_id, text, meta) for doc_id, (text, meta) in merged.items()]
        report["docs_found"] = len(docs)
        if docs:
            _write_snapshot(name, docs)
            if not resuming:
                _save_rebuild_state(name, {"created": time.time(), "failures": 0,
                                           "last_error": None})

        failed: dict[str, str] = {}
        landed: set[str] = set()
        _reopen_client()
        _bump_generation()   # before the delete: a kill from here on still makes others reopen
        try:
            col = None
            todo = docs
            if pending is not None and rebuild_state(name).get("recreated"):
                try:
                    col = _get_collection(name)
                    todo = _resume_todo(col, docs)
                    report["resume_mode"] = "incremental"
                except _SoftTimeLimit:
                    raise
                except Exception as e:
                    log.warning(f"rebuild_collection: '{name}' can't be read for an incremental "
                                f"resume ({e}); rebuilding it in full")
                    col, todo = None, docs
                    _collections.pop(name, None)
            if col is None:
                if pending is not None:
                    report["resume_mode"] = "full"
                client = _get_client()
                try:
                    old_id = client.get_collection(name).id
                except Exception:
                    old_id = None
                deleted = True
                try:
                    client.delete_collection(name)
                except Exception as e:
                    deleted = False   # may already be gone/unreadable; get_or_create below still works
                    log.warning(f"rebuild_collection: deleting '{name}' failed: {e}")
                _collections.pop(name, None)
                col = _get_collection(name)
                if _snapshot_path(name).exists() and deleted and col.id != old_id:
                    # A new collection: from here a resume can fill it in place.
                    # Not set when the delete raised, which may leave the old
                    # index in use; the next resume then rebuilds in full again.
                    _save_rebuild_state(name, {**rebuild_state(name), "recreated": True})
            report["upserted"] = len(todo)
            report["remaining"] = len(todo)    # counts down as batches land
            for lo in range(0, len(todo), _UPSERT_BATCH):
                chunk = todo[lo:lo + _UPSERT_BATCH]
                try:
                    col.upsert(ids=[d[0] for d in chunk],
                               embeddings=embed_many([d[1][:8192] for d in chunk]),
                               documents=[d[1] for d in chunk],
                               metadatas=[d[2] or None for d in chunk])
                    report["remaining"] -= len(chunk)
                    landed.update(d[0] for d in chunk)
                    report["landed"] = len(landed)
                    continue
                except _SoftTimeLimit:
                    raise   # cut off: keep the snapshot, the next run resumes
                except Exception as e:
                    log.warning(f"rebuild_collection: '{name}' batch at {lo} failed ({e}); "
                                f"upserting its documents one by one")
                for doc_id, text, meta in chunk:
                    try:
                        col.upsert(ids=[doc_id], embeddings=embed_many([text[:8192]]),
                                   documents=[text], metadatas=[meta] if meta else None)
                        report["remaining"] -= 1
                        landed.add(doc_id)
                        report["landed"] = len(landed)
                    except _SoftTimeLimit:
                        raise
                    except Exception as e:
                        failed[doc_id] = f"{type(e).__name__}: {e}"[:300]
                        log.warning(f"rebuild_collection: '{name}' doc {doc_id!r} not restored: {e}")
            present: set[str] = set()
            ids = [d[0] for d in docs]
            for lo in range(0, len(ids), _UPSERT_BATCH):
                present.update(col.get(ids=ids[lo:lo + _UPSERT_BATCH], include=[])["ids"])
            report["restored"] = len(present)
            report["docs_after"] = col.count()
        except BaseException:
            _forget_generation()
            raise
        # Any missing id keeps the snapshot, including ids whose upsert raised:
        # a disk-full store fails every upsert, and dropping the snapshot then
        # would lose the collection. _resume_attempt bounds the retries, and
        # an id that fails on _MAX_RESUME_FAILURES resumes in a row is
        # dead-lettered so the rest can finish.
        missing = set(ids) - present
        if resuming and _snapshot_path(name).exists():
            state = rebuild_state(name)
            prior = state.get("id_failures") or {}
            streak = {i: int(prior.get(i) or 0) + 1 for i in failed if i in missing}
            dead = sorted(i for i, n in streak.items() if n >= _MAX_RESUME_FAILURES)
            # Only isolated failures: everything else is back, and the
            # failures are a few. A store that fails every upsert (disk full)
            # must keep its snapshot, not dead-letter the whole collection.
            isolated = (missing <= failed.keys()
                        and len(missing) < len(docs)
                        and len(missing) <= _isolated_cap(len(docs)))
            # And the store must be shown to take writes in this same run: an
            # incremental resume tries only the missing records, so "the rest
            # is back" says nothing when every upsert it tried failed.
            if dead and isolated and not (landed & present):
                isolated = _canary_upsert(name, col)
            if not isolated:
                dead = []
            if dead:
                by_id = {d[0]: d for d in docs}
                _dead_letter(name, [(*by_id[i], failed[i]) for i in dead])
                _write_snapshot(name, [d for d in docs if d[0] not in set(dead)])
                missing -= set(dead)
                report["dead_lettered"] = len(dead)
            state["id_failures"] = {i: n for i, n in streak.items() if i not in set(dead)}
            _save_rebuild_state(name, state)
        report["remaining"] = len(missing)
        if missing:
            report["error"] = (f"{len(missing)} documents missing after the rebuild "
                               f"({len(missing & failed.keys())} failed to upsert)")
            log.error(f"rebuild_collection: '{name}' {report['error']}: {sorted(missing)[:5]}; "
                      f"keeping {_snapshot_path(name).name} for the next run")
        else:
            _snapshot_path(name).unlink(missing_ok=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    log.info(f"rebuild_collection: '{name}' rebuilt ({report['method']}), "
             f"{report['restored']}/{report['docs_found']} documents restored")
    return report


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
