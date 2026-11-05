"""A rebuild on a real Chroma store (a temp dir, never the repo's .chromadb):
a rebuild killed partway loses no document, and analyst retrieval (col.get()
and top-8 search) is the same before and after a rebuild.

Embeddings are a deterministic hash of the text, so a re-embedded document
gets the same vector it was stored with, as all-MiniLM-L6-v2 does."""
import hashlib
import json
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import vector_store as vs

_REPO = Path(__file__).resolve().parent.parent
_N_DOCS = 600          # three upsert batches of 256 / 256 / 88
_QUERIES = [f"doc {k} about AAPL" for k in (3, 77, 150, 420, 599)]


def _det(text):
    r = random.Random(hashlib.sha1(text.encode()).hexdigest())
    return [r.random() for _ in range(16)]


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(vs, "_DB_PATH", tmp_path)
    monkeypatch.setattr(vs, "_WRITE_LOCK_PATH", tmp_path / ".write.lock")
    monkeypatch.setattr(vs, "_seen_generation", None)
    monkeypatch.setattr(vs, "_client", None)
    monkeypatch.setattr(vs, "_collections", {})
    monkeypatch.setattr(vs, "embed", _det)
    monkeypatch.setattr(vs, "embed_many", lambda texts: [_det(t) for t in texts])
    monkeypatch.setattr(vs, "_warm_embedder", lambda: None)
    for i in range(_N_DOCS):
        vs.store("news", f"url-{i}", f"doc {i} about AAPL " * 3,
                 {"ticker": "AAPL", "n": i, "f": i / 3, "b": i % 2 == 0})
    yield tmp_path
    vs._reopen_client()


def _snapshot():
    """What the analyst reads: every record, and the top 8 for a few queries."""
    got = vs._get_collection("news").get(include=["documents", "metadatas"])
    records = sorted(zip(got["ids"], got["documents"], got["metadatas"]))
    hits = [[(r["text"], r["metadata"], round(r["score"], 6)) for r in vs.search("news", q, 8)]
            for q in _QUERIES]
    return records, hits


def test_retrieval_is_identical_across_a_rebuild(store):
    before = _snapshot()
    vs._reopen_client()
    report = vs.rebuild_collection("news")
    assert (report["method"], report["docs_found"], report["restored"], report["docs_after"]) == \
        ("rebuild", _N_DOCS, _N_DOCS, _N_DOCS)
    after = _snapshot()
    assert after[0] == before[0]          # col.get(): ids, documents, metadata and its types
    assert after[1] == before[1]          # top-8 search results and scores
    assert len(after[0]) == _N_DOCS and all(len(h) == 8 for h in after[1])
    assert vs.pending_rebuilds() == []


def test_a_rebuild_killed_partway_loses_no_document(store):
    before = _snapshot()
    vs._reopen_client()
    # A separate process rebuilds and is killed (os._exit, as SIGKILL at the
    # stop grace) while embedding the second batch: batch one is upserted.
    script = textwrap.dedent(f"""
        import hashlib, os, random, sys
        from pathlib import Path
        sys.path.insert(0, {str(_REPO)!r})
        import vector_store as vs
        vs._DB_PATH = Path({str(store)!r}); vs._WRITE_LOCK_PATH = vs._DB_PATH / ".write.lock"
        calls = [0]
        def det(t):
            r = random.Random(hashlib.sha1(t.encode()).hexdigest())
            return [r.random() for _ in range(16)]
        def embed_many(texts):
            calls[0] += 1
            if calls[0] == 2:
                os._exit(137)
            return [det(t) for t in texts]
        vs.embed_many = embed_many
        vs._warm_embedder = lambda: None
        vs.rebuild_collection("news")
    """)
    killed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=300, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert killed.returncode == 137, killed.stderr[-2000:]

    # The kill left a partial collection and the snapshot of every record.
    assert vs.pending_rebuilds() == ["news"]
    assert vs._count_collection_rows("news") == 256
    gen_after_kill = vs._read_generation()

    # A store() from this (long-lived) process before the resume: the killed
    # rebuild bumped the generation, so it reopens instead of writing from
    # its old client, and the newer copy of that id survives the resume.
    vs.store("news", "url-5", "doc 5 about AAPL, updated after the kill", {"ticker": "AAPL"})
    vs.store("news", "late-1", "a headline stored after the kill", {"ticker": "MSFT"})
    assert vs._seen_generation == gen_after_kill + 2

    col_id = vs._get_collection("news").id
    live_before = vs._count_collection_rows("news")   # first batch + what store() added
    report = vs.rebuild_collection("news")
    assert report["method"] == "snapshot_resume"
    assert (report["docs_found"], report["restored"], report["docs_after"], report["error"]) == \
        (_N_DOCS + 1, _N_DOCS + 1, _N_DOCS + 1, None)
    # the killed run had recreated the collection: the resume fills it in
    # place (same collection, no delete) with only the records it lacked
    assert report["resume_mode"] == "incremental"
    assert report["upserted"] == _N_DOCS + 1 - live_before
    vs._reopen_client()
    assert vs._get_collection("news").id == col_id
    assert vs.pending_rebuilds() == []

    records, _ = _snapshot()
    by_id = {r[0]: r for r in records}
    assert by_id["url-5"][1] == "doc 5 about AAPL, updated after the kill"
    assert by_id["late-1"][1] == "a headline stored after the kill"
    expected = [r for r in before[0] if r[0] != "url-5"]
    assert [r for r in records if r[0] not in ("url-5", "late-1")] == expected


def test_a_rebuild_cut_off_by_the_soft_time_limit_keeps_its_snapshot(store, monkeypatch):
    from celery.exceptions import SoftTimeLimitExceeded
    before = _snapshot()
    vs._reopen_client()
    calls = [0]

    def embed_many(texts):
        calls[0] += 1
        if calls[0] == 2:
            raise SoftTimeLimitExceeded()
        return [_det(t) for t in texts]
    monkeypatch.setattr(vs, "embed_many", embed_many)
    with pytest.raises(SoftTimeLimitExceeded):
        vs.rebuild_collection("news")
    assert vs.pending_rebuilds() == ["news"] and vs._seen_generation is None

    monkeypatch.setattr(vs, "embed_many", lambda texts: [_det(t) for t in texts])
    report = vs.rebuild_collection("news", resume_only=True)
    assert (report["method"], report["restored"]) == ("snapshot_resume", _N_DOCS)
    assert _snapshot() == before
    # a runner that queued behind this one finds the snapshot gone
    assert vs.rebuild_collection("news", resume_only=True)["method"] == "already_resumed"
    assert _snapshot() == before


def test_a_store_that_fails_every_upsert_keeps_the_snapshot_and_stops_after_three(store,
                                                                                  monkeypatch):
    before = _snapshot()
    vs._reopen_client()

    def enospc(texts):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(vs, "embed_many", enospc)
    report = vs.rebuild_collection("news")          # the collection is gone, nothing restored
    assert report["restored"] == 0 and "600 documents missing" in report["error"]
    assert vs.pending_rebuilds() == ["news"]        # snapshot kept, not dropped as "failed docs"
    for n in (1, 2, 3):
        assert vs.rebuild_collection("news", resume_only=True)["resume_failures"] == n
    assert vs.rebuild_collection("news", resume_only=True)["method"] == "resume_gave_up"
    assert vs.snapshot_age_hours("news") is not None
    assert not vs._deadletter_path("news").exists()   # a systemic failure dead-letters nothing

    # the cause fixed and the sidecar removed by hand: the next resume restores everything
    monkeypatch.setattr(vs, "embed_many", lambda texts: [_det(t) for t in texts])
    vs._rebuild_state_path("news").unlink()
    report = vs.rebuild_collection("news", resume_only=True)
    assert (report["method"], report["restored"], report["error"]) == ("snapshot_resume", _N_DOCS, None)
    assert vs.pending_rebuilds() == [] and _snapshot() == before


def _kill_after_first_batch(store_path):
    """Runs a rebuild in a subprocess that dies (os._exit) while embedding the
    second batch: the collection is recreated and holds the first 256."""
    script = textwrap.dedent(f"""
        import hashlib, os, random, sys
        from pathlib import Path
        sys.path.insert(0, {str(_REPO)!r})
        import vector_store as vs
        vs._DB_PATH = Path({str(store_path)!r}); vs._WRITE_LOCK_PATH = vs._DB_PATH / ".write.lock"
        calls = [0]
        def det(t):
            r = random.Random(hashlib.sha1(t.encode()).hexdigest())
            return [r.random() for _ in range(16)]
        def embed_many(texts):
            calls[0] += 1
            if calls[0] == 2:
                os._exit(137)
            return [det(t) for t in texts]
        vs.embed_many = embed_many
        vs._warm_embedder = lambda: None
        vs.rebuild_collection("news")
    """)
    killed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=300, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert killed.returncode == 137, killed.stderr[-2000:]


def test_a_resume_after_a_partial_rebuild_never_deletes_and_ends_complete(store, monkeypatch):
    before = _snapshot()
    vs._reopen_client()
    _kill_after_first_batch(store)
    assert vs._count_collection_rows("news") == 256 and vs.rebuild_state("news")["recreated"]

    from chromadb.api.client import Client
    deleted = []
    real_delete = Client.delete_collection
    monkeypatch.setattr(Client, "delete_collection",
                        lambda self, name, *a, **k: deleted.append(name) or real_delete(self, name, *a, **k))
    # search() during the gap reads the 256 already there, not an empty collection
    assert len(vs.search("news", _QUERIES[0], 8)) == 8
    report = vs.rebuild_collection("news", resume_only=True)
    assert deleted == []
    assert (report["method"], report["resume_mode"], report["upserted"], report["docs_after"]) == \
        ("snapshot_resume", "incremental", _N_DOCS - 256, _N_DOCS)
    assert _snapshot() == before and vs.pending_rebuilds() == []


def test_a_resume_against_an_unreadable_collection_rebuilds_in_full(store, monkeypatch):
    before = _snapshot()
    vs._reopen_client()
    _kill_after_first_batch(store)
    col_id = vs._get_collection("news").id

    def broken(col, docs):
        raise RuntimeError("Failed to apply logs to the hnsw segment writer")
    monkeypatch.setattr(vs, "_resume_todo", broken)
    report = vs.rebuild_collection("news", resume_only=True)
    assert (report["resume_mode"], report["upserted"], report["restored"]) == \
        ("full", _N_DOCS, _N_DOCS)
    vs._reopen_client()
    assert vs._get_collection("news").id != col_id          # deleted and recreated
    assert _snapshot() == before and vs.pending_rebuilds() == []


def test_a_resume_of_a_run_stopped_before_its_delete_rebuilds_in_full(store):
    before = _snapshot()
    # snapshot and sidecar written, then killed before the delete: the
    # (possibly broken) original collection is still there, never opened here
    vs._write_snapshot("news", vs._read_collection_docs("news"))
    vs._save_rebuild_state("news", {"created": 0, "failures": 0, "last_error": None})
    vs._reopen_client()
    col_id = vs._get_collection("news").id
    report = vs.rebuild_collection("news", resume_only=True)
    assert (report["resume_mode"], report["upserted"]) == ("full", _N_DOCS)
    vs._reopen_client()
    assert vs._get_collection("news").id != col_id
    assert _snapshot() == before


def test_a_resume_that_kills_the_process_in_count_still_gives_up_and_alerts(store, monkeypatch):
    import alert_policy
    import alerts
    import db
    import tasks
    vs._reopen_client()
    _kill_after_first_batch(store)
    assert vs.rebuild_state("news")["recreated"]
    crash = textwrap.dedent(f"""
        import os, sys
        from pathlib import Path
        sys.path.insert(0, {str(_REPO)!r})
        import vector_store as vs
        vs._DB_PATH = Path({str(store)!r}); vs._WRITE_LOCK_PATH = vs._DB_PATH / ".write.lock"
        vs._warm_embedder = lambda: None
        vs._resume_todo = lambda col, docs: os._exit(139)   # a segfault inside count()
        vs.rebuild_collection("news", resume_only=True)
    """)
    for n in (1, 2, 3):
        died = subprocess.run([sys.executable, "-c", crash], capture_output=True, text=True,
                              timeout=300, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        assert died.returncode == 139, died.stderr[-2000:]
        assert vs.rebuild_state("news")["failures"] == n
    assert "process died" in vs.rebuild_state("news")["last_error"]

    report = vs.rebuild_collection("news", resume_only=True)
    assert report["method"] == "resume_gave_up" and vs.pending_rebuilds() == ["news"]

    state, sent = {}, []
    monkeypatch.setattr(db, "get_state", lambda key, default=None: state.get(key, default))
    monkeypatch.setattr(db, "set_state", lambda key, value: state.__setitem__(key, value))
    monkeypatch.setattr(alerts, "_send", lambda text, **k: sent.append(text))
    assert tasks._alert_unfinished_rebuilds(vs) == ["news"]
    assert "resumes stopped after 3 failures" in sent[0]
    assert alert_policy.classify(sent[0])[0] == alert_policy.CRITICAL


def _run(script):
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=300, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


@pytest.mark.parametrize("mode", ["full", "incremental"])
def test_a_resume_killed_in_an_upsert_counts_and_gives_up_after_three(store, mode):
    if mode == "incremental":
        vs._reopen_client()
        _kill_after_first_batch(store)
        assert vs.rebuild_state("news")["recreated"]
    else:
        # snapshot and sidecar, stopped before the delete: the resume rebuilds in full
        vs._write_snapshot("news", vs._read_collection_docs("news"))
        vs._save_rebuild_state("news", {"created": 0, "failures": 0, "last_error": None})
        vs._reopen_client()
    crash = textwrap.dedent(f"""
        import os, sys
        from pathlib import Path
        sys.path.insert(0, {str(_REPO)!r})
        import vector_store as vs
        vs._DB_PATH = Path({str(store)!r}); vs._WRITE_LOCK_PATH = vs._DB_PATH / ".write.lock"
        vs._warm_embedder = lambda: None
        vs.embed_many = lambda texts: os._exit(137)    # killed while embedding an upsert batch
        vs.rebuild_collection("news", resume_only=True)
    """)
    for n in (1, 2, 3):
        died = _run(crash)
        assert died.returncode == 137, died.stderr[-2000:]
        assert vs.rebuild_state("news")["failures"] == n
    report = vs.rebuild_collection("news", resume_only=True)
    assert report["method"] == "resume_gave_up" and vs.pending_rebuilds() == ["news"]


def test_a_resume_that_makes_progress_resets_the_failure_count(store, monkeypatch):
    from celery.exceptions import SoftTimeLimitExceeded
    vs._reopen_client()
    _kill_after_first_batch(store)                            # 256 of 600 live
    vs._save_rebuild_state("news", {**vs.rebuild_state("news"), "failures": 2,
                                    "last_missing": _N_DOCS - 256})
    calls = [0]

    def cut_on_second(texts):
        calls[0] += 1
        if calls[0] == 2:
            raise SoftTimeLimitExceeded()
        return [_det(t) for t in texts]
    monkeypatch.setattr(vs, "embed_many", cut_on_second)
    with pytest.raises(SoftTimeLimitExceeded):
        vs.rebuild_collection("news", resume_only=True)
    state = vs.rebuild_state("news")
    assert (state["failures"], state["last_missing"]) == (0, _N_DOCS - 512)   # 256 more landed

    def cut_at_once(texts):
        raise SoftTimeLimitExceeded()
    monkeypatch.setattr(vs, "embed_many", cut_at_once)
    with pytest.raises(SoftTimeLimitExceeded):
        vs.rebuild_collection("news", resume_only=True)
    state = vs.rebuild_state("news")
    assert (state["failures"], state["last_missing"]) == (1, _N_DOCS - 512)   # no progress: counted

    monkeypatch.setattr(vs, "embed_many", lambda texts: [_det(t) for t in texts])
    report = vs.rebuild_collection("news", resume_only=True)
    assert (report["remaining"], report["docs_after"]) == (0, _N_DOCS) and vs.pending_rebuilds() == []


def test_a_record_that_fails_three_resumes_in_a_row_is_dead_lettered(store, monkeypatch, caplog):
    import json
    vs.store("news", "poison", "POISON record that never embeds", {"ticker": "X"})
    before = _snapshot()

    def embed_many(texts):
        if any("POISON" in t for t in texts):
            raise ValueError("cannot embed this text")
        return [_det(t) for t in texts]
    monkeypatch.setattr(vs, "embed_many", embed_many)
    vs._reopen_client()
    report = vs.rebuild_collection("news")                    # fresh: 1 missing, snapshot kept
    assert report["remaining"] == 1 and vs.pending_rebuilds() == ["news"]
    for n in (1, 2):
        report = vs.rebuild_collection("news", resume_only=True)
        assert (report["remaining"], report["dead_lettered"], report["landed"]) == (1, 0, 0)
        assert vs.rebuild_state("news")["id_failures"] == {"poison": n}
        assert vs.rebuild_state("news")["failures"] == n         # the canary is not progress
    with caplog.at_level("WARNING", logger="vector_store"):
        report = vs.rebuild_collection("news", resume_only=True)
    assert (report["dead_lettered"], report["remaining"], report["error"]) == (1, 0, None)
    assert vs.pending_rebuilds() == [] and vs.snapshot_age_hours("news") is None   # prune runs again
    [line] = vs._deadletter_path("news").read_text().splitlines()
    rec = json.loads(line)
    assert rec["id"] == "poison" and "cannot embed" in rec["error"]
    assert rec["document"] == "POISON record that never embeds" and rec["metadata"]["ticker"] == "X"
    assert "moved to .rebuild-news.deadletter.jsonl" in caplog.text
    records, _ = _snapshot()
    assert records == [r for r in before[0] if r[0] != "poison"]


def test_every_attempted_upsert_failing_dead_letters_nothing(store, monkeypatch):
    """Reviewer's case: 600 records live, 5 more in the snapshot, and every
    upsert fails (broken embedder, or a store that answers count/get but
    takes no writes). The incremental resume tries only the 5, so "the rest
    is back" holds; the canary re-upsert fails too, so nothing is set aside."""
    vs._reopen_client()
    live = vs._read_collection_docs("news")
    ghosts = [(f"g{i}", f"ghost {i}", {"ticker": "X"}) for i in range(5)]
    vs._write_snapshot("news", live + ghosts)
    vs._save_rebuild_state("news", {"created": 0, "failures": 0, "recreated": True})

    def disk_full(texts):
        raise RuntimeError("chroma: database or disk is full")
    monkeypatch.setattr(vs, "embed_many", disk_full)
    seen = [vs.rebuild_collection("news", resume_only=True) for _ in range(4)]
    assert [(r["method"], r["dead_lettered"]) for r in seen] == \
        [("snapshot_resume", 0)] * 3 + [("resume_gave_up", 0)]
    assert [r["resume_failures"] for r in seen[:3]] == [1, 2, 3]
    assert not vs._deadletter_path("news").exists() and vs.pending_rebuilds() == ["news"]
    vs._reopen_client()
    assert vs._get_collection("news").count() == _N_DOCS


def test_a_full_mode_attempt_is_not_progress(store, monkeypatch):
    from celery.exceptions import SoftTimeLimitExceeded
    vs._write_snapshot("news", vs._read_collection_docs("news"))
    vs._save_rebuild_state("news", {"created": 0, "failures": 1, "last_error": None})
    vs._reopen_client()
    calls = [0]

    def cut_on_second(texts):
        calls[0] += 1
        if calls[0] == 2:
            raise SoftTimeLimitExceeded()
        return [_det(t) for t in texts]
    monkeypatch.setattr(vs, "embed_many", cut_on_second)
    with pytest.raises(SoftTimeLimitExceeded):
        vs.rebuild_collection("news", resume_only=True)      # full: deleted, 256 landed, cut
    state = vs.rebuild_state("news")
    assert state["failures"] == 2 and state["recreated"]     # counted; next resume is incremental


def test_a_cut_before_the_bad_record_fails_three_times_still_sets_it_aside(store, monkeypatch):
    """The professor's sequence: the first resume is cut by the soft limit
    before anything lands, then two resumes where only the bad record fails.
    That is three counted failures with the record at two, so resumes stop
    before it can be dead-lettered; the final check sets it aside instead."""
    from celery.exceptions import SoftTimeLimitExceeded
    vs.store("news", "poison", "POISON record that never embeds", {"ticker": "X"})
    before = _snapshot()

    def embed_many(texts):
        if any("POISON" in t for t in texts):
            raise ValueError("cannot embed this text")
        return [_det(t) for t in texts]
    monkeypatch.setattr(vs, "embed_many", embed_many)
    vs._reopen_client()
    assert vs.rebuild_collection("news")["remaining"] == 1             # fresh: snapshot kept

    def cut(texts):
        raise SoftTimeLimitExceeded()
    monkeypatch.setattr(vs, "embed_many", cut)
    with pytest.raises(SoftTimeLimitExceeded):
        vs.rebuild_collection("news", resume_only=True)
    monkeypatch.setattr(vs, "embed_many", embed_many)
    for _ in range(2):
        vs.rebuild_collection("news", resume_only=True)
    state = vs.rebuild_state("news")
    assert state["failures"] == 3 and state["id_failures"] == {"poison": 2}

    report = vs.rebuild_collection("news", resume_only=True)
    assert (report["method"], report["dead_lettered"], report["remaining"]) == \
        ("gave_up_set_aside", 1, 0)
    assert vs.pending_rebuilds() == [] and vs.snapshot_age_hours("news") is None
    assert [json.loads(l)["id"] for l in vs._deadletter_path("news").read_text().splitlines()] \
        == ["poison"]
    records, _ = _snapshot()
    assert records == [r for r in before[0] if r[0] != "poison"]


def test_the_final_check_runs_once(store, monkeypatch):
    vs._reopen_client()
    live = vs._read_collection_docs("news")
    vs._write_snapshot("news", live + [("g0", "ghost", {"ticker": "X"})])
    vs._save_rebuild_state("news", {"created": 0, "failures": 3, "recreated": True,
                                    "id_failures": {"g0": 2}})
    canaries = []
    monkeypatch.setattr(vs, "_canary_upsert", lambda name, col: canaries.append(name) or False)
    assert vs.rebuild_collection("news", resume_only=True)["method"] == "resume_gave_up"
    assert vs.rebuild_collection("news", resume_only=True)["method"] == "resume_gave_up"
    assert canaries == ["news"] and vs.rebuild_state("news")["final_check_done"]
    assert vs.pending_rebuilds() == ["news"]


def test_the_canary_leaves_nothing_behind(store):
    vs._reopen_client()
    col = vs._get_collection("news")
    assert vs._canary_upsert("news", col) is True
    assert col.count() == _N_DOCS and col.get(ids=[vs._CANARY_ID])["ids"] == []
    assert vs._CANARY_TEXT not in [r["text"] for r in vs.search("news", vs._CANARY_TEXT, 8)]
    assert vs._CANARY_ID not in {i for i, _, _ in vs._read_collection_docs("news")}


def test_a_leftover_canary_is_never_searched_read_or_counted(monkeypatch, tmp_path):
    """A process killed between the canary's upsert and its delete leaves
    the scratch record, unscaled here to be the worst case, in a 3-document
    collection."""
    monkeypatch.setattr(vs, "_DB_PATH", tmp_path)
    monkeypatch.setattr(vs, "_WRITE_LOCK_PATH", tmp_path / ".write.lock")
    monkeypatch.setattr(vs, "_seen_generation", None)
    monkeypatch.setattr(vs, "_client", None)
    monkeypatch.setattr(vs, "_collections", {})
    monkeypatch.setattr(vs, "embed", _det)
    monkeypatch.setattr(vs, "embed_many", lambda texts: [_det(t) for t in texts])
    try:
        for i in range(3):
            vs.store("news", f"u{i}", f"doc {i} about AAPL", {"ticker": "AAPL"})
        col = vs._get_collection("news")
        col.upsert(ids=[vs._CANARY_ID], embeddings=[_det(vs._CANARY_TEXT)],
                   documents=[vs._CANARY_TEXT], metadatas=[{"canary": True}])
        hits = vs.search("news", vs._CANARY_TEXT, 8)
        assert len(hits) == 3 and vs._CANARY_TEXT not in [h["text"] for h in hits]
        assert sorted(i for i, _, _ in vs._read_collection_docs("news")) == ["u0", "u1", "u2"]
        assert vs._count_collection_rows("news") == 3
        assert vs._canary_upsert("news", col) is True       # removes the leftover too
        assert col.get(ids=[vs._CANARY_ID])["ids"] == [] and col.count() == 3
    finally:
        vs._reopen_client()
