"""Every vector-store write starts from the on-disk state, under the lock, and
a rebuild keeps the collection's documents.

Several worker processes each kept their own cached Chroma client. Under the
write lock each one still wrote from its own stale in-memory HNSW copy, which
left the on-disk 'news' index unreadable to a fresh process; the hourly health
check then deleted every news document (8 times, 2026-08-27 to 11-03)."""
import contextlib
import sqlite3
from unittest.mock import MagicMock

import pytest

import vector_store as vs


@pytest.fixture(autouse=True)
def scratch_store(monkeypatch, tmp_path):
    """The lock, generation and snapshot files go to a temp dir, never the
    repo's .chromadb."""
    monkeypatch.setattr(vs, "_DB_PATH", tmp_path)
    monkeypatch.setattr(vs, "_WRITE_LOCK_PATH", tmp_path / ".write.lock")
    monkeypatch.setattr(vs, "_seen_generation", None)
    return tmp_path


@pytest.fixture
def events(monkeypatch):
    seen = []

    @contextlib.contextmanager
    def fake_lock():
        seen.append("lock")
        try:
            yield
        finally:
            seen.append("unlock")

    monkeypatch.setattr(vs, "_write_lock", fake_lock)
    monkeypatch.setattr(vs, "_reopen_client", lambda: seen.append("reopen"))
    monkeypatch.setattr(vs, "embed", lambda text: [0.0])
    monkeypatch.setattr(vs, "embed_many", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(vs, "_count_collection_rows", lambda name: 2)
    monkeypatch.setattr(vs, "_warm_embedder", lambda: seen.append("warm"))
    return seen


def test_store_reopens_the_client_inside_the_lock_before_writing(monkeypatch, events):
    col = MagicMock()
    col.upsert.side_effect = lambda **k: events.append("upsert")
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    vs.store("news", "u1", "headline", {"ticker": "NVDA"})
    assert events == ["lock", "reopen", "upsert", "unlock"]


def test_back_to_back_writes_from_one_process_skip_the_reopen(monkeypatch, events):
    col = MagicMock()
    col.upsert.side_effect = lambda **k: events.append("upsert")
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    monkeypatch.setattr(vs, "_client", object())   # a client is open
    vs.store("news", "u1", "one", {})
    vs.store("news", "u2", "two", {})
    assert events == ["lock", "reopen", "upsert", "unlock", "lock", "upsert", "unlock"]


def test_a_write_from_another_process_makes_the_next_write_reopen(monkeypatch, events,
                                                                  scratch_store):
    col = MagicMock()
    col.upsert.side_effect = lambda **k: events.append("upsert")
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    monkeypatch.setattr(vs, "_client", object())
    vs.store("news", "u1", "one", {})
    gen = int((scratch_store / ".write.gen").read_text())
    (scratch_store / ".write.gen").write_text(str(gen + 1))   # another process wrote
    vs.store("news", "u2", "two", {})
    assert events == ["lock", "reopen", "upsert", "unlock", "lock", "reopen", "upsert", "unlock"]
    assert int((scratch_store / ".write.gen").read_text()) == gen + 2


def test_generation_is_bumped_before_the_write_and_a_failed_write_reopens_next(
        monkeypatch, events, scratch_store):
    col = MagicMock()
    seen_gen = []
    col.upsert.side_effect = lambda **k: seen_gen.append(vs._read_generation())
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    monkeypatch.setattr(vs, "_client", object())
    vs.store("news", "u1", "one", {})
    assert seen_gen == [1]          # a writer killed mid-upsert has already bumped it

    def boom(**k):
        raise RuntimeError("disk")
    col.upsert.side_effect = boom
    with pytest.raises(RuntimeError):
        vs.store("news", "u2", "two", {})
    assert vs._seen_generation is None


def test_prune_deletes_inside_the_lock_from_a_reopened_client(monkeypatch, events):
    col = MagicMock()
    col.count.return_value = 2
    col.get.return_value = {"ids": ["old", "new"],
                            "metadatas": [{"timestamp": "2000-01-01T00:00:00+00:00"},
                                          {"timestamp": "2999-01-01T00:00:00+00:00"}]}
    col.delete.side_effect = lambda ids: events.append(("delete", tuple(ids)))
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    assert vs.prune_old_documents("news", max_age_days=90) == 1
    assert events == ["lock", "reopen", ("delete", ("old",)), "unlock"]


def test_reopen_stops_the_old_system_and_drops_every_handle(monkeypatch):
    from chromadb.api.client import SharedSystemClient
    system = MagicMock()
    monkeypatch.setattr(SharedSystemClient, "_identifier_to_system", {"path": system})
    monkeypatch.setattr(SharedSystemClient, "_identifier_to_refcount", {"path": 1})
    monkeypatch.setattr(vs, "_client", object())
    monkeypatch.setattr(vs, "_collections", {"news": object()})
    vs._reopen_client()
    system.stop.assert_called_once()   # frees the Rust bindings (no leak)
    assert SharedSystemClient._identifier_to_system == {}
    assert vs._client is None and vs._collections == {}


def _fake_chroma_sqlite(path):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE collections (id TEXT, name TEXT);
        CREATE TABLE segments (id TEXT, scope TEXT, collection TEXT);
        CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT);
        CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT,
            int_value INTEGER, float_value REAL, bool_value INTEGER);
        INSERT INTO collections VALUES ('c-news', 'news'), ('c-arxiv', 'arxiv');
        INSERT INTO segments VALUES ('s-meta', 'METADATA', 'c-news'), ('s-vec', 'VECTOR', 'c-news'),
                                    ('s-ax', 'METADATA', 'c-arxiv');
        INSERT INTO embeddings VALUES (1, 's-meta', 'url-1'), (2, 's-meta', 'url-2'),
                                      (3, 's-ax', 'paper-1'), (4, 's-meta', 'no-doc');
        INSERT INTO embedding_metadata VALUES
            (1, 'chroma:document', 'NVDA beats', NULL, NULL, NULL),
            (1, 'ticker', 'NVDA', NULL, NULL, NULL),
            (1, 'timestamp', '2026-10-01T00:00:00+00:00', NULL, NULL, NULL),
            (1, 'n', NULL, 3, NULL, NULL),
            (1, 'x', NULL, NULL, 1.5, NULL),
            (1, 'flag', NULL, NULL, NULL, 1),
            (2, 'chroma:document', 'AMD guides', NULL, NULL, NULL),
            (3, 'chroma:document', 'a paper', NULL, NULL, NULL),
            (4, 'ticker', 'X', NULL, NULL, NULL);
    """)
    con.commit()
    con.close()


def test_documents_are_read_back_from_sqlite_with_their_metadata(monkeypatch, tmp_path):
    _fake_chroma_sqlite(tmp_path / "chroma.sqlite3")
    monkeypatch.setattr(vs, "_DB_PATH", tmp_path)
    docs = sorted(vs._read_collection_docs("news"))
    assert docs == [
        ("url-1", "NVDA beats", {"ticker": "NVDA", "timestamp": "2026-10-01T00:00:00+00:00",
                                 "n": 3, "x": 1.5, "flag": True}),
        ("url-2", "AMD guides", {}),
    ]


def test_rebuild_puts_the_documents_back(monkeypatch, events, scratch_store):
    saved = [("url-1", "NVDA beats", {"ticker": "NVDA", "timestamp": "t1"}),
             ("url-2", "AMD guides", {})]
    monkeypatch.setattr(vs, "_read_collection_docs", lambda name: saved)
    client, col = MagicMock(), MagicMock()
    client.delete_collection.side_effect = lambda name: events.append(
        ("delete_collection", name, vs._snapshot_path(name).exists()))
    col.upsert.side_effect = lambda **k: events.append(("upsert", k["ids"], k["documents"],
                                                        k["metadatas"]))
    col.get.side_effect = lambda ids, include: {"ids": list(ids)}
    col.count.return_value = 2
    monkeypatch.setattr(vs, "_get_client", lambda: client)
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    report = vs.rebuild_collection("news")
    assert (report["method"], report["docs_before"], report["docs_found"], report["restored"],
            report["docs_after"], report["error"]) == ("rebuild", 2, 2, 2, 2, None)
    # the snapshot is on disk before the delete, the upsert is one batch,
    # and the snapshot is removed once every id is back
    assert events == ["warm", "lock", "reopen", ("delete_collection", "news", True),
                      ("upsert", ["url-1", "url-2"], ["NVDA beats", "AMD guides"],
                       [{"ticker": "NVDA", "timestamp": "t1"}, None]),
                      "unlock"]
    assert not vs._snapshot_path("news").exists()


def test_rebuild_deletes_nothing_when_sqlite_cannot_be_read(monkeypatch, events):
    def unreadable(name):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(vs, "_read_collection_docs", unreadable)
    client = MagicMock()
    monkeypatch.setattr(vs, "_get_client", lambda: client)
    report = vs.rebuild_collection("news")
    assert report["method"] == "skipped_read_failed" and "database is locked" in report["error"]
    assert events == ["warm", "lock", "unlock"]
    client.delete_collection.assert_not_called()


def test_rebuild_deletes_nothing_when_no_documents_come_back_from_a_full_collection(
        monkeypatch, events):
    monkeypatch.setattr(vs, "_read_collection_docs", lambda name: [])
    monkeypatch.setattr(vs, "_count_collection_rows", lambda name: 4000)
    client = MagicMock()
    monkeypatch.setattr(vs, "_get_client", lambda: client)
    report = vs.rebuild_collection("news")
    assert report["method"] == "skipped_zero_docs" and report["docs_before"] == 4000
    assert events == ["warm", "lock", "unlock"]
    client.delete_collection.assert_not_called()


def test_rebuild_keeps_the_snapshot_when_documents_are_missing_afterwards(monkeypatch, events):
    monkeypatch.setattr(vs, "_read_collection_docs", lambda name: [("a", "A", {}), ("b", "B", {})])
    client, col = MagicMock(), MagicMock()
    col.get.side_effect = lambda ids, include: {"ids": ["a"]}    # "b" silently lost
    col.count.return_value = 1
    monkeypatch.setattr(vs, "_get_client", lambda: client)
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    report = vs.rebuild_collection("news")
    assert report["restored"] == 1 and "1 documents missing" in report["error"]
    assert vs.pending_rebuilds() == ["news"]


def _check(corrupted, rc=-11, stderr=""):
    result = MagicMock(returncode=rc, stderr=stderr)
    return set(corrupted), set(), result


def _patch_health(monkeypatch, checks):
    import alerts
    import tasks
    calls = iter(checks)
    monkeypatch.setattr(tasks, "_run_vector_store_check", lambda: next(calls))
    rebuilt, sent = [], []
    monkeypatch.setattr(vs, "rebuild_collection",
                        lambda name, **k: rebuilt.append(name) or _report(name))
    monkeypatch.setattr(vs, "_write_lock", contextlib.nullcontext)
    monkeypatch.setattr(vs, "pending_rebuilds", lambda: [])
    monkeypatch.setattr(alerts, "_send", lambda text, **k: sent.append(text))
    monkeypatch.setattr(tasks, "_log_vector_store_rebuild", lambda report: True)
    return tasks, rebuilt, sent


def _report(name, method="rebuild", found=40, restored=39, error=None):
    return {"collection": name, "method": method, "docs_before": found, "docs_found": found,
            "restored": restored, "docs_after": restored, "error": error,
            "started_at": "2026-11-03T14:00:00+00:00", "finished_at": "2026-11-03T14:00:20+00:00"}


def test_health_check_rebuilds_with_documents_only_after_two_failures(monkeypatch):
    tasks, rebuilt, sent = _patch_health(monkeypatch, [
        _check({"news"}, rc=1, stderr="InternalError: Failed to apply logs to the hnsw segment writer"),
        _check({"news"})])
    tasks.check_vector_store_health.run()
    assert rebuilt == ["news"]
    assert len(sent) == 1 and "Vector store auto-recovery" in sent[0]
    assert "39/40 docs restored" in sent[0]


def test_health_check_does_not_rebuild_on_a_one_off_failure(monkeypatch):
    tasks, rebuilt, sent = _patch_health(monkeypatch, [_check({"news"}), _check(set(), rc=0)])
    tasks.check_vector_store_health.run()
    assert rebuilt == [] and sent == []


def test_exit_note_names_a_segfault_and_an_exception_apart():
    import tasks
    assert tasks._check_exit_note(MagicMock(returncode=-11, stderr="")) == "exit -11 (segfault)"
    note = tasks._check_exit_note(MagicMock(returncode=1, stderr="Traceback\nInternalError: boom\n"))
    assert note == "exit 1: InternalError: boom"


def test_health_check_resumes_an_unfinished_rebuild_first(monkeypatch):
    tasks, rebuilt, sent = _patch_health(monkeypatch, [_check(set(), rc=0)])
    monkeypatch.setattr(vs, "pending_rebuilds", lambda: ["news"])
    monkeypatch.setattr(vs, "rebuild_collection",
                        lambda name, **k: rebuilt.append((name, k)) or _report(name, "snapshot_resume", 600, 600))
    tasks.check_vector_store_health.run()
    assert rebuilt == [("news", {"resume_only": True})]
    assert len(sent) == 1 and "resumed" in sent[0] and "600/600" in sent[0]


def test_health_check_writes_one_activity_row_per_rebuild(monkeypatch):
    import db
    import tasks
    real_row_writer = tasks._log_vector_store_rebuild
    tasks, rebuilt, sent = _patch_health(monkeypatch, [_check({"news"}), _check({"news"})])
    monkeypatch.setattr(tasks, "_log_vector_store_rebuild", real_row_writer)
    rows = []
    monkeypatch.setattr(db, "log_activity",
                        lambda stage, ticker, message, sentiment="neutral", extra=None:
                        rows.append((stage, ticker, sentiment, extra)) or True)
    monkeypatch.setattr(vs, "rebuild_collection",
                        lambda name, **k: _report(name, "skipped_read_failed", None, 0, "database is locked"))
    tasks.check_vector_store_health.run()
    assert len(rows) == 1
    stage, ticker, sentiment, extra = rows[0]
    assert stage == "vector_store_rebuild" and ticker == "" and sentiment == "negative"
    assert {k: extra[k] for k in ("collection", "method", "error")} == {
        "collection": "news", "method": "skipped_read_failed", "error": "database is locked"}
    assert {"docs_before", "docs_after", "started_at", "finished_at"} <= set(extra)
    assert "NOT rebuilt" in sent[0]


def test_activity_row_failure_never_reaches_the_rebuild(monkeypatch):
    import db
    import tasks

    def down(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(db, "log_activity", down)
    assert tasks._log_vector_store_rebuild(_report("news")) is False


@pytest.mark.parametrize("seen", [0, 1])
@pytest.mark.parametrize("content", ["", "garbage"])
def test_an_empty_or_unreadable_generation_forces_a_reopen(monkeypatch, events, scratch_store,
                                                           seen, content):
    col = MagicMock()
    col.upsert.side_effect = lambda **k: events.append("upsert")
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    monkeypatch.setattr(vs, "_client", object())
    monkeypatch.setattr(vs, "_seen_generation", seen)
    (scratch_store / ".write.gen").write_text(content)   # a kill mid-write, or a bad file
    vs.store("news", "u1", "one", {})
    assert events == ["lock", "reopen", "upsert", "unlock"]
    # the new value can't be one a surviving process has seen
    assert vs._read_generation() > 10 ** 15 and vs._seen_generation == vs._read_generation()
    assert not (scratch_store / ".write.gen.tmp").exists()


def test_generation_counts_up_from_a_readable_value(scratch_store):
    (scratch_store / ".write.gen").write_text("41")
    vs._bump_generation()
    assert vs._read_generation() == 42 == vs._seen_generation


def test_prune_is_skipped_while_a_rebuild_is_unfinished(monkeypatch, events, scratch_store, caplog):
    vs._snapshot_path("news").write_text('{"id": "old", "document": "x", "metadata": {}}\n')
    col = MagicMock()
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    with caplog.at_level("WARNING", logger="vector_store"):
        assert vs.prune_old_documents("news", max_age_days=90) == 0
    col.delete.assert_not_called()
    col.get.assert_not_called()
    assert "unfinished rebuild; skipped" in caplog.text


def test_a_second_resume_finds_the_snapshot_gone_and_touches_nothing(monkeypatch, events):
    client = MagicMock()
    monkeypatch.setattr(vs, "_get_client", lambda: client)
    monkeypatch.setattr(vs, "_read_collection_docs",
                        lambda name: pytest.fail("the store must not be read"))
    report = vs.rebuild_collection("news", resume_only=True)   # no snapshot on disk
    assert report["method"] == "already_resumed"
    assert events == ["warm", "lock", "unlock"]
    client.delete_collection.assert_not_called()


def test_health_check_sends_no_alert_for_a_resume_another_run_finished(monkeypatch):
    tasks, rebuilt, sent = _patch_health(monkeypatch, [_check(set(), rc=0)])
    monkeypatch.setattr(vs, "pending_rebuilds", lambda: ["news"])
    monkeypatch.setattr(vs, "rebuild_collection",
                        lambda name, **k: _report(name, "already_resumed", 0, 0))
    tasks.check_vector_store_health.run()
    assert sent == []


def test_health_check_stops_at_the_soft_time_limit(monkeypatch):
    from celery.exceptions import SoftTimeLimitExceeded
    tasks, rebuilt, sent = _patch_health(monkeypatch, [_check({"arxiv", "news"}),
                                                       _check({"arxiv", "news"})])

    def cut_off(name, **k):
        rebuilt.append(name)
        raise SoftTimeLimitExceeded()
    monkeypatch.setattr(vs, "rebuild_collection", cut_off)
    tasks.check_vector_store_health.run()            # logged, not raised
    assert rebuilt == ["arxiv"] and sent == []       # no second rebuild after the limit
    assert tasks.check_vector_store_health.soft_time_limit == 600


def test_records_without_text_are_counted_in_the_report(monkeypatch, events):
    monkeypatch.setattr(vs, "_read_collection_docs", lambda name: [("a", "A", {}), ("b", "B", {})])
    monkeypatch.setattr(vs, "_count_collection_rows", lambda name: 3)
    client, col = MagicMock(), MagicMock()
    col.get.side_effect = lambda ids, include: {"ids": list(ids)}
    col.count.return_value = 2
    monkeypatch.setattr(vs, "_get_client", lambda: client)
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    report = vs.rebuild_collection("news")
    assert (report["docs_before"], report["docs_without_text"], report["restored"]) == (3, 1, 2)
    assert report["error"] is None


def test_records_without_text_are_logged(monkeypatch, tmp_path, caplog):
    _fake_chroma_sqlite(tmp_path / "chroma.sqlite3")
    with caplog.at_level("WARNING", logger="vector_store"):
        docs = vs._read_collection_docs("news")
    assert len(docs) == 2
    assert "1 records with no document text" in caplog.text and "no-doc" in caplog.text


def _rows(monkeypatch):
    import db
    rows = []
    monkeypatch.setattr(db, "log_activity",
                        lambda stage, ticker, message, sentiment="neutral", extra=None:
                        rows.append((stage, message, sentiment, extra)) or True)
    return rows


def _pending_snapshot(name="news", age_h=1.0, failures=0, last_error=None):
    import time
    vs._snapshot_path(name).write_text('{"id": "a", "document": "A", "metadata": {}}\n')
    vs._save_rebuild_state(name, {"created": time.time() - age_h * 3600, "failures": failures,
                                  "last_error": last_error})


def test_prune_task_leaves_a_row_for_a_collection_with_a_pending_snapshot(monkeypatch):
    import tasks
    rows = _rows(monkeypatch)
    _pending_snapshot("news", age_h=30)
    pruned = []
    monkeypatch.setattr(vs, "prune_old_documents",
                        lambda collection, max_age_days: pruned.append(collection) or 0)
    tasks.prune_vector_store.run()
    assert "news" not in pruned and "options" in pruned
    [(stage, message, sentiment, extra)] = rows
    assert stage == "vector_store_rebuild" and sentiment == "negative"
    assert extra["method"] == "prune_skipped_snapshot_pending" and extra["collection"] == "news"
    assert 29.9 <= extra["snapshot_age_hours"] <= 30.1


@pytest.mark.parametrize("resume_only,method", [(True, "resume_failed"), (False, "rebuild_failed")])
def test_a_rebuild_that_raises_leaves_a_failed_row(monkeypatch, resume_only, method):
    import tasks
    rows = _rows(monkeypatch)

    def enospc(name, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(vs, "rebuild_collection", enospc)
    with pytest.raises(OSError):
        tasks._rebuild_and_log(vs, "news", resume_only=resume_only)
    [(stage, message, sentiment, extra)] = rows
    assert extra["method"] == method and "No space left on device" in extra["error"]
    assert sentiment == "negative" and extra["started_at"] and extra["finished_at"]


def test_health_check_carries_on_after_a_resume_raises(monkeypatch):
    tasks, rebuilt, sent = _patch_health(monkeypatch, [_check(set(), rc=0)])
    monkeypatch.setattr(vs, "pending_rebuilds", lambda: ["news"])
    monkeypatch.setattr(tasks, "_log_vector_store_rebuild", lambda report: rebuilt.append(report))

    def boom(name, **k):
        raise RuntimeError("count failed")
    monkeypatch.setattr(vs, "rebuild_collection", boom)
    tasks.check_vector_store_health.run()
    assert [r["method"] for r in rebuilt] == ["resume_failed"]


def _fake_state(monkeypatch):
    import db
    state = {}
    monkeypatch.setattr(db, "get_state", lambda key, default=None: state.get(key, default))
    monkeypatch.setattr(db, "set_state", lambda key, value: state.__setitem__(key, value))
    return state


def test_a_snapshot_older_than_a_day_alerts_once_a_day_outside_the_digest(monkeypatch):
    import alert_policy
    import alerts
    import tasks
    _fake_state(monkeypatch)
    sent = []
    monkeypatch.setattr(alerts, "_send", lambda text, **k: sent.append(text))
    _pending_snapshot("news", age_h=30, failures=1, last_error="OSError: disk full")
    assert tasks._alert_unfinished_rebuilds(vs) == ["news"]
    assert tasks._alert_unfinished_rebuilds(vs) == []          # same day: throttled
    [text] = sent
    assert alert_policy.classify(text)[0] == alert_policy.CRITICAL   # not LOW, not digested
    assert "30 h" in text and "OSError: disk full" in text and "Prune is paused" in text


def test_a_young_snapshot_with_retries_left_does_not_alert(monkeypatch):
    import alerts
    import tasks
    _fake_state(monkeypatch)
    sent = []
    monkeypatch.setattr(alerts, "_send", lambda text, **k: sent.append(text))
    _pending_snapshot("news", age_h=2, failures=2)
    assert tasks._alert_unfinished_rebuilds(vs) == [] and sent == []


def test_resumes_stop_after_three_failures_and_alert(monkeypatch, events):
    import alert_policy
    import alerts
    import tasks
    _fake_state(monkeypatch)
    sent = []
    monkeypatch.setattr(alerts, "_send", lambda text, **k: sent.append(text))
    _pending_snapshot("news", age_h=1)
    client = MagicMock()
    monkeypatch.setattr(vs, "_get_client", lambda: client)

    def unreadable(name):
        raise RuntimeError("database disk image is malformed")
    monkeypatch.setattr(vs, "_read_collection_docs", unreadable)
    for n in (1, 2, 3):
        report = vs.rebuild_collection("news", resume_only=True)
        assert report["method"] == "skipped_read_failed" and report["resume_failures"] == n
    report = vs.rebuild_collection("news", resume_only=True)
    assert report["method"] == "resume_gave_up" and "malformed" in report["error"]
    assert vs.pending_rebuilds() == ["news"]                    # snapshot kept
    client.delete_collection.assert_not_called()
    assert tasks._alert_unfinished_rebuilds(vs) == ["news"]
    assert "resumes stopped after 3 failures" in sent[0]
    assert alert_policy.classify(sent[0])[0] == alert_policy.CRITICAL


def test_a_finished_resume_clears_the_failure_count(monkeypatch, events):
    _pending_snapshot("news", failures=2, last_error="x")
    monkeypatch.setattr(vs, "_read_collection_docs", lambda name: [])
    monkeypatch.setattr(vs, "_count_collection_rows", lambda name: 0)
    client, col = MagicMock(), MagicMock()
    col.get.side_effect = lambda ids, include: {"ids": list(ids)}
    col.count.return_value = 1
    monkeypatch.setattr(vs, "_get_client", lambda: client)
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    report = vs.rebuild_collection("news", resume_only=True)
    assert (report["method"], report["restored"], report["resume_failures"]) == ("snapshot_resume", 1, 2)
    assert vs.pending_rebuilds() == [] and not vs._rebuild_state_path("news").exists()


def test_an_orphan_sidecar_does_not_hold_up_prune(monkeypatch):
    import tasks
    rows = _rows(monkeypatch)
    vs._save_rebuild_state("news", {"created": 0, "failures": 3, "last_error": "x"})  # no snapshot
    assert vs.snapshot_age_hours("news") is None
    pruned = []
    monkeypatch.setattr(vs, "prune_old_documents",
                        lambda collection, max_age_days: pruned.append(collection) or 0)
    tasks.prune_vector_store.run()
    assert "news" in pruned and rows == []


def test_health_check_removes_orphan_sidecars_only(monkeypatch, scratch_store, caplog):
    tasks, rebuilt, sent = _patch_health(monkeypatch, [_check(set(), rc=0)])
    vs._save_rebuild_state("news", {"created": 0, "failures": 1})          # orphan
    _pending_snapshot("arxiv", age_h=1, failures=1)                        # has its snapshot
    with caplog.at_level("WARNING", logger="vector_store"):
        tasks.check_vector_store_health.run()
    assert not vs._rebuild_state_path("news").exists()
    assert vs._rebuild_state_path("arxiv").exists() and vs._snapshot_path("arxiv").exists()
    assert "orphan rebuild state for 'news'" in caplog.text


def test_a_soft_limit_during_the_failure_count_still_records_it_once(monkeypatch, events):
    from celery.exceptions import SoftTimeLimitExceeded
    _pending_snapshot("news", failures=1)
    monkeypatch.setattr(vs, "_read_collection_docs",
                        lambda name: (_ for _ in ()).throw(RuntimeError("malformed")))
    real_save, hits = vs._save_rebuild_state, []

    def save_cut_once(name, state):
        real_save(name, state)
        if not hits:
            hits.append(1)
            raise SoftTimeLimitExceeded()     # lands just after the write
    monkeypatch.setattr(vs, "_save_rebuild_state", save_cut_once)
    with pytest.raises(SoftTimeLimitExceeded):
        vs.rebuild_collection("news", resume_only=True)
    assert vs.rebuild_state("news")["failures"] == 2        # counted once, not twice


def test_a_soft_limit_during_the_cleanup_still_removes_the_sidecar(monkeypatch, events):
    from celery.exceptions import SoftTimeLimitExceeded
    _pending_snapshot("news", failures=2)
    monkeypatch.setattr(vs, "_read_collection_docs", lambda name: [])
    monkeypatch.setattr(vs, "_count_collection_rows", lambda name: 0)
    client, col = MagicMock(), MagicMock()
    col.get.side_effect = lambda ids, include: {"ids": list(ids)}
    col.count.return_value = 1
    monkeypatch.setattr(vs, "_get_client", lambda: client)
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    real_path = vs._rebuild_state_path

    class _CutOnUnlink:
        hit = False

        def __init__(self, path):
            self.path = path

        def __fspath__(self):
            return str(self.path)

        def __getattr__(self, attr):
            return getattr(self.path, attr)

        def unlink(self, missing_ok=False):
            if not _CutOnUnlink.hit:
                _CutOnUnlink.hit = True
                raise SoftTimeLimitExceeded()
            self.path.unlink(missing_ok=missing_ok)
    monkeypatch.setattr(vs, "_rebuild_state_path", lambda name: _CutOnUnlink(real_path(name)))
    with pytest.raises(SoftTimeLimitExceeded):
        vs.rebuild_collection("news", resume_only=True)
    assert not vs._snapshot_path("news").exists() and not real_path("news").exists()


def test_a_soft_limit_while_reading_the_sidecar_is_raised_not_read_as_empty(monkeypatch):
    from celery.exceptions import SoftTimeLimitExceeded

    class _Cut:
        def read_text(self):
            raise SoftTimeLimitExceeded()
    monkeypatch.setattr(vs, "_rebuild_state_path", lambda name: _Cut())
    with pytest.raises(SoftTimeLimitExceeded):
        vs.rebuild_state("news")


@pytest.mark.parametrize("delete_fails,new_id,flag", [(True, "old", False), (True, "new", False),
                                                       (False, "old", False), (False, "new", True)])
def test_recreated_is_set_only_after_a_real_delete_and_a_new_collection(monkeypatch, events,
                                                                        delete_fails, new_id, flag):
    monkeypatch.setattr(vs, "_read_collection_docs", lambda name: [("a", "A", {})])
    client, col = MagicMock(), MagicMock()
    client.get_collection.return_value.id = "old"
    if delete_fails:
        client.delete_collection.side_effect = RuntimeError("index unreadable")
    col.id = new_id
    col.get.side_effect = lambda ids, include: {"ids": []}     # keeps the snapshot to inspect
    monkeypatch.setattr(vs, "_get_client", lambda: client)
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    vs.rebuild_collection("news")
    assert vs.pending_rebuilds() == ["news"]
    assert bool(vs.rebuild_state("news").get("recreated")) is flag


def test_the_attempt_is_counted_before_the_live_read_and_undone_after(monkeypatch, events):
    _pending_snapshot("news", failures=1)
    vs._save_rebuild_state("news", {**vs.rebuild_state("news"), "recreated": True})
    seen = []

    def todo(col, docs):
        seen.append(vs.rebuild_state("news")["failures"])    # what a crash here would leave
        return []
    monkeypatch.setattr(vs, "_resume_todo", todo)
    monkeypatch.setattr(vs, "_read_collection_docs", lambda name: [])
    monkeypatch.setattr(vs, "_count_collection_rows", lambda name: 0)
    col = MagicMock()
    col.get.side_effect = lambda ids, include: {"ids": list(ids)}
    monkeypatch.setattr(vs, "_get_collection", lambda name: col)
    report = vs.rebuild_collection("news", resume_only=True)
    assert seen == [2]
    assert (report["resume_mode"], report["upserted"], report["resume_failures"]) == ("incremental", 0, 1)
    assert vs.pending_rebuilds() == []


def test_health_check_alerts_before_it_resumes(monkeypatch):
    tasks, rebuilt, sent = _patch_health(monkeypatch, [_check(set(), rc=0)])
    order = []
    monkeypatch.setattr(vs, "pending_rebuilds", lambda: ["news"])
    monkeypatch.setattr(tasks, "_alert_unfinished_rebuilds", lambda vs_: order.append("alert") or [])
    monkeypatch.setattr(vs, "rebuild_collection",
                        lambda name, **k: order.append("resume") or _report(name, "snapshot_resume"))
    tasks.check_vector_store_health.run()
    assert order[:2] == ["alert", "resume"]
