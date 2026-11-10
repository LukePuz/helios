"""Static provenance read at import: template hashes (M187)."""
import hashlib
import inspect
from unittest.mock import MagicMock, patch

from agents import analyst


def test_template_hashes_are_computed_at_import_with_the_old_definitions():
    import context_engine
    shas = analyst._TEMPLATE_SHAS
    assert shas["prompt_template_sha"] == hashlib.sha256(
        inspect.getsource(analyst._build_prompt).encode()).hexdigest()[:12]
    assert shas["brief_template_sha"] == hashlib.sha256(
        inspect.getsource(context_engine).encode()).hexdigest()[:12]


def test_provenance_uses_the_import_time_hashes_not_the_disk(monkeypatch):
    """A pull between boot and the first verdict must not change them."""
    monkeypatch.setattr(analyst, "_PROVENANCE_STATIC", None)
    monkeypatch.setattr(analyst, "_TEMPLATE_SHAS",
                        {"prompt_template_sha": "boot_prompt1", "brief_template_sha": "boot_brief01"})
    resp = MagicMock()
    resp.json.return_value = {"models": []}
    with patch.object(analyst.requests, "get", return_value=resp), \
         patch.object(analyst.requests, "post", return_value=resp), \
         patch.object(analyst, "_code_mixed_check", return_value={}), \
         patch("inspect.getsource", side_effect=AssertionError("read from disk")), \
         patch.object(analyst, "_module_source", side_effect=AssertionError("read from disk")):
        prov = analyst._provenance("p")
    assert prov["prompt_template_sha"] == "boot_prompt1"
    assert prov["brief_template_sha"] == "boot_brief01"


def test_full_template_hashes_present_at_import():
    shas = analyst._TEMPLATE_SHAS
    for k in ("prompt_template_full_sha", "brief_template_full_sha"):
        assert isinstance(shas[k], str) and len(shas[k]) == 12
    assert shas["brief_template_full_sha"] != shas["brief_template_sha"]


def test_full_prompt_hash_sees_changes_the_old_one_misses(monkeypatch):
    """A track-record wording edit or a philosophy heading change leaves
    prompt_template_sha unchanged; the full hash moves (M171)."""
    base = analyst._template_shas()

    def _build_track_record_text():
        return "reworded"
    monkeypatch.setattr(analyst, "_build_track_record_text", _build_track_record_text)
    after_tr = analyst._template_shas()
    assert after_tr["prompt_template_sha"] == base["prompt_template_sha"]
    assert after_tr["prompt_template_full_sha"] != base["prompt_template_full_sha"]
    monkeypatch.undo()
    monkeypatch.setattr(analyst, "_PHILOSOPHY_KEEP", analyst._PHILOSOPHY_KEEP + ("New Heading",))
    assert analyst._template_shas()["prompt_template_full_sha"] != base["prompt_template_full_sha"]


def test_verdict_provenance_has_full_template_hashes(monkeypatch):
    monkeypatch.setattr(analyst, "_PROVENANCE_STATIC", None)
    with patch.object(analyst, "_code_mixed_check", return_value={}):
        prov = analyst._provenance("p")
    assert prov["prompt_template_full_sha"] == analyst._TEMPLATE_SHAS["prompt_template_full_sha"]
    assert prov["brief_template_full_sha"] == analyst._TEMPLATE_SHAS["brief_template_full_sha"]


def test_db_queries_hash_moves_with_the_track_record_query(monkeypatch):
    """The track record's date bound lives in db, outside the prompt hashes;
    prompt_db_queries_sha covers it and leaves the others alone (m596)."""
    import db
    base = analyst._template_shas()
    assert isinstance(base["prompt_db_queries_sha"], str) and len(base["prompt_db_queries_sha"]) == 12

    def get_conviction_calibration(since=None):
        return []
    monkeypatch.setattr(db, "get_conviction_calibration", get_conviction_calibration)
    after = analyst._template_shas()
    assert after["prompt_db_queries_sha"] != base["prompt_db_queries_sha"]
    assert after["prompt_template_full_sha"] == base["prompt_template_full_sha"]
    monkeypatch.setattr(analyst, "_PROVENANCE_STATIC", None)
    with patch.object(analyst, "_code_mixed_check", return_value={}):
        assert analyst._provenance("p")["prompt_db_queries_sha"] == analyst._TEMPLATE_SHAS["prompt_db_queries_sha"]
