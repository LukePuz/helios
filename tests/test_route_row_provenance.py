"""Routing rows carry the verdict's model and provenance (SEF run 13)."""
from unittest.mock import patch

from agents import crew


def test_route_exit_row_has_verdict_provenance():
    prov = {"code_sha": "abc", "model": "qwen-test", "prompt_sha": "p1"}
    result = {"ticker": "META", "action": "OPEN", "conviction": 6, "provenance": prov}
    with patch("db.log_signal") as log_signal:
        crew._log_route_exit("META", "ROUTE_ZERO_SHARES", result, "c6 < micro floor 7")
    args, kw = log_signal.call_args
    assert args[:2] == ("META", "ROUTE_ZERO_SHARES") and args[2] == 0.6
    assert kw["model"] == "qwen-test" and kw["provenance"] == prov
    assert kw["decision"] == {"route_exit": "ROUTE_ZERO_SHARES", "verdict_action": "OPEN",
                              "verdict_conviction": 6}


def test_route_exit_without_provenance_still_logs():
    with patch("db.log_signal") as log_signal:
        crew._log_route_exit("META", "ROUTE_NO_PRICE", {"conviction": 7}, "no price")
    assert log_signal.call_args.kwargs["provenance"] is None


def test_code_mixed_flags_a_pull_without_restart():
    """Professor M75: HEAD on disk differs from the HEAD read at import."""
    from unittest.mock import patch
    from agents import analyst
    analyst._DISK_CHECK.update(at=0.0, value=None)
    with patch.dict(analyst._CODE_SHA, {"head_sha": "aaa111"}), \
         patch.object(analyst, "_git_code_sha", return_value={"head_sha": "bbb222-dirty"}):
        v = analyst._code_mixed_check()
    assert v == {"disk_head_sha": "bbb222-dirty", "code_mixed": True}
    analyst._DISK_CHECK.update(at=0.0, value=None)
    with patch.dict(analyst._CODE_SHA, {"head_sha": "aaa111"}), \
         patch.object(analyst, "_git_code_sha", return_value={"head_sha": "aaa111-dirty"}):
        assert analyst._code_mixed_check()["code_mixed"] is False
    analyst._DISK_CHECK.update(at=0.0, value=None)
