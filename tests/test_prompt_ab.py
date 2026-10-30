"""prompt_ab swaps exactly the schema block between old and new templates,
and its summary applies the rule fixed before the run."""
import importlib.util
import os
import subprocess

_spec = importlib.util.spec_from_file_location(
    "prompt_ab", os.path.join(os.path.dirname(__file__), "..", "scripts", "prompt_ab.py"))
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)
REPO = os.path.join(os.path.dirname(__file__), "..")


def test_old_and_new_blocks_differ_only_in_order():
    new = ab.schema_block(open(os.path.join(REPO, "agents", "analyst.py")).read())
    old = ab.schema_block(subprocess.run(["git", "-C", REPO, "show", "31d52f8^:agents/analyst.py"],
                                         capture_output=True, text=True, check=True).stdout)
    assert new != old
    assert new.index('"action"') > new.index('"reasoning"')
    assert old.index('"action"') < old.index('"reasoning"')


def test_variant_replaces_block_in_a_rendered_prompt():
    from agents import analyst
    prompt = analyst._build_prompt({"ticker": "NVDA"}, "BRIEF TEXT", "", {"regime": "Neutral"}, "")
    new = ab.schema_block(open(os.path.join(REPO, "agents", "analyst.py")).read())
    assert new in prompt
    old = ab.old_variant(prompt, new, "OLD BLOCK")
    assert "OLD BLOCK" in old and "BRIEF TEXT" in old and new not in old

def _runs(new, old):
    return ([{"ticker": t, "variant": "new", "conviction": c, "expected": e} for t, c, e in new]
            + [{"ticker": t, "variant": "old", "conviction": c, "expected": e} for t, c, e in old])


def test_order_effect_needs_both_thresholds():
    s = ab.summarize(_runs([("A", 6, 6.1), ("B", 6, 5.9)], [("A", 7, 6.9), ("B", 7, 7.0)]))
    assert s["c7_share_diff_pts"] == 100.0 and s["paired_mean_expected_diff"] == 0.95
    assert s["order_effect"] is True
    # share moves but the paired expected conviction barely does
    s = ab.summarize(_runs([("A", 6, 6.6), ("B", 6, 6.6)], [("A", 7, 6.8), ("B", 7, 6.8)]))
    assert s["order_effect"] is False


def test_missing_logprobs_or_unparsed_runs_are_skipped():
    s = ab.summarize(_runs([("A", None, None), ("A", 6, 6.0)], [("A", 6, None)]))
    assert s["new"]["runs"] == 1 and s["paired_prompts"] == 0 and s["order_effect"] is False


def test_round_robin_takes_newest_per_ticker_first():
    rows = [("A", "a1", 5, 11), ("A", "a2", 4, 12), ("A", "a3", 3, 13), ("B", "b1", 6, 14), ("C", "c1", 1, 15)]
    assert ab._round_robin(rows, 4) == [("B", "b1", 14), ("A", "a1", 11), ("C", "c1", 15), ("A", "a2", 12)]


def test_bad_runs_are_counted_and_left_out_of_the_rule():
    runs = [{"ticker": "A", "prompt_id": 0, "variant": "new", "conviction": 6, "expected": 6.0},
            {"ticker": "A", "prompt_id": 0, "variant": "new", "conviction": 7, "expected": 7.0, "salvaged": True},
            {"ticker": "A", "prompt_id": 0, "variant": "old", "conviction": None, "expected": None},
            {"ticker": "A", "prompt_id": 0, "variant": "old", "conviction": 7, "expected": 7.0}]
    s = ab.summarize(runs)
    assert s["new"]["runs"] == 1 and s["new"]["salvaged"] == 1
    assert s["old"]["no_conviction"] == 1 and s["old"]["no_logprobs"] == 1
    assert s["paired_mean_expected_diff"] == 1.0


def test_conviction_parse_survives_nan_and_text():
    assert ab._conviction(float("nan")) is None and ab._conviction("7") == 7 and ab._conviction("x") is None


def test_pilot_prompts_are_excluded_by_default():
    assert {79199, 79189, 79173, 79172, 79150, 79137} == set(ab.PILOT_SIGNAL_IDS)


def test_a_run_without_conviction_is_out_of_the_paired_mean_too():
    runs = [{"ticker": "A", "prompt_id": 0, "variant": "new", "conviction": 6, "expected": 6.0},
            {"ticker": "A", "prompt_id": 0, "variant": "old", "conviction": None, "expected": 9.0},
            {"ticker": "A", "prompt_id": 0, "variant": "old", "conviction": 7, "expected": 7.0}]
    assert ab.summarize(runs)["paired_mean_expected_diff"] == 1.0
