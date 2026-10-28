"""prompt_ab swaps exactly the schema block between old and new templates."""
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
