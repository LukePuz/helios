import ast
import inspect

import tasks


def test_regime_runs_on_ops_not_the_llm_lane():
    assert tasks.app.conf.beat_schedule["assess-regime"]["options"]["queue"] == "ops"


def test_regime_assessment_makes_no_llm_call():
    """The move off the llm lane assumes assess_regime stays LLM-free."""
    from agents import market_regime
    src = inspect.getsource(market_regime)
    names = {n.id if isinstance(n, ast.Name) else n.attr
             for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Name, ast.Attribute))}
    assert not names & {"ollama", "analyze", "_call_ollama", "chat"}
    assert "11434" not in src
