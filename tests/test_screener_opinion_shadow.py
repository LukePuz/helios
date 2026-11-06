"""Log-only opinion_shadow: a deterministic opinion/listicle headline check
recorded in screener provenance. It must not change any screening outcome."""
import copy
from unittest.mock import patch

import pytest

from agents import screener

_OPINION = [
    ("Is Palantir Stock a Buy After its Latest Earnings Report?", "is_x_a_buy"),
    ("Microsoft Corporation (MSFT) Is Betting Big on AI: Is Its Gr", "colon_question"),
    ("Is Nvidia a Good Investment Right Now?", "is_x_a_buy"),
    ("Should You Buy AMD Stock Before Earnings?", "should_you_buy"),
    ("3 Stocks to Buy in November", "stocks_to_buy"),
    ("The Best AI Stocks for 2027", "best_stocks"),
    ("Time to Buy the Dip in Oracle", "buy_the_dip"),
    ("Is It Too Late to Buy Broadcom?", "too_late_to_buy"),
    ("Prediction: This Chip Stock Will Beat Nvidia", "prediction"),
    ("Could IonQ Stock Soar 200%?", "could_soar"),
    ("Here's Why Tesla Shares Jumped", "heres_why"),
    ("Apple Earnings: What Investors Should Know", "what_investors_should_know"),
    ("I'm Buying This Stock Hand Over Fist", "im_buying"),
    ("This Millionaire-Maker Stock Is Still Cheap", "millionaire_maker"),
    ("This Is My Favorite Forever Stock", "forever_stock"),
]

_FACTUAL = [
    "Palantir Nearly Doubles Revenue to Record, Traders Dump the Stock",
    "NVIDIA files 10-Q",
    "Microsoft raises dividend 10%",
    "BTIG main NOW: Buy, price target lowers to $180",
    "AMD wins DoD contract",
    "",
    None,
]


@pytest.mark.parametrize("title,name", _OPINION)
def test_opinion_headlines_match(title, name):
    assert screener._opinion_headline(title) == name


@pytest.mark.parametrize("title", _FACTUAL)
def test_factual_headlines_do_not_match(title):
    assert screener._opinion_headline(title) is None


def test_case_insensitive_and_word_bounded():
    assert screener._opinion_headline("IS PALANTIR STOCK A BUY?") == "is_x_a_buy"
    assert screener._opinion_headline("bestseller stockpile") is None


def test_keyword_fallback_unchanged():
    """The fallback keeps its own list: titles only the new patterns catch are
    judged as before, and the fallback never consults _opinion_headline."""
    with patch("agents.screener._opinion_headline", side_effect=AssertionError):
        for title, _ in _OPINION:
            r = screener._keyword_fallback({"ticker": "NVDA", "title": title, "text": ""})
            old_list = ["is it a buy", "should you buy", "why is", "why did", "stocks that explain",
                        "stocks to watch", "top gainers", "top losers", "here's why", "explained"]
            recap = any(kw in title.lower() for kw in old_list)
            assert (r["reason"].startswith("Keyword heuristic: looks like a recap")) == recap
        # Old keyword still drops; a new-only pattern ("stocks to buy") doesn't.
        assert screener._keyword_fallback({"title": "Is it a buy?", "text": ""})["relevant"] is False
        r = screener._keyword_fallback({"ticker": "NVDA", "title": "3 stocks to buy: NVDA contract", "text": ""})
        assert r["reason"] == "Keyword heuristic (Ollama not available)"


def _screen(title, reply='{"relevant": true, "urgency": "immediate", "reason": "earnings", "new_fact": true}'):
    with patch("agents.screener._call_ollama", return_value=reply), patch("feed.emit"), \
         patch("db.get_position_stop", return_value=None):
        return screener.screen({"ticker": "PLTR", "source": "news", "title": title, "text": "x"})


def test_provenance_carries_opinion_shadow():
    r = _screen("Is Palantir Stock a Buy After its Latest Earnings Report?")
    sha = screener._OPINION_PATTERNS_SHA
    assert r["provenance"]["opinion_shadow"] == {"match": "is_x_a_buy", "matches": ["is_x_a_buy"],
                                                 "would_drop": True, "patterns_sha": sha, "error": None}
    r = _screen("Is Palantir Stock a Buy? Here's Why", reply='{"relevant": false, "urgency": "low", "reason": "opinion"}')
    assert r["provenance"]["opinion_shadow"] == {"match": "is_x_a_buy", "matches": ["is_x_a_buy", "heres_why"],
                                                 "would_drop": False, "patterns_sha": sha, "error": None}
    r = _screen("Palantir Nearly Doubles Revenue to Record, Traders Dump the Stock")
    assert r["provenance"]["opinion_shadow"] == {"match": None, "matches": [],
                                                 "would_drop": False, "patterns_sha": sha, "error": None}
    # Keyword fallback path too.
    r = _screen("Is Palantir Stock a Buy?", reply=None)
    assert r["provenance"]["opinion_shadow"]["match"] == "is_x_a_buy"


def test_shadow_never_raises():
    with patch("agents.screener._opinion_headlines", side_effect=RuntimeError("boom")):
        r = _screen("Is Palantir Stock a Buy?")
    assert r["relevant"] is True
    assert r["provenance"]["opinion_shadow"] == {"match": None, "matches": [], "would_drop": False,
                                                 "patterns_sha": screener._OPINION_PATTERNS_SHA, "error": "RuntimeError"}
    with patch("agents.screener._opinion_headlines", side_effect=RuntimeError("boom")), \
         patch("agents.screener._OPINION_PATTERNS_SHA", None):
        r = _screen("Is Palantir Stock a Buy?")
    assert r["provenance"]["opinion_shadow"] == {"match": None, "matches": [], "would_drop": False,
                                                 "patterns_sha": None, "error": "RuntimeError"}


def test_all_matches_in_list_order():
    title = "Is Palantir Stock a Buy? Here's Why"
    assert screener._opinion_headlines(title) == ["is_x_a_buy", "heres_why"]
    assert screener._opinion_headline(title) == "is_x_a_buy"
    assert screener._opinion_headlines("NVIDIA files 10-Q") == []


def test_patterns_sha_stable_and_changes_with_patterns():
    import re
    pats = screener._OPINION_PATTERNS
    sha = screener._OPINION_PATTERNS_SHA
    assert isinstance(sha, str) and len(sha) == 12 and int(sha, 16) >= 0
    assert screener._opinion_patterns_sha(list(pats)) == sha
    changed = list(pats)
    changed[0] = (pats[0][0], re.compile(pats[0][1].pattern + "x", pats[0][1].flags))
    assert screener._opinion_patterns_sha(changed) != sha
    assert screener._opinion_patterns_sha(list(reversed(pats))) != sha
    flags = list(pats)
    flags[0] = (pats[0][0], re.compile(pats[0][1].pattern))
    assert screener._opinion_patterns_sha(flags) != sha


@pytest.mark.parametrize("reply", [
    '{"relevant": true, "urgency": "immediate", "reason": "earnings", "new_fact": true}',
    '{"relevant": false, "urgency": "low", "reason": "opinion", "new_fact": false}',
    None,
])
@pytest.mark.parametrize("title", [t for t, _ in _OPINION[:2]] + _FACTUAL[:3])
def test_screening_outcome_identical_with_and_without_shadow(title, reply):
    with_shadow = copy.deepcopy(_screen(title, reply))
    with patch("agents.screener._opinion_headlines", return_value=[]):
        without = copy.deepcopy(_screen(title, reply))
    for r in (with_shadow, without):
        r["provenance"].pop("opinion_shadow")
    assert with_shadow == without
    for key in ("relevant", "urgency", "new_fact", "affected_tickers", "override"):
        assert with_shadow.get(key) == without.get(key)
