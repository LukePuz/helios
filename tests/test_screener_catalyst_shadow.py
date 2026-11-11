"""Log-only catalyst_shadow in the screener provenance (audit 2026-11-10 10:53 #5)."""
import pytest

from agents import screener as S


@pytest.fixture(autouse=True)
def _names(monkeypatch):
    # Title-names-ticker reads the watchlist; keep it deterministic.
    names = {"RKLB": "Rocket Lab", "ASTS": "AST SpaceMobile", "TER": "Teradyne",
             "LUNR": "Intuitive Machines", "CEG": "Constellation", "AMD": "AMD",
             "NVDA": "Nvidia", "RGTI": "Rigetti"}
    monkeypatch.setattr(S, "_title_names_ticker",
                        lambda t, title: names[t].lower() in (title or "").lower())


def _shadow(ticker, title, relevant=False):
    op = (S._opinion_headlines(title) or [None])[0]
    return S._catalyst_shadow(ticker, title, relevant, op)


@pytest.mark.parametrize("ticker,title,kind", [
    ("RKLB", "Rocket Lab Sinks After-Hours On Q3 Earnings Miss", "earnings_results"),
    ("ASTS", "AST SpaceMobile Retreats On Q3 Sales Miss", "earnings_results"),
    ("CEG", "Constellation Energy raises 2026 outlook", "guidance"),
    ("LUNR", "Intuitive Machines wins NASA CLPS 2 contract", "contract"),
    ("AMD", "Analyst upgrades AMD, price target to $300", "rating_change"),
])
def test_hard_catalysts_would_pass_and_rescue(ticker, title, kind):
    rec = _shadow(ticker, title)
    assert kind in rec["matches"]
    assert rec["would_pass"] is True and rec["would_rescue"] is True and rec["error"] is None


def test_screener_pass_is_not_a_rescue():
    rec = _shadow("RKLB", "Rocket Lab Posts Q3 Results", relevant=True)
    assert rec["would_pass"] is True and rec["would_rescue"] is False


def test_opinion_headline_does_not_pass():
    rec = _shadow("NVDA", "Is Nvidia a Buy After Q3 Earnings?")
    assert rec["matches"] == ["earnings_results"] and rec["would_pass"] is False


@pytest.mark.parametrize("title", [
    "Rigetti Q3 Earnings Preview: What To Expect From Upcoming Report",
    "Rigetti Computing Q3 earnings on deck: what to expect",
    "Rigetti set to report Q3 results Thursday",
])
def test_earnings_previews_do_not_pass(title):
    rec = _shadow("RGTI", title)
    assert rec["preview"] is True and rec["would_pass"] is False


def test_title_not_naming_the_ticker_does_not_pass():
    rec = _shadow("TER", "Chip stocks: Q3 results roundup")
    assert rec["matches"] and rec["names_ticker"] is False and rec["would_pass"] is False


def test_no_catalyst_form():
    rec = _shadow("NVDA", "Nvidia Stock To $129? - Trefis")
    assert rec["matches"] == [] and rec["would_pass"] is False


def test_never_raises(monkeypatch):
    def boom(t, title):
        raise RuntimeError("x")
    monkeypatch.setattr(S, "_title_names_ticker", boom)
    rec = S._catalyst_shadow("RKLB", "Rocket Lab Q3 results", False, None)
    assert rec["error"] == "RuntimeError" and rec["would_pass"] is False


def test_patterns_sha_is_set():
    assert isinstance(S._CATALYST_PATTERNS_SHA, str) and len(S._CATALYST_PATTERNS_SHA) == 12
