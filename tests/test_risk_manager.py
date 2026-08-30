"""
Pure-logic unit tests for risk_manager.py — no live DB/IBKR/network required,
safe to run in CI. Uses mocks for the IBKR client and db module.
"""
from unittest.mock import MagicMock, patch

import pytest

import risk_manager


def _make_client(portfolio_value=100_000.0, positions=None, cash=None):
    client = MagicMock()
    # cash defaults to full portfolio value (no positions held) so existing
    # sizing tests aren't inadvertently capped to 0 shares by the
    # cash-reserve cap (MAX_TRADE_PCT_OF_CASH_RESERVE) added 2026-07-15 —
    # tests that care about cash-cap behavior specifically pass it in.
    client.get_account.return_value = MagicMock(
        portfolio_value=portfolio_value,
        cash=cash if cash is not None else portfolio_value,
    )
    client.get_all_positions.return_value = positions or []
    return client


# Keys are trading_env-scoped in the real code (see risk_manager.is_halted's
# 2026-08-14 fix) -- every test here pins risk_manager.TRADING_ENV to
# "paper" explicitly rather than relying on whatever this machine's real
# .env currently says, same reasoning as config.AUTO_EXECUTE being patched
# explicitly in test_crew_routing.py.
class TestSizing:
    def test_shares_initial_is_integer(self):
        client = _make_client()
        with patch.object(risk_manager.RiskManager, "calculate_stop", return_value=(120.0, "mocked")), \
             patch("risk_manager.TRADING_ENV", "paper"), \
             patch("db.get_state", side_effect=lambda k, d=None: {"trading_halted_paper": "false",
                                                                    "high_water_mark_paper": "100000"}.get(k, d)), \
             patch("db.set_state"):
            rm = risk_manager.RiskManager(client)
            sizing = rm.size_position("NVDA", conviction=8, bucket="Compounder",
                                      regime="Risk-On", price=130.0)
        assert isinstance(sizing["shares_initial"], int)
        assert sizing["shares_initial"] > 0

    def test_sub_one_share_allocation_skips_gracefully(self):
        client = _make_client()
        with patch.object(risk_manager.RiskManager, "calculate_stop", return_value=(650_000.0, "mocked")), \
             patch("risk_manager.TRADING_ENV", "paper"), \
             patch("db.get_state", side_effect=lambda k, d=None: {"trading_halted_paper": "false",
                                                                    "high_water_mark_paper": "100000"}.get(k, d)), \
             patch("db.set_state"):
            rm = risk_manager.RiskManager(client)
            sizing = rm.size_position("BRKA", conviction=5, bucket="Moonshot",
                                      regime="Risk-Off", price=700_000.0)
        assert sizing["shares_initial"] == 0
        assert sizing["flags"]


class TestMicroTierBandRescaling:
    """Incident 2026-08-21: Compounder/Accelerator never got the same
    tier-rescaling Moonshot got on 2026-08-13 — their BUCKET_SIZE_TABLE
    bands stayed at the values tuned for the ~$1M paper account at every
    tier. On the live account's actual ~$2,000 balance ("micro" tier),
    that meant every Accelerator ADD priced above roughly $45-70/share
    (NVDA, MSFT, GOOG, META, AVGO, ANET, NOW, CRWD, MRVL, TSLA, LRCX —
    real conviction-7/8 analyst output, dozens/day) computed a dollar
    allocation too small to clear the whole-share floor and was silently
    dropped — no DB row, no alert, just a log.info line — while only the
    3 existing Moonshot positions (all under $21/share, since Moonshot's
    bands *did* rescale) could ever size. The account went 4 days without
    a single trade as a result. This asserts a conviction-8 Accelerator
    ADD on a $1,990 micro-tier account actually sizes to >=1 share of a
    ~$217 stock (NVDA's real price that day) instead of silently
    returning 0."""

    def test_accelerator_add_sizes_on_micro_tier_account(self):
        # $1,990 portfolio value -> "micro" tier (< config.ACCOUNT_TIER_REDUCED_MIN)
        client = _make_client(portfolio_value=1_990.26)
        with patch.object(risk_manager.RiskManager, "calculate_stop", return_value=(180.0, "mocked")), \
             patch("risk_manager.TRADING_ENV", "live"), \
             patch("db.get_state", side_effect=lambda k, d=None: {"trading_halted_live": "false",
                                                                    "high_water_mark_live": "2010.54"}.get(k, d)), \
             patch("db.set_state"):
            rm = risk_manager.RiskManager(client)
            sizing = rm.size_position("NVDA", conviction=8, bucket="Accelerator",
                                      regime="Risk-On", price=216.99)
        assert sizing["shares_initial"] >= 1, (
            f"Accelerator conviction-8 ADD failed to size on a micro-tier account: {sizing['flags']}"
        )

    def test_moonshot_still_rescales_unchanged(self):
        # Regression guard: the pre-existing Moonshot tier-rescaling
        # (2026-08-13) must keep working exactly as before this fix
        # generalized the same code path to Compounder/Accelerator.
        client = _make_client(portfolio_value=1_990.26)
        with patch.object(risk_manager.RiskManager, "calculate_stop", return_value=(12.0, "mocked")), \
             patch("risk_manager.TRADING_ENV", "live"), \
             patch("db.get_state", side_effect=lambda k, d=None: {"trading_halted_live": "false",
                                                                    "high_water_mark_live": "2010.54"}.get(k, d)), \
             patch("db.set_state"):
            rm = risk_manager.RiskManager(client)
            sizing = rm.size_position("RGTI", conviction=7, bucket="Moonshot",
                                      regime="Risk-On", price=18.635)
        assert sizing["shares_initial"] >= 1


class TestHalt:
    def test_persisted_halt_flag_blocks_trading(self):
        client = _make_client()
        with patch("risk_manager.TRADING_ENV", "paper"), \
             patch("db.get_state", side_effect=lambda k, d=None: {"trading_halted_paper": "true"}.get(k, d)):
            rm = risk_manager.RiskManager(client)
            assert rm.is_halted() is True

    def test_no_halt_flag_allows_trading(self):
        client = _make_client()
        with patch("risk_manager.TRADING_ENV", "paper"), \
             patch("db.get_state", side_effect=lambda k, d=None: {"trading_halted_paper": "false",
                                                                    "high_water_mark_paper": "100000"}.get(k, d)), \
             patch("db.set_state"):
            rm = risk_manager.RiskManager(client)
            assert rm.is_halted() is False


class TestConcentrationFailsClosed:
    def test_errored_check_fails_closed(self):
        client = MagicMock()
        client.get_account.return_value = MagicMock(portfolio_value=100_000.0)
        client.get_all_positions.side_effect = Exception("IBKR down")

        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("NVDA", 0.05)

        assert result["passed"] is False
        assert "RISK CHECK FAILED" in result["flags"][0]

    def test_healthy_check_passes(self):
        client = _make_client(positions=[])
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("NVDA", 0.05)
        assert result["passed"] is True


def _position(symbol, pct_of_portfolio, portfolio_value=100_000.0):
    return MagicMock(symbol=symbol, market_value=pct_of_portfolio * portfolio_value)


class TestConcentrationSizeCap:
    """2026-08-11: check_concentration used to be advisory only -- a failed
    check just appended text to a field nobody reviews in real time (auto-
    execution has no human-in-the-loop tap). The portfolio sat over its own
    60% AI-infra limit for a week-plus while every new ADD to an AI-infra
    name kept sailing through anyway. ai_infra_size_cap_pct is the actual
    enforcement mechanism now -- the largest proposed_size_pct for THIS
    ticker that keeps the total at or under the limit."""

    def test_cap_reflects_generous_headroom_when_well_under_the_limit(self):
        """cap is always a number for an AI-infra ticker (its own headroom),
        never None just because the check passed -- None specifically means
        "not an AI-infra ticker," not "under the limit." crew.py only acts
        when cap < proposed size, so a generous cap like this is a no-op."""
        client = _make_client(positions=[_position("NVDA", 0.30)])
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("MSFT", 0.10)  # 30% + 10% = 40%, well under 60%

        assert result["passed"] is True
        assert result["ai_infra_size_cap_pct"] == pytest.approx(0.30, abs=1e-6)

    def test_cap_is_none_for_a_non_ai_infra_ticker_even_when_total_already_breached(self):
        """A breach caused entirely by other, pre-existing AI-infra
        positions must not cap an unrelated ticker's size -- capping it
        wouldn't address the actual breach."""
        client = _make_client(positions=[_position("NVDA", 0.45), _position("AMD", 0.20)])
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("JPM", 0.05)  # not in ai_infra_keywords

        assert result["passed"] is False  # still flagged -- the breach is real
        assert result["ai_infra_size_cap_pct"] is None  # but not this ticker's to fix

    def test_cap_reflects_remaining_headroom_for_an_ai_infra_ticker(self):
        """55% already held in AI infra names, proposing another 10% PLTR
        (also AI infra) would hit 65% -- cap should be exactly the 5%
        headroom left under the 60% limit."""
        client = _make_client(positions=[_position("NVDA", 0.35), _position("AMD", 0.20)])
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("PLTR", 0.10)

        assert result["passed"] is False
        assert result["ai_infra_size_cap_pct"] == pytest.approx(0.05, abs=1e-6)

    def test_cap_is_zero_when_already_at_or_over_the_limit(self):
        client = _make_client(positions=[_position("NVDA", 0.45), _position("AMD", 0.20)])
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_concentration("MSFT", 0.05)  # 65% + 5% = 70%, already over before adding

        assert result["ai_infra_size_cap_pct"] == 0.0


def _holding(symbol, qty=10.0, market_value=1000.0, unrealized_pl=0.0):
    return MagicMock(symbol=symbol, qty=qty, market_value=market_value, unrealized_pl=unrealized_pl)


class TestLatestConviction:
    def test_prefers_the_most_recent_signal_over_entry_conviction(self):
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        with patch("db.get_latest_signal_for", return_value={"confidence": 0.4, "signal": "HOLD",
                                                               "timestamp": "2026-08-12"}), \
             patch("db.get_position_stop", return_value={"conviction": 8}):
            assert rm._latest_conviction("NVDA") == 4

    def test_falls_back_to_entry_conviction_when_no_signal_exists(self):
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        with patch("db.get_latest_signal_for", return_value=None), \
             patch("db.get_position_stop", return_value={"conviction": 7}):
            assert rm._latest_conviction("NVDA") == 7

    def test_returns_none_when_neither_exists(self):
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        with patch("db.get_latest_signal_for", return_value=None), \
             patch("db.get_position_stop", return_value=None):
            assert rm._latest_conviction("NVDA") is None


class TestFindSwapCandidate:
    """2026-08-13: added ahead of a live-money switch, so a genuinely
    high-conviction idea isn't just capped/blocked outright whenever cash is
    tight while the portfolio might be sitting on a position nobody's
    excited about anymore. Deliberately conservative by explicit design
    (not just implementation detail) -- see find_swap_candidate's own
    docstring for the full reasoning. These tests guard the three
    guarantees that actually matter: never forced, never touches
    Compounders, and prefers realizing a loss over a gain when there's a
    choice (the tax-aware part)."""

    def _rm(self, positions):
        client = _make_client(positions=positions)
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        return rm

    def test_no_candidate_when_gap_is_too_small(self):
        """new conviction 8, held conviction 7 -- only a 1-point gap,
        short of SWAP_MIN_CONVICTION_GAP (2). Must not force anything."""
        rm = self._rm([_holding("SNOW", qty=10, unrealized_pl=100.0)])
        with patch("db.get_position_stop", return_value={"thesis_bucket": "Accelerator"}), \
             patch("db.get_latest_signal_for", return_value={"confidence": 0.7, "signal": "HOLD"}):
            result = rm.find_swap_candidate("PLTR", new_conviction=8)
        assert result is None

    def test_no_candidate_when_held_conviction_is_too_high(self):
        """This is the core guarantee the user asked for explicitly: a
        held position at 7+/10 must NEVER be sold just because something
        else scores even higher -- the gap alone (9 vs 7 = 2, meets
        SWAP_MIN_CONVICTION_GAP) is not sufficient on its own. It hasn't
        independently earned an exit."""
        rm = self._rm([_holding("SNOW", qty=10, unrealized_pl=100.0)])
        with patch("db.get_position_stop", return_value={"thesis_bucket": "Accelerator"}), \
             patch("db.get_latest_signal_for", return_value={"confidence": 0.7, "signal": "HOLD"}):
            result = rm.find_swap_candidate("PLTR", new_conviction=9)
        assert result is None

    def test_compounder_bucket_is_never_eligible(self):
        """Even a genuinely weak, low-conviction Compounder holding must
        never be swapped out -- that bucket is the long-term core by
        design, exempt regardless of conviction."""
        rm = self._rm([_holding("MSFT", qty=10, unrealized_pl=100.0)])
        with patch("db.get_position_stop", return_value={"thesis_bucket": "Compounder"}), \
             patch("db.get_latest_signal_for", return_value={"confidence": 0.3, "signal": "HOLD"}):
            result = rm.find_swap_candidate("PLTR", new_conviction=9)
        assert result is None

    def test_eligible_accelerator_holding_is_found(self):
        rm = self._rm([_holding("SNOW", qty=10, unrealized_pl=100.0)])
        with patch("db.get_position_stop", return_value={"thesis_bucket": "Accelerator"}), \
             patch("db.get_latest_signal_for", return_value={"confidence": 0.4, "signal": "HOLD"}):
            result = rm.find_swap_candidate("PLTR", new_conviction=9)  # gap=5, candidate=4 <= 6
        assert result is not None
        assert result["ticker"] == "SNOW"
        assert result["conviction"] == 4

    def test_prefers_a_loss_over_a_gain_when_multiple_qualify(self):
        """The tax-aware part: given a choice, realize the loss (or
        smaller gain) rather than the bigger gain, to minimize capital
        gains drag in a real taxable account."""
        rm = self._rm([
            _holding("SNOW", qty=10, unrealized_pl=500.0),   # up big
            _holding("CRWD", qty=5,  unrealized_pl=-200.0),  # down
        ])

        def fake_stop(ticker):
            return {"thesis_bucket": "Accelerator"}

        def fake_signal(ticker):
            return {"confidence": 0.4, "signal": "HOLD"}  # both conviction 4

        with patch("db.get_position_stop", side_effect=fake_stop), \
             patch("db.get_latest_signal_for", side_effect=fake_signal):
            result = rm.find_swap_candidate("PLTR", new_conviction=9)

        assert result["ticker"] == "CRWD"
        assert result["unrealized_pl"] == -200.0

    def test_the_new_ticker_itself_is_never_a_candidate(self):
        """Guards against a held position with the same symbol as the new
        idea (e.g. an ADD to an existing name) being treated as its own
        swap target."""
        rm = self._rm([_holding("PLTR", qty=10, unrealized_pl=100.0)])
        with patch("db.get_position_stop", return_value={"thesis_bucket": "Accelerator"}), \
             patch("db.get_latest_signal_for", return_value={"confidence": 0.4, "signal": "HOLD"}):
            result = rm.find_swap_candidate("PLTR", new_conviction=9)
        assert result is None

    def test_broker_error_returns_none_not_an_exception(self):
        client = MagicMock()
        client.get_all_positions.side_effect = Exception("IBKR down")
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        assert rm.find_swap_candidate("PLTR", new_conviction=9) is None


class TestCorrelation:
    def test_price_correlation_perfect_positive(self):
        dates = [f"2026-01-{d:02d}" for d in range(1, 26)]
        returns = [0.01, -0.02, 0.015, 0.005, -0.01] * 4 + [0.02]
        prices_a, prices_b = [100.0], [200.0]
        for r in returns:
            prices_a.append(prices_a[-1] * (1 + r))
            prices_b.append(prices_b[-1] * (1 + r))
        bars_a = [{"date": d, "close": p} for d, p in zip(dates, prices_a)]
        bars_b = [{"date": d, "close": p} for d, p in zip(dates, prices_b)]

        corr = risk_manager._price_correlation(bars_a, bars_b)
        assert corr == pytest.approx(1.0, abs=1e-9)

    def test_price_correlation_perfect_negative(self):
        dates = [f"2026-01-{d:02d}" for d in range(1, 26)]
        returns = [0.01, -0.02, 0.015, 0.005, -0.01] * 4 + [0.02]
        prices_a, prices_c = [100.0], [50.0]
        for r in returns:
            prices_a.append(prices_a[-1] * (1 + r))
            prices_c.append(prices_c[-1] * (1 - r))
        bars_a = [{"date": d, "close": p} for d, p in zip(dates, prices_a)]
        bars_c = [{"date": d, "close": p} for d, p in zip(dates, prices_c)]

        corr = risk_manager._price_correlation(bars_a, bars_c)
        assert corr == pytest.approx(-1.0, abs=1e-9)

    def test_price_correlation_insufficient_history_returns_none(self):
        bars_a = [{"date": "2026-01-01", "close": 100.0}] * 3
        bars_b = [{"date": "2026-01-01", "close": 200.0}] * 3
        assert risk_manager._price_correlation(bars_a, bars_b) is None

    def test_check_correlation_fails_closed_on_error(self):
        client = MagicMock()
        client.get_all_positions.side_effect = Exception("IBKR down")
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        result = rm.check_correlation("NVDA")
        assert result["reduce_by"] == 0.25
        assert "RISK CHECK FAILED" in result["note"]

    def test_check_correlation_falls_back_when_no_price_history(self):
        pos = MagicMock(symbol="AMD")
        client = MagicMock()
        client.get_all_positions.return_value = [pos]
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        with patch("db.get_price_history_bulk", return_value={}):
            result = rm.check_correlation("ZZZNOTCACHED")
        assert "price history not yet cached" in result["note"] or "Sector unknown" in result["note"]


def _priced_holding(symbol, qty=10.0, current_price=100.0):
    return MagicMock(symbol=symbol, qty=qty, current_price=current_price)


class TestCheckProfitTargets:
    """2026-08-29, user request: the profit-target/partial-exit piece
    missing from the original protective-only stop path -- see
    config.py's PROFIT_TARGET_PCT_BY_STYLE docstring for the full
    rationale. These tests guard: regime-style selection actually changes
    the target used, Compounder is never touched, an already-taken
    position isn't re-sold, and a broker failure doesn't silently mark
    the target as taken anyway."""

    def _rm(self, positions):
        client = _make_client(positions=positions)
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        return rm

    def test_hits_target_in_trending_style_and_partial_sells(self):
        # Accelerator target in Trending/Expanding = (0.20, 0.30) -> 0.25
        # midpoint. 30% gain clears it.
        rm = self._rm([_priced_holding("SNOW", qty=100.0, current_price=130.0)])
        stops = [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "profit_target_taken": False}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Trending/Expanding"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops), \
             patch("db.mark_profit_target_taken") as mock_mark, \
             patch("ibkr.place_market_order", return_value={"order_id": 1}) as mock_order:
            result = rm.check_profit_targets()

        assert len(result) == 1
        assert result[0]["ticker"] == "SNOW"
        assert result[0]["action"] == "PARTIAL_SELL"
        # Trending/Expanding partial-sell fraction is 0.25 -> 25 shares
        assert result[0]["shares_sold"] == 25.0
        mock_order.assert_called_once_with("SNOW", "SELL", 25.0)
        mock_mark.assert_called_once_with("SNOW", 100.0)  # breakeven = entry

    def test_below_target_does_nothing(self):
        # 10% gain, well short of Accelerator's Trending/Expanding target (~25%)
        rm = self._rm([_priced_holding("SNOW", qty=100.0, current_price=110.0)])
        stops = [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "profit_target_taken": False}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Trending/Expanding"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops), \
             patch("db.mark_profit_target_taken") as mock_mark, \
             patch("ibkr.place_market_order") as mock_order:
            result = rm.check_profit_targets()
        assert result == []
        mock_order.assert_not_called()
        mock_mark.assert_not_called()

    def test_choppy_style_uses_a_lower_target_than_trending(self):
        # Same 18% gain: below Trending/Expanding's Accelerator target (~25%)
        # but at/above Choppy/Distribution's (~12.5%) -- proves the regime
        # style actually changes which table is consulted, not just cosmetic.
        rm = self._rm([_priced_holding("SNOW", qty=100.0, current_price=118.0)])
        stops = [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "profit_target_taken": False}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Choppy/Distribution"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops), \
             patch("db.mark_profit_target_taken"), \
             patch("ibkr.place_market_order", return_value={"order_id": 1}) as mock_order:
            result = rm.check_profit_targets()
        assert len(result) == 1
        # Choppy/Distribution partial-sell fraction is 0.45 -> 45 shares
        mock_order.assert_called_once_with("SNOW", "SELL", 45.0)

    def test_compounder_bucket_is_never_touched(self):
        rm = self._rm([_priced_holding("MSFT", qty=100.0, current_price=200.0)])
        stops = [{"ticker": "MSFT", "entry_price": 100.0, "thesis_bucket": "Compounder",
                  "profit_target_taken": False}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Trending/Expanding"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops), \
             patch("ibkr.place_market_order") as mock_order:
            result = rm.check_profit_targets()
        assert result == []
        mock_order.assert_not_called()

    def test_already_taken_is_never_resold(self):
        rm = self._rm([_priced_holding("SNOW", qty=100.0, current_price=200.0)])
        stops = [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "profit_target_taken": True}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Trending/Expanding"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops), \
             patch("ibkr.place_market_order") as mock_order:
            result = rm.check_profit_targets()
        assert result == []
        mock_order.assert_not_called()

    def test_missing_regime_data_defaults_to_conservative_choppy_style(self):
        # No cached regime available -- must default to the more
        # conservative Choppy/Distribution table, not skip the check.
        rm = self._rm([_priced_holding("SNOW", qty=100.0, current_price=118.0)])
        stops = [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "profit_target_taken": False}]
        with patch("agents.market_regime.get_cached_regime", return_value=None), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops), \
             patch("db.mark_profit_target_taken"), \
             patch("ibkr.place_market_order", return_value={"order_id": 1}) as mock_order:
            result = rm.check_profit_targets()
        # 18% gain clears Choppy/Distribution's ~12.5% Accelerator target
        assert len(result) == 1
        mock_order.assert_called_once_with("SNOW", "SELL", 45.0)

    def test_broker_failure_does_not_mark_target_taken(self):
        rm = self._rm([_priced_holding("SNOW", qty=100.0, current_price=130.0)])
        stops = [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "profit_target_taken": False}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Trending/Expanding"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops), \
             patch("db.mark_profit_target_taken") as mock_mark, \
             patch("ibkr.place_market_order", side_effect=Exception("IBKR down")):
            result = rm.check_profit_targets()
        assert result == []
        mock_mark.assert_not_called()


class TestFlagStalePositions:
    """2026-08-29, user request: staleness flagging never exits anything
    itself -- these tests guard the trigger condition (time AND lack of
    movement, not either alone) and that it's regime-style-conditional
    just like the profit-target check."""

    def _rm(self, positions):
        client = _make_client(positions=positions)
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        return rm

    def test_stale_and_flat_position_is_flagged(self):
        from datetime import datetime, timedelta, timezone
        opened = datetime.now(timezone.utc) - timedelta(days=30)
        rm = self._rm([_priced_holding("SNOW", qty=100.0, current_price=101.0)])
        stops = [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "opened_at": opened}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Choppy/Distribution"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops):
            result = rm.flag_stale_positions()
        assert len(result) == 1
        assert result[0]["ticker"] == "SNOW"

    def test_stale_but_moved_a_lot_is_not_flagged(self):
        from datetime import datetime, timedelta, timezone
        opened = datetime.now(timezone.utc) - timedelta(days=30)
        rm = self._rm([_priced_holding("SNOW", qty=100.0, current_price=130.0)])  # +30%
        stops = [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "opened_at": opened}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Choppy/Distribution"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops):
            result = rm.flag_stale_positions()
        assert result == []

    def test_flat_but_not_yet_stale_is_not_flagged(self):
        from datetime import datetime, timedelta, timezone
        opened = datetime.now(timezone.utc) - timedelta(days=2)  # well under any window
        rm = self._rm([_priced_holding("SNOW", qty=100.0, current_price=101.0)])
        stops = [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "opened_at": opened}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Choppy/Distribution"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops):
            result = rm.flag_stale_positions()
        assert result == []

    def test_compounder_bucket_is_never_flagged(self):
        from datetime import datetime, timedelta, timezone
        opened = datetime.now(timezone.utc) - timedelta(days=60)
        rm = self._rm([_priced_holding("MSFT", qty=100.0, current_price=101.0)])
        stops = [{"ticker": "MSFT", "entry_price": 100.0, "thesis_bucket": "Compounder",
                  "opened_at": opened}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Choppy/Distribution"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops):
            result = rm.flag_stale_positions()
        assert result == []


class TestSectorAwareTradeStyle:
    """2026-08-29, user request: a held position should be evaluated
    against ITS OWN sector's regime (agents.market_regime.
    get_trade_style_for_ticker), not a single blanket whole-market read
    -- a sector can trend while the broad market chops, or vice versa.
    These tests guard: the per-ticker sector style actually overrides the
    market-wide one when available, and the market-wide read is still
    used as the fallback when sector data isn't."""

    def _rm(self, positions):
        client = _make_client(positions=positions)
        rm = risk_manager.RiskManager.__new__(risk_manager.RiskManager)
        rm._client = client
        return rm

    def test_sector_style_overrides_market_style_for_profit_target(self):
        # Market-wide regime says Trending/Expanding (25% target, would
        # NOT be hit by an 18% gain) but this ticker's own sector says
        # Choppy/Distribution (~12.5% target, WOULD be hit) -- the sector
        # read must win.
        rm = self._rm([_priced_holding("UNH", qty=100.0, current_price=118.0)])
        stops = [{"ticker": "UNH", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "profit_target_taken": False}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Trending/Expanding"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value="Choppy/Distribution"), \
             patch("db.get_all_position_stops", return_value=stops), \
             patch("db.mark_profit_target_taken"), \
             patch("ibkr.place_market_order", return_value={"order_id": 1}) as mock_order:
            result = rm.check_profit_targets()
        assert len(result) == 1
        assert result[0]["trade_style"] == "Choppy/Distribution"
        mock_order.assert_called_once_with("UNH", "SELL", 45.0)

    def test_sector_lookup_unavailable_falls_back_to_market_style(self):
        # get_trade_style_for_ticker returns None (ticker isn't an S&P 500
        # constituent, or its sector hasn't been assessed yet) -- must
        # fall back to the market-wide read, not silently skip or crash.
        rm = self._rm([_priced_holding("SNOW", qty=100.0, current_price=130.0)])
        stops = [{"ticker": "SNOW", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "profit_target_taken": False}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Trending/Expanding"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value=None), \
             patch("db.get_all_position_stops", return_value=stops), \
             patch("db.mark_profit_target_taken"), \
             patch("ibkr.place_market_order", return_value={"order_id": 1}) as mock_order:
            result = rm.check_profit_targets()
        assert len(result) == 1
        assert result[0]["trade_style"] == "Trending/Expanding"
        mock_order.assert_called_once_with("SNOW", "SELL", 25.0)

    def test_two_positions_in_different_sector_regimes_get_different_targets(self):
        # Two holdings, same gain, opposite sector regimes -- confirms the
        # resolution happens per-ticker inside the loop, not once for the
        # whole run.
        rm = self._rm([
            _priced_holding("UNH", qty=100.0, current_price=118.0),
            _priced_holding("NVDA", qty=100.0, current_price=118.0),
        ])
        stops = [
            {"ticker": "UNH", "entry_price": 100.0, "thesis_bucket": "Accelerator",
             "profit_target_taken": False},
            {"ticker": "NVDA", "entry_price": 100.0, "thesis_bucket": "Accelerator",
             "profit_target_taken": False},
        ]

        def fake_sector_style(ticker):
            return {"UNH": "Choppy/Distribution", "NVDA": "Trending/Expanding"}.get(ticker)

        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Choppy/Distribution"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", side_effect=fake_sector_style), \
             patch("db.get_all_position_stops", return_value=stops), \
             patch("db.mark_profit_target_taken"), \
             patch("ibkr.place_market_order", return_value={"order_id": 1}) as mock_order:
            result = rm.check_profit_targets()

        # UNH (Choppy, ~12.5% target) hits at 18% gain; NVDA (Trending,
        # ~25% target) does not.
        assert len(result) == 1
        assert result[0]["ticker"] == "UNH"
        mock_order.assert_called_once_with("UNH", "SELL", 45.0)

    def test_sector_style_used_for_stale_window_too(self):
        from datetime import datetime, timedelta, timezone
        # 10 calendar days held, flat -- past Choppy/Distribution's
        # Accelerator window (14d... wait must be >=) use a value that
        # clears the Choppy window (14d) but not Trending's (28d), so the
        # sector read (Choppy) is what determines whether this flags.
        opened = datetime.now(timezone.utc) - timedelta(days=20)
        rm = self._rm([_priced_holding("UNH", qty=100.0, current_price=101.0)])
        stops = [{"ticker": "UNH", "entry_price": 100.0, "thesis_bucket": "Accelerator",
                  "opened_at": opened}]
        with patch("agents.market_regime.get_cached_regime",
                    return_value={"trade_style": "Trending/Expanding"}), \
             patch("agents.market_regime.get_trade_style_for_ticker", return_value="Choppy/Distribution"), \
             patch("db.get_all_position_stops", return_value=stops):
            result = rm.flag_stale_positions()
        assert len(result) == 1
        assert result[0]["trade_style"] == "Choppy/Distribution"
