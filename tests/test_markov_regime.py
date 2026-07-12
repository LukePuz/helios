"""
Pure-logic unit tests for agents/markov_regime.py — no network calls.
get_markov_packet() (which hits yfinance) is intentionally NOT exercised here.
"""
import pytest

from agents.markov_regime import (
    STATES,
    WINDOW,
    build_transition_matrix,
    classify_state,
    compute_states,
    k_step_projection,
)


class TestClassifyState:
    def test_above_threshold_is_bull(self):
        assert classify_state(0.051) == "bull"

    def test_below_threshold_is_bear(self):
        assert classify_state(-0.051) == "bear"

    def test_exact_positive_threshold_is_sideways(self):
        assert classify_state(0.05) == "sideways"

    def test_exact_negative_threshold_is_sideways(self):
        assert classify_state(-0.05) == "sideways"

    def test_zero_is_sideways(self):
        assert classify_state(0.0) == "sideways"


class TestComputeStates:
    def test_constant_closes_are_sideways(self):
        closes = [100.0] * 21
        states = compute_states(closes)
        assert states == ["sideways"]

    def test_rising_series_produces_bull_states(self):
        # Each 20-day window return is +10% (rises by 1% per compounding day-ish,
        # but built explicitly so closes[i]/closes[i-20]-1 > 0.05 for all valid i).
        closes = [100.0 * (1.10 ** (i // WINDOW)) for i in range(60)]
        states = compute_states(closes, window=WINDOW)
        assert len(states) == 60 - WINDOW
        assert all(s == "bull" for s in states)

    def test_alignment_last_element_is_most_recent(self):
        closes = [100.0] * 20 + [200.0]  # last window: 200/100 - 1 = 100% -> bull
        states = compute_states(closes)
        assert states[-1] == "bull"

    def test_too_short_returns_empty(self):
        closes = [100.0] * 5
        assert compute_states(closes) == []


class TestBuildTransitionMatrix:
    def test_constant_state_gives_certain_self_transition(self):
        states = ["bull"] * 100
        matrix, counts = build_transition_matrix(states, stride=20)
        assert matrix["bull"]["bull"] == pytest.approx(1.0)
        assert matrix["bull"]["bear"] == pytest.approx(0.0)
        assert matrix["bull"]["sideways"] == pytest.approx(0.0)

    def test_never_seen_state_row_is_uniform(self):
        states = ["bull"] * 100  # "bear" and "sideways" never appear as s1
        matrix, counts = build_transition_matrix(states, stride=20)
        for s2 in STATES:
            assert matrix["bear"][s2] == pytest.approx(1 / 3)
            assert matrix["sideways"][s2] == pytest.approx(1 / 3)
        assert counts["bear"] == 0
        assert counts["sideways"] == 0

    def test_alternating_states_at_stride_boundaries(self):
        # Build a states list long enough that sampling at stride=20 yields
        # bull, bear, bull, bear, ... alternation.
        pattern = (["bull"] * 20) + (["bear"] * 20)
        states = pattern * 5  # 200 entries -> sampled indices 0,20,40,...180 (10 samples)
        matrix, counts = build_transition_matrix(states, stride=20)
        # sampled sequence alternates bull, bear, bull, bear, ...
        # every bull->bear and bear->bull transition, no self-transitions
        assert matrix["bull"]["bear"] == pytest.approx(1.0)
        assert matrix["bull"]["bull"] == pytest.approx(0.0)
        assert matrix["bear"]["bull"] == pytest.approx(1.0)
        assert matrix["bear"]["bear"] == pytest.approx(0.0)
        assert counts["bull"] > 0
        assert counts["bear"] > 0

    def test_counts_are_correct(self):
        states = ["bull", "bull", "bear", "bull"]  # stride=1 for a small hand check
        matrix, counts = build_transition_matrix(states, stride=1)
        # sampled = states itself; transitions: bull->bull, bull->bear, bear->bull
        assert counts["bull"] == 2
        assert counts["bear"] == 1
        assert counts["sideways"] == 0
        assert matrix["bull"]["bull"] == pytest.approx(0.5)
        assert matrix["bull"]["bear"] == pytest.approx(0.5)
        assert matrix["bear"]["bull"] == pytest.approx(1.0)

    def test_rows_sum_to_one(self):
        states = (["bull"] * 20 + ["sideways"] * 20 + ["bear"] * 20) * 3
        matrix, _ = build_transition_matrix(states, stride=20)
        for s1 in STATES:
            row_sum = sum(matrix[s1].values())
            assert row_sum == pytest.approx(1.0)


class TestKStepProjection:
    def test_identity_like_matrix_stays_put(self):
        # stickiness of 1.0 for every state — pure identity matrix
        matrix = {
            "bull":     {"bull": 1.0, "sideways": 0.0, "bear": 0.0},
            "sideways": {"bull": 0.0, "sideways": 1.0, "bear": 0.0},
            "bear":     {"bull": 0.0, "sideways": 0.0, "bear": 1.0},
        }
        for k in (1, 2, 5, 10):
            dist = k_step_projection(matrix, "bull", k)
            assert dist["bull"] == pytest.approx(1.0)
            assert dist["sideways"] == pytest.approx(0.0)
            assert dist["bear"] == pytest.approx(0.0)

    def test_distribution_sums_to_one(self):
        matrix = {
            "bull":     {"bull": 0.6, "sideways": 0.3, "bear": 0.1},
            "sideways": {"bull": 0.3, "sideways": 0.4, "bear": 0.3},
            "bear":     {"bull": 0.1, "sideways": 0.3, "bear": 0.6},
        }
        for k in (0, 1, 3, 5, 10):
            dist = k_step_projection(matrix, "sideways", k)
            assert sum(dist.values()) == pytest.approx(1.0)

    def test_zero_steps_returns_certain_current_state(self):
        matrix = {
            "bull":     {"bull": 0.6, "sideways": 0.3, "bear": 0.1},
            "sideways": {"bull": 0.3, "sideways": 0.4, "bear": 0.3},
            "bear":     {"bull": 0.1, "sideways": 0.3, "bear": 0.6},
        }
        dist = k_step_projection(matrix, "bear", 0)
        assert dist["bear"] == pytest.approx(1.0)
        assert dist["bull"] == pytest.approx(0.0)
