"""
Tests for agents.chase_risk's Kijun-sen distance scoring.

Regression coverage for the 2026-08-04 finding: the negative side (price
below its 26-day equilibrium -- a dip, not a chase) used to floor at 5 for
*any* distance below zero, so a stock 3% under its Kijun-sen and one 40%+
under it scored identically. Reviewing a week of real approvals showed this
system's actual entries skew toward deep pullbacks (-20% to -40% from
Kijun was common, not an edge case), so the missing gradation was silently
discarding real signal on the single most common shape of input. The fix
adds gradation to the negative side while deliberately keeping it more
forgiving than the positive side -- a dip is not a chase -- through -20%,
mirroring the module's original, still-intentional design decision.
"""
from agents.chase_risk import _score_from_pct


class TestPositiveSideUnchanged:
    """The chasing-strength side of the scale is untouched by this fix --
    confirm it still behaves exactly as before."""

    def test_at_kijun_scores_five(self):
        assert _score_from_pct(0.0) == 5

    def test_slightly_above_scores_four(self):
        assert _score_from_pct(3.0) == 4

    def test_ten_percent_above_scores_three(self):
        assert _score_from_pct(10.0) == 3

    def test_twenty_percent_above_scores_two(self):
        assert _score_from_pct(20.0) == 2

    def test_far_above_scores_one(self):
        assert _score_from_pct(35.0) == 1


class TestNegativeSideGradation:
    """The fix: a dip still gets full credit up to -20%, but depth beyond
    that now actually matters instead of collapsing to a flat 5."""

    def test_mild_dip_still_scores_five(self):
        """A few percent under Kijun is a normal, healthy pullback --
        must still score the same as before the fix."""
        assert _score_from_pct(-3.5) == 5

    def test_moderate_dip_still_scores_five(self):
        assert _score_from_pct(-12.0) == 5

    def test_exactly_twenty_percent_below_still_scores_five(self):
        assert _score_from_pct(-20.0) == 5

    def test_deep_pullback_no_longer_scores_five(self):
        """The core regression: -29.4% from Kijun (a real value seen in
        production, e.g. an actual MRVL approval) used to score the same
        "safest" 5 as a 1% wobble. Must now score lower."""
        assert _score_from_pct(-29.4) == 4

    def test_very_deep_pullback_scores_lower_still(self):
        """-39.3% (a real SNDK approval) is deeper than -29.4% and must
        score no higher than it -- monotonic, not just "different"."""
        score_29 = _score_from_pct(-29.4)
        score_39 = _score_from_pct(-39.3)
        assert score_39 <= score_29
        assert score_39 == 3

    def test_extreme_pullback_bottoms_out_at_two_not_one(self):
        """Even a catastrophic drop below Kijun must never score as
        harshly as chasing strength does (which can hit 1) -- a dip is
        still a fundamentally different risk than a chase, by design."""
        assert _score_from_pct(-75.0) == 2
        assert _score_from_pct(-500.0) == 2

    def test_negative_side_is_monotonically_non_increasing_with_depth(self):
        """Scores must never improve as the pullback gets deeper -- would
        indicate a bucket-boundary bug."""
        pcts = [0, -5, -10, -15, -20, -21, -25, -30, -35, -36, -40, -50, -51, -80]
        scores = [_score_from_pct(p) for p in pcts]
        for earlier, later in zip(scores, scores[1:]):
            assert later <= earlier, f"scores {scores} are not monotonic for pcts {pcts}"
