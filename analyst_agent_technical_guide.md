
















































































Technical & Valuation Analysis Framework — Analyst Agent Build Guide

A reference framework for how the Analyst agent should read price action and
valuation when turning a Scanner-flagged setup into a trade proposal. Volume


price trend and Ichimoku are the primary technical lenses; forward-looking
PEG is the primary valuation lens. Everything else in this guide exists to
keep those three honest and to flag when they don't apply.



⚠️ A note on tickers and thresholds. Any ticker or numeric threshold
below (e.g., "PEG < 1") is illustrative only — a way to explain the
framework, not a rule to apply mechanically or a reference to a real
position. Sector, rate environment, and growth stage all shift what
"cheap" or "bullish" actually means; this guide gives you the lens, not
the verdict.




1. Purpose & Scope

This framework governs the technical/valuation half of Analyst's job: once
Scanner flags a setup and tags it Core / Tail / (0DTE-macro), Analyst applies
this read before producing a trade proposal (thesis validation, catalyst
timeline, strike/expiration selection, IV percentile check). The goal isn't
to produce a single buy/sell signal — it's to produce a structured,
confidence-tagged read that Risk can gate against and Monitor can later
check for invalidation.

2. Where This Fits in the Agent Workflow

 Scanner                Analyst                          Risk / Monitor
 ────────               ───────                          ──────────────
 flags setup    ──▶     §3 Volume + Price read     ──┐
 (Core/Tail/             §4 Ichimoku read             ├──▶ structured technical +
  0DTE tag)              §5 combined technical tag    │    valuation confidence tag
                         §6 Forward PEG read         ──┘    feeds:
                         §7 alignment checks               • Risk's sizing/gating
                              (IV%, catalyst calendar,      • Monitor's trend-
                               relative strength,              invalidation watch
                               options positioning,
                               sentiment, confluence)

Scanner's screen is intentionally lighter-weight (price/volume/IV thresholds
to flag a candidate). This framework is the deeper pass Analyst runs before
anything reaches Risk.

3. Volume + Price Trend Analysis (Primary)

Core principle: a price move without volume confirmation is a move you
don't trust yet. Trend continuation should be accompanied by expanding
volume on the trend leg and contracting volume on pullbacks; the reverse
pattern (rising price, fading volume) is an early warning, not a confirmation.

Indicator set:

ToolWhat it's forRelative Volume (RVOL)Today's volume vs. 20/50-day average — spots unusual activity worth investigating before it shows up elsewhereOn-Balance Volume (OBV) / Accumulation-DistributionDivergence detector — price making new highs while OBV/AD doesn't is a classic warning signAnchored VWAPAnchor to the catalyst date (earnings, press event) from the calendar service — price holding above a post-catalyst anchored VWAP is a cleaner "the move is sticking" signal than a simple moving averageVolume Profile (Point of Control, Value Area High/Low)Identifies where the market thinks fair value sits, not just where price is — useful for setting realistic strike levels and stop logicPrice-volume divergence flagA simple boolean Analyst should always compute: is the current leg's volume above or below the prior leg's volume at a comparable price-move magnitude?

Read-rules checklist (each should resolve to pass/fail/neutral, not prose):


Is the trend leg's volume expanding relative to its own recent history?
Is OBV/AD confirming or diverging from price?
Is price holding above/below the catalyst-anchored VWAP?
Is RVOL elevated without a known news catalyst (a Scanner-relevant flag
in its own right — unexplained volume often precedes news)?


4. Ichimoku Analysis (Primary)

Components (standard periods — note if you deviate, since changing
periods changes everything downstream):


Tenkan-sen (conversion line): (9-period high + 9-period low) / 2
Kijun-sen (base line): (26-period high + 26-period low) / 2
Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 periods forward
Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods forward
Chikou Span: current close, plotted 26 periods back
Kumo (cloud): the area between Senkou A and B


How to read it, in priority order:


Cloud position is the trend filter. Price above the cloud = bullish
regime; below = bearish; inside = no clear trend, lower-conviction zone.
This should gate whether a setup even proceeds to strike selection — a
setup inside the cloud needs a much stronger volume/PEG case to justify
sizing.
Cloud color is forward-looking. Because the cloud is plotted 26
periods ahead, a bullish-to-bearish color flip in the future cloud
(Senkou A crossing below Senkou B) is an early warning that's already
visible today — don't ignore the cloud shape ahead of current price.
Cloud thickness is a volatility/support proxy. A thick cloud is a
harder level to break through (more support/resistance); a thin cloud
is a level price can punch through easily. This matters directly for
strike placement — a strike sitting just past a thick cloud edge is a
meaningfully different risk than one past a thin one.
TK cross is the timing trigger. Tenkan crossing above Kijun is
bullish, especially when it happens above the cloud (trend-confirming)
rather than inside or below it (weaker signal, more likely a fakeout).
Chikou span confirms. The lagging span should be clear of the
price action from 26 periods ago, in the direction of the trade — if
it's tangled in past price action, the trend lacks conviction.


Multi-timeframe approach: use weekly Ichimoku as the trend filter for
Core-bucket (thesis/trend) positions with longer holding periods, and daily
Ichimoku for Tail-bucket timing into a specific catalyst window. A Core
position should ideally show cloud agreement on both timeframes before
sizing up; a Tail position primarily needs the daily read to support holding
through the catalyst+buffer window, since the position is fundamentally
catalyst-driven rather than trend-driven.

Feeding into strike/expiration selection: cloud edges and Kijun-sen
levels are reasonable informal support/resistance references for strike
placement — a strike just beyond a level the cloud or Kijun-sen actively
defends is a different bet than one in open space. This is a secondary input
to the existing duration = catalyst + buffer logic, not a replacement for it.

5. Combining Volume+Price and Ichimoku into a Technical Read

Resist the urge to write a paragraph of narrative and call it a read. Score
it:

CheckPass / Fail / NeutralCloud position (trend filter)Cloud color, forward 26 periodsTK cross direction & location (above/below/in cloud)Chikou span clear of priceVolume confirming the trend legOBV/AD agreementPrice holding catalyst-anchored VWAP

A simple rule of thumb: 4+ of 7 passing in the same direction = "strong"
technical read; 2-3 = "moderate"; ≤1, or any direct conflict between
Ichimoku and volume signals = "weak" or "conflicting." Conflicting reads
shouldn't silently average out — they should be flagged the same way the
catalyst calendar flags conflicting dates, so Risk sees the disagreement
rather than a falsely-confident single tag.

6. Valuation: Forward-Looking PEG (Primary)

Calculation: PEG = Forward P/E ÷ Forward EPS growth rate (%), using
consensus forward estimates, not trailing twelve-month figures.

Why forward, not trailing, for this approach: the whole portfolio is
built around catalyst-driven inflections — a name's trailing growth rate is
frequently irrelevant precisely because the thesis is that growth is about
to change shape (a new product cycle, a capacity ramp, a regulatory
approval). Trailing PEG would either look absurdly expensive or wouldn't
register the setup at all. Forward PEG, anchored to consensus post-catalyst
estimates, is the version that actually reflects the thesis being
underwritten.

Data sourcing & growth horizon: use sell-side consensus forward EPS
(next fiscal year, and ideally a 2–3 year forward CAGR for names with a
multi-year ramp thesis, not just next-quarter estimates). Cross-check
consensus growth against the company's own guidance — a large gap between
guided and consensus growth is itself a signal worth flagging, in either
direction.

Interpreting the number: treat PEG as sector- and regime-relative,
not an absolute threshold. The classic "PEG < 1 = cheap" heuristic was built
for a different rate environment and doesn't transfer cleanly to, say, an
AI-capex-cycle semiconductor name trading alongside peers who are all
running hot PEGs — what matters more is PEG relative to the immediate peer
set, not relative to a textbook cutoff.

Guardrails — where PEG breaks down, and what to use instead:

SituationWhy PEG failsFallbackPre-revenue or pre-earnings (binary biotech, early-stage names)No meaningful EPS to anchor either side of the ratioRisk-adjusted NPV of pipeline/contracts, peer M&A comps, cash runway vs. catalyst timelineHyper-growth off a tiny base (>75-100%+ forward growth)Denominator instability makes the ratio swing wildly on small estimate revisionsRevenue-growth-adjusted multiples (EV/Sales vs. growth), rule-of-40-style composite scoresCyclical/commodity-exposed names mid-upcycleForward EPS itself is the volatile variable — consensus can lag the cycle badly in either directionCross-check against capacity/supply data and management capex commentary directly, not just consensus EPSNegative or break-even forward earningsRatio is undefined or meaninglessEV/EBITDA, EV/Sales, or cash-flow-based multiples instead

PEG should never be read in isolation from balance-sheet quality — a
flattering forward PEG built on margin assumptions the balance sheet can't
support (heavy debt load funding the growth, deteriorating free-cash-flow
conversion) is a value trap, not a value signal. Pair every PEG read with a
quick check of: net debt trend, FCF conversion, and whether margin
expansion (not just revenue growth) is actually driving the forward EPS
estimate.

7. Other Considerations for Approach Alignment

These are the pieces that keep §3–§6 from being read in a vacuum, given how
this portfolio is actually run:


IV percentile interplay. A strong bullish technical read at low IV
percentile is a much better long-premium setup than the same read at high
IV percentile heading into a known catalyst — the technical read should
always be reported alongside, not instead of, the existing IV percentile
check.
Catalyst-calendar alignment. Cross-reference §4's Ichimoku trend
against the catalyst calendar service's confidence tags before finalizing
duration = catalyst + buffer. A choppy, inside-the-cloud technical picture
heading into a conflicting-confidence catalyst date is a double caution
flag, not two independent ones — Risk should see them as compounding.
Relative strength vs. sector/thematic basket. Given the portfolio
leans into thematically correlated baskets (AI infrastructure, memory,
photonics, etc.), confirm whether a name's move is name-specific (thesis
working) or just basket beta — compare the technical read against a
sector ETF or peer basket, not price action in isolation.
Options positioning context. Since the data stack already includes
flow and dark-pool data, fold in: open-interest concentration at nearby
strikes (pin-risk awareness near expiration), put/call skew, IV term
structure (contango vs. backwardation heading into the catalyst), and
gamma exposure (GEX) levels — these affect how price actually behaves
near big OI strikes independent of the "fundamental" technical read.
Multi-timeframe confluence requirement. Don't size a Core position on
a daily-only signal that the weekly chart contradicts — require
agreement across both before treating a technical read as "strong."
Social-arbitrage / sentiment cross-check. Consistent with the
social-arbitrage research approach already in use, treat unusual
options flow or elevated news/social velocity as a prompt to re-run
this framework promptly, not as a substitute for it — sentiment moves
price before technicals confirm it, but technicals are what tell you
whether the move has follow-through.
Confidence/conviction tagging discipline. Output a structured tag
(strong/moderate/weak/conflicting), not narrative prose, for the same
reason the catalyst calendar tags confidence explicitly — Risk and
Monitor need something they can gate against programmatically, not a
paragraph they'd have to re-interpret.


8. Suggested Output Structure

This doesn't need its own database service the way the catalyst calendar
does — it fits naturally as a structured block attached to Analyst's
existing trade-proposal output (or, if you want it queryable/historical the
same way the catalyst calendar is, it would extend cleanly into a
lightweight technical_reads table following the same pattern — happy to
build that out the same way if useful).

Example shape, as JSON attached to the trade proposal:

json{
  "ticker": "<TICKER>",
  "as_of": "2026-06-21T00:00:00Z",
  "volume_price_read": {
    "trend_volume_confirming": true,
    "obv_divergence": false,
    "above_anchored_vwap": true
  },
  "ichimoku_read": {
    "timeframe_weekly": "above_cloud",
    "timeframe_daily": "above_cloud",
    "cloud_color_forward": "bullish",
    "tk_cross": "bullish_above_cloud",
    "chikou_confirms": true
  },
  "technical_confidence": "strong",
  "valuation": {
    "forward_pe": null,
    "forward_growth_pct": null,
    "peg_forward": null,
    "peg_applicable": true,
    "peer_relative_peg_note": "<sector-relative context, not absolute threshold>"
  },
  "alignment_checks": {
    "iv_percentile": null,
    "catalyst_calendar_confidence": "<pulled from catalyst service>",
    "relative_strength_vs_basket": null,
    "multi_timeframe_confluence": true
  }
}

9. Review Cadence & Maintenance


Re-run on every Scanner re-flag of a name already in the watchlist.
Re-run before any Risk re-sizing decision, not just at initial entry.
Daily, at minimum, for Tail-bucket names approaching a catalyst
window — the technical read can shift fast heading into a binary event.
Refresh PEG inputs on every guidance update or earnings cycle —
treat consensus estimates as perishable, not static; a forward PEG
computed on stale estimates is worse than no PEG at all.


10. Build Checklist


 Implement the §3 volume/price indicator set (RVOL, OBV/AD, anchored
VWAP, volume profile) as reusable functions Analyst calls per ticker
 Implement standard-period Ichimoku (§4) on both daily and weekly
timeframes; confirm cloud projection (26 periods forward) and Chikou
offset (26 periods back) are implemented correctly — this is the most
common place to introduce an off-by-one error
 Build the §5 scoring rubric as a function returning a structured tag,
not a free-text summary
 Build forward PEG calculation (§6) sourcing consensus forward EPS and
growth, with the guardrail table as explicit branching logic (route
pre-revenue/binary names to the NPV/comps fallback automatically
rather than silently computing a meaningless PEG)
 Wire the §7 alignment checks to pull from the catalyst calendar
service and existing IV percentile/options-flow data sources rather
than recomputing them
 Decide: attach output to the trade-proposal object (§8 default) or
extend into a queryable table mirroring the catalyst calendar's
pattern — only build the latter if you want historical technical
reads for backtesting, not just point-in-time decisions