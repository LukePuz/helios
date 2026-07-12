# Market Regime Rules

## Purpose

This file defines the scoring system, signal definitions, thresholds,
and portfolio impact rules used by the Market Regime Agent.

It is a reference document — the Market Regime Agent imports these rules
as part of its context.

---

## Core Philosophy

Use macro signals to improve:
- Upside capture in favorable environments
- Drawdown protection in deteriorating environments

The regime does not predict the future — it describes the current environment
and adjusts how aggressively the system pursues opportunity.

Do not override regime signals with thesis conviction alone.
A great thesis in a Risk-Off regime still deserves reduced sizing.

---

## Signal Scoring

### Score Values

| Label | Score |
|---|---|
| Very Bullish | +5 |
| Bullish | +2 |
| Neutral | 0 |
| Bearish | -2 |
| Very Bearish | -5 |

### Signal Weights

| Category | Weight | Rationale |
|---|---|---|
| Trend (MAs) | 1x | Lagging — confirms direction |
| Volatility (VIX) | 2x | Fear/greed — fast-moving |
| Liquidity | 2x | Structural driver of risk appetite |
| Credit | 2x | Leading stress indicator |
| Breadth | 1x | Participation quality |
| Volume | 1x | Conviction confirmation |
| Markov persistence | 1x | Empirical state-transition probabilities — quantifies trend stickiness |

### Weighted Score Formula

```
Raw Score = Σ (signal_score × signal_weight)
Normalized Score = ((Raw Score - Min Possible) / (Max Possible - Min Possible)) × 100
```

---

## Signal Definitions

### Trend Signals (1x weight)

| Condition | Signal | Score |
|---|---|---|
| 50DMA > 200DMA (Golden Cross) | Very Bullish | +5 |
| Price > 50DMA and > 200DMA | Bullish | +2 |
| Price > 50DMA, below 200DMA | Neutral | 0 |
| Price < 50DMA, above 200DMA | Bearish | -2 |
| 50DMA < 200DMA (Death Cross) | Very Bearish | -5 |

### Volatility Signals (2x weight)

| Condition | Signal | Score | Note |
|---|---|---|---|
| VIX > 40 | Very Bullish | +5 | Contrarian — capitulation may signal bottom |
| VIX 25–40 | Bearish | -2 | Fear elevated |
| VIX 15–25 | Neutral | 0 | Normal range |
| VIX 12–15 | Neutral | 0 | Mild complacency |
| VIX < 12 | Bearish | -2 | Underpricing risk |

**Note on VIX > 40 contrarian signal:**
Apply only when trend and breadth signals are not simultaneously Very Bearish.
A VIX > 40 during a Death Cross environment may indicate more downside, not a bottom.
Require at least one other confirming signal before treating as Very Bullish.

### Liquidity Signals (2x weight)

| Condition | Signal | Score |
|---|---|---|
| Fed easing + M2 expanding + credit spreads tight | Very Bullish | +5 |
| Stable liquidity + benign credit | Bullish | +2 |
| Mixed signals | Neutral | 0 |
| Fed tightening or credit spreads widening | Bearish | -2 |
| Active QT + credit stress + spreads spiking | Very Bearish | -5 |

### Credit Signals (2x weight)

| Condition | Signal | Score |
|---|---|---|
| HY spreads tightening, LQD stable/rising | Bullish | +2 |
| HY spreads flat, no stress | Neutral | 0 |
| HY spreads widening (>50bps move) | Bearish | -2 |
| Credit stress, spreads spiking (>100bps move) | Very Bearish | -5 |

### Breadth Signals (1x weight)

| Condition | Signal | Score |
|---|---|---|
| >70% stocks above 200DMA | Very Bullish | +5 |
| 55–70% stocks above 200DMA | Bullish | +2 |
| 40–55% stocks above 200DMA | Neutral | 0 |
| 25–40% stocks above 200DMA | Bearish | -2 |
| <25% stocks above 200DMA | Very Bearish | -5 |

### Volume Signals (1x weight)

| Condition | Signal | Score |
|---|---|---|
| Strong accumulation (high volume on up days) | Bullish | +2 |
| Mixed / no clear pattern | Neutral | 0 |
| Distribution (high volume on down days) | Bearish | -2 |

### Markov Persistence Signals (1x weight)

Classifies each trading day's trailing 20-day SPY return into a state:

| Condition | State |
|---|---|
| 20-day return > +5% | bull |
| 20-day return in [-5%, +5%] | sideways |
| 20-day return < -5% | bear |

An empirical first-order transition matrix, P(next state \| current state),
is estimated from the historical state sequence sampled at **non-overlapping**
20-day strides (sampling every day would make consecutive states share 19 of
20 underlying closes and artificially inflate measured stickiness).

The directional signal is:

```
signal = P(next = bull | current state) - P(next = bear | current state)
```

which ranges from -1 (strongly bearish continuation) to +1 (strongly bullish
continuation).

| Condition | Signal | Score |
|---|---|---|
| signal >= +0.30 | Very Bullish | +5 |
| signal >= +0.10 | Bullish | +2 |
| signal > -0.10 | Neutral | 0 |
| signal > -0.30 | Bearish | -2 |
| signal <= -0.30 | Very Bearish | -5 |

**Low-sample suppression:** if the current state has fewer than 10 historical
non-overlapping transition samples, the signal is unreliable and is forced to
Neutral regardless of its computed value.

---

## Regime Thresholds

| Normalized Score | Regime |
|---|---|
| 75–100 | Strong Risk-On |
| 55–74 | Risk-On |
| 40–54 | Neutral |
| 20–39 | Risk-Off |
| 0–19 | Strong Risk-Off |

---

## Portfolio Impact by Regime

### Strong Risk-On (75–100)

- Equity exposure: Maximum allowable
- Accelerator positions: Increase size
- Moonshots: Acceptable with normal sizing
- Options: Tactical options and LEAPS allowed
- Cash target: 5–10%
- Leverage: Selective LEAPS on highest conviction only

### Risk-On (55–74)

- Equity exposure: High
- Accelerator positions: Normal to elevated
- Moonshots: Normal sizing
- Options: Catalyst-driven trades allowed
- Cash target: 10–15%
- Leverage: LEAPS on very high conviction only

### Neutral (40–54)

- Equity exposure: Moderate
- Accelerator positions: Reduce to normal
- Moonshots: Reduce size or hold only
- Options: Reduce speculative positions
- Cash target: 15–25%
- Leverage: Avoid new leverage

### Risk-Off (20–39)

- Equity exposure: Reduced
- Accelerator positions: Reduce materially
- Moonshots: Exit or minimal holds
- Options: Protective puts acceptable; no speculative calls
- Cash target: 25–40%
- Leverage: Exit leveraged positions

### Strong Risk-Off (0–19)

- Equity exposure: Minimal — core compounders only
- Accelerator positions: Exit
- Moonshots: Exit all
- Options: Hedges only
- Cash target: 40%+
- Leverage: None

---

## Cash Management Rules

### When to Deploy Cash

- Regime improves by at least one level
- High-conviction setup appears with near-term catalyst
- VIX spike > 30 creates asymmetric entry on core names
- Risk/reward materially improves on existing watchlist names

### When to Raise Cash

- Regime deteriorates by at least one level
- Liquidity signals turn bearish
- Multiple positions approach stop levels simultaneously
- Thesis confidence declines on core positions
- Valuations become disconnected from any reasonable fundamental path

### Cash is Optionality

Cash is not dead weight — it is the system's ability to exploit dislocations.
Never be fully invested in a Neutral or deteriorating regime.

---

## Regime Change Protocol

When regime changes:

1. If upgrade (e.g., Neutral → Risk-On):
   - Review watchlist for deployment candidates
   - Increase sizing on existing accelerators
   - Reduce excess cash toward new target

2. If downgrade (e.g., Risk-On → Neutral):
   - Review moonshot positions first — reduce or exit
   - Check accelerators against stop levels
   - Move toward cash target for new regime
   - Flag for human review if downgrade is by more than one level

3. If downgrade by two levels or more (rapid deterioration):
   - Immediate human alert required
   - Do not wait for next scheduled run
   - Prepare risk reduction plan before next session

---

## Macro Event Calendar Rules

Monitor the following recurring events and adjust pre-event:

| Event | Typical Action |
|---|---|
| FOMC Decision | Reduce speculative options 2 days prior. Review after. |
| CPI Release | Note sensitivity of rate-sensitive names. |
| Jobs Report (NFP) | Watch for liquidity signal changes. |
| Earnings (major tech) | Handled by Scanner/Analyst — not regime-level. |
| Treasury Auctions | Watch for yield spike and credit impact. |

**Rule:** Do not make large new position additions the day before a major macro event.
Wait for the data and re-score regime if warranted.
