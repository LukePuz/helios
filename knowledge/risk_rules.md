# Risk Rules

## Purpose

This file defines the quantitative risk rules used by the Risk Agent.
It is the authoritative source for position sizing, stop methodology,
concentration limits, and correlation management.

All rules are enforced deterministically. Deviations require human approval.

---

## Core Philosophy

Protect capital while pressing advantage.

The system's edge comes from being right on thesis and conviction —
not from taking excessive risk on individual positions.

Optimize for:
- Long-term CAGR (not short-term win rate)
- Asymmetric upside capture
- Thesis durability over trade frequency

Avoid:
- Unnecessary turnover
- Selling winners too early
- Overreacting to normal volatility
- Excessive concentration in correlated names

Risk controls exist to prevent:
- Catastrophic single-position losses
- Theme-level blowups (correlated positions all down simultaneously)
- Behavioral errors (averaging down on broken theses)

---

## Position Sizing Rules

### Sizing Inputs (in order of priority)

1. Thesis bucket (sets the ceiling)
2. Conviction score from Analyst (sets the target within ceiling)
3. Market regime multiplier (scales the target)
4. Correlation adjustment (reduces if theme is already crowded)

### Base Size Table

| Bucket | Conviction 9–10 | Conviction 7–8 | Conviction 5–6 | Conviction ≤4 |
|---|---|---|---|---|
| Compounder | 8–12% | 5–8% | 3–5% | 2–3% |
| Accelerator | 4–7% | 3–5% | 2–3% | 1–2% |
| Moonshot | 1–2% | 0.5–1% | 0.25–0.5% | WATCH only |

### Regime Multiplier

| Regime | Multiplier |
|---|---|
| Strong Risk-On | 1.2× |
| Risk-On | 1.0× |
| Neutral | 0.75× |
| Risk-Off | 0.5× |
| Strong Risk-Off | 0.25× |

### Correlation Adjustment

If adding this position would mean 3+ positions share the same primary catalyst or bottleneck:
- Reduce size by 25%
- Flag in output packet

### Absolute Position Limits

| Rule | Limit |
|---|---|
| Single Compounder maximum | 15% of portfolio |
| Single Accelerator maximum | 8% of portfolio |
| Single Moonshot maximum | 2% of portfolio |
| Single thesis/sector maximum | 35% of portfolio |
| Total AI infrastructure exposure | 60% of portfolio |
| Top 5 positions combined | 60% of portfolio |

**Growth exception:** A Compounder that grows above its initial allocation through price appreciation
does not need to be trimmed solely due to price gains. Trim only if thesis or valuation warrants.

---

## Entry Strategy

### Default: Hybrid Staged Entry

- Initial entry: 50–60% of full target size
- Remaining 40–50% added only after validation trigger

### Add Triggers (Staged Entry — 2nd tranche)

- Earnings beat with guidance raise
- Margin improvement confirmed
- Technical breakout above key resistance
- New contract, partnership, or design win announced
- Analyst conviction score upgraded

### Conditions for Full Position on Initial Entry

- Catalyst is < 5 days away (must be established now)
- Regime is Strong Risk-On AND conviction is 9–10
- Human explicitly approves full entry

### Averaging Down Rules

| Bucket | Rule |
|---|---|
| Compounder | May add on pullback IF thesis intact, regime supports, and fundamental data not deteriorating |
| Accelerator | Only add if a new fundamental catalyst confirms — not on price decline alone |
| Moonshot | Never average down under any circumstances |

---

## Stop Loss Rules

### Philosophy

Stops are pre-defined before entry and logged immediately after fill.
Stops protect against thesis failure, not normal volatility.

### Stop Methodology by Bucket

**Compounders — Wide, Thesis-Based**
- Set: 2–3× ATR below entry price, or below major long-term support
- Review: Weekly or after 5%+ single-session move
- Exit trigger: Stop breach OR thesis materially breaks (management change, competitive disruption, fundamental deterioration)
- Do not exit on: Normal short-term drawdowns while thesis is intact

**Accelerators — Moderate, Technical and Momentum-Based**
- Set: 1.5–2× ATR below entry price, or below 50DMA
- Review: Weekly or after any catalyst event
- Exit trigger: Technical stop breach, momentum loss, or catalyst fails to materialize
- Do not hold through: Failed catalyst without clear new thesis catalyst

**Moonshots — Defined Maximum Loss**
- Set: Maximum acceptable loss at entry (typically 30–40% of position value)
- Options preferred: Use options to structurally cap maximum loss to premium paid
- Exit trigger: Hard loss limit or binary event negative outcome
- No averaging down: If position moves against, assess fresh; do not add

### Stop Adjustment Rules

**Moving stops against the position (wider):** Never permitted.
**Moving stops with the position (trailing):** Permitted under trailing stop rules below.

---

## Trailing Stop Rules

### When to Activate Trailing Stop

| Bucket | Activate after... | Trail distance |
|---|---|---|
| Compounder | 30%+ gain | 15–20% from high |
| Accelerator | 20%+ gain | 10–15% from high |
| Moonshot | N/A | Use hard stop only |

### When NOT to Use Trailing Stops

- Position is in early thesis formation (< 3 months held)
- Thesis is intact and long-duration in nature
- Short-term price action is noisy but fundamental picture unchanged
- The system would trail out a Compounder during normal sector rotation

---

## Concentration Rules

### Single-Position Concentration

Monitored at entry and weekly thereafter.

| Bucket | Initial Cap | Growth Cap (price appreciation) |
|---|---|---|
| Compounder | 12% at entry | Allow up to 15% via gains |
| Accelerator | 7% at entry | Allow up to 8% via gains |
| Moonshot | 2% maximum, always | 2% hard cap |

If a position grows past the growth cap, it is flagged for human review.
Human decides whether to trim or maintain.

### Theme / Thesis Concentration

| Theme | Maximum |
|---|---|
| Single sector (e.g., AI networking) | 35% |
| Total AI infrastructure | 60% |
| Total speculative (Moonshots) | 10% |

### Correlation Risk Checks

Before adding any position, check:
- Shared customer concentration: Do 3+ positions depend on the same 2–3 customers?
- Shared bottleneck: Do 3+ positions depend on the same constraint (e.g., memory, power)?
- Shared catalyst: Do 3+ positions all move on the same single event?

If any of these triggers, reduce size of new position by 25% and flag.

---

## Loss Management Rules

### Before Entry

Define and log:
- Maximum acceptable loss in dollar terms
- Stop price level
- Position size
- R multiple target (minimum 2:1 upside vs. initial stop)

### After Entry

- Stop order placed in IBKR immediately after fill confirmation
- Do not move stop further from entry (never widen stops)
- Log stop level in SQLite

### If Stop Triggers

1. Execute exit (market order if urgent; limit order if orderly)
2. Log exit in SQLite: entry thesis, exit reason, R multiple achieved
3. Notify human with summary
4. Do not immediately re-enter the same name — wait for Learning Agent debrief
5. Review correlated positions for potential cascade risk

### Losing Trade Rules

- Never add to a losing position without a new fundamental catalyst
- A price decline alone is not a reason to add
- If thesis is intact: hold with stop in place
- If thesis is questionable: trim or exit, do not add

---

## Options Risk Rules

### When Options Are Appropriate

- High conviction (7+)
- Identifiable, near-term catalyst (30–120 DTE preferred)
- Liquidity is sufficient (reasonable bid/ask spread)
- Risk/reward is genuinely asymmetric

### Options Sizing

- Maximum single options position: 1–2% of portfolio (in premium paid)
- Total options exposure: ≤ 5% of portfolio in premium at risk
- Never use options to express low-conviction ideas

### Options Rules

- Never buy options with < 21 DTE on a new position (too much theta decay risk)
- Preferred DTE: 45–90 for catalyst plays, 90–120 for LEAPS
- Always use limit orders — never market orders on options
- Define exit at entry: both profit target and stop (50% premium loss)
- Do not hold through earnings if the thesis was pre-earnings only

### Leverage Rule

LEAPS (long-dated call options) are the only permitted leverage vehicle.
Use only on Compounders with conviction 8+ in a Risk-On or better regime.
LEAPS sizing: follow Accelerator sizing rules maximum (4–7% of portfolio notional equivalent).

---

## Summary Rules Table

| Rule | Value |
|---|---|
| Max single Compounder | 15% |
| Max single Accelerator | 8% |
| Max single Moonshot | 2% |
| Max single theme | 35% |
| Max total AI infrastructure | 60% |
| Min upside/downside ratio for new entry | 2:1 |
| Options premium at risk per position | 1–2% max |
| Total options exposure | ≤5% |
| Initial entry size (staged) | 50–60% of target |
| Correlation trigger | 3+ positions same catalyst/customer |
| Trailing stop: Compounder | Activate at 30% gain, 15–20% trail |
| Trailing stop: Accelerator | Activate at 20% gain, 10–15% trail |
