# Watchlist

> **Portfolio Role is the flag-off fallback.** Every ticker below still
> carries a Compounder / Accelerator / Moonshot role, and that role stays
> the sizing/stop authority as long as `config.SETUP_GATES_ENABLED` is
> False (the default). Once that flag is set True, it becomes INERT for
> any ticker that `agents/setup_gates.py`'s `classify_setup()` currently
> matches to a live setup category (Breakout / Catalyst / Continuation,
> read nightly off `agents/trade_ladder.py`'s rungs) — that dynamic,
> day-to-day classification takes over as the sizing/stop authority
> instead. Role is still used as the fallback when the gates don't
> currently match anything for a ticker (a legacy position, or a genuinely
> quiet period — the gates are strict by design and can run near 0%
> qualification across the whole watchlist for days at a time). Do NOT
> rewrite the Portfolio Role column with Breakout/Catalyst/Continuation
> labels — that would misrepresent the design: the three setup categories
> are live, per-moment classifications, not a static per-ticker label the
> way role is. (2026-10-11, swing-trading cutover Phase 4/5.)

## Purpose

The watchlist is the system's active universe of tracked ideas.
Every ticker in the system passes through the watchlist before
the Scanner evaluates it. The watchlist defines scope.

---

## Structure

Every ticker in the watchlist carries the following fields:

| Field | Description |
|---|---|
| Ticker | Stock symbol |
| Company | Full name |
| Sector / Thesis | Which thesis file this maps to |
| Portfolio Role | Compounder / Accelerator / Moonshot |
| Conviction Tier | Core / Watch / Speculative |
| Primary Catalyst | What changes the investment case |
| Timing | When the thesis inflects |
| Notes | Current status, concerns, or context |

---

## Conviction Tiers

### Core

Definition: Highest conviction ideas. Ready to evaluate for action.

Characteristics:
- Strong thesis alignment with sector files
- Durable competitive advantage or clear inflection
- Long-term ownership candidate
- Near-term catalyst or ongoing compounder dynamic

Treatment:
- Full Scanner + Analyst + Risk evaluation every cycle
- Largest sizing potential
- Monitored continuously

### Watch

Definition: Interesting opportunities that need more evidence before acting.

Characteristics:
- Thesis forming but not yet confirmed
- Timing uncertain or early
- Needs a catalyst or data point to graduate to Core

Treatment:
- Scanner monitors price and catalyst signals
- Analyst evaluates when catalyst becomes visible
- Smaller or no position until conviction increases

### Speculative

Definition: High upside, uncertain outcomes. Moonshot territory.

Characteristics:
- Early technology or commercial stage
- Binary or long-duration risk profile
- Position sizing strictly capped

Treatment:
- Small sizing, defined risk
- Catalyst-driven evaluation
- Never force a position — wait for the right setup

---

## Watchlist by Theme

### AI Infrastructure — Compute / Accelerators

| Ticker | Company | Role | Tier |
|---|---|---|---|
| NVDA | Nvidia | Compounder | Core |
| AMD | Advanced Micro Devices | Compounder | Core |
| ARM | Arm Holdings | Compounder | Core |
| INTC | Intel | Watch | Watch |

### AI Infrastructure — Hyperscalers

| Ticker | Company | Role | Tier |
|---|---|---|---|
| MSFT | Microsoft | Compounder | Core |
| AMZN | Amazon | Compounder | Core |
| GOOG | Alphabet | Compounder | Core |
| META | Meta Platforms | Compounder | Core |

### AI Infrastructure — Memory / Storage

| Ticker | Company | Role | Tier |
|---|---|---|---|
| MU | Micron Technology | Accelerator | Core |
| SNDK | SanDisk | Accelerator | Watch |
| DRAM | Roundhill Memory ETF | Accelerator | Watch |

### AI Infrastructure — Memory Supply Chain

| Ticker | Company | Role | Tier |
|---|---|---|---|
| NLST | Netlist | Speculative | Speculative |
| ENTG | Entegris | Accelerator | Watch |
| KLIC | Kulicke and Soffa | Accelerator | Watch |
| TER | Teradyne | Accelerator | Watch |
| KLAC | KLA Corporation | Accelerator | Core |
| ONTO | Onto Innovation | Accelerator | Watch |
| LRCX | Lam Research | Accelerator | Core |
| AMAT | Applied Materials | Accelerator | Core |
| RMBS | Rambus | Accelerator | Watch |

### AI Infrastructure — Networking / Optical

| Ticker | Company | Role | Tier |
|---|---|---|---|
| AVGO | Broadcom | Compounder | Core |
| ANET | Arista Networks | Compounder | Core |
| CRDO | Credo Technology | Accelerator | Core |
| MRVL | Marvell Technology | Accelerator | Core |
| COHR | Coherent | Accelerator | Watch |
| LITE | Lumentum | Accelerator | Watch |
| AAOI | Applied Optoelectronics | Accelerator | Speculative |

### AI Infrastructure — Semi Packaging / Equipment

| Ticker | Company | Role | Tier |
|---|---|---|---|
| ASML | ASML Holding | Compounder | Core |
| SNPS | Synopsys | Compounder | Core |
| AMKR | Amkor Technology | Accelerator | Watch |

### AI Infrastructure — Neo Clouds / AI Compute Providers

| Ticker | Company | Role | Tier |
|---|---|---|---|
| CRWV | CoreWeave | Accelerator | Core |
| NBIS | Nebius | Accelerator | Watch |
| IREN | IREN | Accelerator | Watch |
| APLD | Applied Digital | Accelerator | Speculative |
| CORZ | Core Scientific | Accelerator | Speculative |
| ORCL | Oracle | Accelerator | Core |
| CLS | Celestica | Accelerator | Core |

### AI Software / Agent Economy

| Ticker | Company | Role | Tier |
|---|---|---|---|
| PLTR | Palantir | Compounder | Core |
| SNOW | Snowflake | Accelerator | Core |
| CRM | Salesforce | Accelerator | Watch |
| NOW | ServiceNow | Accelerator | Watch |
| DDOG | Datadog | Accelerator | Core |
| MDB | MongoDB | Accelerator | Watch |
| APP | AppLovin | Accelerator | Watch |
| NET | Cloudflare | Accelerator | Watch |

### Cyber / AI Security

| Ticker | Company | Role | Tier |
|---|---|---|---|
| CRWD | CrowdStrike | Compounder | Core |
| PANW | Palo Alto Networks | Compounder | Core |
| ZS | Zscaler | Accelerator | Watch |

### Power Infrastructure for AI

| Ticker | Company | Role | Tier |
|---|---|---|---|
| VRT | Vertiv | Accelerator | Core |
| PWR | Quanta Services | Accelerator | Core |
| NVT | nVent Electric | Accelerator | Watch |
| CAT | Caterpillar | Accelerator | Watch |
| GEV | GE Vernova | Accelerator | Core |
| BE | Bloom Energy | Accelerator | Watch |

### Nuclear / Baseload Power

| Ticker | Company | Role | Tier |
|---|---|---|---|
| CEG | Constellation Energy | Accelerator | Core |
| BWXT | BWX Technologies | Accelerator | Watch |
| CCJ | Cameco | Accelerator | Watch |
| SMR | NuScale Power | Moonshot | Speculative |
| OKLO | Oklo | Moonshot | Speculative |

### Robotics / Physical AI

| Ticker | Company | Role | Tier |
|---|---|---|---|
| TSLA | Tesla | Accelerator | Watch |
| ISRG | Intuitive Surgical | Compounder | Watch |
| TER | Teradyne | Accelerator | Watch |
| SYM | Symbotic | Accelerator | Watch |
| AUR | Aurora Innovation | Moonshot | Speculative |

### Quantum Computing

| Ticker | Company | Role | Tier |
|---|---|---|---|
| IONQ | IonQ | Moonshot | Core |
| RGTI | Rigetti Computing | Moonshot | Speculative |
| QBTS | D-Wave Quantum | Moonshot | Speculative |
| QUBT | Quantum Computing Inc | Moonshot | Speculative |
| QTUM | Defiance Quantum ETF | Moonshot | Watch |
| IBM | IBM | Compounder | Watch |

### Space Data Centers

| Ticker | Company | Role | Tier |
|---|---|---|---|
| RKLB | Rocket Lab | Moonshot | Speculative |
| ASTS | AST SpaceMobile | Moonshot | Speculative |
| LUNR | Intuitive Machines | Moonshot | Speculative |

### Financials / AI-Enabled Fintech

| Ticker | Company | Role | Tier |
|---|---|---|---|
| MA | Mastercard | Compounder | Watch |
| V | Visa | Compounder | Watch |
| HOOD | Robinhood | Accelerator | Watch |
| SOFI | SoFi Technologies | Accelerator | Watch |
| DAVE | Dave Inc | Accelerator | Speculative |

---

## Watchlist Management Rules

1. Every ticker must map to a thesis file. If no thesis file exists, create one before adding to the watchlist.

2. Tickers move between conviction tiers based on:
   - Thesis validation events (earnings, contracts, product milestones)
   - Changes in competitive position
   - Learning Agent feedback (consistent underperformance → downgrade)

3. The Learning Agent reviews the watchlist monthly and may recommend:
   - Removing consistently disappointing names
   - Reclassifying tickers based on observed behavior
   - Adding new names discovered through thesis ecosystem research

4. No ticker is added to Core without a clear thesis file entry and an identified catalyst.

5. Speculative tier names should not exceed 10% of the total watchlist count.
   Quality over quantity — a tight watchlist is better than a sprawling one.
