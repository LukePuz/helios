"""
Fundamentals-based valuation ensemble — deterministic (not LLM-estimated),
same house style as agents/chase_risk.py and agents/price_action.py.

Built 2026-09-13 replacing the original 2-anchor (peer-multiple + Street
consensus) fair-value calc, after running it against the full watchlist
showed 60% of tickers landing in "Deep Value" -- a real finding, not a
bug: Wall Street's own consensus price targets are structurally
optimistic almost everywhere (a well-documented sell-side bias), so
blending targetMeanPrice as a co-equal fair-value anchor systematically
skewed the whole watchlist cheap. Per user decision, Street consensus is
now DISPLAYED as a reference range only (street_target_low/mean/high) and
never feeds fair_value_blended or valuation_tier.

In its place: four independent, well-established valuation methods, each
of which computes its own implied fair value or explicitly abstains
(returns None) when the ticker's data doesn't support that method --
never forces a number a method isn't suited for (e.g. DCF on negative
FCF, PEG on negative earnings growth). fair_value_blended is the MEDIAN
of whichever methods actually computed (robust to one outlier method
dominating), and a valuation_tier is only assigned when >=2 independent
methods agree to compute at all -- otherwise "Insufficient Data", same
fail-closed convention used throughout this codebase (Trend Template,
the analyst's UDR/source-count gates, etc.) rather than a misleadingly
precise number from too little support.

  1. DCF (intrinsic)     -- 2-stage discounted free cash flow
  2. Comps (peer)        -- sector-scoped peer-median P/S, dispersion-
                             guarded (see ingestion.fundamentals.
                             get_sector_peer_median)
  3. Rule-of-40 multiple -- growth+FCF-margin score regressed against
                             this watchlist's OWN observed (score, P/S)
                             pairs -- data-driven, not a fixed lookup
  4. PEG (Lynch)          -- fair P/E == earnings growth rate (PEG=1),
                             applied to forward EPS
"""
import logging

log = logging.getLogger(__name__)

# CAPM-style discount rate inputs for the DCF -- not a live risk-free
# rate feed (that's a bigger dependency for a marginal accuracy gain at
# this stage); a fixed, documented estimate, revisit if this ever needs
# to track real rate moves.
DCF_RISK_FREE_RATE = 0.04
DCF_EQUITY_RISK_PREMIUM = 0.05
DCF_DEFAULT_BETA = 1.2          # used only when yfinance has no beta cached
DCF_TERMINAL_GROWTH = 0.03
DCF_PROJECTION_YEARS = 5
# Caps the near-term growth rate FED INTO the DCF projection, not the
# real observed growth rate stored elsewhere -- a ticker genuinely
# growing 150%+ YoY (common pre-scale in this watchlist) can't sustain
# that for 5 years, and extrapolating it directly would produce an
# absurd valuation; this bounds the DCF's own assumption, independent of
# what's reported everywhere else.
DCF_MAX_NEAR_TERM_GROWTH = 0.60
DCF_MIN_NEAR_TERM_GROWTH = -0.20
# DCF abstains (rather than risk a blown-up terminal value) unless the
# discount rate clears the terminal growth rate by at least this much.
DCF_MIN_DISCOUNT_SPREAD = 0.02

# Found 2026-09-13, user pushback on NVDA reading "Overextended": raw
# regression beta is well-documented (Damodaran) to overstate the true
# cost of equity for high-momentum growth names -- their price beta
# reflects trading volatility/momentum, not the underlying business risk
# of a company with NVDA's actual margin/balance-sheet profile. Capping
# beta before it feeds CAPM is a standard, disclosed adjustment (not
# tuned to produce a preferred answer), used consistently by every
# method below via _discount_rate() so the fix applies uniformly rather
# than needing to be re-applied per method.
DCF_MAX_BETA_FOR_DISCOUNT = 1.8

# Hypergrowth names get a LONGER explicit high-growth window before
# fading to terminal, and a higher near-term cap -- a fixed 5-year/60%-cap
# taper for every company regardless of growth stage was found (same
# 2026-09-13 pushback) to systematically undervalue genuine multi-year
# compounders; real DCF practice extends the explicit forecast period for
# companies with a longer expected runway of above-average growth, it
# doesn't apply one taper shape to every company. Threshold and extended
# values are a disclosed heuristic, not derived from a company-specific
# TAM study (this codebase doesn't have TAM data) -- revisit if that ever
# becomes available.
DCF_HYPERGROWTH_THRESHOLD = 0.75
DCF_HYPERGROWTH_PROJECTION_YEARS = 10
DCF_HYPERGROWTH_MAX_NEAR_TERM_GROWTH = 0.80

# Single-stage perpetuity models (Residual Income, Justified Multiple)
# are explicitly scoped in standard practice (CFA curriculum) to mature,
# stable-growth companies -- they have no mechanism to represent "high
# growth now, moderating over many years," unlike DCF's multi-stage
# taper. Applying them to a company still in supernormal growth is a
# textbook misapplication, not a calibration issue: they ABSTAIN above
# this threshold rather than force a number the model can't represent.
SINGLE_STAGE_MAX_APPLICABLE_GROWTH = 0.30


def _discount_rate(fund: dict) -> float:
    """Shared CAPM discount rate, beta-capped -- see
    DCF_MAX_BETA_FOR_DISCOUNT's comment. Every method (DCF, reverse DCF,
    Residual Income, Justified Multiple) calls this instead of computing
    its own, so the beta-cap fix applies uniformly."""
    beta = fund.get("beta") if fund.get("beta") is not None else DCF_DEFAULT_BETA
    beta = min(beta, DCF_MAX_BETA_FOR_DISCOUNT)
    return DCF_RISK_FREE_RATE + beta * DCF_EQUITY_RISK_PREMIUM

RULE40_MIN_PEER_POINTS = 8   # minimum cross-sectional points to trust a fitted regression
# A fitted regression is only trustworthy for x-values inside (or close
# to) the range it was fit on. Abstains rather than extrapolate beyond
# this margin past the observed (post-trim, see below) min/max.
RULE40_EXTRAPOLATION_MARGIN = 0.25
# Found 2026-09-13 testing against real data: a raw, untrimmed fit was
# dominated by a handful of near-zero-revenue-base tickers where % growth
# or % FCF margin becomes mathematically explosive and not meaningfully
# comparable across companies -- QUBT's revenue_growth_yoy=90.0 (9000%
# YoY, a tiny prior-year base) and AUR's fcf_margin=-82.85 (-8285%, FCF
# burning >80x its own revenue) between them warped the fitted slope
# enough that NVDA's evaluation came out at $797/share against a $218
# stock -- NOT because NVDA was mistreated, but because the line it was
# being measured against was already broken. Standard fix: trim points
# outside median +/- RULE40_OUTLIER_IQR_MULTIPLE x IQR before fitting
# (IQR-based, not a raw min/max clip -- robust to the exact kind of skew
# these leverage points create). A ticker whose OWN score falls outside
# that same trimmed range abstains too -- the metric is just as unreliable
# for evaluating it as it is for fitting against it.
RULE40_OUTLIER_IQR_MULTIPLE = 2.0

# Lynch's "fair P/E == growth rate" heuristic was built for the moderate
# 10-30% growers he actually covered -- it breaks down completely applied
# literally to triple-digit-growth mega-caps (NVDA's 128% earnings growth
# implied a 100x-capped P/E of $1,556/share against an actual $218 stock,
# testing this against real data 2026-09-13). Above this growth rate PEG
# ABSTAINS rather than force a number through an ever-higher cap -- this
# is a real methodology boundary, not a parameter to keep retuning
# upward until the output looks reasonable.
PEG_MAX_APPLICABLE_GROWTH = 0.50
PEG_FAIR_PE_CAP = 45.0

VALUATION_TIERS = [
    (0.80, "Deep Value"),
    (0.95, "Value"),
    (1.05, "Fair"),
    (1.25, "Rich"),
    (float("inf"), "Overextended"),
]


def _dcf_equity_value_for_growth(
    fcf: float, shares: float, net_cash: float, discount_rate: float, g0: float,
    terminal_growth: float = DCF_TERMINAL_GROWTH, years: int = DCF_PROJECTION_YEARS,
) -> float:
    """The forward-DCF math, factored out so compute_reverse_dcf can solve
    for g0 given a target equity value instead of solving for equity
    value given g0 -- same 2-stage taper-to-terminal shape as
    compute_dcf_fair_value, just parameterized so both directions share
    one implementation."""
    pv_sum = 0.0
    fcf_t = fcf
    for year in range(1, years + 1):
        g_t = g0 + (terminal_growth - g0) * (year / years)
        fcf_t = fcf_t * (1 + g_t)
        pv_sum += fcf_t / ((1 + discount_rate) ** year)
    terminal_value = fcf_t * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)
    return pv_sum + pv_terminal + net_cash


# Bisection bounds for the reverse-DCF growth-rate solver -- wide enough
# to cover anything from a genuine decline (-90%) to an extreme, likely-
# unsustainable ask (+500%), so "unsolvable within this range" itself is
# informative (the market is pricing in something even more extreme than
# the bound, or the stock is priced for outright decline beyond -90%/yr).
REVERSE_DCF_GROWTH_LO = -0.90
REVERSE_DCF_GROWTH_HI = 5.00
REVERSE_DCF_TOLERANCE = 1e-6
REVERSE_DCF_MAX_ITER = 60

# Thresholds for growth_assumption_flag -- documented as a rule-of-thumb,
# not a rigorous historical base-rate study (this codebase doesn't have a
# database of "% of companies that sustained X% growth for Y years" to
# derive a real percentile from). Sustained >=40% revenue growth for a
# 5-year explicit DCF window is widely cited in equity-research practice
# as rare; treat these bands as a transparent heuristic, not a precise
# statistic.
GROWTH_ASSUMPTION_AGGRESSIVE = 0.40
GROWTH_ASSUMPTION_EXTREME = 0.75


def compute_reverse_dcf(ticker: str, fund: dict, current_price: float | None) -> dict | None:
    """
    "Expectations Investing" (Mauboussin) -- instead of assuming a growth
    rate to compute a price, holds the CURRENT price fixed and solves
    (via bisection, since the forward-DCF relationship is monotonic in
    g0) for the near-term growth rate the market is implicitly requiring.
    Compares that ask against the ticker's own actual revenue growth.

    This is a genuinely different signal from forward DCF/Comps -- a
    plausibility judgment on the assumption embedded in the price, not
    another dollar estimate -- so it is NOT part of fair_value_blended's
    median. Returns None (abstains) under the same conditions forward DCF
    does (non-positive FCF, missing shares/price, discount rate too close
    to terminal growth), plus when the implied ask falls outside the
    solver's bounds entirely.
    """
    fcf = fund.get("free_cash_flow")
    shares = fund.get("shares_outstanding")
    if not fcf or fcf <= 0 or not shares or shares <= 0 or not current_price:
        return None

    discount_rate = _discount_rate(fund)
    if discount_rate - DCF_TERMINAL_GROWTH < DCF_MIN_DISCOUNT_SPREAD:
        return None

    net_cash = fund.get("net_cash") or 0.0
    target_equity_value = current_price * shares

    # Same hypergrowth-aware horizon as forward DCF (compute_dcf_fair_
    # value) -- solving for the implied growth rate using a 5-year window
    # when forward DCF used a 10-year window for the SAME ticker would be
    # internally inconsistent (two different structural assumptions about
    # the same company's own runway).
    actual_growth = fund.get("revenue_growth_yoy")
    years = DCF_HYPERGROWTH_PROJECTION_YEARS if (actual_growth or 0) >= DCF_HYPERGROWTH_THRESHOLD else DCF_PROJECTION_YEARS

    def f(g: float) -> float:
        return _dcf_equity_value_for_growth(fcf, shares, net_cash, discount_rate, g, years=years) - target_equity_value

    lo, hi = REVERSE_DCF_GROWTH_LO, REVERSE_DCF_GROWTH_HI
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:
        # The market's implied ask is outside even this wide a bound --
        # informative on its own, but not a number the solver can return.
        return None

    mid = (lo + hi) / 2
    for _ in range(REVERSE_DCF_MAX_ITER):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid) < REVERSE_DCF_TOLERANCE * abs(target_equity_value) + 1e-9:
            break
        if f_mid < 0:
            lo = mid
        else:
            hi = mid

    implied_growth = mid
    actual_growth = fund.get("revenue_growth_yoy")
    growth_gap = implied_growth - actual_growth if actual_growth is not None else None

    if implied_growth >= GROWTH_ASSUMPTION_EXTREME:
        flag = "extreme"
    elif implied_growth >= GROWTH_ASSUMPTION_AGGRESSIVE:
        flag = "aggressive"
    else:
        flag = "plausible"

    return {
        "implied_growth_rate": round(implied_growth, 4),
        "growth_gap_vs_actual": round(growth_gap, 4) if growth_gap is not None else None,
        "growth_assumption_flag": flag,
    }


# Residual Income (Ohlson model) needs the same discount rate the
# residual is measured against -- reuses DCF_RISK_FREE_RATE/
# DCF_EQUITY_RISK_PREMIUM/DCF_DEFAULT_BETA rather than a second set of
# constants, since it's the same "cost of equity," not a different one.
RI_TERMINAL_GROWTH = DCF_TERMINAL_GROWTH
RI_MIN_DISCOUNT_SPREAD = DCF_MIN_DISCOUNT_SPREAD


def compute_residual_income_fair_value(ticker: str, fund: dict) -> float | None:
    """
    Ohlson residual-income model, single-stage perpetuity form:
        V0 = BV0 + (ROE - r) x BV0 / (r - g)
    Mechanically distinct from DCF (accounting book equity + profitability
    spread, no cash-flow projection) and from Comps (no peer data at
    all). Works even when FCF is negative, AS LONG AS book equity is
    positive and ROE is computable -- exactly the gap DCF can't cover for
    early-stage names with real book equity but poor near-term cash flow.
    Abstains on negative/missing book value, missing ROE, or a discount
    rate too close to the growth rate (the perpetuity term blows up).
    """
    bv0 = fund.get("book_value_per_share")
    roe = fund.get("return_on_equity")
    growth = fund.get("revenue_growth_yoy")
    if not bv0 or bv0 <= 0 or roe is None:
        return None
    if growth is not None and growth > SINGLE_STAGE_MAX_APPLICABLE_GROWTH:
        return None  # single-stage perpetuity can't represent supernormal growth -- abstain, don't misapply

    discount_rate = _discount_rate(fund)
    # A single-stage perpetuity's growth rate must be a plausible FOREVER
    # rate -- RI_TERMINAL_GROWTH (same 3% long-run rate DCF's terminal
    # value uses), never this company's own current/near-term growth. No
    # company grows at, say, 60% in perpetuity; plugging a near-term rate
    # in here was the actual bug (found 2026-09-13 testing against real
    # data -- NVDA/ANET/PLTR all abstained despite perfectly good ROE/book
    # value, because their near-term growth exceeded the discount rate,
    # making (r-g) negative).
    g = RI_TERMINAL_GROWTH
    if discount_rate - g < RI_MIN_DISCOUNT_SPREAD:
        return None

    value = bv0 + (roe - discount_rate) * bv0 / (discount_rate - g)
    return value if value > 0 else None


# Justified Multiple (CFA-standard "fundamentals-derived fair multiple")
# needs a retention/reinvestment rate -- derived from the sustainable-
# growth identity (g = ROE x retention) rather than assumed, so it's
# still fully computed from this ticker's own numbers, not guessed.
JUSTIFIED_MULTIPLE_MAX_RETENTION = 1.0   # a company can't retain >100% of earnings


def compute_justified_multiple_fair_value(ticker: str, fund: dict) -> float | None:
    """
    Justified P/S, CFA-standard closed form:
        Justified P/S = NetMargin x (1 - b) x (1 + g) / (r - g)
    where b (retention rate) is backed out from the sustainable-growth
    identity g = ROE x b, i.e. b = g / ROE. No peer data needed at all --
    entirely self-contained from this ticker's own margin, growth, ROE,
    and a CAPM required return. Abstains when ROE <= 0 (the sustainable-
    growth identity breaks down -- can't back out a meaningful retention
    rate from a negative or zero return on equity), when the implied
    retention rate is nonsensical (<0 or >1, meaning g/ROE doesn't
    describe a real reinvestment policy for this company), or when the
    discount rate doesn't clear growth by a safe margin.
    """
    net_margin = fund.get("net_profit_margin")
    roe = fund.get("return_on_equity")
    market_cap = fund.get("market_cap")
    price_to_sales = fund.get("price_to_sales")
    shares = fund.get("shares_outstanding")
    growth = fund.get("revenue_growth_yoy")
    if net_margin is None or not roe or roe <= 0 or not market_cap or not price_to_sales or not shares:
        return None
    if growth is not None and growth > SINGLE_STAGE_MAX_APPLICABLE_GROWTH:
        return None  # same abstention as Residual Income -- not a stable-growth company

    # Same fix as Residual Income above: g must be the long-run terminal
    # rate this perpetuity-style formula can actually support, not the
    # company's own near-term growth (which would make (r-g) negative for
    # almost every real grower and made this method abstain universally
    # before the fix). Retention is then backed out from the sustainable-
    # growth identity at THAT rate -- e.g. "how much of NVDA's current ROE
    # would need to be retained to sustain a 3% long-run growth rate,"
    # which is a small, realistic number, not near-100% retention implied
    # by trying to sustain 60%+ growth forever.
    g = RI_TERMINAL_GROWTH
    retention = g / roe
    if retention < 0 or retention > JUSTIFIED_MULTIPLE_MAX_RETENTION:
        return None

    discount_rate = _discount_rate(fund)
    if discount_rate - g < RI_MIN_DISCOUNT_SPREAD:
        return None

    justified_ps = net_margin * (1 - retention) * (1 + g) / (discount_rate - g)
    if justified_ps <= 0:
        return None

    revenue = market_cap / price_to_sales
    return (justified_ps * revenue) / shares


def compute_dcf_fair_value(ticker: str, fund: dict) -> float | None:
    """2-stage DCF: near-term growth (capped, see module docstring) tapers
    linearly to DCF_TERMINAL_GROWTH, discounted at a beta-capped CAPM
    rate (_discount_rate), plus a Gordon-growth terminal value, plus net
    cash, over shares outstanding. Abstains on negative/missing FCF,
    missing shares, or a discount rate too close to terminal growth
    (numerically unstable terminal value).

    Genuine hypergrowth names (current growth >= DCF_HYPERGROWTH_
    THRESHOLD) get a LONGER explicit projection window and a higher
    near-term cap before fading to terminal -- found 2026-09-13 (user
    pushback on NVDA reading "Overextended"): a fixed 5-year/60%-cap
    taper applied to every company regardless of growth stage
    systematically undervalues multi-year compounders. Real DCF practice
    extends the explicit forecast period for companies with a longer
    expected runway of above-average growth."""
    fcf = fund.get("free_cash_flow")
    shares = fund.get("shares_outstanding")
    growth = fund.get("revenue_growth_yoy")
    if not fcf or fcf <= 0 or not shares or shares <= 0 or growth is None:
        return None

    discount_rate = _discount_rate(fund)
    if discount_rate - DCF_TERMINAL_GROWTH < DCF_MIN_DISCOUNT_SPREAD:
        return None

    if growth >= DCF_HYPERGROWTH_THRESHOLD:
        years = DCF_HYPERGROWTH_PROJECTION_YEARS
        max_growth = DCF_HYPERGROWTH_MAX_NEAR_TERM_GROWTH
    else:
        years = DCF_PROJECTION_YEARS
        max_growth = DCF_MAX_NEAR_TERM_GROWTH
    g0 = max(min(growth, max_growth), DCF_MIN_NEAR_TERM_GROWTH)
    net_cash = fund.get("net_cash") or 0.0

    pv_sum = 0.0
    fcf_t = fcf
    for year in range(1, years + 1):
        g_t = g0 + (DCF_TERMINAL_GROWTH - g0) * (year / years)
        fcf_t = fcf_t * (1 + g_t)
        pv_sum += fcf_t / ((1 + discount_rate) ** year)

    terminal_value = fcf_t * (1 + DCF_TERMINAL_GROWTH) / (discount_rate - DCF_TERMINAL_GROWTH)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)

    equity_value = pv_sum + pv_terminal + net_cash
    if equity_value <= 0:
        return None
    return equity_value / shares


def compute_comps_fair_value(ticker: str, fund: dict) -> float | None:
    """Peer-median P/S x this ticker's own trailing revenue / shares
    outstanding. Prefers the broadened S&P-500-scale industry search
    (ingestion.fundamentals.get_industry_peer_median, added 2026-09-13 --
    yfinance's own `industry` field, works regardless of index membership,
    ~500-name pool) since it's statistically sturdier; falls back to the
    original watchlist-only sector search when the broad one can't
    compute (no industry classification cached yet, or the broader pool
    itself is too thin/dispersed). Both fail closed on <3-5 peers or
    excessive dispersion rather than force a number (see either
    function's docstring for the concrete LUNR/RKLB/ASTS case)."""
    price_to_sales = fund.get("price_to_sales")
    market_cap = fund.get("market_cap")
    shares = fund.get("shares_outstanding")
    if not price_to_sales or not market_cap or not shares:
        return None

    peer_median_ps = None
    try:
        from ingestion.fundamentals import get_industry_peer_median
        peer_median_ps = get_industry_peer_median(ticker, "price_to_sales")
    except Exception as e:
        log.debug(f"[{ticker}] comps: broad industry peer median lookup failed: {e}")

    if not peer_median_ps:
        try:
            from ingestion.fundamentals import get_sector_peer_median
            peer_median_ps = get_sector_peer_median(ticker, "price_to_sales")
        except Exception as e:
            log.debug(f"[{ticker}] comps: narrow sector peer median lookup failed: {e}")

    if not peer_median_ps:
        return None
    revenue = market_cap / price_to_sales
    return (peer_median_ps * revenue) / shares


def compute_rule40_fair_value(ticker: str, fund: dict, all_fund: list[dict] | None = None) -> float | None:
    """Rule-of-40 score (revenue growth% + FCF margin%) regressed against
    this watchlist's OWN observed (score, price_to_sales) pairs via
    ordinary least squares -- genuinely data-driven (fit from real
    cross-sectional data each run), not a fixed heuristic lookup table.
    Abstains if this ticker's own inputs are missing, or if fewer than
    RULE40_MIN_PEER_POINTS other tickers have enough data to fit a
    trustworthy regression."""
    growth = fund.get("revenue_growth_yoy")
    fcf_margin = fund.get("fcf_margin")
    market_cap = fund.get("market_cap")
    price_to_sales = fund.get("price_to_sales")
    shares = fund.get("shares_outstanding")
    if growth is None or fcf_margin is None or not market_cap or not price_to_sales or not shares:
        return None

    score = growth * 100 + fcf_margin * 100

    import db
    rows = all_fund if all_fund is not None else db.get_fundamentals_all()
    all_points = [
        (r["revenue_growth_yoy"] * 100 + r["fcf_margin"] * 100, r["price_to_sales"])
        for r in rows
        if r["ticker"] != ticker
        and r.get("revenue_growth_yoy") is not None
        and r.get("fcf_margin") is not None
        and r.get("price_to_sales") is not None
    ]
    if len(all_points) < RULE40_MIN_PEER_POINTS:
        return None

    # IQR-based trim BEFORE fitting -- see RULE40_OUTLIER_IQR_MULTIPLE's
    # comment for the concrete QUBT/AUR case this fixes. Quartiles via
    # nearest-rank on the sorted score values (all_points), not the fancier
    # interpolated methods -- adequate for a filter threshold, not a
    # reported statistic.
    sorted_scores = sorted(p[0] for p in all_points)
    n_all = len(sorted_scores)
    q1 = sorted_scores[n_all // 4]
    q3 = sorted_scores[(3 * n_all) // 4]
    iqr = q3 - q1
    if iqr > 0:
        lo_bound = q1 - RULE40_OUTLIER_IQR_MULTIPLE * iqr
        hi_bound = q3 + RULE40_OUTLIER_IQR_MULTIPLE * iqr
        points = [p for p in all_points if lo_bound <= p[0] <= hi_bound]
        # This ticker's OWN score is just as unreliable to evaluate as it
        # would be to fit against, for the identical reason (a near-zero-
        # revenue-base % explosion) -- abstain rather than answer.
        if not (lo_bound <= score <= hi_bound):
            return None
    else:
        points = all_points

    if len(points) < RULE40_MIN_PEER_POINTS:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Extrapolation guard: don't trust the fitted line past the range it
    # was actually fit on (post-trim). A margin (not a hard clip to
    # min/max) allows a LITTLE headroom beyond the observed data without
    # pretending the regression is valid arbitrarily far outside it.
    x_min, x_max = min(xs), max(xs)
    x_range = x_max - x_min
    if x_range > 0:
        margin = x_range * RULE40_EXTRAPOLATION_MARGIN
        if score < x_min - margin or score > x_max + margin:
            return None

    n = len(points)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x

    implied_multiple = intercept + slope * score
    if implied_multiple <= 0:
        return None

    revenue = market_cap / price_to_sales
    return (implied_multiple * revenue) / shares


def compute_peg_fair_value(ticker: str, fund: dict) -> float | None:
    """Peter Lynch's heuristic: fair P/E ~= earnings growth rate (PEG=1),
    applied to forward EPS. Falls back to revenue_growth_yoy as a growth
    proxy when earnings_growth isn't cached (common for names not yet
    consistently profitable). Abstains on non-positive forward EPS or
    non-positive growth -- PEG is meaningless for either."""
    forward_eps = fund.get("forward_eps")
    growth = fund.get("earnings_growth")
    if growth is None:
        growth = fund.get("revenue_growth_yoy")
    if not forward_eps or forward_eps <= 0 or growth is None or growth <= 0:
        return None
    if growth > PEG_MAX_APPLICABLE_GROWTH:
        return None  # outside the range Lynch's heuristic was built for -- abstain, don't cap-and-force
    fair_pe = min(growth * 100, PEG_FAIR_PE_CAP)
    return fair_pe * forward_eps


def compute_valuation_ensemble(ticker: str, current_price: float | None = None) -> dict:
    """
    Runs all four methods, blends via median, and computes a valuation
    tier only when >=2 methods actually produced a value. Returns a dict
    matching agents/trade_ladder.py's ladder schema directly.
    """
    import db

    result = {
        "dcf_fair_value": None, "comps_fair_value": None, "rule40_fair_value": None, "peg_fair_value": None,
        "residual_income_fair_value": None, "justified_multiple_fair_value": None,
        "peer_implied_fair_value": None,  # kept for schema/back-compat -- same value as comps_fair_value
        "fair_value_blended": None, "valuation_methods_used": [], "valuation_confidence": None,
        "valuation_pct_of_fair": None, "valuation_tier": None, "valuation_outlier": False,
        "street_target_low": None, "street_target_high": None, "street_consensus_target": None,
        "implied_growth_rate": None, "growth_gap_vs_actual": None, "growth_assumption_flag": None,
    }

    fund = db.get_fundamentals(ticker)
    if not fund:
        return result

    result["street_target_low"] = fund.get("target_low_price")
    result["street_target_high"] = fund.get("target_high_price")
    result["street_consensus_target"] = fund.get("target_mean_price")

    # PRIMARY blend inputs -- comparable dollar price estimates only.
    # PEG and Rule-of-40 are computed separately below as CONTEXT (same
    # tier as Street consensus): both are useful growth-quality signals
    # but neither should be averaged against a real price estimate, and
    # reverse DCF is a plausibility judgment, not a price, so it's not
    # blend-eligible either (2026-09-13 user decision).
    methods: dict[str, float] = {}
    for name, fn in (
        ("dcf", compute_dcf_fair_value),
        ("comps", compute_comps_fair_value),
        ("residual_income", compute_residual_income_fair_value),
        ("justified_multiple", compute_justified_multiple_fair_value),
    ):
        try:
            value = fn(ticker, fund)
            if value and value > 0:
                methods[name] = value
        except Exception as e:
            log.debug(f"[{ticker}] valuation method '{name}' failed: {e}")

    result["dcf_fair_value"] = round(methods["dcf"], 2) if "dcf" in methods else None
    result["comps_fair_value"] = round(methods["comps"], 2) if "comps" in methods else None
    result["peer_implied_fair_value"] = result["comps_fair_value"]
    result["residual_income_fair_value"] = round(methods["residual_income"], 2) if "residual_income" in methods else None
    result["justified_multiple_fair_value"] = round(methods["justified_multiple"], 2) if "justified_multiple" in methods else None
    result["valuation_methods_used"] = sorted(methods.keys())

    # Context-only signals -- computed regardless of how the blend above
    # turned out, displayed alongside it, never fed into the median/tier.
    try:
        rule40 = compute_rule40_fair_value(ticker, fund)
        result["rule40_fair_value"] = round(rule40, 2) if rule40 else None
    except Exception as e:
        log.debug(f"[{ticker}] rule40 (context) failed: {e}")
    try:
        peg = compute_peg_fair_value(ticker, fund)
        result["peg_fair_value"] = round(peg, 2) if peg else None
    except Exception as e:
        log.debug(f"[{ticker}] peg (context) failed: {e}")
    try:
        reverse = compute_reverse_dcf(ticker, fund, current_price)
        if reverse:
            result.update(reverse)
    except Exception as e:
        log.debug(f"[{ticker}] reverse DCF (context) failed: {e}")

    # Family-grouped blend (2026-09-13, found testing against real data):
    # DCF, Residual Income, and Justified Multiple all share the SAME CAPM
    # discount rate and 3% terminal growth rate -- they are not 3
    # independent votes, they're 3 flavors of one underlying model
    # (pseudo-replication). Pooling all 4 raw values into one median let
    # this correlated "intrinsic" family outvote Comps 3-to-1, making the
    # blend MORE distorted, not less, once RI/Justified came online.
    # Grouping into two genuinely distinct families -- intrinsic
    # (cash-flow/book-value paradigm) vs. market (peer-pricing paradigm,
    # Comps today, joined by the S&P-500 broadening later) -- and blending
    # the two FAMILY-level medians restores an honest 2-perspective
    # comparison instead of a lopsided 4-way vote.
    intrinsic_methods = [v for k, v in methods.items() if k in ("dcf", "residual_income", "justified_multiple")]
    market_methods = [v for k, v in methods.items() if k in ("comps",)]

    def _median(vals: list[float]) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2

    family_estimates = [v for v in (_median(intrinsic_methods), _median(market_methods)) if v is not None]
    if len(family_estimates) < 2:
        return result  # only one PERSPECTIVE represented -- "Insufficient Data," not a guess

    blended = sum(family_estimates) / len(family_estimates)
    result["fair_value_blended"] = round(blended, 2)

    # Confidence now reflects how much the two FAMILIES agree with each
    # other -- the honest disagreement signal (intrinsic vs. market
    # perspective), not how tightly 3 correlated methods happen to
    # cluster against 1 independent one.
    lo, hi = min(family_estimates), max(family_estimates)
    family_gap = (hi - lo) / lo if lo else None
    if family_gap is not None:
        result["valuation_confidence"] = "high" if family_gap < 0.15 else "medium" if family_gap < 0.35 else "low"

    if current_price and blended:
        pct = current_price / blended
        result["valuation_pct_of_fair"] = round(pct, 4)
        for threshold, tier in VALUATION_TIERS:
            if pct <= threshold:
                result["valuation_tier"] = tier
                break
        result["valuation_outlier"] = result["valuation_tier"] in ("Deep Value", "Overextended")

    return result


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

    import db
    tickers = sys.argv[1:] or ["NVDA"]
    for t in tickers:
        price = None
        try:
            bars = db.get_price_history(t, days=5)
            price = bars[-1]["close"] if bars else None
        except Exception:
            pass
        r = compute_valuation_ensemble(t, current_price=price)
        print(f"\n=== {t} (price ${price}) ===")
        for k, v in r.items():
            print(f"  {k}: {v}")
