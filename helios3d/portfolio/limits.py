"""Pure re-implementation of the live sizing limits (risk_manager.RiskManager).

The shadow portfolio applies exactly the live rules without a broker client,
the DB or alerts. Every function takes plain numbers (and plain dicts) and
returns plain numbers; nothing here does I/O.

Values come from ``config`` at call time (``config.NAME``, not copied
constants), so a config change, an env override or a test monkeypatch of
``config`` reaches these functions the same way it reaches live code. The
arithmetic is copied line for line from risk_manager.py, including the order
of floating-point operations and the rounding of returned fields, so results
are identical, not just close. tests/test_helios3d_limits_parity.py runs the
live methods (I/O mocked) and these functions over a grid and asserts equality.

What live reads through I/O becomes an argument here:

==========================  ===============================================
live read                   argument
==========================  ===============================================
portfolio_value()           ``pv``
cash()                      ``cash``
current_position_value(t)   ``held_value``
len(get_all_positions())    ``open_count`` (None = the read failed)
calculate_stop(...)         ``stop_price`` (ATR/bars are I/O)
is_halted()                 ``halted`` / ``halt_unknown``, or drawdown_halt()
db.get_portfolio_heat_pct   portfolio_heat_pct(equity, positions, stops)
load_watchlist()            ``watchlist`` ({ticker: {"sector", "role"}})
classify_setup(...)         ``setup`` ({"category", "stop_price"})
==========================  ===============================================

Not reproduced (documented, see the parity tests): the flag strings (each
outcome has a ``reason`` code instead), the cash-blocked alert, the halt
latch writes, and the setup-gate path's DB lookups (get_trade_ladder,
classify_setup), which are inputs here. If ``config`` fails to import, live
risk_manager falls back to built-in defaults; this module does not.
"""
from __future__ import annotations

import config as _cfg

# risk_manager.size_position: "initial_pct = 0.55  # 55% initial tranche".
# A literal in live code, not a config value, so it is repeated here; the
# parity tests fail if it moves.
FIRST_BUY_FRACTION = 0.55
# risk_manager.size_position: "if remaining < 0.005" (already at target).
AT_TARGET_TOLERANCE = 0.005
# risk_manager.size_position: REGIME_MULTIPLIER.get(regime, 0.75).
DEFAULT_REGIME_MULTIPLIER = 0.75
# risk_manager.size_position: sizes.get(band, (0.01, 0.02)).
DEFAULT_BAND = (0.01, 0.02)
# The AI-infrastructure ticker set is a literal inside
# RiskManager.check_concentration; copied here. test_ai_infra_set_matches_live
# parses the live source and fails if the two differ.
AI_INFRA_TICKERS = frozenset({
    "NVDA", "AMD", "ARM", "INTC", "MSFT", "AMZN", "GOOG", "GOOGL",
    "META", "AVGO", "ANET", "CRDO", "MRVL", "ASML", "SNPS", "KLAC",
    "LRCX", "AMAT", "MU", "CRWV", "ORCL", "CLS", "PLTR", "SNOW",
    "DDOG", "VRT", "PWR", "GEV", "CEG"})


# ---------------------------------------------------------------------------
# Account tiers (config.get_account_tier / get_active_limits)
# ---------------------------------------------------------------------------

def account_tier(pv: float) -> str:
    if pv >= _cfg.ACCOUNT_TIER_STANDARD_MIN:
        return "standard"
    if pv >= _cfg.ACCOUNT_TIER_REDUCED_MIN:
        return "reduced"
    return "micro"


def active_limits(pv: float) -> dict:
    tier = account_tier(pv)
    return {
        "tier": tier,
        "max_concurrent_positions": _cfg.TIER_MAX_CONCURRENT_POSITIONS[tier],
        "min_conviction_to_queue": _cfg.TIER_MIN_CONVICTION_TO_QUEUE[tier],
        "max_position_pct_moonshot": _cfg.TIER_MAX_POSITION_PCT_MOONSHOT[tier],
        "max_position_pct_compounder": _cfg.TIER_MAX_POSITION_PCT_COMPOUNDER[tier],
        "max_position_pct_accelerator": _cfg.TIER_MAX_POSITION_PCT_ACCELERATOR[tier],
        "single_share_cap_pct": _cfg.TIER_SINGLE_SHARE_CAP_PCT[tier],
    }


def standard_bucket_cap(bucket: str) -> float | None:
    return {"Compounder": _cfg.MAX_POSITION_PCT_COMPOUNDER,
            "Accelerator": _cfg.MAX_POSITION_PCT_ACCELERATOR,
            "Moonshot": _cfg.MAX_POSITION_PCT_MOONSHOT}.get(bucket)


def tier_bucket_cap(bucket: str, pv: float) -> float | None:
    """The tier's per-bucket position cap; None for an unknown bucket."""
    lim = active_limits(pv)
    return {"Compounder": lim.get("max_position_pct_compounder"),
            "Accelerator": lim.get("max_position_pct_accelerator"),
            "Moonshot": lim.get("max_position_pct_moonshot")}.get(bucket)


def hard_cap(bucket: str, pv: float) -> float:
    """The hard per-position cap sizing uses. An unknown bucket falls back to
    the STANDARD Accelerator cap at every tier (live behaviour)."""
    tc = tier_bucket_cap(bucket, pv)
    return tc if tc is not None else _cfg.MAX_POSITION_PCT_ACCELERATOR


# ---------------------------------------------------------------------------
# Max positions and the conviction floor
# ---------------------------------------------------------------------------

def max_positions(pv: float) -> int | None:
    return active_limits(pv)["max_concurrent_positions"]


def blocks_new_position(pv: float, open_count: int | None, held_value: float = 0.0) -> bool:
    """True when the tier's max-concurrent cap refuses this buy. Only a NEW
    name (nothing held) uses a slot; an unreadable count (None) does not block."""
    held_pct = held_value / pv if pv > 0 else 0
    cap = max_positions(pv)
    return held_pct == 0 and cap is not None and open_count is not None and open_count >= cap


def conviction_floor(pv: float) -> int:
    return active_limits(pv)["min_conviction_to_queue"]


def below_conviction_floor(conviction: int, pv: float) -> bool:
    return conviction < conviction_floor(pv)


# ---------------------------------------------------------------------------
# Per-bucket target ("max entry") and the first buy
# ---------------------------------------------------------------------------

def conviction_band(conviction: int) -> int:
    if conviction >= 9:
        return 9
    elif conviction >= 7:
        return 7
    elif conviction >= 5:
        return 5
    else:
        return 3


def regime_multiplier(regime: str) -> float:
    return _cfg.REGIME_MULTIPLIER.get(regime, DEFAULT_REGIME_MULTIPLIER)


def target_pct(bucket: str, conviction: int, regime: str, pv: float) -> float | None:
    """Target position weight for a new entry: the bucket's conviction band
    (scaled to the tier cap below standard), midpoint, times the regime
    multiplier, capped at the hard cap. None when the band has no size
    (Moonshot below c5: WATCH only)."""
    band = conviction_band(conviction)
    sizes = _cfg.BUCKET_SIZE_TABLE.get(bucket, _cfg.BUCKET_SIZE_TABLE["Accelerator"])
    lo, hi = sizes.get(band, DEFAULT_BAND)
    if lo is None:
        return None
    std_cap = standard_bucket_cap(bucket)
    tier_cap = tier_bucket_cap(bucket, pv)
    if account_tier(pv) != "standard" and std_cap and tier_cap:
        tier_scale = tier_cap / std_cap
        lo, hi = lo * tier_scale, hi * tier_scale
    base_pct = (lo + hi) / 2
    target = base_pct * regime_multiplier(regime)
    return min(target, hard_cap(bucket, pv))


def remaining_pct(target: float, held_value: float, pv: float) -> float:
    held_pct = held_value / pv if pv > 0 else 0
    return max(0, target - held_pct)


def single_share_ok(price: float, pv: float) -> bool:
    """Tier single-share cap: one share may not exceed this share of pv."""
    cap = active_limits(pv)["single_share_cap_pct"]
    return not (cap is not None and price > pv * cap)


def first_buy_dollars(pv: float, remaining: float, cash: float) -> tuple[float, bool]:
    """(initial_dollar, cash_capped): FIRST_BUY_FRACTION of the remaining
    target, capped at MAX_TRADE_PCT_OF_CASH_RESERVE of current cash."""
    dollar_amount = pv * remaining
    initial_dollar = dollar_amount * FIRST_BUY_FRACTION
    cash_cap = max(cash, 0) * _cfg.MAX_TRADE_PCT_OF_CASH_RESERVE
    if initial_dollar > cash_cap:
        return cash_cap, True
    return initial_dollar, False


def whole_shares(dollars: float, price: float) -> int:
    return int(dollars / price)


def one_share_allowed(shares: int, price: float, pv: float, cap: float, cash: float,
                      held_value: float = 0.0) -> bool:
    """The one-share rule: a sub-1-share first buy becomes 1 share when held
    value plus that share fits the hard cap and the share costs at most
    ONE_SHARE_MAX_PCT_OF_CASH of cash."""
    return (shares < 1 and _cfg.ALLOW_ONE_SHARE_ABOVE_ALLOCATION
            and held_value + price <= pv * cap
            and price <= max(cash, 0) * _cfg.ONE_SHARE_MAX_PCT_OF_CASH)


# ---------------------------------------------------------------------------
# Risk per trade and heat
# ---------------------------------------------------------------------------

def risk_cap_shares(pv: float, price: float, stop_price: float | None) -> int | None:
    """Most shares whose stop-out loses at most RISK_PCT_PER_TRADE of equity.
    None when there is no usable stop (the cap does not apply)."""
    if not (stop_price and stop_price < price):
        return None
    return int(pv * _cfg.RISK_PCT_PER_TRADE / (price - stop_price))


def portfolio_heat_pct(equity: float, positions: dict, stops: list[dict]) -> dict:
    """db.get_portfolio_heat_pct with its DB read (position_stops rows) as an
    argument. positions: {ticker: {"qty": float}}; stops: rows with ticker,
    entry_price, stop_price, setup_category, thesis_bucket."""
    by_category: dict[str, float] = {}
    total = 0.0
    for s in stops:
        pos = positions.get(s["ticker"])
        entry, stop = s.get("entry_price"), s.get("stop_price")
        if not pos or entry is None or stop is None:
            continue
        qty = abs(float(pos.get("qty", 0)))
        dollar_risk = max(0.0, float(entry) - float(stop)) * qty
        cat = s.get("setup_category") or s.get("thesis_bucket") or "Unclassified"
        by_category[cat] = by_category.get(cat, 0.0) + dollar_risk
        total += dollar_risk
    return {"heat_pct": round(total / equity, 4) if equity else 0.0,
            "heat_dollars": round(total, 2),
            "by_category": {k: round(v, 2) for k, v in by_category.items()}}


def heat_cap_pct(regime: str) -> float:
    return _cfg.PORTFOLIO_HEAT_CAP_PCT * regime_multiplier(regime)


def heat_shadow(shares: int, price: float, stop_price, pv: float, mult: float,
                current_heat_pct: float) -> dict | None:
    """RiskManager._heat_shadow (log-only on the legacy path)."""
    if not (pv and stop_price and stop_price < price):
        return None
    cap = _cfg.PORTFOLIO_HEAT_CAP_PCT * mult
    new = shares * (price - stop_price) / pv
    fits = max(0, int(max(0.0, cap - current_heat_pct) * pv / (price - stop_price)))
    return {"current_heat_pct": round(current_heat_pct, 4), "new_trade_risk_pct": round(new, 4),
            "cap_pct": round(cap, 4), "would_cut": current_heat_pct + new > cap,
            "shares_within_cap": min(shares, fits)}


def heat_capped_shares(shares: int, price: float, stop_price: float, pv: float,
                       mult: float, current_heat_pct: float) -> int:
    """The enforced heat trim (setup-gate path): scale down to the budget left
    under PORTFOLIO_HEAT_CAP_PCT x regime multiplier; never scales up."""
    cap = _cfg.PORTFOLIO_HEAT_CAP_PCT * mult
    new = (shares * (price - stop_price)) / pv if pv else 0.0
    if current_heat_pct + new > cap:
        budget = max(0.0, cap - current_heat_pct) * pv
        capped = int(budget / (price - stop_price))
        if capped < shares:
            return capped
    return shares


# ---------------------------------------------------------------------------
# Drawdown halt (RiskManager.is_halted without the latch write)
# ---------------------------------------------------------------------------

def drawdown_halt(pv: float, high_water_mark: float | None, start_value: float,
                  halt_flag: bool = False) -> tuple[bool, bool]:
    """(halted, halt_unknown). halt_unknown means the value read was
    implausible (not latched). Live also writes the latch and ratchets the
    high-water mark; those writes are left to the caller. high_water_mark
    None means no stored mark (live: missing or empty string); a stored 0 is
    a baseline of 0, which disables the drawdown check, as in live."""
    if halt_flag:
        return True, False
    baseline = float(high_water_mark) if high_water_mark is not None else start_value
    if baseline <= 0:
        return False, False
    if pv <= 0 or pv < 0.5 * baseline:
        return True, True
    return (baseline - pv) / baseline >= _cfg.DRAWDOWN_HALT_PCT, False


# ---------------------------------------------------------------------------
# Legacy sizing, composed (RiskManager.size_position with gates off)
# ---------------------------------------------------------------------------

def _zero(reason: str, cash_blocked: bool = False, **extra) -> dict:
    return {"position_size_pct": 0, "dollar_amount": 0, "shares_initial": 0,
            "cash_blocked": cash_blocked, "reason": reason, **extra}


def size_legacy(*, conviction: int, bucket: str, regime: str, price: float, pv: float,
                cash: float, held_value: float = 0.0, open_count: int | None = 0,
                stop_price: float | None = None, halted: bool = False,
                halt_unknown: bool = False, current_heat_pct: float | None = None) -> dict:
    """The legacy sizing path step for step. reason codes for a zero result:
    halted, conviction_floor, no_band, max_positions, at_target, bad_price,
    single_share_cap, under_one_share, risk_cap. A sized result has reason
    "sized" and the live dict's numeric fields. heat_shadow is filled only
    when current_heat_pct is given."""
    if halted:
        return _zero("halted", halt_unknown=halt_unknown)
    if below_conviction_floor(conviction, pv):
        return _zero("conviction_floor")
    target = target_pct(bucket, conviction, regime, pv)
    if target is None:
        return _zero("no_band")
    mult = regime_multiplier(regime)
    cap = hard_cap(bucket, pv)
    if blocks_new_position(pv, open_count, held_value):
        return _zero("max_positions")
    remaining = remaining_pct(target, held_value, pv)
    if remaining < AT_TARGET_TOLERANCE:
        return _zero("at_target")
    if not price or price <= 0:
        return _zero("bad_price")
    if not single_share_ok(price, pv):
        return _zero("single_share_cap")

    dollar_amount = pv * remaining
    initial_dollar, cash_capped = first_buy_dollars(pv, remaining, cash)
    shares = whole_shares(initial_dollar, price)
    one_share = False
    if one_share_allowed(shares, price, pv, cap, cash, held_value):
        shares, initial_dollar, one_share = 1, price, True
    if shares < 1:
        return _zero("under_one_share", cash_blocked=cash <= 0 or cash_capped)

    risk_cut = False
    rc = risk_cap_shares(pv, price, stop_price)
    if rc is not None and rc < shares:
        if rc < 1:
            return _zero("risk_cap")
        shares, initial_dollar, risk_cut = rc, rc * price, True

    initial_cost = shares * price
    max_loss = (initial_cost - shares * stop_price) if stop_price else initial_cost * 0.35
    return {
        "reason": "sized",
        "position_size_pct": round(remaining, 4),
        "dollar_amount": round(dollar_amount, 2),
        "initial_entry_pct": FIRST_BUY_FRACTION,
        "initial_dollar": round(initial_dollar, 2),
        "shares_initial": shares,
        "order_pct": round(shares * price / pv, 4) if pv else 0,
        "stop_loss_price": stop_price,
        "max_loss_if_stopped": round(max_loss, 2),
        "regime_multiplier": mult,
        "target_pct": target,
        "hard_cap": cap,
        "cash_capped": cash_capped,
        "one_share_rule": one_share,
        "risk_capped": risk_cut,
        "heat_shadow": (heat_shadow(shares, price, stop_price, pv, mult, current_heat_pct)
                        if current_heat_pct is not None else None),
    }


# ---------------------------------------------------------------------------
# Setup-gate sizing (RiskManager._size_position_setup_gates; off in live)
# ---------------------------------------------------------------------------

def size_setup_gate(*, bucket: str, regime: str, price: float, pv: float, cash: float,
                    setup: dict | None, held_value: float = 0.0,
                    current_heat_pct: float = 0.0) -> dict | None:
    """Risk-first sizing when SETUP_GATES_ENABLED. ``setup`` is the
    classify_setup best match ({"category", "stop_price"}) or None when there
    is no ladder or no match. Returns None where live falls back to legacy."""
    if bucket not in _cfg.CATEGORY_RISK_PCT_PER_TRADE:
        return None
    if not setup or setup.get("category") != bucket or setup.get("stop_price") is None:
        return None
    stop_price = float(setup["stop_price"])
    if not price or price <= 0 or stop_price >= price:
        return None
    mult = regime_multiplier(regime)
    risk_dollars = pv * _cfg.CATEGORY_RISK_PCT_PER_TRADE[bucket] * mult
    shares = int(risk_dollars / (price - stop_price))
    if shares < 1:
        return _zero("risk_cap", heat_blocked=False)
    _lo, ceiling_hi = _cfg.CATEGORY_SOFT_CEILING_PCT.get(bucket, (None, None))
    if ceiling_hi is not None:
        ceiling_shares = int(max(0.0, pv * ceiling_hi - held_value) / price) if price else 0
        if ceiling_shares < shares:
            shares = ceiling_shares
    if shares < 1:
        return _zero("soft_ceiling", heat_blocked=False)
    cash_cap = max(cash, 0) * _cfg.MAX_TRADE_PCT_OF_CASH_RESERVE
    initial_dollar = shares * price
    cash_capped = initial_dollar > cash_cap
    if cash_capped:
        shares = int(cash_cap / price) if price else 0
        initial_dollar = shares * price
    if shares < 1:
        return _zero("under_one_share", cash_blocked=cash <= 0 or cash_capped, heat_blocked=False)
    trimmed = heat_capped_shares(shares, price, stop_price, pv, mult, current_heat_pct)
    if trimmed != shares:
        shares = trimmed
        initial_dollar = shares * price
    if shares < 1:
        return _zero("heat_cap", heat_blocked=True)
    return {
        "reason": "sized",
        "position_size_pct": round(initial_dollar / pv, 4) if pv else 0,
        "dollar_amount": round(initial_dollar, 2),
        "initial_entry_pct": 1.0,
        "initial_dollar": round(initial_dollar, 2),
        "shares_initial": shares,
        "order_pct": round(initial_dollar / pv, 4) if pv else 0,
        "stop_loss_price": stop_price,
        "max_loss_if_stopped": round(shares * (price - stop_price), 2),
        "regime_multiplier": mult,
        "cash_blocked": False,
        "heat_blocked": False,
    }


# ---------------------------------------------------------------------------
# Concentration caps (RiskManager.check_concentration + shares_after_concentration)
# ---------------------------------------------------------------------------

def concentration(ticker: str, proposed_size_pct: float, bucket: str | None,
                  position_values: dict[str, float], pv: float,
                  watchlist: dict | None) -> dict:
    """position_values: {symbol: market_value} of current holdings.
    watchlist: {ticker: {"sector", "role"}}, or None when it could not be
    loaded (then the theme and moonshot caps are None, as in live)."""
    weights = {sym: (float(mv) / pv if pv else 0) for sym, mv in position_values.items()}
    weights[ticker] = weights.get(ticker, 0) + proposed_size_pct
    breaches: list[str] = []

    ai_total = sum(v for t, v in weights.items() if t in AI_INFRA_TICKERS)
    ai_cap = None
    if ticker in AI_INFRA_TICKERS:
        ai_cap = max(0.0, _cfg.MAX_AI_INFRA_PCT - (ai_total - proposed_size_pct))
    if ai_total > _cfg.MAX_AI_INFRA_PCT:
        breaches.append("ai_infra")
    if weights.get(ticker, 0) > active_limits(pv)["max_position_pct_compounder"]:
        breaches.append("single_position")

    theme_cap = None
    if watchlist is not None:
        sector = watchlist.get(ticker, {}).get("sector", "")
        if sector:
            theme_total = sum(v for t, v in weights.items()
                              if watchlist.get(t, {}).get("sector", "") == sector)
            theme_cap = max(0.0, _cfg.MAX_SINGLE_THEME_PCT - (theme_total - proposed_size_pct))
            if theme_total > _cfg.MAX_SINGLE_THEME_PCT:
                breaches.append("theme")

    moon_cap = None
    if bucket == "Moonshot" and watchlist is not None:
        moon_total = sum(v for t, v in weights.items()
                         if watchlist.get(t, {}).get("role", "") == "Moonshot")
        moon_cap = max(0.0, _cfg.MAX_MOONSHOT_TOTAL_PCT - (moon_total - proposed_size_pct))
        if moon_total > _cfg.MAX_MOONSHOT_TOTAL_PCT:
            breaches.append("moonshot_total")

    def _r(x):
        return round(x, 4) if x is not None else None
    return {"passed": not breaches, "breaches": breaches,
            "ai_infrastructure_total": round(ai_total, 4),
            "ai_infra_size_cap_pct": _r(ai_cap), "theme_size_cap_pct": _r(theme_cap),
            "moonshot_size_cap_pct": _r(moon_cap)}


def shares_after_concentration(shares: int, position_size_pct: float | None,
                               order_pct: float | None, conc: dict) -> int:
    """Shares left after the concentration caps, checked at the larger of
    target and order weight. ``conc`` is concentration()'s result for that
    larger weight."""
    shares = int(shares or 0)
    if shares <= 0:
        return 0
    proposed = max(position_size_pct or 0, order_pct or 0)
    if conc.get("passed", False):
        return shares
    caps = [c for c in (conc.get("ai_infra_size_cap_pct"), conc.get("theme_size_cap_pct"),
                        conc.get("moonshot_size_cap_pct")) if c is not None]
    cap = min(caps) if caps else None
    if cap is None or cap >= proposed:
        return shares
    base = (order_pct or 0) or proposed
    return min(shares, int(shares * cap / base + 1e-9)) if base > 0 else 0
