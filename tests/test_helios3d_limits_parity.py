"""Parity: helios3d.portfolio.limits against the live RiskManager.

Each test runs the live method with its I/O replaced by fakes (broker client,
db reads, watchlist, setup gates) and the pure function on the same numbers,
and asserts identical results (==, not approx). No DB, broker or network.

Not compared, because they are side effects or text rather than limits:
the flag strings (mapped to reason codes instead), the cash-blocked alert,
the halt latch and high-water-mark writes, and the fail-closed results when a
broker or DB read raises (the pure functions take values, so they have no
read to fail).
"""
import ast
import inspect
import itertools
import random
import textwrap
from types import SimpleNamespace

import pytest

import config
import db
import risk_manager
from helios3d.portfolio import limits

PVS = [900.0, 2400.0, 4999.99, 5000.0, 12_000.0, 24_999.0, 25_000.0, 100_000.0]
CONVICTIONS = [3, 4, 5, 6, 7, 8, 9, 10]
BUCKETS = ["Compounder", "Accelerator", "Moonshot", "Mystery"]
REGIMES = ["Strong Risk-On", "Risk-On", "Neutral", "Risk-Off", "Strong Risk-Off", "Unlabelled"]
PRICES = [0.0, 3.2, 18.75, 47.3, 180.0, 455.5, 1300.0]
CASH_FRACS = [-0.05, 0.0, 0.08, 0.35, 0.6, 1.0]
HELD_FRACS = [0.0, 0.0, 0.01, 0.06, 0.3]
OTHERS = [0, 2, 4, 7, 8, 9]
STOP_FRACS = [None, 0.97, 0.9, 0.5, 0.01]


class FakeClient:
    def __init__(self, pv, cash, positions):
        self.acct = SimpleNamespace(portfolio_value=pv, cash=cash, buying_power=cash)
        self.positions = positions

    def get_account(self):
        return self.acct

    def get_all_positions(self):
        return list(self.positions)

    def get_open_position(self, symbol):
        for p in self.positions:
            if p.symbol == symbol:
                return p
        raise Exception(f"No open position in {symbol}")


def _pos(symbol, market_value, qty=1.0):
    return SimpleNamespace(symbol=symbol, market_value=market_value, qty=qty, unrealized_pl=0.0)


@pytest.fixture
def live(monkeypatch):
    """Live RiskManager with every I/O faked. Returns a builder."""
    state = {"stops": [], "stop_price": None, "alerts": 0}
    monkeypatch.setattr(risk_manager, "TRADING_ENV", "paper")
    monkeypatch.setattr(risk_manager, "SETUP_GATES_ENABLED", False)
    monkeypatch.setattr(db, "get_state", lambda k, d=None: {"trading_halted_paper": "false"}.get(k, d))
    monkeypatch.setattr(db, "set_state", lambda *a, **k: None)
    monkeypatch.setattr(db, "get_all_position_stops", lambda: state["stops"])
    monkeypatch.setattr(risk_manager.RiskManager, "calculate_stop",
                        lambda self, t, p, b, **k: (state["stop_price"], "fixture"))

    def _alert(self, cash):
        state["alerts"] += 1
    monkeypatch.setattr(risk_manager.RiskManager, "_alert_cash_blocked", _alert)

    def build(pv, cash, positions):
        return risk_manager.RiskManager(FakeClient(pv, cash, positions))
    build.state = state
    return build


_REASONS = [("HALT", "halted"), ("below the", "conviction_floor"), ("WATCH only", "no_band"),
            ("concurrent positions", "max_positions"), ("at/near target", "at_target"),
            ("Invalid price", "bad_price"), ("single-share cap", "single_share_cap"),
            ("per-trade risk cap", "risk_cap"), ("buys less than 1 share", "under_one_share")]


def _reason(sizing):
    if sizing.get("shares_initial", 0) > 0:
        return "sized"
    last = (sizing.get("flags") or [""])[-1]
    for needle, code in _REASONS:
        if needle in last:
            return code
    return f"unmapped: {last}"


SIZED_KEYS = ["position_size_pct", "dollar_amount", "initial_entry_pct", "initial_dollar",
              "shares_initial", "order_pct", "stop_loss_price", "max_loss_if_stopped",
              "regime_multiplier", "heat_shadow"]


def _legacy_case(build, pv, conv, bucket, regime, price, cash_frac, held_frac, others, stop_frac):
    cash = pv * cash_frac
    held = pv * held_frac
    positions = [_pos(f"OTH{i}", pv * 0.03, qty=2.0) for i in range(others)]
    if held:
        positions.append(_pos("TKR", held, qty=3.0))
    stop = None if stop_frac is None else round(price * stop_frac, 2)
    build.state["stop_price"] = stop
    build.state["stops"] = [{"ticker": f"OTH{i}", "entry_price": 50.0, "stop_price": 50.0 - i,
                             "thesis_bucket": "Accelerator"} for i in range(others)]
    rm = build(pv, cash, positions)
    got_live = rm.size_position("TKR", conv, bucket, regime, price)
    heat = limits.portfolio_heat_pct(pv, {p.symbol: {"qty": float(p.qty)} for p in positions},
                                     build.state["stops"])["heat_pct"]
    got_pure = limits.size_legacy(conviction=conv, bucket=bucket, regime=regime, price=price,
                                  pv=pv, cash=cash, held_value=held, open_count=len(positions),
                                  stop_price=stop, current_heat_pct=heat)
    return got_live, got_pure


def _assert_same(got_live, got_pure, case):
    assert _reason(got_live) == got_pure["reason"], case
    assert got_live.get("cash_blocked", False) == got_pure.get("cash_blocked", False), case
    if got_pure["reason"] == "sized":
        for k in SIZED_KEYS:
            assert got_live[k] == got_pure[k], (case, k, got_live[k], got_pure[k])
    else:
        for k in ("position_size_pct", "dollar_amount", "shares_initial", "cash_blocked"):
            assert got_live[k] == got_pure[k], (case, k)


def _grid(n, seed):
    # Sample flat indices and decode them, rather than materialising the
    # ~10M-case product (about 1.4 GB).
    dims = (PVS, CONVICTIONS, BUCKETS, REGIMES, PRICES, CASH_FRACS, HELD_FRACS, OTHERS, STOP_FRACS)
    total = 1
    for d in dims:
        total *= len(d)
    cases = []
    for i in random.Random(seed).sample(range(total), n):
        case = []
        for d in reversed(dims):
            i, r = divmod(i, len(d))
            case.append(d[r])
        cases.append(tuple(reversed(case)))
    return cases


def test_size_legacy_parity_grid(live):
    seen = set()
    for case in _grid(6000, 20261102):
        got_live, got_pure = _legacy_case(live, *case)
        _assert_same(got_live, got_pure, case)
        seen.add(got_pure["reason"])
        seen.update(k for k in ("one_share_rule", "risk_capped", "cash_capped") if got_pure.get(k))
        if got_live.get("cash_blocked"):
            seen.add("cash_blocked")
    assert {"one_share_rule", "risk_capped", "cash_capped", "cash_blocked"} <= seen
    seen -= {"one_share_rule", "risk_capped", "cash_capped", "cash_blocked"}
    # The grid reaches every legacy outcome except a halt (tested separately)
    # and no_band: Moonshot has no size below c5, but every tier's conviction
    # floor is already 5 or more, so that exit is unreachable under today's
    # config (test_no_band_exit_with_a_lower_floor reaches it).
    assert seen == {"sized", "conviction_floor", "max_positions", "at_target",
                    "bad_price", "single_share_cap", "under_one_share", "risk_cap"}


def test_size_legacy_micro_account_grid(live):
    """Dense grid where live trades today: micro tier, the names that sized to
    zero in the Phase 0 record, every regime and conviction above the floor."""
    n = 0
    for pv, conv, bucket, regime, price, cash_frac, held_frac, others, stop_frac in itertools.product(
            [1800.0, 2400.0, 3100.0], [6, 7, 8, 9], ["Compounder", "Accelerator", "Moonshot"],
            REGIMES[:5], [18.75, 180.0, 455.5, 610.0], [0.1, 0.35, 0.7], [0.0, 0.06],
            [4, 7, 8], [None, 0.93]):
        case = (pv, conv, bucket, regime, price, cash_frac, held_frac, others, stop_frac)
        _assert_same(*_legacy_case(live, *case), case)
        n += 1
    assert n == 3 * 4 * 3 * 5 * 4 * 3 * 2 * 3 * 2


@pytest.mark.parametrize("allow,max_pct", [(False, 0.50), (True, 0.20), (True, 1.0)])
def test_one_share_rule_settings_follow_config(live, monkeypatch, allow, max_pct):
    for mod in (config, risk_manager):
        monkeypatch.setattr(mod, "ALLOW_ONE_SHARE_ABOVE_ALLOCATION", allow)
        monkeypatch.setattr(mod, "ONE_SHARE_MAX_PCT_OF_CASH", max_pct)
    for case in _grid(1500, 7):
        _assert_same(*_legacy_case(live, *case), case)


def test_halt_short_circuits(live, monkeypatch):
    monkeypatch.setattr(db, "get_state", lambda k, d=None: {"trading_halted_paper": "true"}.get(k, d))
    got_live, got_pure = _legacy_case(live, 2400.0, 8, "Compounder", "Risk-On", 47.3, 1.0, 0, 0, None)
    pure = limits.size_legacy(conviction=8, bucket="Compounder", regime="Risk-On", price=47.3,
                              pv=2400.0, cash=2400.0, halted=True)
    assert _reason(got_live) == pure["reason"] == "halted"
    assert got_live["shares_initial"] == pure["shares_initial"] == 0


def test_no_band_exit_with_a_lower_floor(live, monkeypatch):
    # The dict is shared by config.get_active_limits and limits.active_limits.
    for tier in ("standard", "reduced", "micro"):
        monkeypatch.setitem(config.TIER_MIN_CONVICTION_TO_QUEUE, tier, 1)
    for pv, conv in itertools.product([2400.0, 12_000.0, 100_000.0], [1, 3, 4, 5]):
        case = (pv, conv, "Moonshot", "Risk-On", 3.2, 1.0, 0.0, 0, None)
        got_live, got_pure = _legacy_case(live, *case)
        _assert_same(got_live, got_pure, case)
        assert (got_pure["reason"] == "no_band") == (conv < 5)


@pytest.mark.parametrize("pv", PVS + [0.0, -1.0, 1_000_000.0])
def test_active_limits_match_config(pv):
    assert limits.active_limits(pv) == config.get_active_limits(pv)
    assert limits.account_tier(pv) == config.get_account_tier(pv)
    assert limits.max_positions(pv) == config.get_active_limits(pv)["max_concurrent_positions"]
    assert limits.conviction_floor(pv) == config.get_active_limits(pv)["min_conviction_to_queue"]


def test_conviction_band_matches_live():
    for c in range(-1, 12):
        assert limits.conviction_band(c) == risk_manager._conviction_band(c)


def test_first_buy_fraction_matches_live_literal():
    src = inspect.getsource(risk_manager.RiskManager.size_position)
    assert f"initial_pct    = {limits.FIRST_BUY_FRACTION}" in src
    assert f"remaining < {limits.AT_TARGET_TOLERANCE}" in src


@pytest.mark.parametrize("hwm,flag", [(None, False), ("2600", False), ("2700", False),
                                      ("10000", False), ("0", False), (None, True)])
@pytest.mark.parametrize("pv", [2400.0, 2160.0, 2159.0, 0.0])
def test_drawdown_halt_parity(live, monkeypatch, pv, hwm, flag):
    st = {"trading_halted_paper": "true" if flag else "false", "high_water_mark_paper": hwm}
    monkeypatch.setattr(db, "get_state", lambda k, d=None: st.get(k, d) if st.get(k) is not None else d)
    monkeypatch.setattr(db, "set_state_verified", lambda *a, **k: [], raising=False)
    monkeypatch.setattr(db, "log_halt_event", lambda *a, **k: None, raising=False)
    rm = risk_manager.RiskManager(FakeClient(pv, pv, []), start_value=2400.0)
    halted = rm.is_halted()
    assert (halted, rm.halt_unknown) == limits.drawdown_halt(pv, float(hwm) if hwm else None, 2400.0, flag)


# ---------------------------------------------------------------------------
# Risk per trade and heat
# ---------------------------------------------------------------------------

def test_risk_cap_shares_formula():
    for pv, price, stop in itertools.product([900.0, 2400.0, 100_000.0], [3.2, 180.0, 455.5],
                                             [None, 0.0, 1.0, 170.0, 179.99, 200.0, 450.0]):
        want = None if not (stop and stop < price) else int(pv * config.RISK_PCT_PER_TRADE / (price - stop))
        assert limits.risk_cap_shares(pv, price, stop) == want


def test_portfolio_heat_matches_db(monkeypatch):
    rng = random.Random(3)
    for _ in range(300):
        syms = [f"S{i}" for i in range(rng.randint(0, 6))]
        positions = {s: {"qty": rng.choice([-3.0, 0.5, 1.0, 7.0])} for s in syms if rng.random() < 0.8}
        stops = [{"ticker": s, "entry_price": rng.choice([None, 20.0, 455.5]),
                  "stop_price": rng.choice([None, 10.0, 19.0, 470.0]),
                  "setup_category": rng.choice([None, "Breakout"]),
                  "thesis_bucket": rng.choice([None, "Moonshot"])} for s in syms]
        equity = rng.choice([0.0, 2400.0, 100_000.0])
        monkeypatch.setattr(db, "get_all_position_stops", lambda st=stops: st)
        live = db.get_portfolio_heat_pct(equity, positions)
        pure = limits.portfolio_heat_pct(equity, positions, stops)
        for k in ("heat_pct", "heat_dollars", "by_category"):
            assert live[k] == pure[k]


def test_heat_shadow_parity(live, monkeypatch):
    rng = random.Random(11)
    for _ in range(400):
        pv = rng.choice([900.0, 2400.0, 100_000.0])
        price = rng.choice([3.2, 180.0, 455.5])
        stop = rng.choice([None, price * 0.5, price * 0.93, price * 1.1])
        shares = rng.randint(0, 40)
        mult = rng.choice(list(config.REGIME_MULTIPLIER.values()))
        heat = rng.choice([0.0, 0.01, 0.0349, 0.07, 0.2])
        monkeypatch.setattr(db, "get_portfolio_heat_pct", lambda e, p, h=heat: {"heat_pct": h})
        rm = live(pv, pv, [])
        assert rm._heat_shadow("TKR", shares, price, stop, pv, mult) == \
            limits.heat_shadow(shares, price, stop, pv, mult, heat)


# ---------------------------------------------------------------------------
# Setup-gate sizing (the only path that enforces the 7% heat cap; off in live)
# ---------------------------------------------------------------------------

def test_setup_gate_parity_grid(live, monkeypatch):
    import agents.setup_gates as setup_gates
    monkeypatch.setattr(risk_manager, "SETUP_GATES_ENABLED", True)
    rng = random.Random(5)
    seen = set()
    for _ in range(3000):
        pv = rng.choice(PVS)
        bucket = rng.choice(["Breakout", "Catalyst", "Continuation", "Compounder"])
        regime = rng.choice(REGIMES)
        price = rng.choice([0.0, 3.2, 47.3, 180.0, 455.5])
        cash = pv * rng.choice(CASH_FRACS)
        held = pv * rng.choice(HELD_FRACS)
        heat = rng.choice([0.0, 0.02, 0.05, 0.069, 0.1])
        ladder = rng.random() < 0.9
        stop = rng.choice([None, 0.0, price * 0.99, price * 0.95, price * 0.7, price * 1.05])
        setup = (None if rng.random() < 0.05 else
                 {"category": bucket if rng.random() < 0.9 else "Catalyst", "stop_price": stop})
        positions = [_pos("TKR", held)] if held else []
        monkeypatch.setattr(db, "get_trade_ladder", lambda t, l=ladder: [{"rung": 1}] if l else [])
        monkeypatch.setattr(setup_gates, "classify_setup", lambda t, s=setup: (s, {}))
        monkeypatch.setattr(db, "get_portfolio_heat_pct", lambda e, p, h=heat: {"heat_pct": h})
        rm = live(pv, cash, positions)
        got_live = rm._size_position_setup_gates("TKR", 7, bucket, regime, price, pv)
        got_pure = limits.size_setup_gate(bucket=bucket, regime=regime, price=price, pv=pv,
                                          cash=cash, setup=setup if ladder else None,
                                          held_value=held, current_heat_pct=heat)
        case = (pv, bucket, regime, price, cash, held, heat, ladder, setup)
        if got_pure is None:
            assert got_live is None, case
            seen.add(None)
            continue
        seen.add(got_pure["reason"])
        for k in ("position_size_pct", "dollar_amount", "shares_initial", "cash_blocked",
                  "heat_blocked"):
            assert got_live[k] == got_pure[k], (case, k)
        if got_pure["reason"] == "sized":
            for k in ("initial_dollar", "order_pct", "stop_loss_price", "max_loss_if_stopped",
                      "regime_multiplier", "initial_entry_pct"):
                assert got_live[k] == got_pure[k], (case, k)
    assert {None, "sized", "risk_cap", "under_one_share", "heat_cap"} <= seen


# ---------------------------------------------------------------------------
# Concentration caps
# ---------------------------------------------------------------------------

WATCHLIST = {"NVDA": {"sector": "Semis", "role": "Compounder"},
             "AMD": {"sector": "Semis", "role": "Accelerator"},
             "LRCX": {"sector": "Semis", "role": "Accelerator"},
             "RGTI": {"sector": "Quantum", "role": "Moonshot"},
             "QBTS": {"sector": "Quantum", "role": "Moonshot"},
             "LUNR": {"sector": "Space", "role": "Moonshot"},
             "PLTR": {"sector": "Software", "role": "Accelerator"},
             "COST": {"sector": "Retail", "role": "Compounder"},
             "XYZ": {"role": "Accelerator"}}
UNIVERSE = list(WATCHLIST) + ["MSFT", "ZZZ"]


def test_ai_infra_set_matches_live():
    src = textwrap.dedent(inspect.getsource(risk_manager.RiskManager.check_concentration))
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ai_infra_keywords":
            assert frozenset(ast.literal_eval(node.value)) == limits.AI_INFRA_TICKERS
            return
    pytest.fail("ai_infra_keywords not found in check_concentration")


@pytest.mark.parametrize("watchlist_ok", [True, False])
def test_concentration_parity_grid(live, monkeypatch, watchlist_ok):
    import knowledge_loader

    def _wl():
        if not watchlist_ok:
            raise RuntimeError("watchlist unreadable")
        return WATCHLIST
    monkeypatch.setattr(knowledge_loader, "load_watchlist", _wl)
    rng = random.Random(17 if watchlist_ok else 18)
    caps_hit = set()
    for _ in range(1500):
        pv = rng.choice([900.0, 2400.0, 100_000.0])
        held = rng.sample(UNIVERSE, rng.randint(0, 6))
        values = {s: pv * rng.choice([0.01, 0.05, 0.12, 0.25, 0.4]) for s in held}
        ticker = rng.choice(UNIVERSE)
        bucket = rng.choice(["Compounder", "Accelerator", "Moonshot", None])
        proposed = rng.choice([0.0, 0.004, 0.03, 0.1, 0.2, 0.35])
        rm = live(pv, pv, [_pos(s, v) for s, v in values.items()])
        got_live = rm.check_concentration(ticker, proposed, bucket=bucket)
        got_pure = limits.concentration(ticker, proposed, bucket, values, pv,
                                        WATCHLIST if watchlist_ok else None)
        case = (pv, values, ticker, bucket, proposed)
        for k in ("passed", "ai_infrastructure_total", "ai_infra_size_cap_pct",
                  "theme_size_cap_pct", "moonshot_size_cap_pct"):
            assert got_live[k] == got_pure[k], (case, k)
        assert len(got_live["flags"]) == len(got_pure["breaches"]), case
        caps_hit.update(got_pure["breaches"])

        # shares_after_concentration on a sizing at this weight.
        shares = rng.choice([0, 1, 3, 17])
        order_pct = rng.choice([0, proposed, proposed * 1.5])
        sizing = {"shares_initial": shares, "position_size_pct": proposed, "order_pct": order_pct}
        conc = limits.concentration(ticker, max(proposed, order_pct), bucket, values, pv,
                                    WATCHLIST if watchlist_ok else None)
        assert rm.shares_after_concentration(ticker, sizing, bucket) == \
            limits.shares_after_concentration(shares, proposed, order_pct, conc), case
    want = {"ai_infra", "single_position", "theme", "moonshot_total"} if watchlist_ok \
        else {"ai_infra", "single_position"}
    assert caps_hit == want


@pytest.mark.xfail(strict=True, reason=(
    "No pure equivalent: when the broker position read raises, check_concentration "
    "fails closed (passed=False, every cap 0.0, errored=True). limits.concentration "
    "takes the positions as a value, so it has no read that can fail; a shadow "
    "caller with no positions snapshot must apply that block itself."))
def test_concentration_read_error_has_pure_equivalent(live):
    class Broken(FakeClient):
        def get_all_positions(self):
            raise ConnectionError("broker down")
    rm = risk_manager.RiskManager(Broken(2400.0, 2400.0, []))
    got_live = rm.check_concentration("NVDA", 0.1, bucket="Compounder")
    got_pure = limits.concentration("NVDA", 0.1, "Compounder", {}, 2400.0, WATCHLIST)
    assert got_live["passed"] == got_pure["passed"]
