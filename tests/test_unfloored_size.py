"""Record-only size before the whole-share floor (roadmap 04bb, option 5 of
the fractional-shares fact-find): size_position adds unfloored_shares
(pre-floor dollars / price, 4 dp), floored_shares (the count the floor step
gave) and floor_rule ("one_share_above_allocation" / "int_floor" / "none").
The size it returns is unchanged."""
import inspect
from unittest.mock import MagicMock, patch

import pytest

import config
import risk_manager

FIELDS = ("unfloored_shares", "floored_shares", "floor_rule")


def _state(k, d=None):
    return {"trading_halted_paper": "false", "high_water_mark_paper": "100000"}.get(k, d)


def _size(price, one_share=True):
    """Compounder, conviction 8, $100k portfolio and cash, nothing held, a
    tight stop (the per-trade risk cap doesn't bind)."""
    client = MagicMock()
    client.get_account.return_value = MagicMock(portfolio_value=100_000.0, cash=100_000.0)
    client.get_all_positions.return_value = []
    with patch("risk_manager.TRADING_ENV", "paper"), \
         patch("db.get_state", side_effect=_state), patch("db.set_state"), \
         patch.object(config, "VOL_TARGET_SIZING_SHADOW_ENABLED", False), \
         patch.object(risk_manager, "ALLOW_ONE_SHARE_ABOVE_ALLOCATION", one_share), \
         patch.object(risk_manager, "ONE_SHARE_MAX_PCT_OF_CASH", 1.0), \
         patch.object(risk_manager.RiskManager, "calculate_stop",
                      side_effect=lambda t, p, b, **k: (round(p * 0.995, 2), "mocked")):
        rm = risk_manager.RiskManager(client)
        rm.quiet_fallback_alert = True
        return rm.size_position("NVDA", conviction=8, bucket="Compounder", regime="Risk-On",
                                price=price)


def _pre_floor_dollars():
    s = _size(10.0)                                    # many shares: a plain int floor
    assert s["shares_initial"] > 10
    return s["initial_dollar"]                         # unchanged by the floor on this path


def _price_for(unfloored):
    return round(_pre_floor_dollars() / unfloored, 6)


@pytest.mark.parametrize("unfloored, floored, rule", [
    (0.4, 1, "one_share_above_allocation"),
    (1.7, 1, "int_floor"),
    (3.2, 3, "int_floor"),
])
def test_the_fields_for_a_fractional_allocation(unfloored, floored, rule):
    s = _size(_price_for(unfloored))
    if s["shares_initial"] == 0:
        pytest.fail(f"sized 0: {s.get('flags')}")
    assert s["unfloored_shares"] == pytest.approx(unfloored, abs=1e-4)
    assert s["floored_shares"] == floored and s["floor_rule"] == rule
    assert s["shares_initial"] == floored               # what was used (risk cap not binding)


def test_a_whole_share_allocation_is_rule_none():
    price = _price_for(2.0)
    s = _size(price)
    assert s["unfloored_shares"] == pytest.approx(2.0, abs=1e-4)
    assert s["floored_shares"] == 2 and s["floor_rule"] == "none"


def test_under_one_share_without_the_one_share_rule_skips_and_records():
    s = _size(_price_for(0.4), one_share=False)
    assert s["shares_initial"] == 0
    assert s["unfloored_shares"] == pytest.approx(0.4, abs=1e-4)
    assert s["floored_shares"] == 0 and s["floor_rule"] == "int_floor"


@pytest.mark.parametrize("unfloored", [0.4, 1.7, 3.2])
def test_the_returned_size_is_unchanged(unfloored):
    """Everything but the three record fields is what it was before them."""
    price = _price_for(unfloored)
    s = _size(price)
    rest = {k: v for k, v in s.items() if k not in FIELDS}
    src = inspect.getsource(risk_manager.RiskManager.size_position)
    stripped = src.replace("**floor_record,\n", "").replace(", **floor_record", "")
    ns = {}
    exec_globals = dict(vars(risk_manager))
    code = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in stripped.splitlines())
    exec(code, exec_globals, ns)
    with patch.object(risk_manager.RiskManager, "size_position", ns["size_position"]):
        old = _size(price)
    assert not any(k in old for k in FIELDS)
    assert rest == old


def test_the_record_reads_nothing_new():
    src = inspect.getsource(risk_manager.RiskManager.size_position)
    a = src.index("raw_shares = initial_dollar / price")
    b = src.index('floor_record = {"unfloored_shares"')
    block = src[a:b]
    assert "self." not in block and "db." not in block and "ibkr" not in block
