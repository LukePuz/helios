"""scripts/probe_stop_modify.py refuses anything but a paper session: a
non-paper port or a bot clientId before connecting, a non-D managed account
right after connecting (disconnect, nothing placed). Nothing connects here:
a fake IB stands in."""
import importlib.util
import os
from types import SimpleNamespace

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "probe_stop_modify.py")
_spec = importlib.util.spec_from_file_location("probe_stop_modify", _PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class _FakeIB:
    made = []

    def __init__(self, accounts=("DU1234567",)):
        self.accounts, self.calls = list(accounts), []
        self.client = SimpleNamespace(clientId=None)
        _FakeIB.made.append(self)

    def connect(self, host, port, clientId, timeout=10, readonly=False):
        self.calls.append(("connect", port, clientId))
        self.client.clientId = clientId

    def managedAccounts(self):
        self.calls.append("managedAccounts")
        return self.accounts

    def disconnect(self):
        self.calls.append("disconnect")

    def __getattr__(self, name):            # any order call is a test failure
        raise AssertionError(f"probe called {name} on a refused session")


@pytest.fixture(autouse=True)
def _reset():
    _FakeIB.made = []


def _factory(accounts=("DU1234567",)):
    return lambda: _FakeIB(accounts)


@pytest.mark.parametrize("port", ["7496", "4001", "8888"])
def test_live_or_unknown_port_refused_before_connecting(port, capsys):
    assert probe.main(["--port", port], ib_factory=_factory()) == 2
    assert _FakeIB.made == []
    assert "REFUSED" in capsys.readouterr().out


def test_port_is_required():
    assert probe.main([], ib_factory=_factory()) == 2
    assert _FakeIB.made == []


@pytest.mark.parametrize("cid", ["30", "40", "1", "10", "29"])   # 29: its +1 / +2 hit 30
def test_bot_client_ids_refused_before_connecting(cid):
    code = probe.main(["--port", "7497", "--client-id", cid], ib_factory=_factory())
    assert code == 2 and _FakeIB.made == []


def test_bad_stop_frac_refused():
    assert probe.main(["--port", "7497", "--stop-frac", "0.95"], ib_factory=_factory()) == 2
    assert _FakeIB.made == []


@pytest.mark.parametrize("accounts", [("U1234567",), ("DU1234567", "U7654321"), (), ("",)])
def test_non_paper_account_disconnects_and_places_nothing(accounts, capsys):
    assert probe.main(["--port", "7497"], ib_factory=_factory(accounts)) == 3
    assert len(_FakeIB.made) == 1
    ib = _FakeIB.made[0]
    assert ib.calls == [("connect", 7497, 77), "managedAccounts", "disconnect"]
    out = capsys.readouterr().out
    assert "REFUSED" in out
    for a in accounts:
        if len(a) > 2:
            assert a not in out and a[:2] in out          # masked


def test_mask_and_check_accounts():
    assert probe.mask("DU1234567") == "DU*******"
    assert probe.mask("") == "?"
    assert probe.check_accounts(["DU1", "DF22"]) == ["DU*", "DF**"]
    with pytest.raises(probe.Refusal) as e:
        probe.check_accounts(["U1"])
    assert e.value.code == 3


def test_paper_ports_accepted_and_client_ids_checked():
    assert probe.check_port(7497) == 7497 and probe.check_port("4002") == 4002
    probe.check_client_ids(77, 78, 79)
    for ids in ((77, 77, 79), (28, 29, 30)):
        with pytest.raises(probe.Refusal):
            probe.check_client_ids(*ids)


def test_verdicts():
    sent = {"aux": 5.0, "qty": None}
    live = {"perm_id": 9, "client_id": 77, "aux": 5.0, "qty": 2.0, "status": "Submitted"}
    assert probe.verdict("c", sent, {"status": "Submitted"}, live, [live], {77, 78}) == \
        "broker shows the NEW values"
    old = dict(live, aux=4.0)
    v = probe.verdict("c", sent, {"status": "Cancelled"}, old, [old], {77, 78})
    assert "OLD values" in v and "LOCAL says done while the broker says live" in v
    second = dict(live, perm_id=10, client_id=78)
    assert "1 OTHER probe stop(s)" in probe.verdict("c", sent, {"status": "Submitted"}, live,
                                                     [live, second], {77, 78})
    assert probe.verdict("c", sent, {"status": "Cancelled"}, None, [], {77}).startswith(
        "stop GONE")


def test_table_shows_levels_as_percent_only():
    r = {"case": "price_modify", "sent": {"aux": 6.06, "qty": None, "type": None},
         "send_error": None, "status_seq": ["Submitted"],
         "local": {"status": "Submitted", "aux": 6.06, "qty": 2.0, "errors": []},
         "fresh": {"status": "Submitted", "aux": 6.06, "qty": 2.0}, "verdict": "v"}
    text = probe.table([r], 12.0)
    assert "-49.50%" in text and "6.06" not in text
