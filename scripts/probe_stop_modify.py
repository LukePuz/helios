"""PAPER-ONLY probe: what IBKR and ib_async do when a resting SELL stop is
modified in place (re-sent placeOrder with the same orderId). It answers the
[paper] items in helios-research "08 Helios Engineering/
stop-modify-in-place-design.md" section 2, above all: does the old stop rest
unchanged when a modify is rejected, and what does ib_async's local Trade
say then?

NOT part of the bot. It imports only ib_async (no config, no .env, no db),
uses its own clientIds (default 77, plus 78 for the wrong-client case and 79
for a read-only observer), and refuses to run against anything but a paper
session:
  - before connecting: --port must be given and be 7497 (TWS paper) or 4002
    (Gateway paper); 7496 / 4001 (live) and anything else exit 2. A clientId
    the bot uses (1, 2, 10-12, 20-22, 30, 40) exits 2.
  - after connecting: every ib.managedAccounts() entry must start with "D"
    (IBKR paper accounts); otherwise it disconnects at once, places nothing,
    and exits 3. Accounts are printed masked.

What it does, on one cheap liquid symbol (default F):
  0. optionally buys --qty shares at market (needs regular hours; --no-buy
     places the stop with no position, far below the price so it can't
     trigger);
  1. places a GTC SELL STP far below the price (--stop-frac, default 0.5);
  2. cases, each followed by a fresh reqAllOpenOrders on the observer
     connection (independent of the probing connection's local Trade):
       price_modify      level +1% of price          expect confirmed
       qty_modify        qty -1 (when qty >= 2)      expect confirmed
       both_modify       level and qty together      expect confirmed
       bad_tick          level off the 0.01 tick     expect a reject / warning
       qty_zero          qty 0 (Error 434)           expect a reject / warning
       type_change       STP -> TRAIL                expect 105 / 329
       wrong_client      the same modify sent from clientId 78 (--try-wrong-
                         client; ib_async may send it as a NEW order: the
                         observer read shows whether a second stop appears)
       after_cancel      modify sent after the stop is cancelled (--try-
                         after-cancel, via the raw client: ib_async refuses
                         a done Trade)
  3. finally (always, also on an error or Ctrl-C): each connection cancels
     every order it placed on the symbol, the shares it bought are sold at
     market unless --keep-position, and the observer reports anything left.

The results table: case, what was sent, the local status sequence and the
error codes ib_async logged, the local level/qty, the observer's status and
level/qty, and the verdict. Levels are shown as percent of the reference
price. A TWS restart (does clientId 77 still own its stop afterwards?) is
done by hand: run with --keep-stop, restart TWS, run --resume-order-id N.

    venv/bin/python scripts/probe_stop_modify.py --port 7497 [--symbol F]
        [--qty 2] [--no-buy] [--ref-price P] [--try-wrong-client]
        [--try-after-cancel] [--keep-position]
"""
from __future__ import annotations

import argparse
import copy
import math
import sys

PAPER_PORTS = (7497, 4002)
BOT_CLIENT_IDS = frozenset({1, 2, 10, 11, 12, 20, 21, 22, 30, 40})
DONE = ("Cancelled", "ApiCancelled", "Filled", "Inactive")
ACCEPTED = ("PreSubmitted", "Submitted")
POLL_S = 5


class Refusal(SystemExit):
    """A hard refusal: exit code 2 (before connecting) or 3 (after)."""


def check_port(port) -> int:
    if port is None:
        raise Refusal(2)
    port = int(port)
    if port not in PAPER_PORTS:
        print(f"REFUSED: port {port} is not a paper port {PAPER_PORTS}; nothing connected.")
        raise Refusal(2)
    return port


def check_client_ids(*ids) -> None:
    bad = sorted(set(ids) & BOT_CLIENT_IDS)
    if bad or len(set(ids)) != len(ids):
        print(f"REFUSED: clientIds {sorted(ids)} must be distinct and not the bot's "
              f"({sorted(BOT_CLIENT_IDS)}); nothing connected.")
        raise Refusal(2)


def mask(acct: str) -> str:
    s = str(acct or "")
    return (s[:2] + "*" * max(len(s) - 2, 0)) if s else "?"


def check_accounts(accounts) -> list[str]:
    """Masked account list, or a Refusal(3) unless every account starts with D."""
    accts = [str(a) for a in (accounts or []) if str(a).strip()]
    if not accts or not all(a.startswith("D") for a in accts):
        print(f"REFUSED: managed accounts {[mask(a) for a in accts] or 'none'} are not all "
              f"paper (D...); disconnecting, nothing placed.")
        raise Refusal(3)
    return [mask(a) for a in accts]


def _pct(level, ref) -> str:
    try:
        return f"{(float(level) / float(ref) - 1) * 100:+.2f}%"
    except (TypeError, ValueError, ZeroDivisionError):
        return "?"


def _errors_since(trade, n0: int) -> list[int]:
    out = []
    for e in (getattr(trade, "log", None) or [])[n0:]:
        code = getattr(e, "errorCode", 0)
        if code:
            out.append(int(code))
    return out


def _poll(ib, trade, seconds: int = POLL_S) -> list[str]:
    seq = []
    for _ in range(seconds):
        ib.sleep(1)
        st = trade.orderStatus.status
        if not seq or seq[-1] != st:
            seq.append(st)
    return seq


def observe(observer, symbol: str) -> list[dict]:
    """The observer connection's fresh reqAllOpenOrders: this symbol's SELL
    orders as {client_id, order_id, perm_id, type, status, aux, qty}."""
    rows = []
    for t in observer.reqAllOpenOrders():
        if t.contract.symbol != symbol or t.order.action != "SELL":
            continue
        rows.append({"client_id": t.order.clientId, "order_id": t.order.orderId,
                     "perm_id": t.order.permId, "type": t.order.orderType,
                     "status": t.orderStatus.status, "aux": t.order.auxPrice,
                     "qty": float(t.order.totalQuantity)})
    return rows


def _find(rows, perm_id):
    return next((r for r in rows if perm_id and r["perm_id"] == perm_id), None)


def verdict(case: str, sent: dict, local: dict, fresh_target, fresh_all, probe_ids) -> str:
    want_aux, want_qty = sent.get("aux"), sent.get("qty")
    if fresh_target is None:
        base = "stop GONE at the broker"
    else:
        moved = ((want_aux is None or abs(fresh_target["aux"] - want_aux) < 1e-6)
                 and (want_qty is None or abs(fresh_target["qty"] - want_qty) < 1e-6))
        base = "broker shows the NEW values" if moved else "broker shows the OLD values (rests unchanged)"
    extra = [r for r in fresh_all if r["client_id"] in probe_ids
             and r["perm_id"] != (fresh_target or {}).get("perm_id")]
    if extra:
        base += f"; {len(extra)} OTHER probe stop(s) resting (a second order was created)"
    if local.get("status") in DONE and fresh_target is not None and fresh_target["status"] not in DONE:
        base += "; LOCAL says done while the broker says live"
    return base


def run_case(ib, observer, trade, case: str, ref: float, probe_ids, aux=None, qty=None,
             order_type=None, sender=None) -> dict:
    """Send one modify (from sender, default ib) and record both views."""
    sender = sender or ib
    n0 = len(trade.log or [])
    o = copy.copy(trade.order)
    if aux is not None:
        o.auxPrice = aux
    if qty is not None:
        o.totalQuantity = qty
    if order_type is not None:
        o.orderType = order_type
        if order_type == "TRAIL":
            o.auxPrice = round(ref * 0.05, 2)       # a trailing amount
    send_error = None
    try:
        sender.placeOrder(trade.contract, o)
    except Exception as e:                          # AssertionError on a done Trade, etc.
        send_error = f"{type(e).__name__}: {e}"
    seq = _poll(ib, trade)
    local = {"status": trade.orderStatus.status, "aux": trade.order.auxPrice,
             "qty": float(trade.order.totalQuantity), "errors": _errors_since(trade, n0)}
    fresh_all = observe(observer, trade.contract.symbol)
    target = _find(fresh_all, trade.order.permId)
    sent = {"aux": aux, "qty": qty, "type": order_type}
    return {"case": case, "sent": sent, "send_error": send_error, "status_seq": seq,
            "local": local, "fresh": target, "fresh_all": fresh_all,
            "verdict": verdict(case, sent, local, target, fresh_all, probe_ids)}


def table(results: list[dict], ref: float) -> str:
    lines = ["| case | sent | local status seq | local errors | local level / qty "
             "| broker status | broker level / qty | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r["sent"]
        sent = ", ".join(x for x in (
            f"level {_pct(s['aux'], ref)}" if s.get("aux") is not None else "",
            f"qty {s['qty']:g}" if s.get("qty") is not None else "",
            f"type {s['type']}" if s.get("type") else "") if x) or "-"
        if r.get("send_error"):
            sent += f" (send raised {r['send_error'][:40]})"
        loc, fr = r["local"], r["fresh"]
        lines.append(
            f"| {r['case']} | {sent} | {' > '.join(r['status_seq']) or '-'} "
            f"| {','.join(map(str, loc['errors'])) or '-'} "
            f"| {_pct(loc['aux'], ref)} / {loc['qty']:g} "
            f"| {fr['status'] if fr else 'not found'} "
            f"| {(_pct(fr['aux'], ref) + ' / ' + format(fr['qty'], 'g')) if fr else '-'} "
            f"| {r['verdict']} |")
    return "\n".join(lines)


def _ref_price(ib, contract, given):
    if given:
        return float(given)
    ib.reqMarketDataType(3)                         # delayed is fine for a far stop
    t = ib.reqTickers(contract)[0]
    p = t.marketPrice()
    if p is None or (isinstance(p, float) and math.isnan(p)) or p <= 0:
        p = t.close
    if p is None or (isinstance(p, float) and math.isnan(p)) or p <= 0:
        print("REFUSED: no reference price; pass --ref-price.")
        raise Refusal(2)
    return float(p)


def _cleanup(conns: list, symbol: str, bought: float, ib, keep_position: bool, contract) -> None:
    for c in conns:
        if c is None:
            continue
        try:
            for t in c.openTrades():
                if (t.contract.symbol == symbol and t.order.clientId == c.client.clientId
                        and t.orderStatus.status not in DONE):
                    c.cancelOrder(t.order)
            c.sleep(3)
        except Exception as e:
            print(f"cleanup: cancel on clientId {getattr(c.client, 'clientId', '?')} failed: {e}")
    if bought > 0 and not keep_position and ib is not None:
        try:
            import ib_async as ibi
            ib.placeOrder(contract, ibi.MarketOrder("SELL", bought))
            ib.sleep(3)
            print(f"cleanup: sold the {bought:g} shares this probe bought")
        except Exception as e:
            print(f"cleanup: selling the probe's shares failed: {e}; close them in TWS")


def main(argv=None, ib_factory=None) -> int:
    ap = argparse.ArgumentParser(description="PAPER-ONLY stop modify-in-place probe")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--client-id", type=int, default=77)
    ap.add_argument("--symbol", default="F")
    ap.add_argument("--qty", type=float, default=2)
    ap.add_argument("--stop-frac", type=float, default=0.5)
    ap.add_argument("--ref-price", type=float)
    ap.add_argument("--no-buy", action="store_true")
    ap.add_argument("--try-wrong-client", action="store_true")
    ap.add_argument("--try-after-cancel", action="store_true")
    ap.add_argument("--keep-position", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return 2
    try:
        check_port(a.port)
        cid, wrong_cid, obs_cid = a.client_id, a.client_id + 1, a.client_id + 2
        check_client_ids(cid, wrong_cid, obs_cid)
        if not 0 < a.stop_frac < 0.9:
            print("REFUSED: --stop-frac must be between 0 and 0.9")
            raise Refusal(2)
    except Refusal as r:
        return int(r.code)

    if ib_factory is None:
        import ib_async as ibi
        ib_factory = ibi.IB
    ib = observer = wrong = None
    bought, contract, results, ref = 0.0, None, [], None
    try:
        ib = ib_factory()
        ib.connect(a.host, a.port, clientId=cid, timeout=10)
        try:
            accts = check_accounts(ib.managedAccounts())
        except Refusal as r:
            ib.disconnect()
            ib = None
            return int(r.code)
        print(f"paper session ok, accounts {accts}")
        observer = ib_factory()
        observer.connect(a.host, a.port, clientId=obs_cid, timeout=10, readonly=True)
        check_accounts(observer.managedAccounts())

        import ib_async as ibi
        contract = ib.qualifyContracts(ibi.Stock(a.symbol, "SMART", "USD"))[0]
        ref = _ref_price(ib, contract, a.ref_price)
        if not a.no_buy:
            buy = ib.placeOrder(contract, ibi.MarketOrder("BUY", a.qty))
            _poll(ib, buy, 10)
            bought = float(buy.orderStatus.filled or 0)
            print(f"bought {bought:g} of {a.qty:g}")
        stop_px = round(ref * a.stop_frac, 2)
        trade = ib.placeOrder(contract, ibi.StopOrder("SELL", a.qty, stop_px, tif="GTC"))
        _poll(ib, trade)
        if trade.orderStatus.status not in ACCEPTED:
            print(f"the probe stop was not accepted ({trade.orderStatus.status}); stopping")
            return 1
        probe_ids = {cid, wrong_cid}
        lvl = stop_px
        lvl = round(lvl + ref * 0.01, 2)
        results.append(run_case(ib, observer, trade, "price_modify", ref, probe_ids, aux=lvl))
        q = float(trade.order.totalQuantity)
        if q >= 2:
            q -= 1
            results.append(run_case(ib, observer, trade, "qty_modify", ref, probe_ids, qty=q))
        lvl = round(lvl + ref * 0.01, 2)
        results.append(run_case(ib, observer, trade, "both_modify", ref, probe_ids, aux=lvl,
                                qty=q if q >= 1 else None))
        results.append(run_case(ib, observer, trade, "bad_tick", ref, probe_ids,
                                aux=round(lvl + 0.0037, 4)))
        results.append(run_case(ib, observer, trade, "qty_zero", ref, probe_ids, qty=0))
        results.append(run_case(ib, observer, trade, "type_change", ref, probe_ids,
                                order_type="TRAIL"))
        if a.try_wrong_client:
            wrong = ib_factory()
            wrong.connect(a.host, a.port, clientId=wrong_cid, timeout=10)
            check_accounts(wrong.managedAccounts())
            theirs = next((t for t in wrong.reqAllOpenOrders()
                           if t.order.permId == trade.order.permId), None)
            if theirs is not None:
                results.append(run_case(ib, observer, trade, "wrong_client", ref, probe_ids,
                                        aux=round(lvl + ref * 0.01, 2), sender=wrong))
        if a.try_after_cancel:
            ib.cancelOrder(trade.order)
            _poll(ib, trade)
            o = copy.copy(trade.order)
            o.auxPrice = round(lvl + ref * 0.02, 2)
            n0 = len(trade.log or [])
            ib.client.placeOrder(o.orderId, contract, o)      # raw: ib_async refuses a done Trade
            seq = _poll(ib, trade)
            fresh_all = observe(observer, a.symbol)
            local = {"status": trade.orderStatus.status, "aux": trade.order.auxPrice,
                     "qty": float(trade.order.totalQuantity), "errors": _errors_since(trade, n0)}
            target = _find(fresh_all, trade.order.permId)
            sent = {"aux": o.auxPrice, "qty": None, "type": None}
            results.append({"case": "after_cancel", "sent": sent, "send_error": None,
                            "status_seq": seq, "local": local, "fresh": target,
                            "fresh_all": fresh_all,
                            "verdict": verdict("after_cancel", sent, local, target, fresh_all,
                                               probe_ids)})
        return 0
    except Refusal as r:
        return int(r.code)
    finally:
        if ib is not None:
            _cleanup([ib, wrong], a.symbol, bought, ib, a.keep_position, contract)
            if observer is not None:
                try:
                    left = [r for r in observe(observer, a.symbol)
                            if r["client_id"] in (cid, wrong_cid) and r["status"] not in DONE]
                    print(f"after cleanup: {len(left)} probe order(s) still resting"
                          + (" — CANCEL THEM IN TWS" if left else ""))
                except Exception as e:
                    print(f"after cleanup: observer read failed ({e}); check TWS")
        if results and ref:
            print(table(results, ref))
        for c in (wrong, observer, ib):
            try:
                if c is not None:
                    c.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
