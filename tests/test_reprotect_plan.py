"""tasks._reprotect_plan, the re-protect's decision (no I/O), over every
combination of its inputs, with the invariants asserted on each row:

I1 never more stop shares than held: a place (with what stays resting) is
   at most min(cache, fresh) less a working SELL (and the read after a
   fill, or after a cancel status that may mean one); with the fresh read
   unread, only with nothing resting, at most the cache.
I2 never a SELL stop on a short or flat: no place or cancel when a fresh
   read is <= 0 (the one after a fill, or a maybe-fill, too).
I3 never a cancel when the fresh read is unusable and stops may rest
   (covered > 0 or unread).
I4 never False on a fully covered position: cover not being cancelled
   (covered - cancelling) >= the cache count (not a profit-target
   re-place, not short by the fresh read) is "done".
I5 a re-check whenever shares may be over-stopped or bare: every "wait"
   except the sign flip; a "done" with more cover than the fresh count (the
   cache's count when the fresh read is unread);
   after a cancel, any raise, a stop of ours still resting or filled, the
   stops left unread, or more left than held.
I6 a re-check whenever shares may be under-covered: every "done", or
   "place" below the basis, that credits stops still being cancelled
   (before or after the cancel) or a stop whose cancel status may mean
   filled; every place sized from the cache with the fresh read unread;
   every "done" whose basis (min(cache, fresh) less a working SELL, and
   the read after a fill) is below the cache count (one read lags).

A "done" says why (by): "stops" (cover not being cancelled reaches the
cache's count, after the cancel the count held), "sell" (a working SELL
covers the count held), "lag" (stops cover only a fresh count below the
cache's: qty is that count).

REPROTECT_TABLE_OUT=<path> also writes the truth table there (the
handback's table); otherwise nothing is written."""
import itertools
import os

import pytest

import tasks

REM = 20.0
EPS = 1e-6
FRESH = [None, -4.0, 0.0, 3.0, 12.0, 20.0, 25.0]      # unread, short, flat, under a SELL of 5,
                                                       # below, equal, above
COVERED = [0.0, 8.0, 12.0, 20.0, 25.0, None]           # none, partial, = a fresh 12, full, over, unread
CANCELLING = [0.0, 8.0]                                # of covered: stops still being cancelled
SELLING = [0.0, 5.0]
FILLED = [(), (8,)]
# (still, ambiguous): ambiguous is a subset of still (a status that may mean filled)
STILL_AMBIG = [((), ()), ((9,), ()), ((9,), (9,))]
EXC = [None, RuntimeError("socket")]
FRESH_AFTER = [None, 0.0, 3.0, 10.0, 30.0]
LEFT = [None, [], [{"qty": 5.0, "status": "Submitted"}], [{"qty": 30.0, "status": "Submitted"}],
        [{"qty": 5.0, "status": "PendingCancel"}]]
CANCELLING_STATUSES = ("PendingCancel", "ApiPendingCancel")


def _basis(fresh, selling):
    if fresh is None:
        return None
    return round(min(REM, fresh - selling), 4) if fresh > 0 else 0.0


def _pre_rows():
    for path, pt, covered, cancelling, fresh, selling in itertools.product(
            ("queued", "late"), (False, True), COVERED, CANCELLING, FRESH, SELLING):
        if cancelling and (covered is None or covered < cancelling):
            continue      # cancelling is part of a read cover
        yield (dict(path=path, pt=pt, covered=covered, cancelling=cancelling, fresh=fresh,
                    selling=selling),
               tasks._reprotect_plan(REM, covered, fresh, selling, path=path, pt=pt,
                                     cancelling=cancelling))


def _post_rows():
    for (kw, pre) in _pre_rows():
        if pre.action != "cancel":
            continue
        for filled, (still, amb), exc, fa, left in itertools.product(FILLED, STILL_AMBIG, EXC,
                                                                     FRESH_AFTER, LEFT):
            if exc is not None and (fa is not None or left is not None):
                continue      # a raised cancel reads neither
            if not (filled or amb) and fa is not None:
                continue      # the read after the cancel runs only after a (maybe-)fill
            post = dict(filled=filled, still=still, ambiguous=amb, cancel_exc=exc,
                        fresh_after=fa, left=left)
            step = tasks._reprotect_plan(REM, kw["covered"], kw["fresh"], kw["selling"],
                                         path=kw["path"], pt=kw["pt"],
                                         cancelling=kw["cancelling"], cancelled=True, **post)
            yield kw, pre, post, step


def test_the_table_covers_every_combination():
    pre = list(_pre_rows())
    n_cov = sum(1 + sum(1 for c in CANCELLING if c and v is not None and v >= c)
                for v in COVERED)
    assert len(pre) == 2 * 2 * n_cov * len(FRESH) * len(SELLING)
    assert {s.action for _, s in pre} == {"done", "wait", "cancel", "place"}
    post = list(_post_rows())
    assert post and {s.action for *_, s in post} == {"done", "wait", "place"}


@pytest.mark.parametrize("kw, step", list(_pre_rows()),
                         ids=lambda v: (f"{v['path']}-pt{int(v['pt'])}-cov{v['covered']}-canc"
                                        f"{v['cancelling']}-fresh{v['fresh']}-sell{v['selling']}")
                         if isinstance(v, dict) else v.action)
def test_before_the_cancel(kw, step):
    covered, fresh, selling, pt = kw["covered"], kw["fresh"], kw["selling"], kw["pt"]
    cancelling = kw["cancelling"]
    basis = _basis(fresh, selling)
    usable = fresh is not None and fresh > 0
    may_rest = covered is None or covered > EPS
    assert step.action in ("done", "wait", "cancel", "place")
    # I1
    if step.action == "place":
        assert covered is not None and covered <= EPS          # nothing resting to net
        assert step.qty <= REM + EPS
        if fresh is not None:
            assert step.qty <= basis + EPS
        assert step.qty > 0
    if step.action == "cancel":
        assert usable and 0 < step.qty <= basis + EPS
    # I2
    if fresh is not None and fresh <= EPS:
        assert step.action not in ("place", "cancel")
    # I3
    if not usable and may_rest:
        assert step.action != "cancel"
    # I4
    if not pt and covered is not None and covered - cancelling >= REM - EPS \
            and not (fresh is not None and fresh < 0):
        assert step.action == "done"
    # I5
    if step.action == "wait" and step.alert != "sign_flip":
        assert step.recheck
    if step.action == "done" and covered is not None \
            and covered > (REM if basis is None else max(basis, 0.0)) + EPS:
        assert step.recheck
    # why a done is done
    assert (step.by is not None) == (step.action == "done")
    if step.action == "done":
        if basis is not None and basis <= EPS and step.by != "stops":
            assert step.by == "sell"       # stops reaching the cache's count are "stops" first
        if step.by == "sell":
            assert basis is not None and basis <= EPS and step.recheck
        if step.by == "stops":
            assert covered - cancelling >= REM - EPS
        if step.by == "lag":
            assert step.recheck and EPS < basis < REM - EPS and step.qty == basis
            assert covered - cancelling >= basis - EPS
    # I6
    if step.action == "done" and cancelling > EPS:
        assert step.recheck
    if step.action == "done" and basis is not None and basis < REM - EPS:
        assert step.recheck
    if step.action == "place" and fresh is None:
        assert step.recheck
    # a done never counts stops being cancelled as cover
    if step.action == "done" and covered is not None and basis is not None and basis > EPS:
        assert covered - cancelling >= basis - EPS
    # the sign flip: a short by the fresh read waits, with the alert
    assert (step.alert == "sign_flip") == (fresh is not None and fresh < 0)
    if step.alert == "sign_flip":
        assert step.action == "wait"


def test_after_the_cancel():
    n = 0
    for kw, pre, post, step in _post_rows():
        n += 1
        held = pre.qty
        filled, amb, exc = post["filled"], post["ambiguous"], post["cancel_exc"]
        fa, left = post["fresh_after"], post["left"]
        if (filled or amb) and fa is not None and fa > EPS:
            held = min(held, fa - kw["selling"])
        left_cancelling = bool(left) and any(o["status"] in CANCELLING_STATUSES for o in left)
        others = 0.0 if left is None else sum(o["qty"] for o in left
                                              if o["status"] not in CANCELLING_STATUSES)
        ctx = (kw, post, step)
        assert step.action in ("done", "wait", "place"), ctx
        assert (step.alert == "filled") == bool(filled), ctx          # not on a maybe-fill
        if exc is not None:                                          # a raised cancel
            assert step.action == "wait" and step.recheck, ctx
            continue
        if (filled or amb) and (fa is None or fa <= EPS):            # I2 after a (maybe-)fill
            assert step.action == "wait" and step.recheck, ctx
            continue
        if step.action == "place":                                   # I1
            assert step.qty > 0 and step.qty + others <= held + EPS, ctx
        if step.action == "done":
            assert others >= held - EPS, ctx
        if post["still"] or filled or left is None or others > max(held, 0.0) + EPS:   # I5
            assert step.recheck, ctx
        if step.action in ("done", "place") and (amb or left_cancelling
                                                 or kw["cancelling"] > EPS):        # I6
            assert step.recheck, ctx
        if step.action == "done" and held < REM - EPS:                             # I6
            assert step.recheck, ctx
        if step.action == "done":
            want = "sell" if held <= EPS else ("lag" if held < REM - EPS else "stops")
            assert step.by == want, ctx
            if want == "lag":
                assert step.qty == held, ctx
        else:
            assert step.by is None, ctx
    assert n > 0


# ── the truth table (written only when REPROTECT_TABLE_OUT is set) ──────────

def _fmt(v):
    if v is None:
        return "unread"
    return f"{v:g}"


def _out(step):
    s = step.action + (f" {step.qty:g}" if step.action in ("place", "cancel") else "")
    if step.by:
        s += f" ({step.by}" + (f" {step.qty:g}" if step.by == "lag" else "") + ")"
    if step.recheck:
        s += " +re-check"
    if step.alert:
        s += f" +{step.alert}"
    return s


def _left_fmt(left):
    if left is None:
        return "unread"
    if not left:
        return "none"
    return ", ".join(f"{o['qty']:g}" + (" cancelling" if o["status"] in CANCELLING_STATUSES
                                         else "") for o in left)


def test_write_the_truth_table():
    path = os.environ.get("REPROTECT_TABLE_OUT")
    lines = [f"Before the cancel (cache {REM:g}; '+re-check' = queue the coverage re-check; "
             f"cancelling = shares of covered still being cancelled)",
             "pt | selling | covered | cancelling | fresh | queued | late"]
    rows = {}
    for kw, step in _pre_rows():
        rows.setdefault((kw["pt"], kw["selling"], _fmt(kw["covered"]), _fmt(kw["cancelling"]),
                         _fmt(kw["fresh"])), {})[kw["path"]] = _out(step)
    for (pt, selling, cov, canc, fresh), by in rows.items():
        lines.append(f"{'PT' if pt else '-'} | {selling:g} | {cov} | {canc} | {fresh} | "
                     f"{by['queued']} | {by['late']}")
    lines += ["", "After the cancel (planned before it with cache 20, fresh 20, covered 8, none "
                  "cancelling: cancel, sized 20 less the working SELL; 'maybe filled' = a "
                  "still-resting status of NotFound, SentUnknown or none)",
              "selling | raised | filled | maybe filled | fresh after | still resting | left | "
              "outcome"]
    seen = set()
    for kw, pre, post, step in _post_rows():
        if not (kw["fresh"] == 20.0 and kw["covered"] == 8.0
                and kw["cancelling"] == 0.0 and kw["path"] == "queued" and not kw["pt"]):
            continue
        key = (f"{kw['selling']:g}", "yes" if post["cancel_exc"] else "-", "yes" if post["filled"] else "-",
               "yes" if post["ambiguous"] else "-",
               "-" if not (post["filled"] or post["ambiguous"]) else _fmt(post["fresh_after"]),
               "yes" if post["still"] else "-", _left_fmt(post["left"]))
        if key in seen:
            continue
        seen.add(key)
        lines.append(" | ".join(key) + f" | {_out(step)}")
    assert len(lines) > 20
    if path:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
