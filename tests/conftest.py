import pytest


@pytest.fixture(autouse=True)
def _default_to_orders_worker_context(monkeypatch):
    """Most tests exercise stop-order logic (placement, cancellation,
    reconciliation) without caring about WHICH process is allowed to touch
    the broker. tasks.manage_stop_place and its siblings (added 2026-08-11,
    see their module-level docstring in tasks.py) route through Celery
    dispatch-and-wait unless IB_CLIENT_ID matches the dedicated orders
    worker -- a bare test environment has no IB_CLIENT_ID set at all, which
    would otherwise send every test down the dispatch branch and try to
    actually talk to Redis (hangs on .get(timeout=...) since nothing is
    consuming the queue). Default every test to "already the orders worker"
    -- the direct-call branch, which is what every existing ibkr.* mock
    already targets. Tests that specifically exercise the routing logic
    itself (tests/test_order_routing.py) override this locally per-test."""
    monkeypatch.setenv("IB_CLIENT_ID", "30")
