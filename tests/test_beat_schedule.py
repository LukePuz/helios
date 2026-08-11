"""
Regression coverage for the 2026-08-11 finding: reconcile-unconfirmed-orders'
expires (280s) was tighter than its own 300s schedule -- unlike every other
periodic safety task, which all give themselves the full interval before
expiring (reconcile-stops 900/900, check-stops 1800/1800, check-drawdown
1800/1800). Celery's expires DROPS a task outright rather than collapsing
duplicates, so that narrower-than-interval gap meant a real fraction of
scheduled reconciliation runs (~24% on a sampled trading day) were silently
discarded whenever the shared ops queue fell even slightly behind -- not a
rare pileup case, a routine one. A NET order sat unreconciled for over a day
because every scheduled check for it that day either got dropped this way or
hit a real IBKR connectivity gap.

This doesn't re-verify every task's tuning, just guards the specific,
already-identified failure mode: a periodic safety task whose expiry is
tighter than its own schedule interval, which guarantees runs get dropped
during any queue backup lasting less than one full cycle -- not just
extreme ones.
"""
import tasks

# Tasks explicitly reasoned about here — see this module's docstring above
# for why each one matters (late-fill / stop / drawdown safety nets).
_SAFETY_CRITICAL_SCHEDULE_KEYS = [
    "reconcile-unconfirmed-orders",
    "reconcile-stops",
    "check-stops",
    "check-drawdown",
]


class TestSafetyCriticalTasksDoNotExpireBeforeTheirOwnInterval:
    def test_expires_is_at_least_the_schedule_interval(self):
        schedule = tasks.app.conf.beat_schedule
        failures = []
        for key in _SAFETY_CRITICAL_SCHEDULE_KEYS:
            entry = schedule[key]
            interval = entry["schedule"]
            expires = entry["options"]["expires"]
            if expires < interval:
                failures.append(f"{key}: expires={expires} < schedule={interval}")
        assert not failures, (
            "These periodic safety tasks expire before their own schedule "
            "interval elapses, guaranteeing dropped runs during any ops-queue "
            "backup shorter than a full cycle: " + "; ".join(failures)
        )
