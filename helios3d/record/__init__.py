"""Decision records and labels. Phase 1 adds one decision record per candidate
(inputs snapshot, every analyst's output, the PM verdict, the portfolio action
and risk checks, version hashes), written by a recorder that tails the live
signals table instead of hooking the live code. Wave 0 holds the
forward-return labels for existing verdicts (labels.py).
"""
