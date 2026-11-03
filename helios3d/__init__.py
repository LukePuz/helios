"""Helios 3D: the shadow research architecture built beside the live pipeline.

Helios 3D turns the one-line pipeline (screen, one LLM conviction score,
rules, order) into three dimensions: many perspectives (an analyst panel),
the portfolio (a risk-budget view of the whole book), and time and feedback
(labels, post-mortems, calibration). Everything here is shadow-only.

Isolation rules, enforced by tests/test_helios3d_isolation.py:
- No live module (main.py, tasks.py, agents/, execution.py, risk_manager.py,
  ibkr.py, db.py, alerts.py) ever imports this package.
- This package never imports the broker layer (ibkr.py, execution.py).
- Its tables live in the separate Postgres schema ``h3d``, created by its own
  migration runner (helios3d.schema.migrate). It reads public-schema tables
  only, and never writes to them.
"""
