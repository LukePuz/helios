"""exempt_by_bucket log: which automatic exit routes a held bucket has (audit A1031-1453 N3)."""
from unittest.mock import patch

import risk_manager
import tasks


def test_compounders_have_no_target_stale_or_swap_route():
    routes = risk_manager.exit_routes_for_bucket("Compounder")
    assert routes["profit_target"] is False and routes["stale_flag"] is False
    assert routes["swap_eligible"] is False and routes["trailing_stop"] is True
    acc = risk_manager.exit_routes_for_bucket("Accelerator")
    assert acc["profit_target"] and acc["stale_flag"] and acc["swap_eligible"]


def test_review_run_logs_one_row_and_never_raises():
    stops = [{"ticker": "AAA", "thesis_bucket": "Compounder"},
             {"ticker": "BBB", "thesis_bucket": "Accelerator"}]
    with patch("db.get_all_position_stops", return_value=stops), patch("db.log_activity") as act:
        tasks._log_exit_routes()
    stage, _, msg, _, extra = act.call_args.args
    assert stage == "exit_routes" and extra["no_auto_exit_route"] == ["AAA"]
    assert extra["held"]["AAA"]["exempt"] == ["profit_target", "stale_flag", "swap_eligible"]
    assert msg.startswith("1 of 2")
    with patch("db.get_all_position_stops", side_effect=RuntimeError("db down")):
        tasks._log_exit_routes()
