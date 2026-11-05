from unittest.mock import patch

import pandas as pd
import pytest

from agents import analyst


@pytest.mark.parametrize("action, changed, pt, want", [
    ("main", False, "Raises", True),     # the ANET 11-04 shape: rating kept, target raised
    ("reit", False, "Lowers", True),
    ("up", True, "Raises", False),       # a real rating change
    ("main", False, None, False),        # maintained, no target change
    ("init", False, "Announces", False),
    ("reit", False, "Maintains", False),   # target unchanged (m544)
    ("main", False, "Announces", False),
])
def test_pt_only(action, changed, pt, want):
    rec = analyst.trigger_record("event", {"source": "analyst_ratings", "title": "x",
                                           "rating": {"action": action, "grade_changed": changed,
                                                      "pt_action": pt}})
    assert rec["rating"]["pt_only"] is want and rec["rating"]["action"] == action


def test_no_rating_key_for_other_sources():
    assert "rating" not in analyst.trigger_record("event", {"source": "news", "title": "x"})


def test_ingestion_adds_the_structured_fields():
    from ingestion import analyst_ratings as ar
    now = pd.Timestamp.now(tz="UTC")
    df = pd.DataFrame([{"Firm": "JPMorgan", "ToGrade": "Overweight", "FromGrade": "Overweight",
                        "Action": "main", "priceTargetAction": "Raises",
                        "currentPriceTarget": 270.0, "priorPriceTarget": 240.0}], index=[now])
    with patch("yfinance.Ticker") as T, patch.object(ar, "_is_seen", return_value=False):
        T.return_value.upgrades_downgrades = df
        (ev,) = ar._fetch_for_ticker("ANET")
    assert ev["rating"] == {"action": "main", "grade_changed": False, "pt_action": "Raises"}
    assert "price target raises to $270" in ev["title"]


def test_nan_cells_map_to_none():
    from ingestion import analyst_ratings as ar
    now = pd.Timestamp.now(tz="UTC")
    df = pd.DataFrame([{"Firm": "Citi", "ToGrade": "Buy", "FromGrade": float("nan"),
                        "Action": "main", "priceTargetAction": float("nan"),
                        "currentPriceTarget": None, "priorPriceTarget": None}], index=[now])
    with patch("yfinance.Ticker") as T, patch.object(ar, "_is_seen", return_value=False):
        T.return_value.upgrades_downgrades = df
        (ev,) = ar._fetch_for_ticker("ANET")
    assert ev["rating"] == {"action": "main", "grade_changed": False, "pt_action": None}
    assert analyst.trigger_record("event", ev)["rating"]["pt_only"] is False
