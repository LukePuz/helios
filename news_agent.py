"""
Sentiment pre-filter using Alpaca's news API and keyword matching.
LLM integration can replace _score_text() later without changing the interface.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

_API_KEY = os.getenv("ALPACA_API_KEY")
_API_SECRET = os.getenv("ALPACA_API_SECRET")
_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

_POSITIVE = {
    "surge", "soar", "rally", "gain", "beat", "record", "growth", "profit",
    "upgrade", "outperform", "bullish", "strong", "rises", "jumps", "higher",
    "boost", "expand", "positive", "success", "revenue", "earnings", "deal",
    "acquisition", "launch", "wins", "partnership", "innovation", "milestone",
    "guidance", "raised", "exceeded", "demand", "approved", "dividend",
}

_NEGATIVE = {
    "crash", "plunge", "fall", "miss", "loss", "lawsuit", "fine", "penalty",
    "downgrade", "underperform", "bearish", "weak", "drops", "slump", "lower",
    "decline", "cuts", "layoffs", "disappoints", "recall", "fraud", "probe",
    "investigation", "fails", "bankruptcy", "default", "warning", "concern",
    "tariff", "sanction", "breach", "hack", "shortage", "delay", "suspended",
}


def _score_text(text: str) -> float:
    """Scores text in [-1, 1] using positive/negative keyword ratio."""
    words = set(text.lower().split())
    pos = len(words & _POSITIVE)
    neg = len(words & _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


def get_sentiment(symbols: list[str]) -> dict[str, float]:
    """
    Returns sentiment score per symbol in [-1.0, 1.0].
    Averages scores across the last 24h of headlines.
    Returns 0.0 per symbol if news fetch fails.
    """
    if not _API_KEY or not _API_SECRET:
        return {s: 0.0 for s in symbols}

    headers = {
        "APCA-API-KEY-ID": _API_KEY,
        "APCA-API-SECRET-KEY": _API_SECRET,
    }
    start = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    scores: dict[str, list[float]] = {s: [] for s in symbols}

    for symbol in symbols:
        try:
            resp = requests.get(
                _NEWS_URL,
                headers=headers,
                params={"symbols": symbol, "start": start, "limit": 10, "sort": "desc"},
                timeout=10,
            )
            resp.raise_for_status()
            for article in resp.json().get("news", []):
                text = f"{article.get('headline', '')} {article.get('summary', '')}"
                scores[symbol].append(_score_text(text))
        except Exception as e:
            logging.warning(f"News fetch failed for {symbol}: {e}")

    return {
        symbol: round(sum(v) / len(v), 4) if v else 0.0
        for symbol, v in scores.items()
    }


if __name__ == "__main__":
    from market_data import SYMBOLS
    logging.basicConfig(level=logging.INFO)
    sentiments = get_sentiment(SYMBOLS)
    print(f"\n{'Symbol':<8} {'Score':>7}  Label")
    print("-" * 28)
    for sym, score in sentiments.items():
        label = "POSITIVE" if score > 0.1 else "NEGATIVE" if score < -0.1 else "NEUTRAL"
        print(f"{sym:<8} {score:>+7.3f}  {label}")
