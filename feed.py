"""
Activity feed — fire-and-forget emitter called by every pipeline stage.
Writes to the activity_feed DB table so the dashboard can show live
commentary on what the bots are doing and thinking.

Never blocks the caller — all DB writes run in a daemon thread.
"""
from __future__ import annotations

import threading

STAGE_ICONS: dict[str, str] = {
    "screener":  "🔍",
    "analyst":   "🧠",
    "risk":      "📊",
    "telegram":  "📱",
    "execution": "✅",
    "system":    "⚙️",
    "edgar":     "📄",
    "options":   "📈",
    "macro":     "🌐",
    "insider":   "👤",
}


def emit(
    stage:     str,
    ticker:    str,
    message:   str,
    sentiment: str = "neutral",
    extra:     dict | None = None,
) -> None:
    """
    Stream a status message to the dashboard activity feed.

    Args:
        stage:     Pipeline stage key (screener/analyst/risk/telegram/system/…)
        ticker:    Symbol being processed, or "" for system-level messages
        message:   Human-readable description of what's happening
        sentiment: "bullish" | "bearish" | "neutral" | "alert"
        extra:     Optional dict of additional data stored as JSON
    """
    def _write() -> None:
        try:
            import db
            db.log_activity(stage, ticker, message, sentiment, extra)
        except Exception:
            pass  # never let the feed crash the pipeline

    threading.Thread(target=_write, daemon=True, name="feed-emit").start()
