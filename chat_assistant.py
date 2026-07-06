"""
Group chat assistant — handles @mention messages from whitelisted Telegram users.

Provides:
  respond(user_message, username) -> str
      Builds live portfolio/price context, optionally web-searches, then asks
      the local Ollama 7b model to reply in plain English.

  parse_trade_intent(text) -> dict | None
      Parses "buy 5 NVDA" / "sell TSLA 10 shares" style commands.
      Returns {"side": "BUY"|"SELL", "qty": float|None, "ticker": str} or None.
"""
import logging
import os
import re
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common English words that look like tickers — blocklist to avoid false matches
# ---------------------------------------------------------------------------
_WORD_BLOCKLIST = {
    "I", "A", "ME", "MY", "WE", "US", "IT", "IS", "IN", "OR", "AN", "AS",
    "AT", "BE", "BY", "DO", "GO", "IF", "NO", "OF", "ON", "SO", "TO", "UP",
    "AM", "ARE", "CAN", "DID", "FOR", "GET", "GOT", "HAD", "HAS", "HIM",
    "HIS", "HOW", "ITS", "LET", "NOT", "NOW", "OLD", "OUR", "OUT", "OWN",
    "SAY", "SHE", "THE", "TOO", "TWO", "USE", "WAS", "WAY", "WHO", "WHY",
    "YET", "YOU", "AND", "BUT", "DAD", "BUY", "SELL", "ASK", "LOOK", "SHOW",
    "TELL", "WHAT", "WHEN", "WILL", "WITH", "THAT", "THIS", "FROM", "HAVE",
    "BEEN", "WERE", "THEM", "THEY", "THEN", "THAN", "THAN", "MUCH", "MANY",
    "SOME", "JUST", "LIKE", "ALSO", "WELL", "EVEN", "BACK", "GOOD", "INTO",
    "MOST", "OVER", "SUCH", "EACH", "MAKE", "LAST", "LONG", "DOES", "COME",
    "YEAR", "HIGH", "NEWS", "BEST", "HOLD", "STOP", "TAKE", "GIVE", "KNOW",
    "FIND", "OPEN", "SEEN", "SURE", "TRUE", "FEEL", "REAL", "BOTH", "DOWN",
    "ONLY", "NEXT", "MORE", "PLAN", "RISK", "LOSS", "GAIN", "SELL", "CALL",
    "EXIT", "TRIM", "WAIT", "SAME", "LEFT", "ONCE", "KEEP", "IDEA", "HELP",
    "OKAY", "YEAH", "COOL", "NICE", "DONE", "ASAP", "CASH",
}

_SEARCH_KEYWORDS = {"news", "latest", "recent", "search", "find", "happening",
                    "update", "today", "report", "earnings", "analyst", "target",
                    "forecast", "outlook", "guidance"}

_TRADE_PATTERN = re.compile(
    r"\b(buy|sell)\b"
    r"(?:\s+(\d+(?:\.\d+)?)\s+(?:shares?\s+of\s+|share\s+of\s+)?)?"
    r"\s*\b([A-Z]{1,5})\b"
    r"(?:\s+(\d+(?:\.\d+)?)\s+shares?)?"  # qty may come after ticker
    ,
    re.IGNORECASE,
)

_TICKER_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")


# ---------------------------------------------------------------------------
# Trade intent parser
# ---------------------------------------------------------------------------

def parse_trade_intent(text: str) -> Optional[dict]:
    """
    Returns {"side": "BUY"|"SELL", "qty": float|None, "ticker": str} or None.

    Understands:
      buy 5 NVDA
      sell TSLA 10 shares
      buy AAPL
      sell 2.5 shares of AMD
    """
    clean = re.sub(r"!urgent", "", text, flags=re.IGNORECASE).strip()
    m = _TRADE_PATTERN.search(clean)
    if not m:
        return None

    side_raw, qty_before, ticker, qty_after = m.groups()
    side = "BUY" if side_raw.lower() == "buy" else "SELL"
    ticker = ticker.upper()

    if ticker in _WORD_BLOCKLIST:
        return None

    qty_str = qty_before or qty_after
    qty = float(qty_str) if qty_str else None

    return {"side": side, "qty": qty, "ticker": ticker}


# ---------------------------------------------------------------------------
# Ticker extraction
# ---------------------------------------------------------------------------

def _extract_tickers(text: str) -> list[str]:
    found = _TICKER_PATTERN.findall(text)
    return [t for t in found if t not in _WORD_BLOCKLIST]


# ---------------------------------------------------------------------------
# Context fetching
# ---------------------------------------------------------------------------

def _ibkr_reachable() -> bool:
    """Check if the ibkr singleton is already connected — never spawns a new connection."""
    try:
        import ibkr as _ibkr_mod
        ib = _ibkr_mod._ib
        result = ib is not None and ib.isConnected()
        log.info(f"[chat] _ibkr_reachable={result} (ib={ib})")
        return result
    except Exception as e:
        log.error(f"[chat] _ibkr_reachable check failed: {e}")
        return False


def _fetch_ticker_context(tickers: list[str]) -> str:
    if not tickers:
        return ""
    if not _ibkr_reachable():
        return "(IBKR offline — live position data unavailable)"
    try:
        import ibkr as _ibkr_mod
        ib = _ibkr_mod._ib
        portfolio = {item.contract.symbol: item for item in ib.portfolio()}
        lines: list[str] = []
        for ticker in tickers:
            item = portfolio.get(ticker)
            if item:
                lines.append(
                    f"{ticker}: {item.position} shares held, avg entry ${item.averageCost:.2f}, "
                    f"market value ${item.marketValue:,.2f}, unrealized P&L ${item.unrealizedPNL:+.2f}"
                )
            else:
                lines.append(f"{ticker}: not currently held")
        return "\n".join(lines)
    except Exception as e:
        log.error(f"Ticker context fetch failed: {e}")
        return "(IBKR offline — live position data unavailable)"


def _fetch_portfolio_snapshot() -> str:
    if not _ibkr_reachable():
        return "(IBKR offline — portfolio data unavailable)"
    try:
        import ibkr as _ibkr_mod
        ib = _ibkr_mod._ib
        vals = {v.tag: v.value for v in ib.accountValues() if v.currency == "USD"}
        net_liq = float(vals.get("NetLiquidation", 0))
        portfolio = [item for item in ib.portfolio() if float(item.position) != 0]
        if not portfolio:
            return f"Portfolio value: ${net_liq:,.2f}. No open positions."
        lines = [f"Portfolio value: ${net_liq:,.2f}. Open positions:"]
        for item in portfolio:
            lines.append(
                f"  {item.contract.symbol}: {item.position} shares, "
                f"market value ${item.marketValue:,.2f}, unrealized P&L ${item.unrealizedPNL:+.2f}"
            )
        return "\n".join(lines)
    except Exception as e:
        log.error(f"Portfolio snapshot failed: {e}")
        return "(IBKR offline — portfolio data unavailable)"


def _fetch_trade_history(tickers: list[str]) -> str:
    if not tickers:
        return ""
    try:
        import db
        lines: list[str] = []
        for ticker in tickers:
            try:
                trades = db.get_trades_for_symbol(ticker, limit=5)
                if trades:
                    for t in trades:
                        side = t.get("side", "?")
                        qty = t.get("quantity", "?")
                        price = t.get("price", 0)
                        reason = t.get("reason", "")
                        ts = str(t.get("executed_at", ""))[:10]
                        lines.append(f"{ticker} trade ({ts}): {side} {qty} shares @ ${price:.2f} — {reason}")
            except Exception:
                pass
        return "\n".join(lines)
    except Exception as e:
        log.debug(f"Trade history fetch failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def _maybe_web_search(text: str, tickers: list[str]) -> str:
    words = set(text.lower().split())
    if not words.intersection(_SEARCH_KEYWORDS):
        return ""
    query_ticker = tickers[0] if tickers else ""
    query = f"{query_ticker} stock {text[:120]}" if query_ticker else text[:120]
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=5)
        if not results:
            return ""
        snippets = []
        for r in results:
            body = (r.get("body") or "")[:300]
            title = (r.get("title") or "")
            snippets.append(f"• {title}: {body}")
        return "Web search results:\n" + "\n".join(snippets)
    except Exception as e:
        log.debug(f"Web search failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _call_ollama(system_prompt: str, user_message: str) -> str:
    try:
        from config import OLLAMA_HOST, CHAT_MODEL
    except ImportError:
        OLLAMA_HOST = "http://localhost:11434"
        CHAT_MODEL  = "qwen2.5:7b"

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": CHAT_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        log.error(f"Ollama chat failed: {e}")
        return "Sorry, I couldn't reach the local AI model right now."


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def respond(user_message: str, username: str) -> str:
    """
    Build live context and return a natural-language response via Ollama 7b.
    Safe to call from the worker thread; no order execution happens here.
    """
    tickers = _extract_tickers(user_message)

    ticker_ctx   = _fetch_ticker_context(tickers)
    portfolio    = _fetch_portfolio_snapshot()
    trade_hist   = _fetch_trade_history(tickers)
    web_results  = _maybe_web_search(user_message, tickers)

    context_parts = [portfolio]
    if ticker_ctx:
        context_parts.append("Ticker details:\n" + ticker_ctx)
    if trade_hist:
        context_parts.append("Recent trade history:\n" + trade_hist)
    if web_results:
        context_parts.append(web_results)

    context = "\n\n".join(p for p in context_parts if p)

    system = (
        "You are Helios, a read-only trading assistant for a small private investment fund. "
        "You have access to live portfolio data and recent trade history shown below. "
        "Answer questions clearly and concisely. Never confirm, suggest, or execute trades — "
        "only describe existing positions and factual information. "
        "Keep responses short (under 200 words unless the question requires detail). "
        "Do not add disclaimers about being an AI.\n\n"
        f"=== LIVE CONTEXT ===\n{context}"
    )

    return _call_ollama(system, f"@{username} asks: {user_message}")
