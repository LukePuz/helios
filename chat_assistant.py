"""
Group chat assistant — handles @mention messages from whitelisted Telegram
users as a genuine tool-calling agent, not a heuristic context-stuffing
pipeline. Previously this extracted tickers via regex, always injected a
portfolio snapshot, and only searched the web when the message happened to
contain a keyword like "news" or "latest" — the model itself never decided
what it needed. Now the model is handed a real toolset (portfolio, ticker
detail, trade history, web search, market regime) via Ollama's native
tool-calling API and decides for itself what to look up, in what order,
possibly across multiple rounds.

Provides:
  respond(user_message, username) -> str
      Runs the tool-calling agent loop and returns the final natural-
      language reply.

  parse_trade_intent(text) -> dict | None
      Parses "buy 5 NVDA" / "sell TSLA 10 shares" style commands. Used only
      by the separate, explicit !urgent command path in telegram_bot.py —
      unrelated to and untouched by the agent rewrite below.
      Returns {"side": "BUY"|"SELL", "qty": float|None, "ticker": str} or None.

Deliberately NOT a tool available to the agent: trade execution. The
!urgent command path is the only way this bot places real orders — it
requires explicit syntax and goes through the full risk/safety pipeline
(execution.py). A natural-language agent must never be able to trigger a
trade through free-text interpretation ("can you buy me some NVDA" should
never become a real order because a model decided to call a tool).
"""
import json
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

_TRADE_PATTERN = re.compile(
    r"\b(buy|sell)\b"
    r"(?:\s+(\d+(?:\.\d+)?)\s+(?:shares?\s+of\s+|share\s+of\s+)?)?"
    r"\s*\b([A-Z]{1,5})\b"
    r"(?:\s+(\d+(?:\.\d+)?)\s+shares?)?"  # qty may come after ticker
    ,
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Trade intent parser — used only by the separate !urgent command path
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
# Tools — all read-only, callable by the agent. See module docstring for
# why trade execution is deliberately excluded.
# ---------------------------------------------------------------------------

def _ibkr_reachable() -> bool:
    """Check if the ibkr singleton is already connected — never spawns a new connection."""
    try:
        import ibkr as _ibkr_mod
        ib = _ibkr_mod._ib
        return ib is not None and ib.isConnected()
    except Exception as e:
        log.error(f"[chat] _ibkr_reachable check failed: {e}")
        return False


def _tool_get_portfolio(_args: dict) -> str:
    """Current account value, cash, and every open position."""
    if not _ibkr_reachable():
        return "IBKR is offline right now — live portfolio data is unavailable."
    try:
        import ibkr as _ibkr_mod
        ib = _ibkr_mod._ib
        vals = {v.tag: v.value for v in ib.accountValues() if v.currency == "USD"}
        net_liq = float(vals.get("NetLiquidation", 0))
        cash = float(vals.get("CashBalance", 0))
        portfolio = [item for item in ib.portfolio() if float(item.position) != 0]
        if not portfolio:
            return f"Portfolio value: ${net_liq:,.2f}. Cash: ${cash:,.2f}. No open positions."
        lines = [f"Portfolio value: ${net_liq:,.2f}. Cash: ${cash:,.2f}. Open positions:"]
        for item in portfolio:
            lines.append(
                f"  {item.contract.symbol}: {item.position} shares, avg entry "
                f"${item.averageCost:.2f}, market value ${item.marketValue:,.2f}, "
                f"unrealized P&L ${item.unrealizedPNL:+.2f}"
            )
        return "\n".join(lines)
    except Exception as e:
        log.error(f"[chat] get_portfolio tool failed: {e}")
        return "Portfolio lookup failed — IBKR may be offline."


def _tool_get_ticker_details(args: dict) -> str:
    """Position detail + latest price for one specific ticker."""
    ticker = str(args.get("ticker", "")).upper().strip()
    if not ticker:
        return "No ticker provided."
    if not _ibkr_reachable():
        return f"IBKR is offline — can't look up live data for {ticker} right now."
    try:
        import ibkr as _ibkr_mod
        ib = _ibkr_mod._ib
        portfolio = {item.contract.symbol: item for item in ib.portfolio()}
        item = portfolio.get(ticker)
        price = _ibkr_mod.get_latest_price(ticker)
        price_line = f"Latest price: ${price:.2f}" if price else "Latest price unavailable"
        if item:
            return (
                f"{ticker}: {item.position} shares held, avg entry ${item.averageCost:.2f}, "
                f"market value ${item.marketValue:,.2f}, unrealized P&L ${item.unrealizedPNL:+.2f}. "
                f"{price_line}."
            )
        return f"{ticker}: not currently held. {price_line}."
    except Exception as e:
        log.error(f"[chat] get_ticker_details tool failed for {ticker}: {e}")
        return f"Lookup failed for {ticker}."


def _tool_get_trade_history(args: dict) -> str:
    """Recent executed trades, optionally filtered to one ticker."""
    ticker = str(args.get("ticker", "")).upper().strip() or None
    limit = int(args.get("limit") or 10)
    try:
        import db
        trades = db.get_recent_trades(limit=50 if ticker else limit)
        if ticker:
            trades = [t for t in trades if t.get("symbol") == ticker][:limit]
        if not trades:
            scope = f" for {ticker}" if ticker else ""
            return f"No trade history found{scope}."
        lines = []
        for t in trades:
            ts = str(t.get("timestamp", ""))[:10]
            lines.append(
                f"{ts}: {t['side']} {t['quantity']} {t['symbol']} @ ${t['price']:.2f} "
                f"— {t.get('reason', '')}"
            )
        return "\n".join(lines)
    except Exception as e:
        log.error(f"[chat] get_trade_history tool failed: {e}")
        return "Trade history lookup failed."


def _tool_get_market_regime(_args: dict) -> str:
    """Current market regime assessment (Risk-On/Neutral/Risk-Off, score, reasoning)."""
    try:
        from agents.market_regime import get_cached_regime
        regime = get_cached_regime()
        if not regime:
            return "No regime assessment available right now."
        return (
            f"Regime: {regime.get('regime')} | Score: {regime.get('score')}/100 | "
            f"Confidence: {regime.get('confidence')}%\n{regime.get('reasoning', '')}"
        )
    except Exception as e:
        log.error(f"[chat] get_market_regime tool failed: {e}")
        return "Regime lookup failed."


def _tool_search_web(args: dict) -> str:
    """Live web search — the agent decides when it needs current
    information, instead of the old keyword-triggered heuristic."""
    query = str(args.get("query", "")).strip()
    if not query:
        return "No search query provided."
    try:
        # duckduckgo_search was renamed to ddgs upstream — the old package
        # name is deprecated and was returning zero results in practice,
        # silently making every web-search tool call a no-op.
        from ddgs import DDGS
        results = DDGS().text(query, max_results=5)
        if not results:
            return f"No web results found for: {query}"
        snippets = []
        for r in results:
            title = r.get("title") or ""
            body = (r.get("body") or "")[:300]
            snippets.append(f"• {title}: {body}")
        return "\n".join(snippets)
    except Exception as e:
        log.debug(f"[chat] search_web tool failed: {e}")
        return f"Web search failed for: {query}"


_TOOL_FUNCTIONS = {
    "get_portfolio": _tool_get_portfolio,
    "get_ticker_details": _tool_get_ticker_details,
    "get_trade_history": _tool_get_trade_history,
    "get_market_regime": _tool_get_market_regime,
    "search_web": _tool_search_web,
}

_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "Get the current total portfolio value, cash balance, and every open position with P&L.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticker_details",
            "description": "Get position detail (if held) and the latest price for one specific stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. NVDA"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trade_history",
            "description": "Get recently executed trades, optionally filtered to one ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Optional ticker to filter to"},
                    "limit": {"type": "integer", "description": "Max trades to return, default 10"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_regime",
            "description": "Get the current market regime assessment (Risk-On/Neutral/Risk-Off), its score, and the reasoning behind it.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the live web for current information — news, prices, events, anything not covered by the other tools.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are Helios, a read-only research assistant for a small private investment fund's "
    "Telegram group. You have tools to look up live portfolio data, ticker details, trade "
    "history, the current market regime, and to search the web. Use them whenever they'd "
    "help answer the question — don't guess at data you can look up. You may use multiple "
    "tools, including more than once, before giving your final answer. "
    "Never confirm, suggest, or execute trades — you are informational only; if asked to "
    "trade, say that trades go through the explicit !urgent command, not chat. "
    "Keep your final answer concise (under 200 words unless the question needs more) and "
    "do not add disclaimers about being an AI."
)

_MAX_TOOL_ITERATIONS = 4


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _call_ollama_chat(messages: list[dict], tools: list[dict] | None = None) -> dict | None:
    try:
        from config import OLLAMA_HOST, CHAT_MODEL
    except ImportError:
        OLLAMA_HOST = "http://localhost:11434"
        CHAT_MODEL  = "qwen2.5:3b"

    payload = {
        "model": CHAT_MODEL,
        "stream": False,
        "messages": messages,
        # The 72B analyst can hold ~45GB combined VRAM across both GPUs
        # (confirmed via `ollama ps`), leaving no room for a concurrently
        # loaded small model — a chat call without num_gpu=0 measurably
        # timed out in production while the analyst was busy. Forcing CPU
        # means this never contends with the analyst for VRAM; keep_alive
        # keeps it resident so a multi-round tool-calling conversation
        # doesn't pay reload cost on every round.
        "options": {"num_gpu": 0},
        "keep_alive": "30m",
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json().get("message")
    except requests.exceptions.Timeout:
        log.error("Ollama chat timed out")
        return None
    except Exception as e:
        log.error(f"Ollama chat failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def respond(user_message: str, username: str) -> str:
    """
    Runs the tool-calling agent loop and returns the final natural-language
    reply. Safe to call from the worker thread; no order execution ever
    happens here (see module docstring).
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"@{username} asks: {user_message}"},
    ]

    for _ in range(_MAX_TOOL_ITERATIONS):
        msg = _call_ollama_chat(messages, tools=_TOOL_SPECS)
        if msg is None:
            return ("The local AI model didn't respond in time — it may be busy running "
                    "a deep analysis right now. Try again in a minute.")

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            content = (msg.get("content") or "").strip()
            return content or "I didn't get a usable response — try rephrasing the question."

        messages.append(msg)
        for call in tool_calls:
            fn_name = call.get("function", {}).get("name", "")
            fn_args = call.get("function", {}).get("arguments", {}) or {}
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    fn_args = {}
            fn = _TOOL_FUNCTIONS.get(fn_name)
            log.info(f"[chat] tool call: {fn_name}({fn_args})")
            result = fn(fn_args) if fn else f"Unknown tool: {fn_name}"
            messages.append({"role": "tool", "content": str(result)})

    return "I wasn't able to finish looking that up after a few tries — try asking again, maybe more specifically."
