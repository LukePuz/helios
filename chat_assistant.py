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
        # CashBalance isn't reliably sent by IBKR for this account — see
        # ibkr.py's get_account() for the full story. TotalCashValue is the
        # field that's actually populated.
        cash = float(vals.get("TotalCashValue", vals.get("CashBalance", 0)))
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


def _tool_request_deep_analysis(args: dict, requested_by: str = "") -> str:
    """Enqueues the real analyst pipeline (situation brief, 72B analysis,
    conviction scoring) for a ticker — not a web-search summary. Runs
    async on llm_p2, the lowest-priority analysis queue (always behind
    held-position and organic news-driven analysis, see
    tasks.LLM_QUEUE_RESEARCH), and delivers its result back to this chat
    as a follow-up message rather than blocking the reply. Deliberately
    read-only: tasks.research_ticker_on_demand -> agents.crew.research_
    ticker never routes to risk sizing or execution, no matter what the
    analyst concludes — a chat question must never become a real trade."""
    ticker = str(args.get("ticker", "")).upper().strip()
    if not ticker:
        return "No ticker provided."
    try:
        from tasks import research_ticker_on_demand
        research_ticker_on_demand.apply_async(
            args=[ticker, requested_by], queue="llm_p2", expires=6 * 3600,
        )
        return (
            f"Submitted {ticker} for full deep analysis — the same 72B model, real "
            f"data, and conviction scoring used for live trading decisions, not just "
            f"a web search. It runs behind any analysis already queued for held "
            f"positions or organic news, so it can take a few minutes or longer "
            f"during busy periods. I'll send the result here (evidence, reasoning, "
            f"conviction score) as a follow-up message when it's done. This never "
            f"places a trade on its own, no matter what it concludes."
        )
    except Exception as e:
        log.error(f"[chat] request_deep_analysis failed to enqueue for {ticker}: {e}")
        return f"Couldn't submit {ticker} for deep analysis right now — try again shortly."


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
            "description": (
                "Search the live web for current information — news, prices, events, anything "
                "not covered by the other tools. A bare ticker or short word is a bad query — "
                "tickers routinely collide with unrelated things (e.g. 'MU' alone returns the "
                "Greek letter, a mythical continent, and Mercer University before it returns "
                "Micron Technology). Fix this by appending 'stock', 'ticker', or 'quote' to the "
                "ticker itself, e.g. 'MU stock' or 'MU ticker quote' — do NOT try to guess or "
                "spell out the full company name from the ticker, since that can be wrong (MU "
                "is Micron Technology, not Microchip Technology — an easy mix-up); the ticker "
                "plus a disambiguating word like 'stock' is enough and safer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Ticker + 'stock'/'ticker'/'quote', e.g. 'MU stock' — never a bare ticker, and never a guessed-at company name.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_deep_analysis",
            "description": (
                "Submit a ticker to the real analyst pipeline for a properly researched "
                "recommendation — situation brief, 72B-model analysis, conviction score "
                "(1-10), thesis, catalysts, valuation math, and key risks. This is NOT "
                "instant: it runs in the background (behind held-position and organic-news "
                "analysis in the queue) and the result arrives as a separate follow-up "
                "message, not in your current reply. Use this whenever someone asks "
                "whether a ticker is a good buy/sell/hold, wants a real recommendation, or "
                "asks for a proper/full/deep analysis — search_web alone only gives news "
                "snippets, not an actual evaluated recommendation with a conviction score. "
                "It's fine to also give a quick informal take from search_web in the same "
                "reply while this runs in the background."
            ),
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. MU"}},
                "required": ["ticker"],
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
    "get_ticker_details only returns a raw price and position status — it is NOT analysis. "
    "For any question asking for an opinion, a take, or whether a ticker is a good buy/sell/"
    "hold, you must also call search_web for recent news and sentiment before answering; "
    "a price with no context is an incomplete answer to that kind of question. You must ALSO "
    "call request_deep_analysis for that same ticker in the same turn — search_web only gets "
    "you a quick informal take, request_deep_analysis is the real, properly-researched "
    "recommendation with a conviction score, and its result arrives as a separate follow-up "
    "message. Give your quick search_web-based take now, and tell the user the full analysis "
    "is on its way. "
    "search_web results for a short ticker sometimes include unrelated noise (wrong company, "
    "a person's name, an unrelated topic that happens to share the letters) — silently ignore "
    "anything in the results that isn't actually about the company in question; never mention "
    "an irrelevant result in your answer just because it showed up in the search. "
    "Be concrete, not just persuasive: pull actual numbers out of the search results (price "
    "moves, revenue/earnings figures, growth rates, analyst price targets — whatever is "
    "actually in the snippets) instead of vague language like 'strong growth' or 'mixed "
    "outlook.' Name the source for each claim (e.g. 'per Yahoo Finance' or the article title) "
    "so it's clear where a number came from. Never state a statistic that isn't actually in "
    "the tool results — if the search results are thin on hard numbers, say so plainly rather "
    "than filling the gap with confident-sounding generalities. "
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

    # Dedicated CPU-only server first (deploy/systemd/ollama-cpu.service), so
    # chat can't queue behind or block the 72B analyst on the main server; see
    # agents/screener.py OLLAMA_CPU_HOST for the 2026-10-25 finding. Falls back
    # to the main server only when the CPU one isn't running at all.
    cpu_host = os.getenv("OLLAMA_CPU_HOST", "http://localhost:11435")
    for host in (cpu_host, OLLAMA_HOST):
        try:
            resp = requests.post(f"{host}/api/chat", json=payload, timeout=90)
            resp.raise_for_status()
            return resp.json().get("message")
        except requests.exceptions.ConnectionError:
            continue  # this server isn't running; try the next
        except requests.exceptions.Timeout:
            log.error(f"Ollama chat timed out ({host})")
            return None
        except Exception as e:
            log.error(f"Ollama chat failed ({host}): {e}")
            return None
    log.error("Ollama chat failed: no Ollama server reachable")
    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def respond(user_message: str, username: str, user_id: str | int | None = None) -> str:
    """
    Runs the tool-calling agent loop and returns the final natural-language
    reply. Safe to call from the worker thread; no order execution ever
    happens here (see module docstring).

    user_id keys persisted conversation history (see db.get/append_chat_
    history) so a follow-up question later has context from what was just
    discussed — without it (e.g. a direct/test call) each call is stateless,
    same as before. Keyed on the numeric Telegram user_id rather than the
    display name, and DB-backed rather than in-memory, so history survives
    process restarts and is shared between the full pipeline's telegram_bot
    .py and the overnight standby_bot.py.
    """
    history: list[dict] = []
    if user_id is not None:
        try:
            import db
            history = db.get_chat_history(user_id)
        except Exception as e:
            log.warning(f"[chat] get_chat_history failed: {e}")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"@{username} asks: {user_message}"},
    ]

    # request_deep_analysis needs to know who asked (for the eventual
    # follow-up alert) — the model can't supply that itself, so it's bound
    # here per-call rather than looked up from the shared _TOOL_FUNCTIONS map.
    tool_functions = dict(_TOOL_FUNCTIONS)
    tool_functions["request_deep_analysis"] = (
        lambda args: _tool_request_deep_analysis(args, requested_by=username)
    )

    # Tracked so the "quick take" disclaimer below is added deterministically
    # rather than relying on the (3B, inconsistent) model to remember to say
    # it every time — found 2026-07-18: a user got a quick take and a full
    # analysis 15 minutes apart on the same ticker (SNDK) with different
    # verdicts and no clear signal that the first one was the provisional
    # one. The model's own wording of this was unreliable; a fixed label
    # appended here never is.
    deep_analysis_requested = False

    for _ in range(_MAX_TOOL_ITERATIONS):
        msg = _call_ollama_chat(messages, tools=_TOOL_SPECS)
        if msg is None:
            return ("The local AI model didn't respond in time — it may be busy running "
                    "a deep analysis right now. Try again in a minute.")

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            content = (msg.get("content") or "").strip()
            reply = content or "I didn't get a usable response — try rephrasing the question."
            if deep_analysis_requested and content:
                reply += (
                    "\n\n⚠️ This is a quick, informal take — not a verdict. The real, "
                    "properly-researched analysis (conviction score, evidence, sources) "
                    "is still coming as a separate message."
                )
            # Only a genuine, complete exchange goes into history — an error/
            # timeout reply here would just confuse a later follow-up.
            if user_id is not None and content:
                try:
                    import db
                    db.append_chat_message(user_id, "user", user_message)
                    db.append_chat_message(user_id, "assistant", reply)
                except Exception as e:
                    log.warning(f"[chat] append_chat_message failed: {e}")
            return reply

        messages.append(msg)
        for call in tool_calls:
            fn_name = call.get("function", {}).get("name", "")
            fn_args = call.get("function", {}).get("arguments", {}) or {}
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    fn_args = {}
            if fn_name == "request_deep_analysis":
                deep_analysis_requested = True
            fn = tool_functions.get(fn_name)
            log.info(f"[chat] tool call: {fn_name}({fn_args})")
            result = fn(fn_args) if fn else f"Unknown tool: {fn_name}"
            messages.append({"role": "tool", "content": str(result)})

    return "I wasn't able to finish looking that up after a few tries — try asking again, maybe more specifically."
