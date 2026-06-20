"""
Central configuration. Edit this file to change the symbol universe.
All other modules import SYMBOLS from here.
"""

SYMBOLS: list[str] = [
    # Mega-cap tech / AI hyperscalers
    "META", "AAPL", "NVDA", "GOOGL", "ORCL",
    "MSFT", "AMZN", "TSLA",

    # AI chips & hardware
    "AMD", "AVGO", "QCOM", "MU", "ARM",

    # Semiconductor equipment — picks & shovels for the AI buildout
    "AMAT", "LRCX", "ASML", "KLAC",

    # AI software & data platforms
    "PLTR", "CRM", "NOW", "SNOW",

    # Cloud & AI infrastructure
    "NET", "DDOG",
]

# Strategy thresholds — edit here, applies everywhere
MOMENTUM_BUY_THRESHOLD  =  0.02   # 5-day momentum > +2% triggers BUY
MOMENTUM_SELL_THRESHOLD = -0.02   # 5-day momentum < -2% triggers SELL
TAKE_PROFIT_PCT         =  0.08   # Exit when position is up 8%
SENTIMENT_BLOCK_BUY     = -0.3    # Block BUY if news sentiment below this
SENTIMENT_BLOCK_SELL    =  0.3    # Block SELL if news sentiment above this

# Risk rules — must match risk_manager.py
MAX_RISK_PER_TRADE  = 0.02   # 2% of portfolio per trade
MAX_POSITION_PCT    = 0.20   # 20% max per position
DRAWDOWN_HALT_PCT   = 0.10   # Halt if portfolio drops 10%

START_VALUE = 100_000.0
POLL_INTERVAL_SECONDS = 300  # 5 minutes
