from alpaca.trading.client import TradingClient
from alpaca.trading.models import TradeAccount

MAX_RISK_PER_TRADE = 0.02      # 2% of portfolio per trade
MAX_POSITION_PCT = 0.20        # 20% max per stock
DRAWDOWN_HALT_PCT = 0.10       # halt trading if portfolio drops 10%


class RiskManager:
    def __init__(self, client: TradingClient, start_value: float | None = None):
        self._client = client
        account = self._get_account()
        self.start_value: float = start_value or float(account.portfolio_value)

    def _get_account(self) -> TradeAccount:
        return self._client.get_account()

    def portfolio_value(self) -> float:
        return float(self._get_account().portfolio_value)

    def is_halted(self) -> bool:
        """Returns True if the portfolio has dropped >= 10% from start value."""
        current = self.portfolio_value()
        drawdown = (self.start_value - current) / self.start_value
        if drawdown >= DRAWDOWN_HALT_PCT:
            print(
                f"[RiskManager] HALT — drawdown {drawdown:.1%} exceeds "
                f"{DRAWDOWN_HALT_PCT:.0%} limit "
                f"(start=${self.start_value:,.2f}, current=${current:,.2f})"
            )
            return True
        return False

    def max_trade_value(self) -> float:
        """Max dollar amount to risk on a single trade (2% of portfolio)."""
        return self.portfolio_value() * MAX_RISK_PER_TRADE

    def max_position_value(self, symbol: str) -> float:
        """Max dollar value allowed in any one position (20% of portfolio)."""
        return self.portfolio_value() * MAX_POSITION_PCT

    def current_position_value(self, symbol: str) -> float:
        """Returns current market value held in a symbol, or 0 if none."""
        try:
            position = self._client.get_open_position(symbol)
            return float(position.market_value)
        except Exception:
            return 0.0

    def shares_affordable(self, symbol: str, price: float) -> int:
        """
        Returns the maximum number of whole shares that can be bought for
        `symbol` at `price`, respecting both the per-trade and per-position caps.
        Returns 0 if trading is halted or no shares can be bought.
        """
        if self.is_halted():
            return 0

        trade_cap = self.max_trade_value()
        position_cap = self.max_position_value(symbol)
        held = self.current_position_value(symbol)
        remaining_position_room = max(0.0, position_cap - held)

        dollar_budget = min(trade_cap, remaining_position_room)
        shares = int(dollar_budget // price)

        if shares <= 0:
            print(
                f"[RiskManager] {symbol}: 0 shares allowed "
                f"(budget=${dollar_budget:.2f}, price=${price:.2f})"
            )
        return shares

    def check_sell_allowed(self, symbol: str, shares: int) -> bool:
        """Basic check: can only sell what we own."""
        try:
            position = self._client.get_open_position(symbol)
            owned = int(position.qty)
            if shares > owned:
                print(f"[RiskManager] Sell rejected — only {owned} shares of {symbol} owned.")
                return False
            return True
        except Exception:
            print(f"[RiskManager] Sell rejected — no position in {symbol}.")
            return False


if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        raise EnvironmentError("ALPACA_API_KEY or ALPACA_API_SECRET not found in .env")

    trading_client = TradingClient(api_key, api_secret, paper=True)
    rm = RiskManager(trading_client)

    pv = rm.portfolio_value()
    print(f"Portfolio value:       ${pv:,.2f}")
    print(f"Max trade size (2%):   ${rm.max_trade_value():,.2f}")
    print(f"Max position (20%):    ${rm.max_position_value('AAPL'):,.2f}")
    print(f"Halted:                {rm.is_halted()}")
    print(f"AAPL shares at $200:   {rm.shares_affordable('AAPL', 200.0)}")
