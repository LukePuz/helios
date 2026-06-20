import os
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Anchor to script directory so load_dotenv works regardless of cwd
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

if not API_KEY or not API_SECRET:
    raise EnvironmentError("ALPACA_API_KEY or ALPACA_API_SECRET not found in .env")

client = TradingClient(API_KEY, API_SECRET, paper=True)
account = client.get_account()

print(f"Account ID:      {account.id}")
print(f"Cash:            ${float(account.cash):,.2f}")
print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
print(f"Buying Power:    ${float(account.buying_power):,.2f}")
print(f"Status:          {account.status}")
