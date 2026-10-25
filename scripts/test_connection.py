"""
Quick broker connectivity smoke test.
Usage: python scripts/test_connection.py

Requires TWS or IB Gateway running with API access enabled (see ibkr.py's
module docstring for port/config details). This used to test an Alpaca
connection — execution moved to Interactive Brokers a while back and this
was never updated, so it was silently checking a broker the system no
longer trades through.
"""

import sys
from pathlib import Path

# Run from scripts/; the modules it imports live at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ibkr


def main() -> None:
    print(f"Connecting to IBKR TWS/Gateway at {ibkr.IB_HOST}:{ibkr.IB_PORT} "
          f"(clientId={ibkr._get_client_id()})...")
    try:
        client = ibkr.get_client()
        account = client.get_account()
    except Exception as e:
        print(f"FAILED to connect: {e}")
        print("Make sure TWS or IB Gateway is running with API connections enabled.")
        sys.exit(1)

    print(f"Portfolio Value: ${account.portfolio_value:,.2f}")
    print(f"Cash:            ${account.cash:,.2f}")
    print(f"Buying Power:    ${account.buying_power:,.2f}")

    positions = client.get_all_positions()
    if positions:
        print(f"\nOpen positions ({len(positions)}):")
        for p in positions:
            print(f"  {p.symbol:<6} {p.qty:>8.2f} shares  "
                 f"unrealized P&L ${p.unrealized_pl:+,.2f}")
    else:
        print("\nNo open positions.")

    print("\nConnection OK.")


if __name__ == "__main__":
    main()
