"""
Manual pipeline test -- fires a fake IONQ quantum computing event through
every stage: Screen → Analyst → Risk Manager → DB → Telegram.

Forces each stage through even if it would normally be filtered out.
Run with:
    venv\Scripts\python.exe fire_test.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Mock IBKR client so the test doesn't conflict with the running main.py connection.
# RiskManager only needs get_account() and get_all_positions().
class _MockIBKR:
    class _Acct:
        portfolio_value = 100_000.0
        buying_power    = 100_000.0
        cash            = 100_000.0
    def get_account(self):
        return self._Acct()
    def get_all_positions(self):
        return []
    def get_open_position(self, symbol):
        raise Exception(f"No position in {symbol}")

TICKER = "IONQ"
FAKE_PRICE = None  # will try to fetch live; falls back to 8.50

TEST_EVENT = {
    "source": "sec_filing",
    "ticker": TICKER,
    "title":  "IonQ Awarded $54.5M DoD Contract -- Largest Government Quantum Deal to Date",
    "text": (
        "IonQ Inc. (NYSE: IONQ) announced today it has been awarded a $54.5 million "
        "contract by the US Department of Defense for quantum computing services over "
        "the next 36 months. The contract covers fault-tolerant quantum systems for "
        "cryptographic applications and optimization problems. Revenue guidance raised "
        "40% for fiscal year. Partnership with Lockheed Martin confirmed for joint "
        "quantum-classical hybrid systems. IonQ's trapped-ion architecture has now "
        "surpassed superconducting approaches in error-correction benchmarks. "
        "This is the largest government quantum computing contract ever awarded to "
        "a publicly traded pure-play quantum company."
    ),
    "url": "https://test.fire_test/ionq-dod-contract",
}


def _divider(label):
    print("\n" + "-"*55)
    print(f"  {label}")
    print("-"*55)


def main():
    print("\n" + "="*55)
    print("  PIPELINE TEST  --  IONQ quantum event")
    print("  All stages forced through for beta validation")
    print("="*55)

    # ── 1. SCREENER ──────────────────────────────────────────────────────────
    _divider("STAGE 1 / 4 -- Screener (7B filter)")
    from agents.screener import screen
    screening = screen(TEST_EVENT)
    print(f"  relevant  : {screening['relevant']}")
    print(f"  urgency   : {screening['urgency']}")
    print(f"  reason    : {screening['reason']}")
    print(f"  model     : {screening.get('model_used', '?')}")

    if not screening["relevant"]:
        print("\n  [OVERRIDE] Screener said not relevant -- forcing through.")
        screening["relevant"] = True
        screening["urgency"]  = "immediate"

    # ── 2. ANALYST ───────────────────────────────────────────────────────────
    _divider("STAGE 2 / 4 -- Analyst (LLM thesis + conviction)")
    print("  This may take 30-90s depending on model size...")

    from agents.analyst import analyze
    from agents.market_regime import get_cached_regime

    regime = get_cached_regime()
    print(f"  Regime: {regime.get('regime', 'Unknown')} (score={regime.get('score', '?')})")

    candidate = {
        "ticker":          TICKER,
        "role":            "Moonshot",
        "tier":            "Speculative",
        "sector":          "Quantum Computing",
        "composite_score": 85,
    }
    brief = f"BREAKING EVENT:\n{TEST_EVENT['title']}\n\n{TEST_EVENT['text']}"

    result = analyze(candidate, situation_brief=brief, regime_packet=regime)

    print(f"  action    : {result.get('action')}")
    print(f"  conviction: {result.get('conviction')}/10")
    print(f"  bucket    : {result.get('thesis_bucket')}")
    print(f"  u/d ratio : {result.get('upside_downside_ratio')}")
    print(f"  pass_to_risk: {result.get('pass_to_risk')}")
    if result.get("reasoning"):
        print(f"  thesis    : {result['reasoning'][:200]}...")

    # Force through even if analyst was cautious
    if not result.get("pass_to_risk"):
        print("\n  [OVERRIDE] Forcing pass_to_risk=True for test.")
        result["pass_to_risk"] = True
    if result.get("conviction", 0) < 5:
        print(f"  [OVERRIDE] Conviction {result.get('conviction')} too low -- raising to 7.")
        result["conviction"] = 7
    if result.get("action") not in ("OPEN", "ADD", "TRIM", "EXIT"):
        print(f"  [OVERRIDE] Action '{result.get('action')}' not actionable -- setting OPEN.")
        result["action"] = "OPEN"
    if not result.get("thesis_bucket"):
        result["thesis_bucket"] = "Moonshot"
    if not result.get("upside_downside_ratio") or result["upside_downside_ratio"] < 2.0:
        result["upside_downside_ratio"] = 2.5

    # ── 3. RISK MANAGER ──────────────────────────────────────────────────────
    _divider("STAGE 3 / 4 -- Risk Manager (position sizing)")

    # Get live price
    price = FAKE_PRICE
    if not price:
        try:
            from ingestion.polygon_stream import get_snapshot
            snap = get_snapshot(TICKER)
            price = snap.get("price")
            if price:
                print(f"  Live price (Polygon): ${price:.2f}")
        except Exception:
            pass
    if not price:
        try:
            from market_data import get_latest_quotes
            quotes = get_latest_quotes([TICKER])
            price = quotes.get(TICKER, {}).get("mid")
            if price:
                print(f"  Live price (IBKR):   ${price:.2f}")
        except Exception:
            pass
    if not price:
        price = 8.50
        print(f"  Could not fetch live price -- using fallback ${price:.2f}")

    from risk_manager import RiskManager
    client = _MockIBKR()
    rm     = RiskManager(client)
    sizing  = rm.size_position(
        TICKER,
        result["conviction"],
        result["thesis_bucket"],
        regime.get("regime", "Neutral"),
        price,
    )

    print(f"  shares    : {sizing.get('shares_initial')}")
    print(f"  size      : {sizing.get('position_size_pct', 0):.2%} of portfolio")
    print(f"  stop loss : ${sizing.get('stop_loss_price', 0):.2f}")
    print(f"  flags     : {sizing.get('flags', [])}")

    if not sizing.get("shares_initial") or sizing["shares_initial"] <= 0:
        print("  [OVERRIDE] 0 shares returned -- forcing 1 share for test.")
        sizing["shares_initial"]  = 1
        sizing["initial_dollar"]  = round(price, 2)
        sizing["position_size_pct"] = round(price / 100_000, 6)
        sizing["initial_entry_pct"] = 0.55
        sizing.setdefault("stop_loss_price", round(price * 0.65, 2))
        sizing.setdefault("flags", [])
        sizing["flags"].append("TEST_OVERRIDE_1_SHARE")

    entry_strategy = (
        f"{sizing.get('initial_entry_pct', 0.55):.0%} initial entry -- "
        f"{sizing['shares_initial']} share(s) @ ~${price:.2f}. "
        f"TEST RUN (paper trade only). Flags: {', '.join(sizing['flags']) or 'none'}."
    )

    # ── 4. TELEGRAM ──────────────────────────────────────────────────────────
    _divider("STAGE 4 / 4 -- DB log + Telegram approval request")

    import db
    import alerts

    reasoning = result.get("reasoning") or "TEST RUN -- IonQ DoD contract forced through pipeline for beta validation."

    approval_id = db.log_pending_approval(
        ticker        = TICKER,
        action        = result["action"],
        conviction    = result["conviction"],
        bucket        = result["thesis_bucket"],
        size_pct      = sizing["position_size_pct"],
        stop_price    = sizing.get("stop_loss_price"),
        entry_strategy= entry_strategy,
        reasoning     = f"[TEST RUN] {reasoning}",
        analyst_json  = result,
        risk_json     = sizing,
    )
    print(f"  Logged to DB -- approval ID: #{approval_id}")

    msg_id = alerts.send_approval_request(
        approval_id   = approval_id,
        ticker        = TICKER,
        action        = result["action"],
        conviction    = result["conviction"],
        bucket        = result["thesis_bucket"],
        size_pct      = sizing["position_size_pct"],
        stop_price    = sizing.get("stop_loss_price"),
        entry_strategy= entry_strategy,
        reasoning     = f"[TEST Run] {reasoning}",
    )

    if msg_id:
        db.set_approval_telegram_id(approval_id, msg_id)
        print(f"\n  Telegram message sent! (msg_id={msg_id})")
        print("  Check your phone -- tap Approve or Reject.")
    else:
        print("\n  Telegram not configured or message failed.")
        print(f"  Approval #{approval_id} is in the DB -- check Pending Approvals tab.")

    print("\n" + "="*55)
    print("  TEST COMPLETE")
    print("="*55)
    print()


if __name__ == "__main__":
    main()
