import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

print("--- Fix 1: vector_store import ---")
try:
    import vector_store
    print("  OK: vector_store imported cleanly")
except Exception as e:
    print("  FAIL:", e)

print()
print("--- Fix 2: edgar date filter ---")
try:
    from ingestion.edgar import _MAX_AGE_HOURS
    print("  OK: date filter =", _MAX_AGE_HOURS, "hours")
except Exception as e:
    print("  FAIL:", e)

print()
print("--- Fix 3: yfinance price fallback ---")
try:
    import yfinance as yf
    t = yf.Ticker("NVDA")
    info = t.fast_info
    price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
    print("  OK: NVDA price =", price)
except Exception as e:
    print("  FAIL:", e)

print()
print("--- Fix 4: env model config ---")
print("  ANALYST_MODEL  =", os.getenv("ANALYST_MODEL", "NOT SET"))
print("  SCREENER_MODEL =", os.getenv("SCREENER_MODEL", "NOT SET"))
print("  IB_CLIENT_ID   =", os.getenv("IB_CLIENT_ID", "NOT SET"))

print()
print("--- Fix 5: chat model (3b) reachable ---")
# Was posting to ANALYST_MODEL (72B) with a 15s timeout despite the "3b
# model" label — a 72B cold load routinely takes well over 15s, so this
# always failed and told you nothing about whether the actual chat model
# was reachable.
try:
    import requests
    resp = requests.post("http://localhost:11434/api/generate",
        json={"model": os.getenv("CHAT_MODEL","qwen2.5:3b"),
              "prompt": "Say OK", "stream": False}, timeout=60)
    if resp.status_code == 200:
        print("  OK: model responded:", resp.json().get("response","")[:40])
    else:
        print("  FAIL: status", resp.status_code)
except Exception as e:
    print("  FAIL:", e)

if "--full" in sys.argv:
    print()
    print("--- Fix 6: analyst model (72b) reachable ---")
    try:
        import requests
        resp = requests.post("http://localhost:11434/api/generate",
            json={"model": os.getenv("ANALYST_MODEL","qwen2.5:72b"),
                  "prompt": "Say OK", "stream": False}, timeout=300)
        if resp.status_code == 200:
            print("  OK: model responded:", resp.json().get("response","")[:40])
        else:
            print("  FAIL: status", resp.status_code)
    except Exception as e:
        print("  FAIL:", e)
