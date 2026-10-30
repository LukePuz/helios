"""Luke D8 (10-29): a Google News item for a ticker must name the company."""
from ingestion.news import _company_core, mentions_company


def test_company_core_strips_suffixes():
    assert _company_core("Arm Holdings") == "Arm"
    assert _company_core("Dave Inc") == "Dave"
    assert _company_core("Advanced Micro Devices") == "Advanced Micro Devices"


def test_wordlike_tickers_need_the_company_or_cashtag():
    assert not mentions_company("NOW", "ServiceNow", "Is Now the Right Time to Buy the Dip?")
    assert not mentions_company("NOW", "ServiceNow", "Should You Buy Microsoft Stock NOW")
    assert mentions_company("NOW", "ServiceNow", "ServiceNow Stock Plunges 18%")
    assert mentions_company("NOW", "ServiceNow", "Why $NOW fell today")
    assert mentions_company("V", "Visa", "Visa (V) Pushes Further Into Payments")
    assert not mentions_company("V", "Visa", "Apple Eyes an Apple Pay Launch in India")


def test_plain_tickers_and_case():
    assert mentions_company("AMD", "Advanced Micro Devices", "AMD Just Hit $1 Trillion")
    assert not mentions_company("NVDA", "Nvidia", "Intel CEO Has News for AMD Stock Investors")
    assert not mentions_company("ARM", "Arm Holdings", "an arms race in AI chips")
    assert mentions_company("ARM", "Arm Holdings", "Arm's royalty rates rise")
