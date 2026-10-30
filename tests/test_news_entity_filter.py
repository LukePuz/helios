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


def test_short_names_and_other_casing_still_count():
    """Review 4 #1: the exact-case full-name match dropped real news."""
    from ingestion.news import mentions_company as m
    assert m("SNDK", "SanDisk Corporation", "Sandisk Drops 23% From 52-Week High")
    assert m("QBTS", "D-Wave Quantum Inc.", "Lawsuit alleges D-Wave misled investors")
    assert m("GOOG", "Alphabet Inc.", "Google faces EU fine")
    assert m("GOOG", "Alphabet Inc.", "GOOGL slides after earnings")
    assert m("MSFT", "Microsoft Corporation", "MICROSOFT CUTS 5,000 JOBS")
    assert m("SMR", "NuScale Power Corporation", "NuScale Stock Faces Contract Doubts")
    assert m("DAVE", "Dave Inc.", "Is DAVE Stock Still Worth Buying")
    assert m("KLIC", "Kulicke and Soffa Industries", "Kulicke & Soffa beats estimates")


def test_ordinary_word_names_and_tickers_still_filtered():
    from ingestion.news import mentions_company as m
    assert not m("DRAM", "Roundhill Memory ETF", "DRAM prices surge on AI demand")
    assert not m("QUBT", "Quantum Computing Inc.", "3 Quantum Computing Stocks to Buy Now")
    assert m("QUBT", "Quantum Computing Inc.", "QUBT jumps 12%")
    assert not m("AUR", "Aurora Innovation", "aurora borealis visible tonight")


def test_aliases_that_name_other_companies_are_gone():
    """Review 5 #6."""
    from ingestion.news import mentions_company as m
    assert not m("CEG", "Constellation Energy Corporation", "Constellation Brands plunges on beer sales")
    assert m("CEG", "Constellation Energy Corporation", "Constellation Energy signs nuclear deal")
    assert not m("PANW", "Palo Alto Networks, Inc.", "Palo Alto startup raises $50M")
    assert not m("NOW", "ServiceNow, Inc.", "BUY NOW STOCK BEFORE IT'S TOO LATE")
    assert not m("DRAM", "Roundhill Memory ETF", "DRAM shares of Samsung's output rise")
