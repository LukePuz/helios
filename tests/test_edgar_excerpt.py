"""EDGAR events carry a source and an excerpt that starts with the news, not
the SEC cover page (limitations audit 2026-10-28 05:00, finding 2)."""
from unittest.mock import MagicMock, patch

from ingestion import edgar

_INDEX = """<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/x/main8k.htm">main8k.htm</a></td><td>8-K</td><td>1</td></tr>
<tr><td>2</td><td>EX-99.1</td><td><a href="/Archives/x/pr.htm">pr.htm</a></td><td>EX-99.1</td><td>1</td></tr>
<tr><td></td><td>Complete submission text file</td><td><a href="/Archives/x/all.txt">all.txt</a></td><td></td><td>1</td></tr>
</table>"""
_MAIN = ("<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p><p>Delaware 001-1 Palo Alto</p>"
         "<p>Item 2.02 Results of Operations.</p><p>See Exhibit 99.1.</p>"
         "<p>Item 9.01 Financial Statements and Exhibits.</p>")
_PR = "<p>ACME Announces Record Quarter</p><p>Revenue up 40%.</p>"


def _get(url, **kw):
    body = {"https://www.sec.gov/Archives/x/main8k.htm": _MAIN,
            "https://www.sec.gov/Archives/x/pr.htm": _PR}.get(url, _INDEX)
    resp = MagicMock(text=body)
    resp.raise_for_status.return_value = None
    return resp


def test_8k_excerpt_leads_with_items_and_press_release():
    with patch("ingestion.edgar.requests.get", side_effect=_get):
        text = edgar._fetch_filing_text("https://www.sec.gov/Archives/x/idx-index.htm", "8-K")
    assert text.startswith("8-K items: 2.02 (results of operations), 9.01 (exhibits)")
    assert "ACME Announces Record Quarter" in text[:500]
    assert "Palo Alto" not in text
    assert "Filing body:\nItem 2.02" in text


def test_documents_strip_the_ixbrl_viewer_prefix_and_keep_types():
    with patch("ingestion.edgar.requests.get", side_effect=_get):
        docs = edgar._filing_documents("https://www.sec.gov/Archives/x/idx-index.htm")
    assert docs == [("8-K", "https://www.sec.gov/Archives/x/main8k.htm"),
                    ("EX-99.1", "https://www.sec.gov/Archives/x/pr.htm")]


def test_10q_skips_to_the_mdna_section_not_the_table_of_contents():
    text = ("Cover ... Contents: Item 2. Management's Discussion and Analysis ... 12\n"
            "Part I ... Item 2. Management’s Discussion and Analysis\nRevenue grew 30%.")
    out = edgar._skip_cover(text, "10-Q")
    assert out.startswith("Management’s Discussion and Analysis\nRevenue grew 30%.")


def test_cover_kept_when_no_marker():
    assert edgar._skip_cover("just text", "8-K") == "just text"
