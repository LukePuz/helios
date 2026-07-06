"""
Unit tests for ingestion/insider.py's Form 4 XML parsing — network is
mocked, so this verifies the parsing/filtering logic (open-market BUY
detection) without hitting SEC EDGAR.
"""
from unittest.mock import MagicMock, patch

from ingestion import insider

_PURCHASE_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>JENSEN HUANG</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <isDirector>1</isDirector>
      <officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.50</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

_GRANT_ONLY_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>JANE DOE</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <isDirector>0</isDirector>
      <officerTitle>CFO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def _mock_response(xml_text: str):
    resp = MagicMock()
    resp.content = xml_text.encode()
    resp.raise_for_status = lambda: None
    return resp


class TestForm4Parsing:
    def test_open_market_purchase_is_extracted(self):
        with patch("requests.get", return_value=_mock_response(_PURCHASE_XML)):
            result = insider._parse_form4_transactions("http://fake-url")

        assert result is not None
        assert result["owner_name"] == "JENSEN HUANG"
        assert result["is_officer"] is True
        assert result["is_director"] is True
        assert result["total_shares_bought"] == 1000.0
        assert result["total_value_bought"] == 150_500.0

    def test_grant_only_filing_is_not_a_buy_signal(self):
        """A restricted-stock grant (code 'A') is not an open-market purchase
        and must not be surfaced as a bullish insider-buy signal."""
        with patch("requests.get", return_value=_mock_response(_GRANT_ONLY_XML)):
            result = insider._parse_form4_transactions("http://fake-url")

        assert result is None

    def test_xsl_viewer_path_is_stripped_to_raw_xml_url(self):
        """
        The filing index page links to the XSLT-rendered HTML viewer path
        (.../xslF345X06/wk-form4_1.xml), which returns human-readable HTML,
        not the machine-readable XML — that segment must be stripped or
        every single filing silently fails to parse (this was a real bug
        caught by testing against a live SEC filing).
        """
        index_html = (
            '<html><body>'
            '<a href="/Archives/edgar/data/123/000123/xslF345X06/wk-form4_1.xml">form4</a>'
            '</body></html>'
        )
        resp = MagicMock()
        resp.text = index_html
        resp.raise_for_status = lambda: None

        with patch("requests.get", return_value=resp):
            url = insider._find_primary_doc_url("http://fake-index-url")

        assert url is not None
        assert "xslF345X06" not in url
        assert url.endswith("/Archives/edgar/data/123/000123/wk-form4_1.xml")
