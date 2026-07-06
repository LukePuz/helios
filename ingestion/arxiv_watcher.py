"""
arXiv paper watcher — monitors quantum computing and AI research.
New breakthroughs here often precede stock moves in IONQ, RGTI, QBTS, etc.
by days or weeks. No API key required.

Relevant arXiv categories:
  quant-ph  — quantum physics / quantum computing
  cs.AI     — artificial intelligence
  cs.ET     — emerging technologies
  cond-mat  — condensed matter (relevant for quantum hardware)
"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"

SEARCH_QUERIES = [
    # Quantum computing breakthroughs
    'ti:"quantum advantage" OR ti:"quantum supremacy" OR ti:"quantum error correction"',
    'ti:"trapped ion" OR ti:"superconducting qubit" OR ti:"quantum annealing"',
    # Company-specific (papers often mention commercial platforms)
    'abs:IonQ OR abs:Rigetti OR abs:"D-Wave" OR abs:IBM Quantum',
    # AI/chip intersection
    'ti:"large language model" AND ti:"inference" AND ti:"hardware"',
    'ti:"neural processing unit" OR ti:"AI accelerator" OR ti:"CUDA"',
]

_SOURCE = "arxiv"


def _is_seen(paper_id: str) -> bool:
    try:
        import db
        return db.is_seen(_SOURCE, paper_id)
    except Exception:
        return False


def _mark_seen(paper_id: str) -> None:
    try:
        import db
        db.mark_seen(_SOURCE, paper_id)
    except Exception as e:
        log.debug(f"mark_seen failed for {paper_id}: {e}")


def _search(query: str, max_results: int = 10) -> list[dict]:
    """Queries arXiv API and returns new papers."""
    params = {
        "search_query": query,
        "start":        0,
        "max_results":  max_results,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }
    try:
        resp = requests.get(ARXIV_API, params=params, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"arXiv fetch failed: {e}")
        return []

    papers = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(resp.text)
        for entry in root.findall("atom:entry", ns):
            paper_id = entry.findtext("atom:id", namespaces=ns) or ""
            if _is_seen(paper_id):
                continue

            title   = (entry.findtext("atom:title",   namespaces=ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip().replace("\n", " ")
            updated = entry.findtext("atom:updated",  namespaces=ns) or ""

            authors = [
                a.findtext("atom:name", namespaces=ns) or ""
                for a in entry.findall("atom:author", namespaces=ns)
            ]

            _mark_seen(paper_id)
            papers.append({
                "source":    "arxiv",
                "paper_id":  paper_id,
                "title":     title,
                "summary":   summary[:2000],
                "authors":   authors[:5],
                "timestamp": updated,
                "url":       paper_id,
                "text":      f"TITLE: {title}\n\nABSTRACT: {summary}",
            })
    except ET.ParseError as e:
        log.warning(f"arXiv XML parse error: {e}")

    return papers


def poll_once() -> list[dict]:
    """Checks all search queries and returns new papers."""
    all_papers = []
    for query in SEARCH_QUERIES:
        papers = _search(query)
        all_papers.extend(papers)
        time.sleep(3)  # arXiv asks for 3 second delay between requests
    return all_papers


def run_forever(on_new_paper, poll_interval_seconds: int = 3600) -> None:
    """
    Polls arXiv every hour. Calls on_new_paper(paper_dict) for each new paper.
    New papers are published daily — hourly polling is more than sufficient.
    """
    log.info("arXiv watcher started.")
    while True:
        try:
            papers = poll_once()
            for paper in papers:
                log.info(f"New arXiv paper: {paper['title'][:80]}")
                try:
                    on_new_paper(paper)
                except Exception as e:
                    log.error(f"on_new_paper callback failed: {e}")
        except Exception as e:
            log.error(f"arXiv poll error: {e}")
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching recent quantum/AI papers from arXiv...\n")
    papers = poll_once()
    for p in papers[:5]:
        print(f"📄 {p['title']}")
        print(f"   {p['url']}")
        print(f"   {p['summary'][:200]}...\n")
