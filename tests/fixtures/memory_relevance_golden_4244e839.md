# Memory relevance readout (live, verdicts 2026-11-20 to before 2026-11-27 ET)

Generated 2026-11-27 01:00 ET. Code version: pinned0. Engineering readout of the stored prompts' memory sections; counts, scores and collection names only. On company: the ingestion.news.mentions_company rule on the item text. No score floor, ticker filter or relevance label is applied (Luke's).

- verdicts: 5; with memory items: 2; items: 8
- without items: prompt NULL 1, empty memory 1, no memory section 1
- below 0: 4 (50.0% of items); on company: 4 (50.0% of items)
- verdicts whose ticker has no watchlist name (ticker forms only): 0

## By held (provenance.plan.held)

| held | verdicts | with items | items | score: median [p25, p75], min..max | below 0 | below 0 % of items | on company | on company % of items | per-verdict below-0 frac, median | per-verdict on-company frac, median | per-verdict min score, median | collections | FILINGS forms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held | 3 | 1 | 4 | 0.06 [-0.02, 0.19], -0.08..0.41 | 2 | 50.0 | 2 | 50.0 | 0.5 | 0.5 | -0.08 | NEWS 1, FILINGS 1, ARXIV 1, OPTIONS 1 | 8-K 1 |
| not held | 1 | 1 | 4 | 0.03 [-0.16, 0.28], -0.21..0.5 | 2 | 50.0 | 2 | 50.0 | 0.5 | 0.5 | -0.21 | NEWS 1, MACRO 1, FILINGS 1, INSIDER 1 | 10-Q 1 |
| unknown | 1 | 0 | 0 | - | 0 | - | 0 | - | - | - | - | - | - |

## By day (New York date of the verdict)

| day | verdicts | with items | items | score: median [p25, p75], min..max | below 0 | below 0 % of items | on company | on company % of items | per-verdict below-0 frac, median | per-verdict on-company frac, median | per-verdict min score, median | collections | FILINGS forms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-25 | 1 | 1 | 4 | 0.06 [-0.02, 0.19], -0.08..0.41 | 2 | 50.0 | 2 | 50.0 | 0.5 | 0.5 | -0.08 | NEWS 1, FILINGS 1, ARXIV 1, OPTIONS 1 | 8-K 1 |
| 2026-11-26 | 4 | 1 | 4 | 0.03 [-0.16, 0.28], -0.21..0.5 | 2 | 50.0 | 2 | 50.0 | 0.5 | 0.5 | -0.21 | NEWS 1, MACRO 1, FILINGS 1, INSIDER 1 | 10-Q 1 |

## Held x day

| held / day | verdicts | with items | items | score: median [p25, p75], min..max | below 0 | below 0 % of items | on company | on company % of items | per-verdict below-0 frac, median | per-verdict on-company frac, median | per-verdict min score, median | collections | FILINGS forms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held / 2026-11-25 | 1 | 1 | 4 | 0.06 [-0.02, 0.19], -0.08..0.41 | 2 | 50.0 | 2 | 50.0 | 0.5 | 0.5 | -0.08 | NEWS 1, FILINGS 1, ARXIV 1, OPTIONS 1 | 8-K 1 |
| held / 2026-11-26 | 2 | 0 | 0 | - | 0 | - | 0 | - | - | - | - | - | - |
| not held / 2026-11-26 | 1 | 1 | 4 | 0.03 [-0.16, 0.28], -0.21..0.5 | 2 | 50.0 | 2 | 50.0 | 0.5 | 0.5 | -0.21 | NEWS 1, MACRO 1, FILINGS 1, INSIDER 1 | 10-Q 1 |
| unknown / 2026-11-26 | 1 | 0 | 0 | - | 0 | - | 0 | - | - | - | - | - | - |

## Collections, all verdicts

| collection | items | below 0 | below 0 % | on company | on company % | off company | score: median [p25, p75], min..max |
|---|---|---|---|---|---|---|---|
| FILINGS | 2 | 0 | 0.0 | 1 | 50.0 | 1 | 0.16 [0.14, 0.18], 0.12..0.2 |
| NEWS | 2 | 0 | 0.0 | 2 | 100.0 | 0 | 0.45 [0.43, 0.48], 0.41..0.5 |
| ARXIV | 1 | 1 | 100.0 | 0 | 0.0 | 1 | -0.08 [-0.08, -0.08], -0.08..-0.08 |
| INSIDER | 1 | 1 | 100.0 | 0 | 0.0 | 1 | -0.21 [-0.21, -0.21], -0.21..-0.21 |
| MACRO | 1 | 1 | 100.0 | 0 | 0.0 | 1 | -0.14 [-0.14, -0.14], -0.14..-0.14 |
| OPTIONS | 1 | 1 | 100.0 | 1 | 100.0 | 0 | -0.0 [-0.0, -0.0], -0.0..-0.0 |

## Collections, held verdicts

| collection | items | below 0 | below 0 % | on company | on company % | off company | score: median [p25, p75], min..max |
|---|---|---|---|---|---|---|---|
| ARXIV | 1 | 1 | 100.0 | 0 | 0.0 | 1 | -0.08 [-0.08, -0.08], -0.08..-0.08 |
| FILINGS | 1 | 0 | 0.0 | 0 | 0.0 | 1 | 0.12 [0.12, 0.12], 0.12..0.12 |
| NEWS | 1 | 0 | 0.0 | 1 | 100.0 | 0 | 0.41 [0.41, 0.41], 0.41..0.41 |
| OPTIONS | 1 | 1 | 100.0 | 1 | 100.0 | 0 | -0.0 [-0.0, -0.0], -0.0..-0.0 |

## Collections, not held verdicts

| collection | items | below 0 | below 0 % | on company | on company % | off company | score: median [p25, p75], min..max |
|---|---|---|---|---|---|---|---|
| FILINGS | 1 | 0 | 0.0 | 1 | 100.0 | 0 | 0.2 [0.2, 0.2], 0.2..0.2 |
| INSIDER | 1 | 1 | 100.0 | 0 | 0.0 | 1 | -0.21 [-0.21, -0.21], -0.21..-0.21 |
| MACRO | 1 | 1 | 100.0 | 0 | 0.0 | 1 | -0.14 [-0.14, -0.14], -0.14..-0.14 |
| NEWS | 1 | 0 | 0.0 | 1 | 100.0 | 0 | 0.5 [0.5, 0.5], 0.5..0.5 |

## Collections, unknown verdicts

| collection | items | below 0 | below 0 % | on company | on company % | off company | score: median [p25, p75], min..max |
|---|---|---|---|---|---|---|---|

## By trigger path

| path | verdicts | with items | items | score: median [p25, p75], min..max | below 0 | below 0 % of items | on company | on company % of items | per-verdict below-0 frac, median | per-verdict on-company frac, median | per-verdict min score, median | collections | FILINGS forms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event | 3 | 1 | 4 | 0.06 [-0.02, 0.19], -0.08..0.41 | 2 | 50.0 | 2 | 50.0 | 0.5 | 0.5 | -0.08 | NEWS 1, FILINGS 1, ARXIV 1, OPTIONS 1 | 8-K 1 |
| scan | 1 | 0 | 0 | - | 0 | - | 0 | - | - | - | - | - | - |
| scheduled_review | 1 | 1 | 4 | 0.03 [-0.16, 0.28], -0.21..0.5 | 2 | 50.0 | 2 | 50.0 | 0.5 | 0.5 | -0.21 | NEWS 1, MACRO 1, FILINGS 1, INSIDER 1 | 10-Q 1 |

## By ticker

| ticker | verdicts | with items | items | score: median [p25, p75], min..max | below 0 | below 0 % of items | on company | on company % of items | per-verdict below-0 frac, median | per-verdict on-company frac, median | per-verdict min score, median | collections | FILINGS forms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ANET | 3 | 1 | 4 | 0.06 [-0.02, 0.19], -0.08..0.41 | 2 | 50.0 | 2 | 50.0 | 0.5 | 0.5 | -0.08 | NEWS 1, FILINGS 1, ARXIV 1, OPTIONS 1 | 8-K 1 |
| NVDA | 2 | 1 | 4 | 0.03 [-0.16, 0.28], -0.21..0.5 | 2 | 50.0 | 2 | 50.0 | 0.5 | 0.5 | -0.21 | NEWS 1, MACRO 1, FILINGS 1, INSIDER 1 | 10-Q 1 |
