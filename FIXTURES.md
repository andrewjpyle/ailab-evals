# FIXTURES — provenance

Every fixture in this repository is **synthetic** and generated deterministically by
`scripts/build_fixtures.py` from content written in that file. No fixture is derived from any private,
customer, or production dataset, and no row describes a real business or person: all towns, shops,
products, and services are invented. Re-run `python scripts/build_fixtures.py` to regenerate them
byte-for-byte.

**How labels were set:** by construction. Each item is written to belong to a category whose label is
defined before any model sees it. The labels are not opinions about borderline items, and borderline
items are deliberately absent. That keeps the gold set defensible. It also makes these fixtures easier
than real data, so read the results as a demonstration of the harness, not a leaderboard.

| fixture | rows | label rule |
|---|---:|---|
| `substance/substance.jsonl` | 60 (40 scored + 20 calibration) | `yes` = page text with ≥ 3 concrete, checkable specifics (numbers, hours, prices, steps, part names). `no` = boilerplate, vague praise, or specific-*sounding* copy with no actual data ("prices vary, contact us"). |
| `qa_judge/qa_judge.jsonl` | 112 (16 questions × 7 candidate kinds) | `pass` = paraphrase of the reference, correct with extra true context, or a bare correct answer. `fail` = one quantity or fact changed, the wrong entity, an evasive non-answer, or a correct answer carrying one false side claim. Questions are stable, widely documented general-knowledge facts. |
| `qa_judge/pairwise.jsonl` | 16 questions × 3 systems | `concise` and `detailed` are correct; `confident_wrong` changes one fact. A sound judge should rank `confident_wrong` last. |
| `retrieval/corpus.jsonl` + `queries.jsonl` | 20 docs, 17 queries | Fictional ops-runbook snippets. Each gold doc id is the snippet that answers the query. `critical` marks the queries whose miss would matter most. |

`fixtures/cassettes/*.jsonl` hold recorded model outputs **for these synthetic prompts only**. Each line
contains the model id, the output text, token counts, and latency. They contain no credentials.
