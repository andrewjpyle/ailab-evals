# ailab-evals

An evals harness for LLM decisions. It answers one question that moves money: **which is the cheapest model that is
good enough?** It also refuses to give a fake answer when the data can't support one.

- **Datasets:** JSONL, sha256-hashed, so every result names the exact data it measured.
- **Runners:** any `predict` / `predict_batch` candidate, whether a prompted LLM, an encoder, a baseline or a cached
  table. Errors are scored as wrong, never skipped.
- **Metrics:** accuracy, macro-F1 with bootstrap CIs, and per-class floors that print `INSUFFICIENT` below them.
  ECE, Platt scaling on a held-out split, precision-at-threshold, Cohen's κ, recall@k, MRR and nDCG.
- **LLM-as-judge:** pointwise grading, plus **calibration against gold labels**. False-pass and false-fail rates are
  reported separately. A judge is `trusted` only if both κ and false-pass clear their bars.
- **Pairwise mode:** A/B position-swap debiasing, win rates and Bradley–Terry.
- **Regression-gate CLI:** pre-committed bars (absolute or relative to a reference model), `cheapest_passing`, and
  `ailab-evals gate` (exit 1 on a regression or a changed dataset).
- **Record / replay cassettes:** CI re-scores real recorded model outputs offline, so it is hermetic and free.

Zero runtime dependencies (stdlib only), so a host application can pin it by commit SHA without lock-file conflicts.

## Results on the public fixtures

Recorded 2026-10-01 against real models (temperature 0), replayed and re-verified in CI on every push. The fixtures
are synthetic and labeled by construction (see [`FIXTURES.md`](FIXTURES.md)). Costs use list prices. `-local`
models ran on local hardware via Ollama, so their API cost is $0 but compute is not free.

### 1. Judge calibration: can each model be trusted to grade answers?
112 graded answers (16 questions × 7 answer kinds; 64 of them deliberately wrong).

| judge | n | agreement | Cohen's κ | false-pass | false-fail | trusted |
|---|---:|---:|---:|---:|---:|---|
| claude-sonnet-4.5 | 112 | 0.973 | 0.946 | 0.047 | 0.000 | yes |
| claude-haiku-4.5 | 112 | 0.982 | 0.964 | 0.031 | 0.000 | yes |
| qwen3-32b-local | 112 | 0.964 | 0.927 | 0.047 | 0.021 | yes |
| gemma3-27b-local | 112 | 0.920 | 0.840 | 0.141 | 0.000 | **NO** |

**What it shows:**
- The easy answer kinds were graded perfectly by every judge: paraphrases, wrong numbers, wrong entities, evasions.
- **Every judge error landed on the two trap kinds**, and 17 of 18 errors on the same one: a *correct answer that
  also carries one false side claim* ("32 °F, which equals 10 °C"). Judges tend to pass an answer once its headline
  is right.
- gemma3-27b passed 9 of 16 such answers. Its κ of 0.84 looks respectable. Its 14% false-pass rate is what
  disqualifies it as a grader, which is why the harness reports false-pass on its own line.

### 2. Pairwise preference: position-swapped, with claude-haiku-4.5 as judge
Each pair is judged twice, with A/B swapped. A winner has to win in both orders, otherwise the pair counts as a tie.

| system | games | wins | losses | ties | win rate | Bradley-Terry |
|---|---:|---:|---:|---:|---:|---:|
| detailed | 32 | 17 | 0 | 15 | 0.766 | 1159.7 |
| concise | 32 | 14 | 7 | 11 | 0.609 | 1068.7 |
| confident_wrong | 32 | 0 | 24 | 8 | 0.125 | 771.7 |

- The wrong system never wins.
- The judge was position-consistent on **70.8%** of comparisons. Without the swap, roughly 3 in 10 verdicts would
  have reflected answer order rather than content.
- Haiku also prefers the longer of two equally correct answers (`detailed` over `concise`): verbosity bias, now
  measured.

### 3. Cheapest model that passes: binary "does this page have substance?" classification
The bar is written before the run: macro-F1 no more than 3 points below the reference (Sonnet). There are 40
scored rows, and 20 more held out for calibration.

| candidate | n | accuracy | macro-F1 | 95% CI | ECE | error rate | $ / 1k | p50 ms | bar |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| majority | 40 | 0.500 | 0.333 | [0.26, 0.39] | 0.500 | 0.000 | 0.0000 | 0 | baseline |
| random | 40 | 0.575 | 0.575 | [0.42, 0.72] | 0.075 | 0.000 | 0.0000 | 0 | baseline |
| gemma3-27b-local | 40 | 1.000 | 1.000 | [1.00, 1.00] | 0.053 | 0.000 | 0.0000 | 4942 | **PASS** ✅ cheapest passing |
| qwen3-32b-local | 40 | 1.000 | 1.000 | [1.00, 1.00] | 0.060 | 0.000 | 0.0000 | 3617 | **PASS** |
| claude-haiku-4.5 | 40 | 1.000 | 1.000 | [1.00, 1.00] | 0.039 | 0.000 | 0.7701 | 1571 | **PASS** |
| claude-sonnet-4.5 | 40 | 1.000 | 1.000 | [1.00, 1.00] | 0.035 | 0.000 | 1.8878 | 3055 | reference |

**Cheapest model that passes the bar: `gemma3-27b-local`.**

Honest reading:
- This fixture is **too easy to separate the models**. All four saturate it, so selection falls to cost. What it
  demonstrates is the mechanism: the baselines, the bar relative to a reference, the cost column, and the named
  pick.
- The same model that wins as a *classifier* here fails as a *judge* in table 1. "Good enough" is per task, which
  is the reason to measure each one.

### 4. Retrieval baseline: BM25 over a 20-document runbook corpus

| retriever | n | recall@3 | hit@3 | MRR | nDCG@3 | critical recall@3 |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 17 | 0.941 | 1.000 | 1.000 | 0.954 | 0.889 |

Recall on the `critical` subset trails overall recall, so the misses that matter are reported on their own line.

## Quick start

```bash
uv sync
make demo        # replays every suite from the committed cassettes: no network, no API key
make test
```

Record against live models (needs `ANTHROPIC_API_KEY` and/or a local Ollama):

```bash
uv run ailab-evals run suites/substance.json --mode record
```

Gate a change against a stored baseline (exits 1 on a regression or a changed dataset):

```bash
uv run ailab-evals gate new/claude-haiku-4.5.json results/substance/claude-haiku-4.5.json \
  --tolerances suites/tolerances.json
```

Use it as a library with your own data and model client:

```python
from ailab_evals.datasets import Dataset
from ailab_evals.gate import cheapest_passing
from ailab_evals.runner import FunctionCandidate, ScoringConfig, run
from ailab_evals.types import Prediction

ds = Dataset.load_jsonl("my_labeled_set.jsonl")
cfg = ScoringConfig(labels=["yes", "no"], floor_per_class=10, calibrate_split="calibrate")
results = [run(ds, FunctionCandidate(name, fn), config=cfg) for name, fn in my_candidates.items()]
pick = cheapest_passing(results, {"macro_f1": {"min_vs_reference": -0.03}}, reference="current-model")
print(pick.winner)
```

## Layout

```
src/ailab_evals/   datasets · providers (Ollama, Anthropic, replay) · runner · metrics · judge · gate · retrieval · report · cli
suites/            suite configs (dataset, prompt, candidates + prices, bar)
fixtures/          synthetic datasets + recorded cassettes      (FIXTURES.md = provenance)
results/           committed results; CI re-derives these and diffs them
docs/              DESIGN.md (tradeoffs) · LEARNING.md (concepts + 10 interview questions)
```

## Privacy wall

This repo holds an interface and synthetic fixtures only. Every push runs:
- **gitleaks** over the full history, both as a pre-push hook (`make hooks`) and as a required CI job;
- a **deny-list**: private terms stored as sha256 hashes, plus private-network, secret-token and email patterns.

Applications keep their data adapters in their own codebases.

License: MIT.
