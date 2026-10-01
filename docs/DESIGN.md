# DESIGN — tradeoffs considered, chosen, and rejected

## Zero runtime dependencies
**Chosen:** stdlib only (urllib for HTTP, `random`/`math`/`statistics` for metrics).
**Why:** the harness is meant to be pinned by commit SHA inside larger applications. Any runtime dependency
becomes a lock-file negotiation with the host. NumPy/SciPy would make bootstrap and Platt faster, but eval sets are
hundreds of rows, not millions, so the speed gain doesn't matter.
**Rejected:** `openai`/`anthropic` SDKs as hard deps. A host passes its own provider callable instead, which also
means its keys, retries and spend caps never pass through this package.

## Candidates, not models
**Chosen:** anything with `predict(example)` or `predict_batch(examples)` is a candidate. That covers a prompted LLM,
a zero-shot encoder, a cached prediction table, or a host-side client.
**Rejected:** a model-centric API (`run(model_name, ...)`). It can't express a non-LLM baseline or a batched
encoder, and real bake-offs need both.

## Failures are scored, never skipped
An error, timeout, unparseable reply, or off-menu label is a wrong answer that stays in the denominator. Skipping
it would reward a model for failing on exactly the hard items.

## Bars are data, written before the run
**Chosen:** a bar is a JSON object (`{"macro_f1": {"min_vs_reference": -0.03}}`) committed next to the suite.
`cheapest_passing` sorts by cost and returns the first candidate that clears it. Baselines are excluded from
winning. A missing or `INSUFFICIENT` metric fails the bar.
**Rejected:** a composite "quality-per-dollar" score. It hides the trade-off in a weighting nobody agreed to.

## Record / replay
**Chosen:** a JSONL cassette keyed by `sha256(provider, model, params, system, prompt, json_mode)`. Replay mode
treats a miss as an error.
**Why:** CI must be hermetic and free. Keying on the full request means a prompt edit invalidates the recording
rather than silently reusing stale outputs. Latency in replay is the recorded latency, so the table is
reproducible.
**Rejected:** mocking providers in CI. A mock proves the plumbing, but the README table must come from real model
outputs.

## Judge calibration before judge use
A judge is a model with its own error rate. `calibrate()` treats judge errors as disagreement. It reports false
pass and false fail separately, and refuses `trusted` on a high false-pass rate even when kappa looks fine,
because a lenient judge in a quality gate ships bad output.

## Pairwise: position swap, then Bradley–Terry
Each comparison is asked twice with the order swapped. Only an order-invariant winner counts. Ties feed
Bradley–Terry as half-wins. Elo was rejected because it is order-dependent over a fixed batch of games.

## Privacy wall
This repo holds only the interface and synthetic fixtures. Applications keep their adapters (data access,
filtering, credentials) in their own codebase. Enforcement:
- a pre-push hook and a required CI job run gitleaks over full history, plus a deny-list;
- the deny-list is stored as sha256 hashes of private terms, so publishing it discloses nothing, and a plaintext
  extension can live outside the repo (`AILAB_DENYLIST_FILE`).
