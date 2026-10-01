# MODEL CARD — ailab-evals

This repository trains no model. It is a **measurement instrument for models**. This card covers what the instrument
measures, how far its numbers can be trusted, and where they mislead.

## Intended use
- Comparing candidate models (LLMs, encoders, baselines) on a **fixed, hashed, labeled** dataset, and selecting the
  cheapest one that clears a bar written down before the run.
- Calibrating an LLM judge against gold labels before trusting it as a grader.
- Gating changes in CI with a regression check over replayed results.

## Not intended for
- Leaderboards. The bundled fixtures are synthetic and labeled by construction, so they are deliberately easy, and
  several models saturate them (see README). They exist to exercise the harness and show its outputs.
- Measuring anything about a production workload. That requires the host application's own labeled data, scored
  through an adapter that lives in the host (never in this public repo).

## Models exercised in the committed results
| name | provider | priced at (USD / 1M tokens, in / out) |
|---|---|---|
| claude-sonnet-4.5 | Anthropic API | 3.00 / 15.00 |
| claude-haiku-4.5 | Anthropic API | 1.00 / 5.00 |
| qwen3-32b-local | Ollama, run locally (thinking disabled) | 0 / 0 marginal API cost (hardware and power not priced) |
| gemma3-27b-local | Ollama, run locally | 0 / 0 marginal API cost (hardware and power not priced) |

All outputs were recorded on 2026-10-01 at temperature 0 and are replayed from `fixtures/cassettes/`. Latency columns
are the recorded latencies: local-model latency depends on the host hardware, and API latency depends on load at
recording time.

## Known limitations
- **Labels by construction stand in for human labels.** The harness's judge calibration is designed for human labels.
  The public fixtures use labels fixed by how each item was built (see `FIXTURES.md`), which is cleaner than real
  human labeling and has no inter-annotator noise.
- **Small n.** 40 scored rows per suite means wide confidence intervals; the README prints them.
- **Cost excludes local compute.** A $0 local model is not free to run; it moves the cost onto hardware.
- **Verbalized confidence.** Confidence comes from the model's self-reported number, not logprobs. ECE measures how
  honest that number is; it is not a property of the underlying distribution.
