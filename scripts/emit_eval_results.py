"""Emit eval_results.json (schema_version 1) from a replayed substance run, for the CI artifact.

Contract (read by the host console's CI-eval sync): {"schema_version": 1, "lab", "dataset",
"provider", "model", "primary_metric", "metrics": {name: number}, "threshold", "passed",
"commit", "generated_at"}. The selected model is the cheapest candidate that clears the suite's
pre-committed bar; `passed` is whether any candidate did.

    python scripts/emit_eval_results.py RESULTS_DIR SUITE.json OUT.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from ailab_evals.cli import baseline_candidates
from ailab_evals.datasets import Dataset
from ailab_evals.gate import cheapest_passing
from ailab_evals.runner import RunResult, ScoringConfig, run


def main(results_dir: str, suite_path: str, out_path: str) -> int:
    suite_file = Path(suite_path)
    suite = json.loads(suite_file.read_text())
    results = [RunResult.load(p) for p in sorted(Path(results_dir).glob("*.json"))]
    ds = Dataset.load_jsonl(suite_file.parent / suite["dataset"], suite["name"])
    cfg = ScoringConfig(labels=suite["labels"], **suite.get("scoring", {}))
    baselines = suite.get("baselines", [])
    for cand in baseline_candidates(ds, suite["labels"], cfg.calibrate_split):
        if cand.name in baselines:
            results.append(run(ds, cand, config=cfg))
    sel = cheapest_passing(results, suite["bar"], reference=suite.get("reference"), exclude=baselines)
    by_name = {r.candidate: r for r in results}
    chosen = by_name.get(sel.winner) or by_name.get(suite.get("reference"))
    spec = {c["name"]: c for c in suite["candidates"]}.get(chosen.candidate, {})
    ref = by_name.get(suite.get("reference"))
    rule = suite["bar"].get("macro_f1", {})
    threshold = (ref.metrics["macro_f1"] + rule.get("min_vs_reference", 0)) if ref else rule.get("min", 0)
    # The host parser requires primary_metric to be a key of metrics: the selected model's score.
    metrics = {"macro_f1": float(chosen.metrics["macro_f1"])}
    metrics |= {f"{r.candidate}.macro_f1": r.metrics["macro_f1"] for r in results
               if isinstance(r.metrics.get("macro_f1"), (int, float))}
    metrics |= {f"{r.candidate}.cost_usd_per_1k": r.metrics["cost_usd_per_1k"] for r in results}
    payload = {
        "schema_version": 1,
        "lab": "ailab-evals",
        "dataset": f"{suite['name']} (synthetic fixture, {len(ds)} rows)",
        "provider": spec.get("provider", ""),
        "model": spec.get("model", chosen.candidate),
        "primary_metric": "macro_f1",
        "metrics": metrics,
        "threshold": round(float(threshold), 4),
        "passed": sel.winner is not None,
        "commit": os.environ.get("GITHUB_SHA"),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("model", "passed", "threshold")}))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
