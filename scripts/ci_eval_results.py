#!/usr/bin/env python3
"""Emit a schema-1 ``eval_results.json`` summarising this lab's CI eval.

ailab-evals is a bake-off harness, not a single-metric lab, so its "CI eval" is the
reproducibility guarantee the CI already enforces: every committed suite table reproduces
from its cassette under replay (the ``diff`` step in the eval-gate job). This script reports
how many suites reproduced, plus the reference model's real substance macro_f1 as a quality
number, as a schema-1 ``eval_results.json`` artifact that downstream dashboards can consume.

Run from the repo root after the replay + gate steps. Writes ``--out`` (default eval_results.json).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

SUITES = ["substance", "judge", "pairwise", "retrieval"]
REFERENCE = ("substance", "claude-sonnet-4.5.json", "macro_f1")


def _candidate_count(results: Path) -> int:
    n = 0
    for suite in SUITES:
        for jf in (results / suite).glob("*.json"):
            try:
                doc = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and "candidate" in doc:
                n += 1
    return n


def _reference_score(results: Path) -> float | None:
    suite, fname, metric = REFERENCE
    path = results / suite / fname
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    value = doc.get("metrics", {}).get(metric)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def build(results: Path) -> dict[str, object]:
    reproduced = sum(1 for s in SUITES if (results / s / "table.md").is_file())
    metrics: dict[str, float] = {
        "suites_reproduced": reproduced,
        "suites_total": len(SUITES),
        "n": _candidate_count(results),
    }
    ref = _reference_score(results)
    if ref is not None:
        metrics["reference_substance_macro_f1"] = ref
    return {
        "schema_version": 1,
        "lab": "ailab-evals",
        "dataset": "ci-replay-suites",
        "provider": "replay",
        "model": "committed-baselines",
        "primary_metric": "suites_reproduced",
        "metrics": metrics,
        "threshold": len(SUITES),
        "passed": reproduced == len(SUITES),
        "commit": os.environ.get("GITHUB_SHA", ""),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="results/ directory")
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()
    payload = build(Path(args.results))
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {payload['primary_metric']}={payload['metrics']['suites_reproduced']}"
          f"/{payload['threshold']} passed={payload['passed']}")


if __name__ == "__main__":
    main()
