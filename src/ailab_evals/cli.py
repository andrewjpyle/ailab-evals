"""``ailab-evals`` command line.

    ailab-evals run SUITE.json [--mode replay|record|live] [--only a,b] [--out DIR]
    ailab-evals judge SUITE.json [--mode ...]          # judge calibration vs human labels
    ailab-evals pairwise SUITE.json [--mode ...]       # position-swapped pairwise + Bradley-Terry
    ailab-evals retrieval SUITE.json                   # BM25 baseline over a corpus
    ailab-evals gate CURRENT.json BASELINE.json --tolerances TOL.json   # exit 1 on regression

Suites are JSON files (see ``suites/``). ``--mode replay`` (the default) serves model calls from
the suite's cassette and fails on a miss, so CI is hermetic and costs nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import judge as judge_mod
from .datasets import Dataset
from .gate import cheapest_passing, regression_gate
from .providers import ProviderRegistry, ReplayProvider
from .report import classification_table, retrieval_table
from .retrieval import BM25, RetrieverCandidate, score_retrieval
from .runner import FunctionCandidate, LLMClassifier, RunResult, ScoringConfig, run
from .types import ModelSpec, Prediction


def _load_suite(path: str) -> tuple[dict[str, Any], Path]:
    p = Path(path)
    return json.loads(p.read_text()), p.parent


def _provider(suite: dict, base: Path, mode: str):
    registry = ProviderRegistry.default()
    if mode == "live":
        return registry
    return ReplayProvider(base / suite["cassette"], inner=registry, mode=mode)


def _specs(suite: dict, only: str | None) -> list[ModelSpec]:
    specs = [ModelSpec(**c) for c in suite.get("candidates", [])]
    if only:
        wanted = set(only.split(","))
        specs = [s for s in specs if s.name in wanted]
    return specs


def baseline_candidates(dataset: Dataset, labels: list[str], calibrate_split: str | None, seed: int = 0):
    """``majority`` (most common label in the calibration split, else the whole set) and ``random``."""
    pool = [ex.expected for ex in dataset if not calibrate_split or ex.meta.get("split") == calibrate_split]
    majority = Counter(pool or [ex.expected for ex in dataset]).most_common(1)[0][0]
    rng = random.Random(seed)
    draws = {ex.id: rng.choice(labels) for ex in dataset}
    return [
        FunctionCandidate("majority", lambda ex: Prediction(ex.id, majority, confidence=1.0)),
        FunctionCandidate("random", lambda ex: Prediction(ex.id, draws[ex.id], confidence=1.0 / len(labels))),
    ]


def cmd_run(args) -> int:
    suite, base = _load_suite(args.suite)
    ds = Dataset.load_jsonl(base / suite["dataset"], suite["name"])
    labels = suite["labels"]
    cfg = ScoringConfig(labels=labels, **suite.get("scoring", {}))
    provider = _provider(suite, base, args.mode)
    out_dir = Path(args.out or base / suite.get("results_dir", "results") / suite["name"])
    results = []
    for spec in _specs(suite, args.only):
        cand = LLMClassifier(spec, provider, labels, suite["system"], suite["template"])
        res = run(ds, cand, config=cfg, workers=args.workers)
        res.save(out_dir / f"{spec.name}.json")
        results.append(res)
        print(f"  {spec.name}: macro_f1={res.metrics['macro_f1']} errors={res.metrics['error_rate']:.2%}", file=sys.stderr)
    baselines = suite.get("baselines", [])
    for cand in baseline_candidates(ds, labels, cfg.calibrate_split):
        if cand.name in baselines:
            results.append(run(ds, cand, config=cfg))
    sel = cheapest_passing(results, suite["bar"], reference=suite.get("reference"), exclude=baselines) \
        if suite.get("bar") else None
    table = classification_table(results, sel)
    (out_dir / "table.md").write_text(table + "\n")
    print(table)
    return 0


def cmd_judge(args) -> int:
    suite, base = _load_suite(args.suite)
    ds = Dataset.load_jsonl(base / suite["dataset"], suite["name"])
    human = {ex.id: ex.expected for ex in ds}
    provider = _provider(suite, base, args.mode)
    head = ("| judge | n | agreement | Cohen's κ | false-pass | false-fail | trusted |\n"
            "|---|---:|---:|---:|---:|---:|---|")
    rows, reports = [], {}
    for spec in _specs(suite, args.only):
        verdicts = judge_mod.PointwiseJudge(spec, provider).judge_all(list(ds), workers=args.workers)
        rep = judge_mod.calibrate(verdicts, human, **suite.get("calibration", {}))
        reports[spec.name] = rep.to_dict()
        rows.append(f"| {spec.name} | {rep.n} | {rep.agreement:.3f} | {rep.kappa:.3f} | {rep.false_pass_rate:.3f} "
                    f"| {rep.false_fail_rate:.3f} | {'yes' if rep.trusted else 'NO'} |")
    out_dir = Path(args.out or base / suite.get("results_dir", "results") / suite["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(reports, indent=2) + "\n")
    table = head + "\n" + "\n".join(rows)
    (out_dir / "table.md").write_text(table + "\n")
    print(table)
    return 0


def cmd_pairwise(args) -> int:
    suite, base = _load_suite(args.suite)
    ds = Dataset.load_jsonl(base / suite["dataset"], suite["name"])
    provider = _provider(suite, base, args.mode)
    spec = _specs(suite, args.only)[0]
    pj = judge_mod.PairwiseJudge(spec, provider)
    outcomes = []
    for ex in ds:
        answers = ex.input["answers"]  # {"system-name": "answer text", ...}
        names = sorted(answers)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                outcomes.append(pj.compare(ex.id, ex.input["question"], (names[i], answers[names[i]]),
                                           (names[j], answers[names[j]])))
    rates, bt = judge_mod.win_rates(outcomes), judge_mod.bradley_terry(outcomes)
    consistent = sum(o.position_consistent for o in outcomes) / len(outcomes)
    head = "| system | games | wins | losses | ties | win rate | Bradley-Terry |\n|---|---:|---:|---:|---:|---:|---:|"
    rows = [f"| {m} | {r['games']} | {r['wins']} | {r['losses']} | {r['ties']} | {r['win_rate']:.3f} | {bt[m]} |"
            for m, r in sorted(rates.items(), key=lambda kv: -bt[kv[0]])]
    table = head + "\n" + "\n".join(rows) + f"\n\nJudge `{spec.name}` position-consistency: {consistent:.1%} of comparisons."
    out_dir = Path(args.out or base / suite.get("results_dir", "results") / suite["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table.md").write_text(table + "\n")
    print(table)
    return 0


def cmd_retrieval(args) -> int:
    suite, base = _load_suite(args.suite)
    ds = Dataset.load_jsonl(base / suite["dataset"], suite["name"])
    corpus = {}
    with (base / suite["corpus"]).open() as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                corpus[d["id"]] = d["text"]
    k = suite.get("k", 5)
    results = [run(ds, RetrieverCandidate("bm25", BM25(corpus), k=k), scorer=lambda d, p: score_retrieval(d, p, k))]
    table = retrieval_table(results)
    out_dir = Path(args.out or base / suite.get("results_dir", "results") / suite["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results[0].save(out_dir / "bm25.json")
    (out_dir / "table.md").write_text(table + "\n")
    print(table)
    return 0


def cmd_gate(args) -> int:
    cur, base = RunResult.load(args.current), RunResult.load(args.baseline)
    tol = json.loads(Path(args.tolerances).read_text())
    res = regression_gate(cur, base, tol)
    print("\n".join(res.lines))
    print("GATE PASS" if res.passed else "GATE FAIL")
    return 0 if res.passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ailab-evals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("judge", cmd_judge), ("pairwise", cmd_pairwise), ("retrieval", cmd_retrieval)):
        p = sub.add_parser(name)
        p.add_argument("suite")
        p.add_argument("--mode", choices=["replay", "record", "live"], default="replay")
        p.add_argument("--only")
        p.add_argument("--out")
        p.add_argument("--workers", type=int, default=2)
        p.set_defaults(fn=fn)
    g = sub.add_parser("gate")
    g.add_argument("current")
    g.add_argument("baseline")
    g.add_argument("--tolerances", required=True)
    g.set_defaults(fn=cmd_gate)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
