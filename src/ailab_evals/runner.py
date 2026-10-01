"""Run candidates over a dataset and score them.

A *candidate* is anything with a ``name`` and a ``predict(example) -> Prediction``. An
LLM prompt is one kind (``LLMClassifier``); a host application can wrap a non-LLM
classifier, a cached prediction table, or its own client the same way. The runner only
cares that every prediction carries its own cost and latency.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from . import __version__, metrics
from .datasets import Dataset
from .providers import Provider
from .types import Example, ModelSpec, Prediction


class Candidate(Protocol):
    name: str

    def predict(self, example: Example) -> Prediction: ...


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_label_json(text: str, labels: Sequence[str]) -> tuple[str | None, float | None]:
    """Pull ``{"label": ..., "confidence": ...}`` out of a model reply. Unknown label -> None."""
    match = _JSON_OBJ.search(text or "")
    if not match:
        return None, None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None
    label = str(obj.get("label", "")).strip().lower()
    by_lower = {lab.lower(): lab for lab in labels}
    conf = obj.get("confidence")
    try:
        conf = None if conf is None else min(1.0, max(0.0, float(conf)))
    except (TypeError, ValueError):
        conf = None
    return by_lower.get(label), conf


@dataclass
class LLMClassifier:
    """Prompted single-label classifier. ``template`` is ``str.format``-ed with the example input."""

    spec: ModelSpec
    provider: Provider
    labels: Sequence[str]
    system: str
    template: str
    json_mode: bool = True

    @property
    def name(self) -> str:
        return self.spec.name

    def render(self, example: Example) -> str:
        fields = example.input if isinstance(example.input, dict) else {"input": example.input}
        return self.template.format(labels=" | ".join(self.labels), **fields)

    def predict(self, example: Example) -> Prediction:
        comp = self.provider(self.spec, self.system, self.render(example), json_mode=self.json_mode)
        label, conf = (None, None) if comp.error else parse_label_json(comp.text, self.labels)
        error = comp.error or (None if label is not None else "unparseable label")
        return Prediction(
            example_id=example.id,
            output=label,
            raw=comp.text,
            confidence=conf,
            input_tokens=comp.input_tokens,
            output_tokens=comp.output_tokens,
            latency_ms=comp.latency_ms,
            cost_usd=self.spec.cost(comp.input_tokens, comp.output_tokens),
            error=error,
        )


@dataclass
class FunctionCandidate:
    """Wrap any ``fn(example) -> Prediction`` (non-LLM model, cached table, host client)."""

    name: str
    fn: Callable[[Example], Prediction]

    def predict(self, example: Example) -> Prediction:
        return self.fn(example)


@dataclass
class RunResult:
    candidate: str
    dataset: str
    dataset_sha256: str
    predictions: list[Prediction]
    metrics: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "dataset": self.dataset,
            "dataset_sha256": self.dataset_sha256,
            "metrics": self.metrics,
            "meta": self.meta,
            "predictions": [p.to_dict() for p in self.predictions],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        d = json.loads(Path(path).read_text())
        return cls(
            candidate=d["candidate"],
            dataset=d["dataset"],
            dataset_sha256=d["dataset_sha256"],
            predictions=[Prediction(**p) for p in d.get("predictions", [])],
            metrics=d["metrics"],
            meta=d.get("meta", {}),
        )


@dataclass
class ScoringConfig:
    """How to score a classification run.

    ``floor_per_class``: if any gold class has fewer scored rows than this, macro-F1 is reported
    as ``"INSUFFICIENT"`` instead of a number (a 3-example class makes F1 noise look like signal).
    Rows whose ``meta["split"] == calibrate_split`` are held out of scoring and used only to fit
    Platt scaling; ``precision_threshold`` is then applied to the *calibrated* confidence.
    """

    labels: Sequence[str] | None = None
    floor_per_class: int = 0
    score_split: str | None = None
    calibrate_split: str | None = None
    precision_threshold: float | None = None
    calibrate_floor_per_class: int = 0
    n_boot: int = 1000
    seed: int = 0


def _pct(sorted_vals: list[float], q: float) -> float | None:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))] if sorted_vals else None


def score_classification(
    dataset: Dataset, predictions: Sequence[Prediction], labels: Sequence[str] | None = None,
    config: ScoringConfig | None = None,
) -> dict[str, Any]:
    """Score predictions against gold. An errored / unparseable prediction counts as WRONG, never skipped.

    Silently dropping failures would let a flaky model look accurate on the subset it answered.
    """
    cfg = config or ScoringConfig(labels=labels)
    by_id = {p.example_id: p for p in predictions}

    def rows(split: str | None):
        return [ex for ex in dataset if split is None or ex.meta.get("split") == split]

    scored = rows(cfg.score_split) if cfg.score_split else [
        ex for ex in dataset if not cfg.calibrate_split or ex.meta.get("split") != cfg.calibrate_split]
    gold, pred, conf_rows = [], [], []
    for ex in scored:
        p = by_id.get(ex.id)
        ok = p is not None and p.error is None
        out = p.output if ok else "__error__"
        gold.append(ex.expected)
        pred.append(out)
        if ok and p.confidence is not None:
            conf_rows.append((p.confidence, out == ex.expected))
    labs = list(cfg.labels or labels or sorted({g for g in gold}, key=str))
    n = len(scored)
    counts = {lab: sum(g == lab for g in gold) for lab in labs}
    insufficient = [f"{lab}: {c} < floor {cfg.floor_per_class}" for lab, c in counts.items() if c < cfg.floor_per_class]
    errors = sum(1 for ex in scored if ex.id not in by_id or by_id[ex.id].error is not None)
    preds_scored = [by_id[ex.id] for ex in scored if ex.id in by_id]
    costs = [p.cost_usd for p in predictions]
    lats = sorted(p.latency_ms for p in preds_scored if p.error is None)
    out: dict[str, Any] = {
        "n": n,
        "class_counts": counts,
        "accuracy": metrics.accuracy(gold, pred) if n else None,
        "per_class": metrics.per_class_prf(gold, pred, labs),
        "error_rate": errors / n if n else 0.0,
        "cost_usd_total": sum(costs),
        "cost_usd_per_1k": 1000 * sum(costs) / len(predictions) if predictions else 0.0,
        "latency_ms_p50": statistics.median(lats) if lats else None,
        "latency_ms_p95": _pct(lats, 0.95),
    }
    if insufficient or not n:
        out["macro_f1"] = "INSUFFICIENT"
        out["insufficient"] = insufficient or ["no scored rows"]
    else:
        out["macro_f1"] = metrics.macro_f1(gold, pred, labs)
        out["macro_f1_ci95"] = list(metrics.bootstrap_ci(
            list(zip(gold, pred)),
            lambda rs: metrics.macro_f1([g for g, _ in rs], [q for _, q in rs], labs),
            n_boot=cfg.n_boot, seed=cfg.seed))
    if conf_rows:
        out["ece"] = metrics.expected_calibration_error([c for c, _ in conf_rows], [k for _, k in conf_rows])
        out["confidence_coverage"] = len(conf_rows) / n if n else 0.0
        if cfg.calibrate_split:
            cal = []
            for ex in rows(cfg.calibrate_split):
                p = by_id.get(ex.id)
                if p is not None and p.error is None and p.confidence is not None:
                    cal.append((p.confidence, p.output == ex.expected))
            cal_counts = Counter(ex.expected for ex in rows(cfg.calibrate_split))
            short = [lab for lab in labs if cal_counts.get(lab, 0) < cfg.calibrate_floor_per_class]
            if short:
                out["platt"] = "INSUFFICIENT"
                out["platt_insufficient"] = [f"{lab}: {cal_counts.get(lab, 0)} < floor "
                                             f"{cfg.calibrate_floor_per_class}" for lab in short]
                conf_for_threshold = None
            elif len(cal) >= 2 and len({k for _, k in cal}) == 2:
                params = metrics.fit_platt([c for c, _ in cal], [k for _, k in cal])
                calibrated = [metrics.apply_platt(c, params) for c, _ in conf_rows]
                out["platt"] = {"a": params[0], "b": params[1], "n_fit": len(cal)}
                out["ece_platt"] = metrics.expected_calibration_error(calibrated, [k for _, k in conf_rows])
                conf_for_threshold = calibrated
            else:
                out["platt"] = "INSUFFICIENT"
                conf_for_threshold = [c for c, _ in conf_rows]
        else:
            conf_for_threshold = [c for c, _ in conf_rows]
        if cfg.precision_threshold is not None and conf_for_threshold is not None:
            prec, cov = metrics.precision_at_threshold(
                conf_for_threshold, [k for _, k in conf_rows], cfg.precision_threshold)
            out["precision_at_threshold"] = prec
            out["coverage_at_threshold"] = cov
    return out


def run(
    dataset: Dataset,
    candidate: Candidate,
    *,
    labels: Sequence[str] | None = None,
    workers: int = 4,
    scorer: Callable[[Dataset, Sequence[Prediction]], dict] | None = None,
    config: ScoringConfig | None = None,
) -> RunResult:
    """Run ``candidate`` over every example. A candidate exposing ``predict_batch(examples)``
    is called once with the whole dataset (for batched / non-LLM models)."""
    t0 = time.perf_counter()
    if hasattr(candidate, "predict_batch"):
        predictions = list(candidate.predict_batch(list(dataset)))
        if len(predictions) != len(dataset):
            raise ValueError(f"{candidate.name}: predict_batch returned {len(predictions)} for {len(dataset)} examples")
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            predictions = list(pool.map(candidate.predict, dataset))
    scored = scorer(dataset, predictions) if scorer else score_classification(dataset, predictions, labels, config)
    return RunResult(
        candidate=candidate.name,
        dataset=dataset.name,
        dataset_sha256=dataset.sha256,
        predictions=predictions,
        metrics=scored,
        meta={
            "harness_version": __version__,
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **({"model": candidate.spec.model, "provider": candidate.spec.provider}
               if isinstance(candidate, LLMClassifier) else {}),
        },
    )
