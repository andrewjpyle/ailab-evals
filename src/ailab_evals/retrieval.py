"""Retrieval evals: a retriever is any ``fn(query, k) -> ranked list of ids``.

Gold rows are ``{"id", "input": "<query>", "expected": ["doc-id", ...], "meta": {"critical": bool}}``.
Reports recall@k, hit@k, MRR and nDCG@k, plus the same on the ``critical`` subset, because
the misses that matter are rarely spread evenly.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from . import metrics
from .datasets import Dataset
from .types import Example, Prediction

Retriever = Callable[[str, int], Sequence[str]]


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def score_retrieval(dataset: Dataset, predictions: Sequence[Prediction], k: int) -> dict[str, Any]:
    by_id = {p.example_id: p for p in predictions}
    per: dict[str, list[float]] = {"recall": [], "hit": [], "rr": [], "ndcg": []}
    crit_recall: list[float] = []
    misses = []
    for ex in dataset:
        p = by_id.get(ex.id)
        ranked = list(p.output) if p and p.error is None and p.output else []
        rel = set(ex.expected or [])
        r = metrics.recall_at_k(ranked, rel, k)
        per["recall"].append(r)
        per["hit"].append(metrics.hit_at_k(ranked, rel, k))
        per["rr"].append(metrics.reciprocal_rank(ranked[:k], rel))
        per["ndcg"].append(metrics.ndcg_at_k(ranked, rel, k))
        if ex.meta.get("critical"):
            crit_recall.append(r)
        if r < 1.0:
            misses.append({"id": ex.id, "missing": sorted(rel - set(ranked[:k]))})
    costs = [p.cost_usd for p in predictions]
    return {
        "n": len(dataset),
        "k": k,
        f"recall@{k}": _mean(per["recall"]),
        f"hit@{k}": _mean(per["hit"]),
        "mrr": _mean(per["rr"]),
        f"ndcg@{k}": _mean(per["ndcg"]),
        "critical_n": len(crit_recall),
        f"critical_recall@{k}": _mean(crit_recall),
        "error_rate": sum(1 for p in predictions if p.error) / len(dataset) if len(dataset) else 0.0,
        "cost_usd_per_1k": 1000 * sum(costs) / len(predictions) if predictions else 0.0,
        "misses": misses,
    }


@dataclass
class RetrieverCandidate:
    name: str
    retriever: Retriever
    k: int = 6

    def predict(self, example: Example) -> Prediction:
        try:
            ranked = list(self.retriever(str(example.input), self.k))
        except Exception as exc:  # a retriever crash is a scored miss, never a skipped row
            return Prediction(example.id, [], error=f"{type(exc).__name__}: {exc}")
        return Prediction(example.id, ranked)


_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Small Okapi BM25 baseline (k1=1.5, b=0.75). Good enough to anchor a retrieval table."""

    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs)
        self.toks = [tokenize(docs[i]) for i in self.ids]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(t for doc in self.toks for t in set(doc))
        n = len(self.toks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.tf = [Counter(doc) for doc in self.toks]

    def __call__(self, query: str, k: int) -> list[str]:
        q = tokenize(query)
        scores = []
        for i, (tf, doc) in enumerate(zip(self.tf, self.toks)):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            scores.append((s, self.ids[i]))
        scores.sort(key=lambda t: (-t[0], t[1]))
        return [doc_id for s, doc_id in scores[:k] if s > 0]
