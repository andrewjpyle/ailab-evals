"""Metrics: classification, agreement, calibration, ranking, and bootstrap intervals.

Pure functions over plain lists so they are trivially testable and reusable by a host
application that already has its own predictions.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Sequence


def accuracy(gold: Sequence, pred: Sequence) -> float:
    _check(gold, pred)
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0


def per_class_prf(gold: Sequence, pred: Sequence, labels: Sequence | None = None) -> dict:
    _check(gold, pred)
    labels = list(labels) if labels is not None else sorted(set(gold) | set(pred), key=str)
    out = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def macro_f1(gold: Sequence, pred: Sequence, labels: Sequence | None = None) -> float:
    """Unweighted mean F1 over ``labels`` (default: labels present in gold). A minority class counts fully."""
    labels = list(labels) if labels is not None else sorted(set(gold), key=str)
    prf = per_class_prf(gold, pred, labels)
    return sum(v["f1"] for v in prf.values()) / len(labels) if labels else 0.0


def confusion(gold: Sequence, pred: Sequence) -> dict[tuple, int]:
    _check(gold, pred)
    return dict(Counter(zip(gold, pred)))


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Chance-corrected agreement between two raters. 1 = perfect, 0 = chance."""
    _check(a, b)
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    _check(x, y)
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx and sy else 0.0


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(_ranks(x), _ranks(y))


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """ECE: |accuracy - confidence| per equal-width bin, weighted by bin size. 0 = perfectly calibrated."""
    _check(confidences, correct)
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0)]
        if not idx:
            continue
        acc = sum(bool(correct[i]) for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return total


def brier(probs: Sequence[float], outcomes: Sequence[bool]) -> float:
    _check(probs, outcomes)
    return sum((p - float(o)) ** 2 for p, o in zip(probs, outcomes)) / len(probs) if probs else 0.0


def fit_platt(scores: Sequence[float], outcomes: Sequence[bool], iters: int = 2000, lr: float = 0.1) -> tuple[float, float]:
    """Fit p = sigmoid(a*s + b) by gradient descent on log-loss. Returns (a, b)."""
    _check(scores, outcomes)
    a, b = 1.0, 0.0
    n = len(scores) or 1
    for _ in range(iters):
        ga = gb = 0.0
        for s, o in zip(scores, outcomes):
            p = 1 / (1 + math.exp(-(a * s + b)))
            ga += (p - float(o)) * s
            gb += p - float(o)
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def apply_platt(score: float, params: tuple[float, float]) -> float:
    a, b = params
    return 1 / (1 + math.exp(-(a * score + b)))


def precision_at_threshold(confidences: Sequence[float], correct: Sequence[bool], threshold: float) -> tuple[float | None, float]:
    """(precision among items with confidence >= threshold, coverage). Precision is None if nothing clears it."""
    _check(confidences, correct)
    kept = [k for c, k in zip(confidences, correct) if c >= threshold]
    cov = len(kept) / len(confidences) if confidences else 0.0
    return (sum(kept) / len(kept) if kept else None), cov


# --- ranking / retrieval -------------------------------------------------------------


def recall_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence, relevant: set) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


# --- uncertainty ---------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence,
    stat: Callable[[Sequence], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat`` over resampled rows. Seeded so reports are reproducible."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def _check(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
