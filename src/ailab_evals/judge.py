"""LLM-as-judge: pointwise grading, pairwise preference, and calibration against human labels.

The rule this module enforces: a judge's scores are not evidence until the judge has been
calibrated against human labels on the same kind of item. ``calibrate`` reports agreement,
Cohen's kappa, and the two error directions separately, because a judge that passes bad
answers (false pass) is far more dangerous in a quality gate than one that is too strict.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from . import metrics
from .providers import Provider
from .runner import _JSON_OBJ
from .types import Example, ModelSpec

POINTWISE_SYSTEM = (
    "You are a strict evaluator. Grade the CANDIDATE answer against the QUESTION and the REFERENCE. "
    "A candidate passes only if it is factually consistent with the reference and answers the question. "
    'Reply with JSON only: {"verdict": "pass" | "fail", "score": 1-5, "reason": "<one sentence>"}'
)

PAIRWISE_SYSTEM = (
    "You compare two answers to the same question. Prefer the answer that is correct and complete; "
    "ignore length and style unless they change correctness. "
    'Reply with JSON only: {"winner": "A" | "B" | "tie", "reason": "<one sentence>"}'
)


def _parse(text: str) -> dict:
    m = _JSON_OBJ.search(text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


@dataclass
class JudgeVerdict:
    example_id: str
    verdict: str | None  # "pass" | "fail" | None on error
    score: float | None
    reason: str = ""
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class PointwiseJudge:
    spec: ModelSpec
    provider: Provider
    system: str = POINTWISE_SYSTEM

    def render(self, item: dict) -> str:
        return (
            f"QUESTION:\n{item['question']}\n\nREFERENCE:\n{item.get('reference', '(none)')}\n\n"
            f"CANDIDATE:\n{item['candidate']}"
        )

    def judge(self, example: Example) -> JudgeVerdict:
        comp = self.provider(self.spec, self.system, self.render(example.input), json_mode=True)
        cost = self.spec.cost(comp.input_tokens, comp.output_tokens)
        if comp.error:
            return JudgeVerdict(example.id, None, None, cost_usd=cost, error=comp.error)
        obj = _parse(comp.text)
        verdict = str(obj.get("verdict", "")).lower()
        if verdict not in {"pass", "fail"}:
            return JudgeVerdict(example.id, None, None, cost_usd=cost, error="unparseable verdict")
        try:
            score = float(obj.get("score"))
        except (TypeError, ValueError):
            score = None
        return JudgeVerdict(example.id, verdict, score, str(obj.get("reason", ""))[:300], cost)

    def judge_all(self, examples: Sequence[Example], workers: int = 4) -> list[JudgeVerdict]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(self.judge, examples))


@dataclass
class CalibrationReport:
    n: int
    agreement: float
    kappa: float
    false_pass_rate: float  # human=fail, judge=pass  (over human fails)
    false_fail_rate: float  # human=pass, judge=fail  (over human passes)
    judge_error_rate: float
    score_spearman: float | None
    trusted: bool
    min_kappa: float
    max_false_pass: float

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate(
    verdicts: Sequence[JudgeVerdict],
    human: dict[str, str],
    *,
    human_scores: dict[str, float] | None = None,
    min_kappa: float = 0.6,
    max_false_pass: float = 0.10,
) -> CalibrationReport:
    """Compare judge verdicts with human pass/fail labels keyed by example id.

    A judge error counts as disagreement (it is not dropped). ``trusted`` requires both
    kappa >= ``min_kappa`` and a false-pass rate <= ``max_false_pass``.
    """
    pairs = [(human[v.example_id], v.verdict or "__error__") for v in verdicts if v.example_id in human]
    if not pairs:
        raise ValueError("no overlap between judge verdicts and human labels")
    h, j = [p[0] for p in pairs], [p[1] for p in pairs]
    fails = [jj for hh, jj in pairs if hh == "fail"]
    passes = [jj for hh, jj in pairs if hh == "pass"]
    fpr = sum(x == "pass" for x in fails) / len(fails) if fails else 0.0
    ffr = sum(x != "pass" for x in passes) / len(passes) if passes else 0.0
    kappa = metrics.cohen_kappa(h, j)
    rho = None
    if human_scores:
        sc = [(human_scores[v.example_id], v.score) for v in verdicts
              if v.example_id in human_scores and v.score is not None]
        if len(sc) >= 3:
            rho = metrics.spearman([a for a, _ in sc], [b for _, b in sc])
    return CalibrationReport(
        n=len(pairs),
        agreement=metrics.accuracy(h, j),
        kappa=kappa,
        false_pass_rate=fpr,
        false_fail_rate=ffr,
        judge_error_rate=sum(x == "__error__" for x in j) / len(j),
        score_spearman=rho,
        trusted=kappa >= min_kappa and fpr <= max_false_pass,
        min_kappa=min_kappa,
        max_false_pass=max_false_pass,
    )


# --- pairwise ------------------------------------------------------------------------


@dataclass
class PairwiseOutcome:
    example_id: str
    model_a: str
    model_b: str
    winner: str  # model name, or "tie"
    position_consistent: bool
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class PairwiseJudge:
    """Asks the judge twice with A/B swapped. A winner must survive the swap, else it is a tie.

    This cancels position bias (judges measurably prefer whichever answer they see first)
    at the price of 2x judge calls.
    """

    spec: ModelSpec
    provider: Provider
    system: str = PAIRWISE_SYSTEM

    def _ask(self, question: str, first: str, second: str) -> tuple[str | None, float, str | None]:
        prompt = f"QUESTION:\n{question}\n\nANSWER A:\n{first}\n\nANSWER B:\n{second}"
        comp = self.provider(self.spec, self.system, prompt, json_mode=True)
        cost = self.spec.cost(comp.input_tokens, comp.output_tokens)
        if comp.error:
            return None, cost, comp.error
        w = str(_parse(comp.text).get("winner", "")).strip().upper()
        return (w if w in {"A", "B", "TIE"} else None), cost, None if w in {"A", "B", "TIE"} else "unparseable"

    def compare(self, example_id: str, question: str, a: tuple[str, str], b: tuple[str, str]) -> PairwiseOutcome:
        (name_a, text_a), (name_b, text_b) = a, b
        w1, c1, e1 = self._ask(question, text_a, text_b)
        w2, c2, e2 = self._ask(question, text_b, text_a)
        err = e1 or e2
        # normalise both answers to a model name
        first = {"A": name_a, "B": name_b}.get(w1 or "", "tie")
        second = {"A": name_b, "B": name_a}.get(w2 or "", "tie")
        consistent = first == second
        winner = first if consistent else "tie"
        return PairwiseOutcome(example_id, name_a, name_b, winner, consistent, c1 + c2, err)


def win_rates(outcomes: Sequence[PairwiseOutcome]) -> dict[str, dict[str, float]]:
    """Per-model wins / losses / ties, with ties counted as half a win in ``win_rate``."""
    stats: dict[str, Counter] = defaultdict(Counter)
    for o in outcomes:
        for m in (o.model_a, o.model_b):
            stats[m]["games"] += 1
        if o.winner == "tie":
            stats[o.model_a]["ties"] += 1
            stats[o.model_b]["ties"] += 1
        else:
            loser = o.model_b if o.winner == o.model_a else o.model_a
            stats[o.winner]["wins"] += 1
            stats[loser]["losses"] += 1
    return {
        m: {
            "games": c["games"],
            "wins": c["wins"],
            "losses": c["losses"],
            "ties": c["ties"],
            "win_rate": (c["wins"] + 0.5 * c["ties"]) / c["games"] if c["games"] else 0.0,
        }
        for m, c in stats.items()
    }


def bradley_terry(outcomes: Sequence[PairwiseOutcome], iters: int = 200) -> dict[str, float]:
    """Bradley-Terry strengths via the MM algorithm, ties split as half-wins. Returned as Elo-like ratings."""
    models = sorted({o.model_a for o in outcomes} | {o.model_b for o in outcomes})
    wins: dict[tuple[str, str], float] = defaultdict(float)
    for o in outcomes:
        if o.winner == "tie":
            wins[(o.model_a, o.model_b)] += 0.5
            wins[(o.model_b, o.model_a)] += 0.5
        else:
            loser = o.model_b if o.winner == o.model_a else o.model_a
            wins[(o.winner, loser)] += 1
    p = {m: 1.0 for m in models}
    for _ in range(iters):
        new = {}
        for i in models:
            w_i = sum(wins[(i, j)] for j in models if j != i)
            denom = sum(
                (wins[(i, j)] + wins[(j, i)]) / (p[i] + p[j]) for j in models if j != i and wins[(i, j)] + wins[(j, i)]
            )
            new[i] = max(w_i, 1e-6) / denom if denom else p[i]
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {k: v / g for k, v in new.items()}
    return {m: round(1000 + 400 * math.log10(p[m]), 1) for m in models}
