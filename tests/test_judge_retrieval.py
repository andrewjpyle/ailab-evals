import pytest

from ailab_evals.datasets import Dataset
from ailab_evals.judge import (
    JudgeVerdict,
    PairwiseJudge,
    PairwiseOutcome,
    PointwiseJudge,
    bradley_terry,
    calibrate,
    win_rates,
)
from ailab_evals.retrieval import BM25, RetrieverCandidate, score_retrieval
from ailab_evals.runner import run
from ailab_evals.types import Completion, Example, ModelSpec

SPEC = ModelSpec("judge", "fake", "j")


def test_calibrate_separates_false_pass_from_false_fail():
    human = {"a": "pass", "b": "pass", "c": "fail", "d": "fail"}
    lenient = [JudgeVerdict("a", "pass", 5), JudgeVerdict("b", "pass", 5),
               JudgeVerdict("c", "pass", 4), JudgeVerdict("d", "fail", 1)]
    rep = calibrate(lenient, human, min_kappa=0.0, max_false_pass=0.1)
    assert rep.false_pass_rate == 0.5 and rep.false_fail_rate == 0.0
    assert rep.trusted is False  # passes bad answers -> never trusted, whatever kappa says


def test_calibrate_counts_judge_errors_as_disagreement():
    human = {"a": "pass", "b": "fail"}
    rep = calibrate([JudgeVerdict("a", None, None, error="x"), JudgeVerdict("b", "fail", 1)], human)
    assert rep.judge_error_rate == 0.5 and rep.agreement == 0.5


def test_calibrate_requires_overlap():
    with pytest.raises(ValueError):
        calibrate([JudgeVerdict("z", "pass", 5)], {"a": "pass"})


def test_pointwise_parses_and_flags_garbage():
    def prov(spec, system, prompt, *, json_mode=False):
        good = "CANDIDATE:\nright" in prompt
        return Completion('{"verdict": "pass", "score": 5}' if good else "I think it's fine")

    j = PointwiseJudge(SPEC, prov)
    ok = j.judge(Example("1", {"question": "q", "reference": "r", "candidate": "right"}))
    bad = j.judge(Example("2", {"question": "q", "reference": "r", "candidate": "wrong"}))
    assert ok.verdict == "pass" and ok.score == 5
    assert bad.verdict is None and bad.error == "unparseable verdict"


def test_pairwise_position_bias_becomes_tie():
    always_a = lambda spec, system, prompt, *, json_mode=False: Completion('{"winner": "A"}')  # noqa: E731
    out = PairwiseJudge(SPEC, always_a).compare("x", "q", ("m1", "t1"), ("m2", "t2"))
    assert out.winner == "tie" and out.position_consistent is False

    def content_judge(spec, system, prompt, *, json_mode=False):
        a_text = prompt.split("ANSWER A:\n")[1].split("\n\nANSWER B")[0]
        return Completion('{"winner": "A"}' if a_text == "good" else '{"winner": "B"}')

    out = PairwiseJudge(SPEC, content_judge).compare("x", "q", ("bad_model", "bad"), ("good_model", "good"))
    assert out.winner == "good_model" and out.position_consistent


def test_win_rates_and_bradley_terry_order():
    outs = [PairwiseOutcome(str(i), "a", "b", "a", True) for i in range(8)]
    outs += [PairwiseOutcome("t", "a", "b", "tie", False)]
    outs += [PairwiseOutcome(str(i), "b", "c", "b", True) for i in range(6)]
    wr = win_rates(outs)
    assert wr["a"]["wins"] == 8 and wr["a"]["ties"] == 1
    bt = bradley_terry(outs)
    assert bt["a"] > bt["b"] > bt["c"]


def test_bm25_and_retrieval_scoring_crash_is_a_miss():
    corpus = {"d1": "backup retention fourteen days", "d2": "certificate renewal alarm", "d3": "deploy rollback"}
    ds = Dataset.from_records("r", [
        {"id": "q1", "input": "how long is backup retention", "expected": ["d1"], "meta": {"critical": True}},
        {"id": "q2", "input": "rollback a deploy", "expected": ["d3"]},
    ])
    res = run(ds, RetrieverCandidate("bm25", BM25(corpus), k=1), scorer=lambda d, p: score_retrieval(d, p, 1))
    assert res.metrics["recall@1"] == 1.0 and res.metrics["critical_recall@1"] == 1.0

    def boom(q, k):
        raise RuntimeError("index down")

    res = run(ds, RetrieverCandidate("boom", boom, k=1), scorer=lambda d, p: score_retrieval(d, p, 1))
    assert res.metrics["recall@1"] == 0.0 and res.metrics["error_rate"] == 1.0
