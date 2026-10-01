import math

import pytest

from ailab_evals import metrics as m


def test_accuracy_and_macro_f1_weight_minority_class():
    gold = ["yes"] * 9 + ["no"]
    pred = ["yes"] * 10
    assert m.accuracy(gold, pred) == pytest.approx(0.9)
    # majority-always scores 0.9 accuracy but macro-F1 punishes the missed minority class
    assert m.macro_f1(gold, pred) == pytest.approx((2 * 0.9 / 1.9 + 0.0) / 2)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        m.accuracy([1, 2], [1])


def test_cohen_kappa_perfect_chance_and_inverse():
    assert m.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert m.cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]) == pytest.approx(-1.0)
    # constant rater: agreement is pure chance
    assert m.cohen_kappa(["a", "a", "b", "b"], ["a", "a", "a", "a"]) == pytest.approx(0.0)


def test_ece_zero_when_confidence_matches_accuracy():
    conf = [0.75] * 4
    correct = [True, True, True, False]
    assert m.expected_calibration_error(conf, correct) == pytest.approx(0.0)
    assert m.expected_calibration_error([1.0] * 4, [False] * 4) == pytest.approx(1.0)


def test_platt_recovers_direction():
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 5
    outcomes = [False, False, False, True, True, True] * 5
    a, b = m.fit_platt(scores, outcomes)
    assert m.apply_platt(0.9, (a, b)) > 0.5 > m.apply_platt(0.1, (a, b))


def test_precision_at_threshold():
    prec, cov = m.precision_at_threshold([0.95, 0.92, 0.5, 0.3], [True, False, True, True], 0.9)
    assert prec == pytest.approx(0.5) and cov == pytest.approx(0.5)
    assert m.precision_at_threshold([0.1], [True], 0.9) == (None, 0.0)


def test_ranking_metrics():
    ranked = ["d3", "d1", "d2"]
    rel = {"d1", "d2"}
    assert m.recall_at_k(ranked, rel, 2) == 0.5
    assert m.reciprocal_rank(ranked, rel) == 0.5
    assert m.hit_at_k(ranked, rel, 1) == 0.0
    ideal = 1 + 1 / math.log2(3)
    assert m.ndcg_at_k(ranked, rel, 3) == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / ideal)


def test_spearman_handles_ties():
    assert m.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert m.spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


def test_bootstrap_ci_is_seeded_and_brackets_point():
    vals = [1] * 70 + [0] * 30
    lo, hi = m.bootstrap_ci(vals, lambda xs: sum(xs) / len(xs), n_boot=500, seed=1)
    assert lo < 0.7 < hi
    assert (lo, hi) == m.bootstrap_ci(vals, lambda xs: sum(xs) / len(xs), n_boot=500, seed=1)
