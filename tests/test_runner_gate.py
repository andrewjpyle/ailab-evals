import json

import pytest

from ailab_evals.datasets import Dataset
from ailab_evals.gate import cheapest_passing, check_bar, regression_gate
from ailab_evals.providers import ReplayProvider
from ailab_evals.runner import FunctionCandidate, LLMClassifier, RunResult, ScoringConfig, parse_label_json, run
from ailab_evals.types import Completion, ModelSpec, Prediction


def _ds(n_yes=12, n_no=12, cal=4):
    rows = []
    for lab, n in (("yes", n_yes), ("no", n_no)):
        for i in range(n):
            rows.append({"id": f"{lab}{i}", "input": {"text": f"{lab} {i}"}, "expected": lab,
                         "meta": {"split": "calibrate" if i < cal else "score"}})
    return Dataset.from_records("t", rows)


def oracle(name, flip=(), cost=0.0, err=()):
    def fn(ex):
        if ex.id in err:
            return Prediction(ex.id, None, error="boom", cost_usd=cost)
        out = ex.expected if ex.id not in flip else ("no" if ex.expected == "yes" else "yes")
        return Prediction(ex.id, out, confidence=0.9, cost_usd=cost)
    return FunctionCandidate(name, fn)


CFG = ScoringConfig(labels=["yes", "no"], floor_per_class=5, calibrate_split="calibrate", precision_threshold=0.5)


def test_errors_count_as_wrong_not_skipped():
    res = run(_ds(), oracle("x", err={"yes5", "yes6"}), config=CFG)
    assert res.metrics["n"] == 16  # calibrate rows held out
    assert res.metrics["error_rate"] == pytest.approx(2 / 16)
    assert res.metrics["accuracy"] == pytest.approx(14 / 16)


def test_floor_reports_insufficient_not_a_number():
    res = run(_ds(n_no=6), oracle("x"), config=CFG)  # only 2 scored "no" rows
    assert res.metrics["macro_f1"] == "INSUFFICIENT"
    assert check_bar(res, {"macro_f1": {"min": 0.0}}).passed is False


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        Dataset.from_records("d", [{"id": "a", "input": 1}, {"id": "a", "input": 2}])


def test_dataset_hash_changes_with_content():
    a = _ds()
    b = Dataset.from_records("t", [{"id": "yes0", "input": {"text": "changed"}, "expected": "yes"}])
    assert a.sha256 != b.sha256 and a.sha256 == _ds().sha256


def test_cheapest_passing_picks_cheapest_within_tolerance_of_reference():
    ds = _ds()
    results = [
        run(ds, oracle("ref", cost=0.010), config=CFG),
        run(ds, oracle("mid", flip={"yes7"}, cost=0.002), config=CFG),       # 1/16 wrong: within 0.10
        run(ds, oracle("cheap", flip={f"yes{i}" for i in range(4, 12)}, cost=0.0001), config=CFG),  # far worse
        run(ds, oracle("majority", flip={f"no{i}" for i in range(12)}), config=CFG),
    ]
    sel = cheapest_passing(results, {"macro_f1": {"min_vs_reference": -0.10}}, reference="ref", exclude=["majority"])
    assert sel.winner == "mid"
    assert [name for name, _, _ in sel.ranked] == ["cheap", "mid", "ref"]


def test_no_winner_when_nothing_passes():
    ds = _ds()
    results = [run(ds, oracle("ref"), config=CFG), run(ds, oracle("bad", flip={"yes5", "no5", "no6"}), config=CFG)]
    sel = cheapest_passing(results, {"macro_f1": {"min": 0.999}}, reference="ref", exclude=["ref"])
    assert sel.winner is None


def test_regression_gate_fails_on_drop_and_on_dataset_change(tmp_path):
    ds = _ds()
    base = run(ds, oracle("m"), config=CFG)
    worse = run(ds, oracle("m", flip={"yes5", "yes6", "no7"}), config=CFG)
    tol = {"macro_f1": {"higher_is_better": True, "max_drop": 0.05}, "error_rate": {"higher_is_better": False}}
    assert regression_gate(base, base, tol).passed
    assert not regression_gate(worse, base, tol).passed
    other = run(_ds(n_yes=13), oracle("m"), config=CFG)
    assert not regression_gate(other, base, tol).passed

    base.save(tmp_path / "b.json")
    assert RunResult.load(tmp_path / "b.json").metrics == json.loads(json.dumps(base.metrics))


def test_parse_label_json_tolerates_fences_and_rejects_unknown():
    assert parse_label_json('```json\n{"label": "YES", "confidence": 1.4}\n```', ["yes", "no"]) == ("yes", 1.0)
    assert parse_label_json('{"label": "maybe"}', ["yes", "no"]) == (None, None)
    assert parse_label_json("no json here", ["yes", "no"]) == (None, None)


def test_replay_records_then_replays_and_misses_fail(tmp_path):
    calls = []

    def inner(spec, system, prompt, *, json_mode=False):
        calls.append(prompt)
        return Completion(text='{"label": "yes", "confidence": 0.8}', input_tokens=100, output_tokens=10)

    spec = ModelSpec("m", "fake", "m-1", input_per_mtok=1.0, output_per_mtok=5.0)
    cassette = tmp_path / "c.jsonl"
    rec = LLMClassifier(spec, ReplayProvider(cassette, inner, "record"), ["yes", "no"], "sys", "{text}")
    ds = _ds()
    first = run(ds, rec, config=CFG, workers=1)
    assert len(calls) == len(ds)
    assert first.metrics["cost_usd_total"] == pytest.approx(len(ds) * (100 * 1 + 10 * 5) / 1e6)

    rep = LLMClassifier(spec, ReplayProvider(cassette, mode="replay"), ["yes", "no"], "sys", "{text}")
    again = run(ds, rep, config=CFG)
    assert len(calls) == len(ds)  # no new calls
    assert again.metrics["accuracy"] == first.metrics["accuracy"]

    miss = LLMClassifier(spec, ReplayProvider(cassette, mode="replay"), ["yes", "no"], "sys", "CHANGED {text}")
    assert run(ds, miss, config=CFG).metrics["error_rate"] == 1.0


def test_predict_batch_length_is_enforced():
    class Bad:
        name = "bad"

        def predict_batch(self, examples):
            return [Prediction(examples[0].id, "yes")]

    with pytest.raises(ValueError):
        run(_ds(), Bad(), config=CFG)


def test_calibration_floor_blocks_platt_and_precision():
    cfg = ScoringConfig(labels=["yes", "no"], floor_per_class=5, calibrate_split="calibrate",
                        calibrate_floor_per_class=10, precision_threshold=0.5)
    res = run(_ds(), oracle("x"), config=cfg)  # only 4 calibrate rows per class
    assert res.metrics["platt"] == "INSUFFICIENT"
    assert "precision_at_threshold" not in res.metrics
    assert check_bar(res, {"precision_at_threshold": {"min": 0.9}}).passed is False
