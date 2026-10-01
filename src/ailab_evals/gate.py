"""Quality bars, cheapest-passing selection, and the regression gate.

A *bar* is a pre-committed set of thresholds written down BEFORE results are seen, e.g.
``{"macro_f1": {"min": 0.80}, "error_rate": {"max": 0.02}}``. ``cheapest_passing`` then
answers the only question that moves money: which is the least expensive candidate that
clears the bar?  The regression gate compares a fresh run against a stored baseline and
fails (exit code 1) on any metric that got worse by more than its tolerance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .runner import RunResult


def _get(metrics: dict, dotted: str) -> Any:
    cur: Any = metrics
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


@dataclass
class BarCheck:
    candidate: str
    passed: bool
    failures: list[str] = field(default_factory=list)


def check_bar(result: RunResult, bar: dict[str, dict[str, float]], reference: RunResult | None = None) -> BarCheck:
    """Every bar metric must be present and numeric; missing / ``"INSUFFICIENT"`` is a failure, not a pass.

    Rules per metric: ``min`` / ``max`` (absolute) and ``min_vs_reference`` (relative to the same
    metric on ``reference``, e.g. ``-0.03`` = no more than 3 points below the reference model).
    """
    failures = []
    for metric, rule in bar.items():
        val = _get(result.metrics, metric)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            failures.append(f"{metric}: {val if val is not None else 'missing'}")
            continue
        if "min" in rule and val < rule["min"]:
            failures.append(f"{metric}={val:.4f} < min {rule['min']}")
        if "max" in rule and val > rule["max"]:
            failures.append(f"{metric}={val:.4f} > max {rule['max']}")
        if "min_vs_reference" in rule:
            ref = _get(reference.metrics, metric) if reference else None
            if not isinstance(ref, (int, float)):
                failures.append(f"{metric}: reference value unavailable ({ref})")
            elif val < ref + rule["min_vs_reference"]:
                failures.append(f"{metric}={val:.4f} < reference {ref:.4f}{rule['min_vs_reference']:+}")
    return BarCheck(result.candidate, not failures, failures)


@dataclass
class Selection:
    winner: str | None
    checks: list[BarCheck]
    ranked: list[tuple[str, float, bool]]  # (candidate, cost_per_1k, passed) cheapest first
    reference: str | None = None


def cheapest_passing(
    results: Sequence[RunResult],
    bar: dict,
    *,
    reference: str | None = None,
    exclude: Sequence[str] = (),
    cost_metric: str = "cost_usd_per_1k",
) -> Selection:
    """Cheapest candidate that clears ``bar``; ties on cost break by name for determinism.

    ``exclude`` drops candidates that must never win (e.g. majority/random baselines).
    """
    ref = next((r for r in results if r.candidate == reference), None) if reference else None
    if reference and ref is None:
        raise ValueError(f"reference candidate {reference!r} not in results")
    eligible = [r for r in results if r.candidate not in set(exclude)]
    checks = {r.candidate: check_bar(r, bar, ref) for r in eligible}
    ranked = sorted(
        ((r.candidate, float(_get(r.metrics, cost_metric) or 0.0), checks[r.candidate].passed) for r in eligible),
        key=lambda t: (t[1], t[0]),
    )
    winner = next((name for name, _, ok in ranked if ok), None)
    return Selection(winner, list(checks.values()), ranked, reference)


@dataclass
class GateResult:
    passed: bool
    lines: list[str]


def regression_gate(
    current: RunResult,
    baseline: RunResult,
    tolerances: dict[str, dict[str, float]],
    *,
    require_same_dataset: bool = True,
) -> GateResult:
    """``tolerances`` per metric: ``{"higher_is_better": bool, "max_drop": float}``.

    ``max_drop`` is absolute (0.02 = two points of F1). A dataset hash mismatch fails the
    gate by default: comparing runs on different data is how regressions hide.
    """
    lines, ok = [], True
    if require_same_dataset and current.dataset_sha256 != baseline.dataset_sha256:
        return GateResult(False, [f"FAIL dataset changed: {baseline.dataset_sha256[:12]} -> {current.dataset_sha256[:12]}"])
    for metric, tol in tolerances.items():
        cur, base = _get(current.metrics, metric), _get(baseline.metrics, metric)
        if cur is None or base is None:
            ok = False
            lines.append(f"FAIL {metric}: missing (current={cur}, baseline={base})")
            continue
        higher = tol.get("higher_is_better", True)
        drop = (base - cur) if higher else (cur - base)
        allowed = tol.get("max_drop", 0.0)
        status = "ok  " if drop <= allowed + 1e-12 else "FAIL"
        ok &= status == "ok  "
        lines.append(f"{status} {metric}: baseline={base:.4f} current={cur:.4f} (worse by {max(drop, 0):.4f}, allowed {allowed})")
    return GateResult(ok, lines)
