"""Markdown tables for results. The table is the artifact people read, so it never hides a gap:
INSUFFICIENT and missing values print as such, and the selected winner is named explicitly."""

from __future__ import annotations

from collections.abc import Sequence

from .gate import Selection
from .runner import RunResult


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _ci(m: dict) -> str:
    ci = m.get("macro_f1_ci95")
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—"


def classification_table(results: Sequence[RunResult], selection: Selection | None = None) -> str:
    passed = {c.candidate: c for c in selection.checks} if selection else {}
    head = ("| candidate | n | accuracy | macro-F1 | 95% CI | ECE | error rate | $ / 1k | p50 ms | bar |\n"
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---|")
    rows = []
    order = {name: i for i, (name, _, _) in enumerate(selection.ranked)} if selection else {}
    for r in sorted(results, key=lambda r: (order.get(r.candidate, -1), r.candidate)):
        m = r.metrics
        chk = passed.get(r.candidate)
        if selection and r.candidate == selection.reference:
            bar = "reference"
        elif chk is None:
            bar = "baseline" if selection else "—"
        else:
            bar = "**PASS**" if chk.passed else "fail: " + "; ".join(chk.failures)
        if selection and r.candidate == selection.winner:
            bar += " ✅ cheapest passing"
        rows.append(
            f"| {r.candidate} | {m.get('n')} | {_fmt(m.get('accuracy'))} | {_fmt(m.get('macro_f1'))} | {_ci(m)} "
            f"| {_fmt(m.get('ece'))} | {_fmt(m.get('error_rate'))} | {_fmt(m.get('cost_usd_per_1k'), 4)} "
            f"| {_fmt(m.get('latency_ms_p50'), 0)} | {bar} |"
        )
    out = head + "\n" + "\n".join(rows)
    if selection:
        out += "\n\n" + (f"**Cheapest model that passes the bar: `{selection.winner}`.**" if selection.winner
                         else "**No candidate passes the bar.**")
    return out


def retrieval_table(results: Sequence[RunResult]) -> str:
    if not results:
        return ""
    k = results[0].metrics.get("k")
    head = (f"| retriever | n | recall@{k} | hit@{k} | MRR | nDCG@{k} | critical recall@{k} |\n"
            "|---|---:|---:|---:|---:|---:|---:|")
    rows = [
        f"| {r.candidate} | {r.metrics['n']} | {_fmt(r.metrics.get(f'recall@{k}'))} | {_fmt(r.metrics.get(f'hit@{k}'))} "
        f"| {_fmt(r.metrics.get('mrr'))} | {_fmt(r.metrics.get(f'ndcg@{k}'))} | {_fmt(r.metrics.get(f'critical_recall@{k}'))} |"
        for r in results
    ]
    return head + "\n" + "\n".join(rows)
