"""Build the README graphics for ailab-evals.

Four graphics, Dark Workshop system:
  hero       wheel of the four eval suites                              (structural)
  flow       how a run becomes a gate: suite -> run/replay -> score     (structural)
  anatomy    the four real results, every number parsed from a capture  (capture-driven)
  catalog    the make-target command surface                           (structural)

Only `anatomy` carries data, and it reads every number from captures/demo.json (a committed
`make demo` run), never from a typed-in literal. Run:
  uv run --with playwright --with pillow python docs/assets/src/render.py docs/assets/src docs/assets
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import readme_kit as k  # noqa: E402

CAP = HERE / "captures" / "demo.json"


def _cell(line: str, col: int) -> str:
    """Return the stripped value in markdown-table column `col` (0-based) of a `| a | b |` line."""
    return [c.strip() for c in line.strip().strip("|").split("|")][col]


def parse_capture() -> dict:
    """Pull the four headline results out of the committed `make demo` capture. No typed numbers."""
    cap = k.load_capture(CAP)
    out = cap["output"]
    rows = out.splitlines()
    vals: dict[str, str] = {"commit": cap["commit"][:9], "captured_at": cap["captured_at"]}

    # 1. Cheapest-passing classifier: the line the harness prints, plus sonnet's reference cost.
    m = re.search(r"Cheapest model that passes the bar: `([^`]+)`", out)
    vals["cheapest"] = m.group(1)
    for r in rows:
        if r.startswith("| claude-sonnet-4.5 |") and "reference" in r:
            vals["sonnet_cost"] = _cell(r, 7)        # $ / 1k column
        if r.startswith("| gemma3-27b-local |") and "cheapest passing" in r:
            vals["cheapest_f1"] = _cell(r, 3)        # macro-F1 column

    # 2. Judge calibration: haiku (trusted) and gemma3 (disqualified on false-pass).
    for r in rows:
        if r.startswith("| claude-haiku-4.5 |") and r.count("|") == 8:
            vals["haiku_kappa"] = _cell(r, 3)
            vals["haiku_fp"] = _cell(r, 4)
        if r.startswith("| gemma3-27b-local |") and ("NO" in r):
            vals["gemma_judge_fp"] = _cell(r, 4)
            vals["gemma_judge_kappa"] = _cell(r, 3)

    # 3. Pairwise preference.
    for r in rows:
        if r.startswith("| confident_wrong |"):
            vals["cw_winrate"] = _cell(r, 5)
    m = re.search(r"position-consistency: ([\d.]+%)", out)
    vals["consistency"] = m.group(1)

    # 4. Retrieval baseline.
    for r in rows:
        if r.startswith("| bm25 |"):
            vals["recall3"] = _cell(r, 2)
            vals["crit_recall3"] = _cell(r, 6)
    return vals


def build_hero() -> str:
    wheel = k.wheel(
        ["JUDGE", "PAIRWISE", "CHEAPEST", "RETRIEVAL"],
        center_top="FOUR", center_main="SUITES", size=540, node_r=48,
    )
    return k.hero(
        kicker="AILAB · LAB 1 · EVALS",
        title="An evals harness",
        accent="for LLM decisions",
        lede_html=(
            "One question that moves money: <b style='color:#F4EFE6'>which is the cheapest "
            "model that is good enough?</b> And an honest refusal when the data cannot answer."
        ),
        rules=[
            ("Hashed, labeled data", "Every result names the sha256 of the exact dataset it measured."),
            ("Calibrated judges", "A model judge is trusted only after kappa and false-pass clear preset bars."),
            ("Replay gate in CI", "Recorded outputs re-score offline, so the gate is hermetic and costs nothing."),
        ],
        pill="ZERO RUNTIME DEPENDENCIES",
        right_html=wheel,
        footer_left="AILAB-EVALS · HOW IT WORKS",
    )


def build_flow() -> str:
    boxes = "".join([
        k.box(56, 250, 250, 150, "SUITE · JSON", [
            "dataset + sha256", "prompt + candidates", "prices and the bar"]),
        k.box(390, 250, 250, 150, "RUN / REPLAY", [
            "cassette keyed by", "sha256(model, prompt)", "a miss fails, never guesses"]),
        k.box(724, 250, 250, 150, "SCORE", [
            "macro-F1 + bootstrap CI", "judge kappa, false-pass", "cost $/1k, p50 ms"]),
        k.box(1058, 250, 286, 150, "GATE", [
            "vs a committed baseline", "exit 1 on a regression", "or on a changed dataset"], accent=True),
        k.box(390, 470, 250, 120, "CASSETTE · RECORDED", [
            "real model outputs", "temperature 0, offline", "free and deterministic"]),
        k.box(1058, 470, 286, 120, "COMMITTED RESULTS", [
            "results/*/table.md", "CI re-derives and diffs", "schema-1 json for dashboards"]),
    ])
    arrows = [
        (306, 325, 388, 325),
        (640, 325, 722, 325),
        (974, 325, 1056, 325),
        (515, 470, 515, 402, "replay", False, "right"),
        (1201, 402, 1201, 470, "writes", False, "right"),
    ]
    return k.flow(
        kicker="HOW IT WORKS",
        title_html="How a run becomes a " + k.em("gate"),
        subline="A suite names its data by hash · replay re-scores recorded outputs offline · the gate fails on any regression",
        boxes_html=boxes,
        arrow_specs=arrows,
        footer_left="AILAB-EVALS · HOW IT WORKS",
    )


def build_anatomy(v: dict) -> str:
    amber = "color:#E8912D;font-weight:600"
    doc_lines = [
        ("h1", "make demo · four suites, one offline run"),
        ("m", f"replayed from committed cassettes · commit {v['commit']} · {v['captured_at']}"),
        ("h2", "1 · Judge calibration: can each model be trusted to grade?"),
        ("li", f"claude-haiku-4.5: kappa {v['haiku_kappa']}, false-pass {v['haiku_fp']} &rarr; "
               f"<span style='{amber}'>trusted</span>"),
        ("li", f"gemma3-27b-local: kappa {v['gemma_judge_kappa']}, false-pass {v['gemma_judge_fp']} &rarr; "
               f"<span style='{amber}'>NOT trusted</span>"),
        ("i", "false-pass, not kappa, is what disqualifies a grader"),
        ("h2", "2 · Pairwise preference, position-swapped"),
        ("li", f"confident_wrong win rate {v['cw_winrate']}: the wrong system is ranked last"),
        ("li2", f"judge position-consistency {v['consistency']} of comparisons"),
        ("h2", "3 · Cheapest model that passes the bar"),
        ("li", f"pick: <span style='{amber}'>{v['cheapest']}</span> "
               f"(macro-F1 {v['cheapest_f1']}, $0 marginal API cost)"),
        ("li2", f"claude-sonnet-4.5 is the reference at ${v['sonnet_cost']} / 1k"),
        ("h2", "4 · BM25 retrieval baseline"),
        ("li", f"recall@3 {v['recall3']} overall · critical recall@3 {v['crit_recall3']}"),
    ]
    notes = [
        (120, "Every number here is parsed from the committed capture, never typed."),
        (250, "The same gemma3 that wins as a classifier fails as a judge. Good enough is per task."),
        (430, "All four models saturate this easy fixture, so the pick falls to cost."),
        (560, "The misses that matter get their own line: critical recall trails overall."),
    ]
    return k.anatomy(
        kicker="ANATOMY OF A REAL RUN",
        doc_lines=doc_lines,
        notes=notes,
        footer_left="AILAB-EVALS · REAL RUN 2026-10-03",
    )


def build_catalog() -> str:
    cards = [
        ("make demo", "Replay every suite from the committed cassettes", "4 suites, offline", "no network, no API key"),
        ("make test", "Run the unit test suite", "uv run pytest -q", "runner, metrics, judge, gate"),
        ("make lint", "Ruff over the whole tree", "uv run ruff check .", "style and errors"),
        ("make hooks", "Enable the mandatory pre-push secret wall", "core.hooksPath .githooks", "gitleaks + deny-list"),
        ("make fixtures", "Rebuild the synthetic fixtures", "python build_fixtures.py", "deterministic, byte-for-byte"),
        ("make record", "Re-record cassettes against live models", "needs API key + Ollama", "temperature 0"),
    ]
    return k.catalog(
        kicker="THE COMMAND SURFACE",
        title_html="Everything you run is a " + k.em("make target"),
        sub_html="Six targets. The demo and the tests need no key and no network.",
        cards=cards,
        footer_left="AILAB-EVALS · HOW IT WORKS",
        cols=3,
        card_height=150,
    )


def main() -> None:
    v = parse_capture()
    pages = {
        "hero": build_hero(),
        "architecture": build_flow(),
        "anatomy": build_anatomy(v),
        "catalog": build_catalog(),
    }
    k.write_pages(HERE, pages)


if __name__ == "__main__":
    main()
