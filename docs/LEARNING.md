# LEARNING — what building this taught, and how I'd explain it

## The one-paragraph version

An eval harness is a measurement instrument, and the failure modes of instruments are boring and
deadly: dropped rows, unpinned data, unvalidated judges, and thresholds chosen after looking at the
results. Almost every design decision here exists to stop one of those. Errors are scored as wrong
(never skipped). Every result carries the sha256 of the data it measured. A model judge isn't trusted
until it has been calibrated against labels whose correctness is known. Bars are written down before the run, and
a class with too few examples prints `INSUFFICIENT` instead of a number. The money question —
*which is the cheapest model that is good enough?* — only becomes answerable once all of that holds.

## Things that were less obvious than expected

1. **Macro-F1, not accuracy, for imbalanced decisions.** A majority-class baseline gets 90% accuracy on
   a 90/10 split and a macro-F1 of ~0.47. The baselines (`majority`, `random`) belong in every table
   precisely so this is visible.
2. **Judges have position bias.** Pairwise judges prefer whichever answer they see first. Asking twice
   with A/B swapped and only accepting a winner that survives the swap turns bias into ties, which is
   honest; ignoring it silently inflates whichever system you listed first.
3. **The dangerous judge error is the false pass.** A judge that rejects good answers costs you a
   re-run; a judge that passes bad answers ships them. `calibrate()` reports the two directions
   separately and refuses `trusted` on a high false-pass rate whatever kappa says.
4. **Kappa, not raw agreement.** Two raters who both say "pass" 90% of the time agree 82% by chance.
   Cohen's kappa subtracts that.
5. **Calibration is fitted on held-out rows.** Platt scaling fitted on the scoring rows grades its own
   homework. The `calibrate` split exists only to fit it.
6. **Record/replay makes evals cheap enough to gate on.** Live model calls are slow, cost money and
   drift. A cassette keyed by `sha256(provider, model, params, system, prompt)` turns a recorded run
   into a free, deterministic CI job — and any prompt change is a cassette *miss*, so a stale
   recording can never silently pass.
7. **"Cheapest that passes" needs a reference, not an absolute.** "Within 3 macro-F1 points of the
   current production model" is a bar you can defend; "F1 ≥ 0.85" invites arguing about the number.
8. **An easy fixture can't tell models apart.** On the public `substance` fixture every model scores
   1.0, so it proves the plumbing and the cost column, not model quality. The judge-calibration suite
   is where models actually differ.

## Interview questions (with the short answer I'd give)

1. **How do you know your LLM-as-judge is any good?** Calibrate it against labels whose correctness is
   known on the same item type: agreement, Cohen's kappa, and false-pass / false-fail separately. Don't
   use it as a gate until it clears a pre-set kappa and false-pass ceiling.
2. **Why macro-F1 over accuracy for a classifier bake-off?** It weights every class equally, so a
   model can't win by ignoring the minority class. Always show a majority baseline next to it.
3. **What is position bias in pairwise evaluation and how do you control it?** Judges favour the first
   answer shown. Run both orders; a winner must win in both, otherwise it's a tie. Report the
   position-consistency rate as a judge-quality signal.
4. **How would you pick the cheapest model that's good enough?** Pre-commit a bar relative to the
   current model (e.g. macro-F1 ≥ reference − 0.03, error rate ≤ 2%), run every candidate on the
   same hashed dataset, then select the lowest $/1k that clears it. Baselines can't win.
5. **What makes an eval regression gate trustworthy in CI?** Hermetic replay (no network, no key), a
   dataset hash check (comparing runs on different data hides regressions), errors counted as wrong,
   and tolerances written down before the change, not after.
6. **How do you handle tiny classes?** Set a per-class floor in advance; below it, report
   `INSUFFICIENT` rather than a number with a meaningless confidence interval.
7. **Why bootstrap confidence intervals?** Eval sets are small. Two models 2 points apart with
   overlapping CIs are a tie; the CI tells you whether you need more labels.
8. **What's ECE and when does it matter?** Expected calibration error: the gap between stated
   confidence and actual accuracy, averaged over confidence bins. It matters the moment you route on
   confidence (e.g. "auto-accept above 0.9, send the rest to review").
9. **How do you keep private data out of a public eval repo?** Synthetic or public fixtures only, a
   provenance file, a pre-push hook plus a required CI job running gitleaks and a deny-list — with the
   deny-list stored as hashes so the list itself leaks nothing.
10. **What's the difference between offline evals and production monitoring?** Offline evals compare
    candidates on a fixed, labeled set before a change; monitoring watches live traffic for drift
    after it. You need both: a model that passed offline can still degrade when the input
    distribution moves.
