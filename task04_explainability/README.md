# Task 1.4 — Explainability (20 pts)

This component lives **outside** `solution/` per the PDF
("`<other-files>` (e.g. code for task 1.4)"). It is **not** invoked by the
grader's pipeline; the discussion goes into `report/report.md` §1.4.

## Goal

Reason about *why* the final model makes its decisions and where it fails —
not just report a score. Treat explanations critically: they may be plausible,
they may reveal shortcuts (e.g. JPEG artifacts, color-cast bias), or they may
just be noise.

## Reasonable directions (PDF §1.4 — pick at least one and justify)

1. **Saliency / gradient-based explanations** — what pixels does the model rely on?
2. **Occlusion / perturbation analysis** — which image regions, when masked,
   change the score most?
3. **Failure analysis** — qualitative inspection of false positives (real images
   flagged as AI) and false negatives (AI images flagged as real).
4. **Real vs. AI comparison** — does the attention pattern systematically differ
   between the two classes?

## Inputs

Reads the trained model artifacts from `solution/artifacts/`:

- `solution/artifacts/task02/best.pt` (Task 2 model)
- `solution/artifacts/task03/best.pt` (Task 3 robust model — usually the "final")
- corresponding `threshold.json` files

If you run this **after** the grader pipeline locally, those files exist. If
you ship only the explainability code in the zip, copy the relevant artifact
into a sibling folder before running.

## Deliverables

- Code in [explain.py](explain.py).
- Figures referenced from `report/report.md` §1.4 (place under `report/figures/`).
- A discussion paragraph in the report on plausibility, shortcut behavior, and
  dataset bias.
