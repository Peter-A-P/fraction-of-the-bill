# Working notes for Claude Code

This repository is Frontier Quality at a Fraction of the Bill, package `smallprint`:
fine-tuning three small open-weights models on structured extraction from SEC filings with
XBRL truth, quantising them, serving them on rented GPUs, and publishing the
quality-latency-cost frontier and the break-even against frontier APIs. Started
2026-09-14 and built as fast as the work allows: the original April 2027 schedule and the
week-by-week dates in the plan are void. Treat PLAN.md section 5 as the order of work only,
and never wait on a calendar date. The plan is in [PLAN.md](PLAN.md).

## Read first

- [README.md](README.md): what this is and the current result tables.
- [PLAN.md](PLAN.md): the design. Do not deviate from it silently; if something in it turns
  out wrong, change the plan in the same commit as the code and say why in the commit
  message.
- `docs/models.md` and `docs/gpu-prices.md` once they exist: the bases, their licences and
  cutoffs, and the GPU decision with the day-one price table.

## Engineering standard

- Python 3.13. Typed throughout; `mypy --strict` and `ruff` clean in CI.
- Tests that fail meaningfully: the grader against adversarial fixtures, the company-level
  split test, the price-entry arithmetic.
- `pyproject.toml` with pinned major versions and a comment saying why for each pin.
- Docs ship in the same commit as the change. Every published model has a card with
  failure modes in the same commit as the upload.
- Never commit credentials, provider keys, Hugging Face tokens, or anything from `.env`.

## Rules specific to this repository

- **No model grades another model.** Truth is the XBRL fact; grading is programmatic.
- **Headline numbers come from the post-cutoff test set.** The pre-cutoff set is reported
  as the contamination gap, never as the headline.
- **No filer appears in more than one split.** The test enforces it.
- **Nothing is measured on the laptop** except the one labelled 4 GB edge point.
- **Checkpoint from the first step.** Every training run resumes from object storage.
- **Every model call goes through the gateway**, frontier and self-hosted alike, so cost
  per call has one source.
- **Quantisation is measured, not assumed.** No format ships without its paired delta.
- **Every reported number carries a confidence interval**, and every price carries its
  date.
- **Plain punctuation** in everything written here: no em-dashes or other typographic
  dashes, straight quotes only.

## What goes in the README

The README opens with the one-liner, the results tables and the honest limitation, before
any installation instructions. The report command fills the tables; do not hand-edit them.
Record one approach that was tried and rejected, with the evidence, once the work has
produced it.
