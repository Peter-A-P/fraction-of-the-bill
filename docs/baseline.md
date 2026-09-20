# The prompt, and running a model over held-out filings

Every accuracy number this project publishes comes from one runner, one prompt and one
grader: the frontier APIs, the untuned bases, the fine-tunes and the quantised builds all
answer the same question in the same words. If the frontier baseline ran on a
better-written prompt than the small models, the gap in the headline table would be a gap
between two prompts.

The code is [`smallprint/prompts`](../smallprint/prompts/__init__.py) and
[`smallprint/baseline.py`](../smallprint/baseline.py); the commands are
`smallprint baseline prompt | run | report`.

## The prompt

The field list is generated from `SCHEMA`, so the prompt, the training target and the
grader cannot drift apart: a field added to the schema appears in the prompt on the next
run without anyone remembering to add it. What is written by hand is the part a schema
cannot express, which is which of the several numbers on a page is the answer. Five rules,
each of which exists because it is a failure the grader has a name for:

| Rule | The miss it is aimed at |
|---|---|
| Values are in whole units; apply the scale printed above the statement | `scale` |
| Report the period the filing covers, not the columns beside it | `wrong_period` |
| A loss or a deficit is negative; parentheses mean negative | `sign` |
| If the filing does not report a field, the value is null | `hallucinated` |
| Answer with the JSON object alone | `malformed` |

The scale rule is worked rather than asserted ("a line reading 1,234 under a heading that
says in thousands is 1234000") and it says that share counts carry their own scale, which
is the case the corpus build found filers printing differently from the money columns.

**Fingerprint.** Every prompt has a sha256 over everything the model is shown except the
filing itself, and every run records it. Two accuracy numbers are comparable only if the
fingerprints match. An edit to a rule that reads as cosmetic changes the fingerprint,
which is the point: it makes the comparison refuse rather than quietly shift.

**Reasoning.** Gemma 4 opts into reasoning with a `<|think|>` token at the head of the
system prompt. `--thinking` puts it there, and the manifest records it, so the untuned
Gemma baselines can be measured both ways and the fine-tunes, which are trained to answer
without it, are measured against the better of the two.

## Few-shot examples

Examples come from the **training pool only**. `examples_from` raises if it is handed a
pool holding a validation or test filing rather than filtering it out, because an example
drawn from the test set puts the answer to a held-out item into the prompt of its
neighbours and inflates every number measured afterwards.

Within the training pool, two rules:

- **Forms take turns**, so a two-shot prompt shows one annual report and one quarterly
  one. Not only for the wording: an annual report has an auditor to name and a quarterly
  one does not, so the pair demonstrates a filled field and a null one without a rule
  having to describe what a null looks like. On the full corpus the two chosen are
  `0001437749-24-030208` (10-K, auditor Haskell & White LLP) and `0001558370-24-012226`
  (10-Q, auditor null).
- **Drawn from the shorter half** of the pool by input length, by keyed hash, so a
  few-shot prompt costs about what the zero-shot one does per example rather than being
  decided by one very long filing, and so every machine builds the same prompt.

## What a run sends, measured on the full corpus

Whole prompt, system text and examples included, over the 716 post-cutoff test items:

| Style | Median characters | p90 | Roughly, in tokens |
|---|---:|---:|---:|
| Zero-shot | 10,549 | 12,824 | 2,600 |
| Two-shot | 22,889 | 25,164 | 5,700 |

The token column is characters over four and is an estimate; the ledger records what each
vendor actually counted, and that is what the cost table uses. `PLAN.md` section 6 budgeted
3.5k tokens in per call, so the zero-shot runs come in under the estimate and the two-shot
runs over it; the plan's total is not changed by this and is not amended.

## How a run behaves

**Pass-through by default.** A published number is a measurement, so the runner asks the
gateway for pass-through mode: no retries, no development cache, no filled-in defaults,
and the request and response bytes kept in the run's own `raw/`. A retry turns an error
rate into a latency, and a cache hit turns a measurement into a memory of one. Standard
mode exists for development and the manifest records which mode was used, so a number
measured with the cache on cannot be published as if it were not. `summary.json` reports
how many answers came from the cache.

**Temperature zero.** Not because vendors promise determinism at zero, which they do not,
but so that a rerun differs by the vendor's nondeterminism alone.

**Resumable.** Predictions are appended one line at a time and the manifest is written
before the first call, so an interrupted run over two thousand filings is finished rather
than paid for twice. Resuming into a directory whose manifest names a different model,
prompt, split, mode or token ceiling is refused: that is two runs, not one.

**One split per run.** Mixing the post-cutoff test set with anything else would produce a
directory whose headline number averages two different claims.

**A failed call is not a wrong answer.** A call that never returned text is counted and
named, not graded as zero. A model measured on the nine tenths of items it managed to
answer is not measured on the task, so the failure count is printed beside the accuracy.

## A run directory

    run.json            what was run: model, split, prompt fingerprint, mode, when
    predictions.jsonl   one line per filing: the answer, the ledger row id, cost, latency, TTFT, tokens
    raw/                pass-through request and response bytes
    summary.json        accuracy, exact match, unparseable share, cost per 1,000, latencies, per-field report

Every number in `summary.json` carries a 95% interval, bootstrapped over filings.

## The gateway

Calls go through [`boundary`](https://github.com/Peter-A-P/compliant-ai-gateway), pinned
at **v0.3.0** in the `gateway` extra. That release adds the two things this project needs
beyond ordinary chat: `ttft_ms` on a streamed call, which is the latency axis, and the
self-hosted price overlay, which is how a GPU-hour rate becomes a cost per call in the
same ledger as a vendor bill.

This repository holds no `boundary.yaml`. The configuration names providers, credential
sources and price files, and a copy of it here would be a second place for a rate to be
wrong. `--config` points at one. It must:

- give the project the name `fraction-of-the-bill`, which is the key its spend cap is
  filed under in the gateway's caps file, currently US$100 a month and US$30 a run;
- name the vendor price files, so every row is costed and dated.

Install the extra with `uv sync --extra gateway`.

## Running one

    smallprint baseline prompt --style few_shot --build-dir data/build/full
    smallprint baseline run --model anthropic/claude-haiku-4-5-20251001 \
      --build-dir data/build/full --split test_post_cutoff --style zero_shot \
      --out data/baseline/haiku-zero --config ../boundary.yaml --limit 20
    smallprint baseline report --run-dir data/baseline/haiku-zero --fields

`--limit` runs the first N items and is how a smoke run costs cents rather than dollars.
The run is resumable, so a smoke run of 20 becomes the full run by dropping the flag.

## Not measured yet

No baseline has been run. Nothing has been spent through the gateway on this project. The
result tables in the README stay empty until a run fills them.
