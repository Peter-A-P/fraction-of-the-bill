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

## Which models, and why those

Chosen 2026-09-20. The set spans the price range on purpose, because the two jobs a
baseline does pull in opposite directions.

**The quality bar.** "A 4B model an organisation owns matches frontier quality" is only
interesting against the best thing money buys, so one model is the top of the range.

**The cost anchor.** The break-even asks at what volume self-hosting beats paying per
call. If the only API in the comparison is the most expensive one, self-hosting wins
trivially and the finding is worth nothing. The anchor is the cheapest API that does this
task acceptably, because that is what a team running a hundred thousand extractions a day
would be paying.

| Role | Model | US$/Mtok in/out | A 716-item run, zero-shot | Two-shot |
|---|---|---|---:|---:|
| Ceiling | `openai/gpt-5.6-sol` | 4 / 20 | US$14.24 | US$25.30 |
| Middle, second vendor | `anthropic/claude-sonnet-5` | 2 / 10 | US$7.12 | US$12.65 |
| Floor, the cost anchor | `openai/gpt-5.6-luna` | 0.2 / 1.2 | US$0.76 | US$1.31 |

Rates are from the price list boundary v0.3.0 ships, dated 2026-09-14. The run costs are
worked from the token counts the vendor actually returned on the smoke run below, a median
3,301 in and 334 out, rather than from the characters-over-four guess this page carried
first; that guess was 25% low on the input side. Six runs come to about **US$61**, and the
ceiling model two-shot at US$25.30 is 84% of the US$30 per-run cap. Two vendors rather
than one, so the bar is not a single lab's quirk on a single task.

**What the floor implies, stated before it is measured.** Luna costs about US$0.83 per
1,000 extractions. A rented GPU at US$2 an hour serving five requests a second at 50%
utilisation costs about US$0.22 per 1,000. So the honest expectation is that self-hosting
wins by something like four times, not by the fifty times the framing of a project called
"a fraction of the bill" might suggest, and the reasons to do it are control, latency and
data residency as much as price. The break-even curve is where that gets settled.

Rejected: `gpt-5-mini` and `gpt-5`, which were in the first draft of this table. They are
a generation and a half old, and they were in it because they were the identifiers
easiest to recognise rather than because they were the right comparison.

## The gateway

Calls go through [`boundary`](https://github.com/Peter-A-P/compliant-ai-gateway), pinned
at **v0.3.0** in the `gateway` extra. That release adds the two things this project needs
beyond ordinary chat: `ttft_ms` on a streamed call, which is the latency axis, and the
self-hosted price overlay, which is how a GPU-hour rate becomes a cost per call in the
same ledger as a vendor bill. Install it with `uv sync --extra gateway`.

The configuration is [`boundary.yaml`](../boundary.yaml) in this repository, with
[`caps.yaml`](../caps.yaml) beside it. An earlier draft of this page said the repository
would hold neither, on the grounds that a copy of the configuration is a second place for
a rate to be wrong. That reasoning does not apply to this file, because it holds no rates:
`prices: builtin` uses the dated price files inside the pinned library, so the version pin
decides the costing and a published cost table is reproducible from a checkout alone. It
holds no credentials either. A provider entry names the environment variable its key is
read from, never the key, and the two variables are `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY`.

`caps.yaml` is the one real duplication. boundary needs a caps file beside the
configuration it is given, and a caps file has to state the portfolio figure as well as
this project's, so the portfolio's US$400 appears here as well as in the gateway
repository, which owns it. The file says so at the top: if the two disagree, the gateway's
copy is right.

The project name is `fraction-of-the-bill`, which is the key the spend cap is filed under:
US$100 a month, US$30 a run.

## Running one

    smallprint baseline prompt --style few_shot --build-dir data/build/full
    smallprint baseline run --model openai/gpt-5.6-luna \
      --build-dir data/build/full --split test_post_cutoff --style zero_shot \
      --out data/baseline/luna-zero --limit 20
    smallprint baseline report --run-dir data/baseline/luna-zero --fields

`--limit` runs the first N items and is how a smoke run costs cents rather than dollars.
The run is resumable, so a smoke run of 20 becomes the full run by dropping the flag.

## What the first smoke run found, 2026-09-20

Twenty post-cutoff items on `gpt-5.6-luna`, zero-shot, pass-through. Three attempts,
sixty ledger rows, forty of them errors, **US$0.0207 in total**. Everything below is why a
smoke run of twenty exists before a run of seven hundred.

**Two provider rejections, both 400 on every call.** First, `max_tokens` is not accepted
by the current OpenAI models: "Use `max_completion_tokens` instead." boundary has a
provider setting for exactly this, `max_tokens_field`, now set in `boundary.yaml`. Second,
those models refuse a temperature other than the default: "Unsupported value:
'temperature' does not support 0.0 with this model. Only the default (1) value is
supported."

**So temperature is not the same across the table, and cannot be.** Anthropic accepts 0;
the current OpenAI models accept only 1. `--vendor-temperature` sends none and takes the
vendor's default, the manifest records which of the two a run used, and two runs that
differ in it cannot resume one another. It means the OpenAI rows are measured at a
temperature that samples and the Anthropic rows at one that does not, which is a real
asymmetry in the results table and is stated there rather than smoothed over.

**A run that fails everywhere is now retried rather than stuck.** The first attempt wrote
twenty failed predictions, and resuming skipped all twenty as "done". A filing that was
answered is still never called twice; a filing whose last attempt failed is called again,
because the usual reason a whole run fails is one the next run has fixed. Errors are not
billed, so this cannot spend twice in any case that matters.

**The result, on twenty items.** Field accuracy 94.3% (91.7% to 96.7%), every field right
on 45% of filings, nothing unparseable, US$1.00 per 1,000 calls against an estimate of
US$0.83, median latency 2.66 s. Thirteen of the fifteen fields were at or near 100%.

**One failure mode accounts for almost all of it, and it is a real one.** On
`total_liabilities` the model scored 45%, with eleven of twenty marked `hallucinated`:
truth null, model confident. Those filers print only "Total liabilities and shareholders'
equity" and tag no `us-gaap:Liabilities` fact, so a correct reading of the page answers
null. The model subtracted equity from the total and returned the difference. The prompt
forbids that in two places, rule 4 and the field's own description ("Not total liabilities
and equity"), and the grader named it correctly. This is the single most useful thing the
smoke run produced: the frontier model's exact-match rate is dragged from a 94% field
score to 45% almost entirely by deriving a figure it was told not to derive, which is
exactly the behaviour a fine-tune on this corpus should train out.

**Time to first token is not measured yet, and cannot be in this mode.** Streaming is
where `ttft_ms` comes from, and boundary v0.3.0 refuses to stream in pass-through mode,
which is the mode every published accuracy number runs in. So the latency column for the
frontier models will need either a separate, clearly labelled streaming run in standard
mode with retries off, or a change in the gateway. Not decided.

## Not measured yet

No full baseline has been run: the spend so far is two cents. The result tables in the
README stay empty until a run fills them.
