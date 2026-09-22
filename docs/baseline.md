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

| Style | Median characters | p90 | Tokens, OpenAI | Tokens, Anthropic |
|---|---:|---:|---:|---:|
| Zero-shot | 10,549 | 12,824 | 2,900 | 4,300 |
| Two-shot | 22,889 | 25,164 | 6,600 | 9,800 |

The token columns are the measured tokens per character for each vendor's tokeniser, 0.28
and 0.41, applied to the median; the ledger records what each vendor actually counted and
that is what the cost table uses. `PLAN.md` section 6 budgeted 3.5k tokens in per call, so
the zero-shot runs land near the estimate and the two-shot runs above it; the plan's total
is not changed by this and is not amended.

## How a run behaves

**Pass-through by default.** A published number is a measurement, so the runner asks the
gateway for pass-through mode: no retries, no development cache, no filled-in defaults,
and the request and response bytes kept in the run's own `raw/`. A retry turns an error
rate into a latency, and a cache hit turns a measurement into a memory of one. Standard
mode exists for development and the manifest records which mode was used, so a number
measured with the cache on cannot be published as if it were not. `summary.json` reports
how many answers came from the cache.

**No temperature.** Not a preference: all three frontier models reject a value outright.
The runner leaves the field out and the vendor's default applies, and the manifest records
that. See the rehearsal below.

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
| Ceiling | `openai/gpt-5.6-sol` | 4 / 20 | US$10.74 | US$22.21 |
| Middle, second vendor | `anthropic/claude-sonnet-5` | 2 / 10 | US$7.96 | US$16.76 |
| Floor, the cost anchor | `openai/gpt-5.6-luna` | 0.2 / 1.2 | US$0.64 | US$1.17 |

Rates are from the price list boundary v0.3.0 ships, dated 2026-09-14. The run costs are
projections from the rehearsal below: tokens per character measured per model against the
character count of every item in the split, and mean output tokens measured per model and
style. Six runs come to about **US$59**, and the most expensive single run is 74% of the
US$30 per-run cap. Two vendors rather than one, so the bar is not a single lab's quirk on
a single task.

The two vendors do not count the same text the same way. Anthropic's tokeniser takes
**0.41 tokens per character** of these prompts against OpenAI's **0.28**, about 47% more,
which is worth knowing before reading anything into two vendors' prices per million.

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

**So no frontier run pins its temperature, because none of them may.** The rehearsal
found Anthropic refusing it too: "`temperature` is deprecated for this model." All three
models in the set reject a value, so the runner now sends none by default and takes the
vendor's default, and the manifest records what was sent. A measurement would rather pin
it, so that a rerun differed by the vendor's nondeterminism alone and not by ours as well;
that is no longer on offer from the frontier, and pretending otherwise by passing a value
that gets rejected is worse than saying so. A served open-weights model does accept it,
and the quantisation deltas are small enough that sampling noise would swamp them, so
those runs will pass `--temperature 0` explicitly.

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

## The rehearsal, 2026-09-20

Before spending US$59, 36 calls costing **US$0.51**: six filings spanning the whole length
range of the split, from 3,507 to 12,782 characters, through all three models in both
styles. A twenty-item run on one model could not have found what this was looking for.
Anthropic had never been called at all, which is how the deprecated-temperature rejection
turned up. The ceiling model had never been called, and if it reasoned it would have run
into the 2,048-token ceiling. The two-shot prompt had never gone out.

All 36 answered, all 36 parsed, none truncated: the largest answer was 659 output tokens
against a 2,048 ceiling. Field accuracy on those six items was 95.6% for luna, 95.6% for
sol and 97.8% to 98.9% for sonnet, which is six items and settles nothing, but it does say
the ceiling model is not obviously ahead on this task.

**Time to first token is not measured yet, and cannot be in this mode.** Streaming is
where `ttft_ms` comes from, and boundary v0.3.0 refuses to stream in pass-through mode,
which is the mode every published accuracy number runs in. So the latency column for the
frontier models will need either a separate, clearly labelled streaming run in standard
mode with retries off, or a change in the gateway. Not decided.

## The six runs, 2026-09-20

All three models over all 716 post-cutoff filings, zero-shot and two-shot: 4,296 calls,
US$59.63, no failures, one answer in 4,296 that would not parse. The table is in
[results-baseline.md](results-baseline.md), written by `smallprint report`, and the three
findings are these.

**Re-graded 2026-09-22 on the audited build**, from the saved answers and at no cost. The
hand audit tightened the filter (see [data.md](data.md#the-hand-audit-2026-09-22)), which
dropped 11 of the 716 post-cutoff filings and 29 of the 1,425 pre-cutoff ones, and
corrected labels on others. The numbers below are over the 705 and 1,396 that remain; the
answers to the dropped filings stay on disk and are counted in each summary, not graded.
Every finding survived, and no paired delta moved by more than a tenth of a point.

**The cheapest model is as accurate as the most expensive one.** Paired over the same
filings, `gpt-5.6-sol` zero-shot is -0.3% (-0.8% to +0.2%) against `gpt-5.6-luna`
zero-shot, and `claude-sonnet-5` zero-shot is -0.1% (-0.4% to +0.1%). Two intervals
straddling zero, from models costing 18 and 14 times as much. Only `gpt-5.6-sol` two-shot
clears it, by +0.7% (+0.5% to +1.0%), at 35 times the price. Field accuracy on this task
saturates around 96% and the money buys nothing above it.

**Whole filings are where paying more shows.** Every field right on one filing goes from
58.3% for luna zero-shot to 67.0% for sol two-shot, nearly nine filings in every hundred.
Fifteen fields at 96% each leaves a lot of filings with one thing wrong, and which
fifteenths go wrong is not the same across models. This is the number a user of an
extractor actually feels, and it is the one the fine-tunes have to move.

**Worked examples are worth it for one model out of three.** Two-shot minus zero-shot,
paired: luna +0.0% (-0.3% to +0.3%), sonnet +0.2% (-0.1% to +0.6%), sol +1.0% (+0.6% to
+1.5%). Two nothings and one real gain, at roughly twice the input cost in every case. So
the headline comparison uses zero-shot, and sol two-shot is carried as the best frontier
result anyone paid for.

**What the bar actually is.** The fine-tunes have to reach **96.2% of fields and 58.3% of
filings, at US$0.87 per 1,000 calls**, which is luna zero-shot. Not the most expensive
model: the cheapest one that does the job, because that is what a team running this
volume would be paying, and self-hosting has to beat what they would otherwise spend
rather than what they could have spent.

**The failure is one behaviour, not fifteen.** Sorted worst first, luna's misses are
`cost_of_revenue` 86.5%, `total_liabilities` 90.5%, `eps_basic` 91.9%, and in each case
the reason is `hallucinated`: the filing does not report the field and the model answers
anyway, deriving cost of revenue from gross profit, total liabilities by subtraction, per
share figures the statement omits. 82 of 705 on the first one. The prompt forbids it
twice, and the frontier models do it anyway. Abstention is the behaviour a fine-tune on
this corpus should be able to buy, and it is worth more in production than a point of
accuracy: an abstention can be routed to a person, an invention cannot. The one field
that fails differently is `shares_diluted`, with 17 scale errors, thousands read as units,
though since the audit it hallucinates too: on the filings whose statement prints no share
count, every model supplies one anyway, net income divided by earnings per share.

## The contamination gap, 2026-09-20: there is not one

`gpt-5.6-luna` zero-shot over all 1,425 pre-cutoff filings, US$1.24, re-graded on the 1,396
the audited build keeps. These are filings
published before the base models' training cutoff, so they may be in a pre-training
corpus; the post-cutoff set cannot be. If memorisation were helping, the pre-cutoff set
would score higher.

It scores **lower**: 94.2% (93.7% to 94.6%) against 96.2% post-cutoff, a difference of
**-2.0% (-2.7% to -1.4%)**. Unpaired, because the two sets are different filings, though
129 of the filers appear in both.

So there is no contamination premium to subtract on this task, and the reason is what the
task is. The answer is printed on the page the model is given. Having read the filing in
2024 does not help you copy a number out of it in 2026, and the thing the model actually
gets wrong is not recall.

**Where the older filings are harder, and how much of it is the corpus.** Split every
field by whether the filing reports it at all:

| | Field is reported | Field is not reported | Share unreported |
|---|---|---|---:|
| Pre-cutoff | 98.3% (98.1 to 98.5) | 67.3% (65.5 to 68.9) | 13.3% |
| Post-cutoff | 99.0% (98.8 to 99.2) | 73.7% (71.3 to 76.2) | 11.2% |

The pre-cutoff set has more unreported fields, and unreported fields are where the model
falls over. Re-weighting the pre-cutoff set to the post-cutoff mix moves it from 94.2% to
94.8%, so about a third of the gap is composition and the rest is that the older filings
are genuinely a little harder, mostly on knowing when to say nothing.

**The single clearest number in this whole exercise is in that table.** A frontier model
gets 99% of fields right when the filing reports them and **73.7%** when it does not. The
weakness is not reading, it is abstention, and it is worth more in production than a point
of accuracy: an abstention can be routed to a person, an invention cannot.

Measured on one model. Whether the fine-tunes show a contamination gap is a separate
question and the split stays as it is, because the methodology is right even where this
model shows nothing to correct for.

## Not measured yet

Time to first token, which needs the streaming question above settled, and the
contamination gap for the other two frontier models, which costs US$12 and US$22 and
would be worth running only if the fine-tunes turn out to show a gap this one does not.
