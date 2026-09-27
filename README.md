# Frontier Quality at a Fraction of the Bill

Proof, with measurements, that a small model an organisation owns and hosts matches a
frontier API on a high-volume extraction task at a small fraction of the per-call cost, and
the exact volume at which the switch pays for itself. For a team spending five figures a
month on API calls, this is the project that finds most of it.

**Status: building, started 2026-09-14.** Built and tested so far: the fifteen-field
extraction schema, the programmatic grader with its named failure modes and paired
bootstrap intervals, the SEC fair-access client, XBRL fact selection, the company-level
splits, the statement locator that finds what the model is shown inside a filing, the
locatability filter that keeps a filing only when its labels are printed where they are
read from, the corpus build that runs all of it over EDGAR and reproduces offline from its
cache, the datasheet and checksums written from a build's own outputs, the hand-audit
tooling, the prompt and its fingerprint, the baseline runner that takes a model over a
split through the gateway and grades what comes back, the load-test, cost-per-call and
break-even arithmetic ([docs/cost.md](docs/cost.md)), the training code with its
checkpointing ([docs/training.md](docs/training.md)), the three bases pinned to exact
revisions, the self-hosted price overlay the gateway validates, the quantisation gate,
the server command lines and the closed-loop load client
([docs/serving.md](docs/serving.md)), the model-card generator, and everything between a
finished fine-tune and its rows in the tables: the checked merge to bf16, the AWQ, GPTQ
and GGUF builds calibrated on training filings only, serving a fine-tune through the
gateway, the fine-tuned and quantisation tables, the files the release gate reads, the
load test on disk and the price overlay made from it, and the serving, break-even and
Pareto outputs of the report.
All of it is tested without a GPU; the rented card only has to run it.

The full corpus is built: **7,874 items from 1,600 companies**, 19,574 filings tried, the
post-cutoff test set 705 filings from 144 filers ([docs/data.md](docs/data.md)). **The
200-item hand audit is done: 3.0% of filings carried a wrong label (1.4% to 6.4%), 6 of
200, none of them in the test sets.** Five of the six had one cause, the filter taking a
number within the grader's tolerance for a location. The rebuild that fixed it dropped
157 filings and corrected 173 labels across the corpus, five of the six among them.

The three frontier baselines are chosen and the gateway is configured: `gpt-5.6-sol` as
the quality ceiling, `claude-sonnet-5` as a second vendor, and `gpt-5.6-luna` as the cost
anchor the break-even has to beat ([docs/baseline.md](docs/baseline.md)).

All three have now been measured on the whole post-cutoff test set, zero-shot and
two-shot: 4,296 calls, **US$59.63**, in the table below. **All three sizes are
fine-tuned**, 24 runs of the sweep, each graded on the validation set, and **the first,
the 2B, clears the cost anchor on the post-cutoff test set** (below;
[docs/training.md](docs/training.md)). The chosen recipes are
seeded three times, quantised, measured against their own bf16 and load-tested on a
rented data center card, which gives the cost per call and the break-even below. The GPU provider is chosen by the portfolio's rule: Runpod, one RTX
4090 on Community Cloud for training, at US$0.34 an hour on 2026-09-22, with checkpoints
pushed to a network volume over its S3 API ([docs/gpu-prices.md](docs/gpu-prices.md)).
The plan is in [PLAN.md](PLAN.md); the dataset design and the open
questions are in [docs/data.md](docs/data.md), the prompt and the runner in
[docs/baseline.md](docs/baseline.md).

## Result

Three sizes, three seeds each, every format paired with its own bf16.

**Quality, on held-out filings published after the base models' training cutoff**

| Model | Size | Format | Fields correct (95% CI) | Every field right | Paired delta vs openai/gpt-5.6-sol few_shot | Pre-cutoff minus post-cutoff |
|---|---|---|---|---|---|---|
| `2b-r64-lr1e-4-nall-s0-e1` | 2b | bf16 | 96.9% (96.5% to 97.3%) | 65.7% (62.1% to 69.1%) | -0.0% (-0.4% to +0.3%) | -1.8% (-2.3% to -1.2%) |
| `2b-r64-lr1e-4-nall-s0-e1` | 2b | gguf-q4_k_m | 96.2% (95.4% to 96.8%) | 64.4% (60.9% to 67.8%) | -0.7% (-1.5% to -0.2%) | not measured |
| `2b-r64-lr1e-4-nall-s0-e1` | 2b | gguf-q8_0 | 96.7% (96.2% to 97.2%) | 65.2% (61.7% to 68.7%) | -0.2% (-0.7% to +0.2%) | not measured |
| `2b-r64-lr1e-4-nall-s1-e1` | 2b | bf16 | 97.0% (96.5% to 97.4%) | 68.7% (65.1% to 72.1%) | +0.0% (-0.3% to +0.4%) | not measured |
| `2b-r64-lr1e-4-nall-s2-e1` | 2b | bf16 | 96.5% (96.1% to 97.0%) | 63.5% (59.9% to 67.0%) | -0.4% (-0.8% to -0.0%) | not measured |
| `4b-r16-lr2e-4-nall-s0-e1` | 4b | bf16 | 96.9% (96.5% to 97.3%) | 66.5% (63.0% to 69.9%) | -0.1% (-0.3% to +0.2%) | -1.8% (-2.4% to -1.3%) |
| `4b-r16-lr2e-4-nall-s0-e1` | 4b | gguf-q4_k_m | 0.1% (0.0% to 0.4%) | 0.0% (0.0% to 0.0%) | -96.8% (-97.3% to -96.3%) | not measured |
| `4b-r16-lr2e-4-nall-s0-e1` | 4b | gguf-q8_0 | 96.9% (96.5% to 97.3%) | 66.5% (63.0% to 69.9%) | -0.0% (-0.3% to +0.2%) | not measured |
| `4b-r16-lr2e-4-nall-s1-e1` | 4b | bf16 | 96.5% (96.1% to 96.9%) | 63.5% (59.9% to 67.1%) | -0.5% (-0.8% to -0.1%) | not measured |
| `4b-r16-lr2e-4-nall-s2-e1` | 4b | bf16 | 96.2% (95.7% to 96.7%) | 63.4% (59.9% to 67.0%) | -0.7% (-1.1% to -0.3%) | not measured |
| `7b-r64-lr1e-4-nall-s0-e1` | 7b | awq | 97.2% (96.8% to 97.5%) | 69.8% (66.2% to 73.0%) | +0.2% (-0.1% to +0.5%) | not measured |
| `7b-r64-lr1e-4-nall-s0-e1` | 7b | bf16 | 97.4% (97.0% to 97.7%) | 71.9% (68.5% to 75.2%) | +0.4% (+0.1% to +0.7%) | -1.3% (-1.9% to -0.8%) |
| `7b-r64-lr1e-4-nall-s0-e1` | 7b | gguf-q4_k_m | 97.3% (96.9% to 97.6%) | 71.2% (67.8% to 74.5%) | +0.3% (-0.0% to +0.7%) | not measured |
| `7b-r64-lr1e-4-nall-s0-e1` | 7b | gguf-q8_0 | 97.3% (96.9% to 97.7%) | 71.8% (68.4% to 75.0%) | +0.4% (+0.1% to +0.7%) | not measured |
| `7b-r64-lr1e-4-nall-s0-e1` | 7b | gptq | 97.3% (96.9% to 97.7%) | 71.6% (68.1% to 74.9%) | +0.4% (+0.0% to +0.7%) | not measured |
| `7b-r64-lr1e-4-nall-s1-e1` | 7b | bf16 | 97.1% (96.6% to 97.5%) | 70.2% (66.8% to 73.6%) | +0.1% (-0.3% to +0.5%) | not measured |
| `7b-r64-lr1e-4-nall-s2-e1` | 7b | bf16 | 97.2% (96.8% to 97.6%) | 71.8% (68.4% to 75.0%) | +0.3% (-0.0% to +0.6%) | not measured |

Every fine-tune, every seed and every format, written by `smallprint report` from the runs.
The 7B is ahead of the best frontier result on whole filings at every seed, and against the
cost anchor, `gpt-5.6-luna` zero-shot, every seed of every size is ahead on fields; the
4B's Q4_K_M forgets the output format and does not ship. Detail, the pre-cutoff runs and the
quantisation gate in [docs/results-finetuned.md](docs/results-finetuned.md), the recipe
and the seeds in [docs/training.md](docs/training.md).

**What it costs to serve**, on a Secure Cloud A40 at US$0.49 an hour, 2026-09-26, at 50%
utilisation, against luna's US$0.87 per 1,000:

| Model | Format | Accuracy delta vs bf16 (95% CI) | Req/s at c=32 | TTFT p99 ms | Cost per 1,000 | Break-even volume at 50% utilisation |
|---|---|---|---:|---:|---:|---:|
| `2b-r64-lr1e-4-nall-s0-e1` | bf16 | reference | 3.29 (3.29 to 3.29) | 3,628 (3,032 to 4,415) | US$0.083 (US$0.083 to US$0.083) | 0.4M |
| `2b-r64-lr1e-4-nall-s0-e1` | gguf-q4_k_m | -0.7% (-1.4% to -0.2%) | 0.67 (0.67 to 0.67) | 27,538 (21,364 to 30,202) | US$0.404 (US$0.404 to US$0.404) | 0.4M |
| `2b-r64-lr1e-4-nall-s0-e1` | gguf-q8_0 | -0.2% (-0.5% to +0.0%) | 0.70 (0.70 to 0.70) | 24,862 (18,454 to 28,287) | US$0.387 (US$0.387 to US$0.387) | 0.4M |
| `7b-r64-lr1e-4-nall-s0-e1` | awq | -0.2% (-0.4% to -0.0%) | 1.39 (1.39 to 1.39) | 11,107 (8,308 to 12,223) | US$0.196 (US$0.196 to US$0.196) | 0.4M |
| `7b-r64-lr1e-4-nall-s0-e1` | bf16 | reference | 1.16 (1.16 to 1.16) | 17,322 (16,672 to 17,468) | US$0.235 (US$0.235 to US$0.235) | 0.4M |
| `7b-r64-lr1e-4-nall-s0-e1` | gptq | -0.1% (-0.2% to +0.1%) | 1.39 (1.39 to 1.39) | 11,925 (8,543 to 12,351) | US$0.196 (US$0.196 to US$0.196) | 0.4M |

The 2B in bf16 answers for **US$0.083 per 1,000, about a tenth of the cheapest frontier
model, and more accurately**; the 7B for US$0.196 in GPTQ, a quarter, and the most accurate
model measured. A dedicated card pays for itself from about 0.4 million extractions a month.
llama.cpp serves GGUF well to one caller and badly to many: its throughput falls as the load
rises, which puts its cost per call four to five times vLLM's on the same card. The 7B's
GGUF and the 4B were not load-tested; the pod was stopped before their turn
([docs/serving.md](docs/serving.md)).

**The frontier baselines, the bar the above has to reach.** Written by `smallprint report`
from the runs; not edited by hand. 705 post-cutoff filings, every call through the
gateway, US$59.63; re-graded on the audited build from the saved answers.

| Model | Prompt | Fields correct (95% CI) | Every field right | Paired delta vs openai/gpt-5.6-luna zero_shot | Cost per 1,000 | Latency p50 | p99 |
|---|---|---|---|---|---:|---:|---:|
| `openai/gpt-5.6-luna` | zero-shot | 96.2% (95.8% to 96.6%) | 58.3% (54.6% to 62.0%) | baseline | US$0.87 | 2.33 s | 4.95 s |
| `openai/gpt-5.6-luna` | few-shot | 96.2% (95.7% to 96.7%) | 60.4% (56.7% to 64.0%) | -0.0% (-0.3% to +0.3%) | US$1.62 | 2.17 s | 4.59 s |
| `anthropic/claude-sonnet-5` | zero-shot | 96.1% (95.7% to 96.5%) | 57.4% (53.8% to 61.0%) | -0.1% (-0.4% to +0.1%) | US$11.96 | 3.04 s | 8.95 s |
| `openai/gpt-5.6-sol` | zero-shot | 95.9% (95.3% to 96.5%) | 64.8% (61.3% to 68.4%) | -0.3% (-0.8% to +0.2%) | US$15.46 | 2.76 s | 6.49 s |
| `anthropic/claude-sonnet-5` | few-shot | 96.3% (95.8% to 96.8%) | 61.1% (57.4% to 64.7%) | +0.1% (-0.3% to +0.4%) | US$22.91 | 3.32 s | 9.86 s |
| `openai/gpt-5.6-sol` | few-shot | 96.9% (96.6% to 97.3%) | 67.0% (63.4% to 70.5%) | +0.7% (+0.5% to +1.0%) | US$30.43 | 2.92 s | 6.53 s |

The cheapest model in the set matches the most expensive one on field accuracy. Whole
filings are where paying more shows: the ceiling model gets 8.7 more of every hundred
filings entirely right, for 35 times the price. Detail and failure modes in
[docs/results-baseline.md](docs/results-baseline.md).

**No contamination premium.** The same model on the 1,396 filings published *before* the
training cutoff scores 94.2%, which is 2.0 points **worse** than the post-cutoff set, not
better. Memorising a filing does not help you copy a number off the page you were handed.

**What a frontier model is actually bad at.** 99.0% of fields right when the filing
reports them; **73.7% when it does not**, where the right answer is null and the model
answers anyway. The weakness is abstention, not reading.

**Quantisation cost, serving and money**

| Model | Format | Accuracy delta vs bf16 (95% CI) | Req/s at c=32 | TTFT p99 ms | Cost per 1,000 | Break-even volume at 50% utilisation |
|---|---|---|---|---|---|---|
| _not yet_ | | | | | | |

## What this does not do

- It does not train anything above about 8B parameters. The argument is that small models
  suffice for a well-specified task; a bigger model is a different argument.
- It does not claim generality. One task, structured extraction from financial statements
  with machine-checkable truth, measured to the bottom.
- It does not measure on the laptop. Every serving number comes from rented hardware
  named in the table, except one clearly labelled 4 GB edge point.
- The break-even is a curve over utilisation with its inputs published, not a single
  number. A reader substitutes their own volume and prices.

## Tried and rejected: one tolerance for grading and for finding a label

The first corpus decided a label was on the page when some number in the right statement
was within the grader's half percent of it. One rule for both seemed the honest choice: the
filter could never keep a filing no correct reading could pass. The 200-item hand audit
rejected it. Six labels were wrong, and five of them had been "found" on the wrong number:
a share count on a depreciation line, another on interest expense, cash tagged 236,000
beside a page printing $236,340. Across the corpus the same rule had kept 128 share counts
that no statement printed, about one in fifty. Grading a reading and
locating a label need different widths: the filter now requires the printed figure at its
own precision, and a share count on a row that is one
([docs/data.md](docs/data.md#the-hand-audit-2026-09-22)).

## How it works

See [PLAN.md](PLAN.md). A dataset of SEC filings paired with the XBRL facts the companies
filed, so every field has programmatic truth and no model grades another. Three
open-weights bases fine-tuned with QLoRA, ablated on rank, learning rate and data volume,
three seeds. Merged to bf16 and quantised to AWQ, GPTQ and GGUF, each measured for its
quality cost. Served with vLLM and llama.cpp on rented GPUs, load-tested at four
concurrencies, registered with the portfolio gateway so self-hosted cost per call is
computed in the same ledger as the frontier APIs. Frontier models run the same items
through the same gateway. The release-gate project runs the non-inferiority comparison. The
Pareto chart and the break-even curve come out of those tables.

## Part of a portfolio

One of fifteen projects. Measured with the AI Release Gate; every
model call, frontier or self-hosted, goes through the Compliant AI Gateway; the filings
dataset and the small extractor are reused by the Verified Filings Analyst.

## How this was built

Design, methodology, evaluation choices and judgement are Peter Parker's. AI coding
assistants (Claude Code) were used for implementation and drafting, the way a senior
engineer uses them in 2026. Every number in the results tables is reproducible from this
repository with one command and a GPU, and that reproducibility is the evidence that
matters.
