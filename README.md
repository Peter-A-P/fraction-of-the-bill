# Frontier Quality at a Fraction of the Bill

Proof, with measurements, that a small model an organisation owns and hosts matches a
frontier API on a high-volume extraction task at a small fraction of the per-call cost, and
the exact volume at which the switch pays for itself. For a team spending five figures a
month on API calls, this is the project that finds most of it.

**The answer.** Fine-tuned on 5,060 filings with the companies' own XBRL facts as labels,
a 2B model gets **96.9% of fields right** on filings published after its base was trained,
level with the most expensive frontier result measured (`gpt-5.6-sol` two-shot, 96.9%) and
above the cheapest model that does the job (`gpt-5.6-luna`, 96.2%). The 7B gets 97.4%.
Served on a rented A40, the 2B answers for **US$0.083 per 1,000 extractions against luna's
US$1.01**, and a dedicated card pays for itself from about **353,000 extractions a
month** against luna, about 10,000 against `gpt-5.6-sol`. **The limitation**: field by field through the release gate, only the 7B is
shown non-inferior to luna on every field; the 2B is better on average and cannot rule out
a three-point loss on operating income and net income, and no fine-tune is shown
non-inferior to the two-shot ceiling. Every number below has its interval, and every
price its date.

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

**How much data it takes**, three seeds a point, rank 16, one epoch, on the post-cutoff
filings, written by `smallprint report`:

| Size | Training filings | Seed 0 | Seed 1 | Seed 2 | Mean of seeds (95% CI) | Every field right, mean |
|---|---:|---:|---:|---:|---|---:|
| 2b | 1,000 | 88.1% | 85.4% | 83.1% | 85.6% (84.3% to 86.8%) | 29.7% |
| 2b | 2,500 | 93.1% | 92.1% | 91.4% | 92.2% (91.4% to 93.0%) | 43.3% |
| 2b | 5,000 | 95.2% | 95.1% | 95.2% | 95.2% (94.7% to 95.7%) | 52.6% |
| 4b | 1,000 | 89.3% | 88.8% | 91.9% | 90.0% (89.2% to 90.8%) | 36.5% |
| 4b | 2,500 | 95.0% | 95.3% | 95.8% | 95.4% (94.9% to 95.8%) | 53.5% |
| 4b | 5,000 | 96.2% | 95.8% | 93.7% | 95.2% (94.8% to 95.7%) | 55.7% |
| 7b | 1,000 | 91.4% | 90.7% | 91.1% | 91.1% (90.1% to 92.0%) | 45.2% |
| 7b | 2,500 | 95.9% | 96.0% | 96.0% | 96.0% (95.5% to 96.4%) | 58.7% |
| 7b | 5,000 | 97.0% | 96.9% | 96.8% | 96.9% (96.5% to 97.3%) | 68.0% |

Size buys most when data is scarce: the 7B leads the 2B by 5.5 points at 1,000 filings and
1.7 at 5,000. The 7B reaches luna from 2,500 labelled filings. One 4B seed at 5,000 learned to
leave the diluted per-share figures empty and is why that point dips. Every seed with its
interval in [docs/results-finetuned.md](docs/results-finetuned.md); the reading in
[docs/training.md](docs/training.md#the-data-scaling-curve).

**What it costs to serve**, on a Secure Cloud A40 at US$0.49 an hour, 2026-09-26, at 50%
utilisation, against luna's US$1.01 per 1,000:

| Model | Format | Accuracy delta vs bf16 (95% CI) | Req/s at c=32 | TTFT p99 ms | Cost per 1,000 | Break-even volume at 50% utilisation |
|---|---|---|---:|---:|---:|---:|
| `2b-r64-lr1e-4-nall-s0-e1` | bf16 | reference | 3.29 (3.29 to 3.29) | 3,628 (3,032 to 4,415) | US$0.083 (US$0.083 to US$0.083) | 353k |
| `2b-r64-lr1e-4-nall-s0-e1` | gguf-q4_k_m | -0.7% (-1.4% to -0.2%) | 0.67 (0.67 to 0.67) | 27,538 (21,364 to 30,202) | US$0.404 (US$0.404 to US$0.404) | 353k |
| `2b-r64-lr1e-4-nall-s0-e1` | gguf-q8_0 | -0.2% (-0.5% to +0.0%) | 0.70 (0.70 to 0.70) | 24,862 (18,454 to 28,287) | US$0.387 (US$0.387 to US$0.387) | 353k |
| `7b-r64-lr1e-4-nall-s0-e1` | awq | -0.2% (-0.4% to -0.0%) | 1.39 (1.39 to 1.39) | 11,107 (8,308 to 12,223) | US$0.196 (US$0.196 to US$0.196) | 353k |
| `7b-r64-lr1e-4-nall-s0-e1` | bf16 | reference | 1.16 (1.16 to 1.16) | 17,322 (16,672 to 17,468) | US$0.235 (US$0.235 to US$0.235) | 353k |
| `7b-r64-lr1e-4-nall-s0-e1` | gguf-q4_k_m | | not measured at c=32, or retried | | | |
| `7b-r64-lr1e-4-nall-s0-e1` | gguf-q8_0 | | not measured at c=32, or retried | | | |
| `7b-r64-lr1e-4-nall-s0-e1` | gptq | -0.1% (-0.2% to +0.1%) | 1.39 (1.39 to 1.39) | 11,925 (8,543 to 12,351) | US$0.196 (US$0.196 to US$0.196) | 353k |

**The release gate** (project 03, field by field, three-point margin; [docs/gate.md](docs/gate.md))
passes the 7B against the cost anchor on all fifteen fields, and blocks the 2B on operating
income and net income, where it cannot rule out a three-point loss; neither passes against
the two-shot ceiling. The 2B in bf16 answers for **US$0.083 per 1,000, about a twelfth of the
cheapest frontier model, and better on average**, though not on those two lines; the 7B for US$0.196 in GPTQ, a fifth, and the most accurate
model measured. A dedicated card pays for itself from about 353,000 extractions a month: while one card
carries the volume, the break-even is simply a month's rent (US$357.95) over the API's
price a call, so it is the same for every model and format, and it falls in step with the
rent. Against `gpt-5.6-sol` two-shot, the frontier result the 2B matches on fields, it is
about 10,000 a month.
llama.cpp serves GGUF well to one caller and badly to many: its throughput falls as the load
rises, which puts the 2B's GGUF at four to five times vLLM's cost on the same card. The 7B's
GGUF managed 0.2 requests a second at one and at eight in flight and failed every request at
32 and 64, whose prompts did not fit its cache together; at its best that is US$1.36 per
1,000, **more than the API**. The 4B was not load-tested: the 2B beats it on both axes
([docs/serving.md](docs/serving.md)).

**The 4 GB edge point**, the one number measured off rented hardware: the 2B's Q4_K_M on a
GTX 1650 with 4 GB, through the same gateway, answers the first 50 post-cutoff filings with
97.9% of fields right (96.3% to 99.2%), 48 of the 50 answers identical to the A40's, at
12.2 seconds a filing at the median. Enough for one analyst on a machine already owned;
not a server ([docs/serving.md](docs/serving.md#the-4-gb-edge-point-2026-09-30)).

**The frontier baselines, the bar the above has to reach.** Written by `smallprint report`
from the runs; not edited by hand. 705 post-cutoff filings, every call through the
gateway; re-graded on the audited build from the saved answers. **OpenAI's costs are
recomputed** from the response bytes each run kept: the gateway this project first pinned
(boundary v0.3.0) priced GPT-5.6's cache writes, which OpenAI bills at 1.25x input, as plain
input, and every OpenAI call here wrote its whole prompt to the cache. Project 04's
September invoice check found it against OpenAI's console. The ledger rows stay as written;
the table is 16 to 22% higher for OpenAI's runs than they recorded, Anthropic's are
unchanged, and luna zero-shot, the cost anchor, is US$1.01 per 1,000 rather than US$0.87.

| Model | Prompt | Fields correct (95% CI) | Every field right | Paired delta vs openai/gpt-5.6-luna zero_shot | Cost per 1,000 | Latency p50 | p99 |
|---|---|---|---|---|---:|---:|---:|
| `openai/gpt-5.6-luna` | zero-shot | 96.2% (95.8% to 96.6%) | 58.3% (54.6% to 62.0%) | baseline | US$1.01 | 2.33 s | 4.95 s |
| `openai/gpt-5.6-luna` | few-shot | 96.2% (95.7% to 96.7%) | 60.4% (56.7% to 64.0%) | -0.0% (-0.3% to +0.3%) | US$1.95 | 2.17 s | 4.59 s |
| `anthropic/claude-sonnet-5` | zero-shot | 96.1% (95.7% to 96.5%) | 57.4% (53.8% to 61.0%) | -0.1% (-0.4% to +0.1%) | US$11.96 | 3.04 s | 8.95 s |
| `openai/gpt-5.6-sol` | zero-shot | 95.9% (95.3% to 96.5%) | 64.8% (61.3% to 68.4%) | -0.3% (-0.8% to +0.2%) | US$18.39 | 2.76 s | 6.49 s |
| `anthropic/claude-sonnet-5` | few-shot | 96.3% (95.8% to 96.8%) | 61.1% (57.4% to 64.7%) | +0.1% (-0.3% to +0.4%) | US$22.91 | 3.32 s | 9.86 s |
| `openai/gpt-5.6-sol` | few-shot | 96.9% (96.6% to 97.3%) | 67.0% (63.4% to 70.5%) | +0.7% (+0.5% to +1.0%) | US$37.07 | 2.92 s | 6.53 s |

The cheapest model in the set matches the most expensive one on field accuracy. Whole
filings are where paying more shows: the ceiling model gets 8.7 more of every hundred
filings entirely right, for 37 times the price. Detail and failure modes in
[docs/results-baseline.md](docs/results-baseline.md).

**Published** on Hugging Face, public since 2026-10-03, each model with the card
generated from its records ([cards/](cards)):

| Repository | What |
|---|---|
| `Peter-A-P/smallprint-2b` | The 2B, merged, bf16 |
| `Peter-A-P/smallprint-2b-lora` | Its LoRA adapter |
| `Peter-A-P/smallprint-2b-gguf` | Q8_0. Its Q4_K_M lost 0.7 points and is not published |
| `Peter-A-P/smallprint-7b` | The 7B, merged, bf16 |
| `Peter-A-P/smallprint-7b-lora` | Its LoRA adapter |
| `Peter-A-P/smallprint-7b-awq`, `-gptq` | W4A16, for vLLM |
| `Peter-A-P/smallprint-7b-gguf` | Q8_0 and Q4_K_M, for llama.cpp on one machine |
| `Peter-A-P/smallprint-sec-extraction` | The corpus, with its datasheet (a dataset repository) |

| `Peter-A-P/smallprint-4b`, `-4b-lora`, `-4b-gguf` | The 4B, its adapter and its Q8_0. Its Q4_K_M forgot the output format and is not published |

The 4B's Q8_0 was rebuilt from its adapter for publishing, 2026-09-30, because the file
graded was not kept; the merge it came from passed the same check (100% of the answer's
tokens). Every card carries the release gate's decisions on its model.

**No contamination premium.** The same model on the 1,396 filings published *before* the
training cutoff scores 94.2%, which is 2.0 points **worse** than the post-cutoff set, not
better. Memorising a filing does not help you copy a number off the page you were handed.

**What a frontier model is actually bad at.** 99.0% of fields right when the filing
reports them; **73.7% when it does not**, where the right answer is null and the model
answers anyway. The weakness is abstention, not reading.

## Status, 2026-10-03

Built 2026-09-14 to 2026-10-03 on rented GPUs, with US$72.29 of API calls at what the
vendors bill (US$67.09 for the frontier baselines and their rehearsals, US$5.20 for the
distillation targets). The corpus is **7,874 items from 1,600 companies**, 19,574
filings tried, the post-cutoff test set 705 filings from 144 filers, and a 200-item hand
audit found 3.0% of labels wrong (1.4% to 6.4%), none in the test sets, all since fixed
([docs/data.md](docs/data.md)). The frontier baselines are `gpt-5.6-sol` as the quality
ceiling, `claude-sonnet-5` as a second vendor and `gpt-5.6-luna` as the cost anchor
([docs/baseline.md](docs/baseline.md)). Training ran on Runpod
([docs/gpu-prices.md](docs/gpu-prices.md)), the recipe and every run are in
[docs/training.md](docs/training.md), serving and the formats in
[docs/serving.md](docs/serving.md), and the order the work was done in is
[docs/runbook.md](docs/runbook.md).

PLAN.md's definition of done:

- [x] Dataset published on Hugging Face with datasheet, checksums, construction script and company-level split test
- [x] Three model sizes fine-tuned; adapters and merged weights published with model cards including failure modes
- [x] Untuned bases and three frontier APIs measured on the same held-out items through the gateway
- [x] Non-inferiority against the best frontier model through the 03 gate, delta stated, interval shown ([docs/gate.md](docs/gate.md))
- [x] Pre-cutoff against post-cutoff accuracy reported
- [x] Three quantisation formats measured for quality cost, paired CIs, per-field breakdown
- [x] Data-scaling curve with three seeds ([docs/training.md](docs/training.md#the-data-scaling-curve))
- [x] Throughput, TTFT and p99 at four concurrencies on real hardware, GPU and prices stated
- [x] Cost per 1,000 extractions from the gateway ledger for both frontier and self-hosted, OpenAI's recomputed at its cache-write rate
- [x] Pareto chart published with the break-even curve over utilisation and the inputs table
- [x] GPU provider decision recorded with the day-one price table
- [x] Reproducible training recipe and serving container (the container is built and checked in CI, not yet started on a GPU)
- [x] One rejected approach documented with evidence ([docs/rejected.md](docs/rejected.md))
- [x] Repository public, `v0.1.0` tagged

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

## Tried and rejected: distilling from the frontier

The usual way to get training labels without truth is to have a frontier model write them.
Trained on luna's answers to the same 5,060 filings instead of the XBRL facts, the same 2B
recipe scores **95.6% of fields against 96.9%, paired -1.3 points (-1.7 to -1.0), and below
its own teacher**, with 52.1% of filings entirely right against 65.7%. It repeats 69.8% of
luna's wrong answers word for word against 52.5% for the model trained on the facts. What it
learned is luna's one real weakness, answering where the filing has no answer: on total
liabilities it invents a value 71 times, luna 66, the facts-trained model once. Distillation
is for when there is no truth; it hands the student the teacher's failure modes
([docs/rejected.md](docs/rejected.md)). Prompting the untuned bases was also measured and
rejected: the best, the 4B with reasoning on, is 7.0 points under luna at 39 seconds a filing.

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
