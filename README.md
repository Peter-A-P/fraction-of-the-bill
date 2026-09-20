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
split through the gateway and grades what comes back, and the load-test, cost-per-call and
break-even arithmetic ([docs/cost.md](docs/cost.md)).

The full corpus is built: **8,031 items from 1,600 companies**, 19,574 filings tried, the
post-cutoff test set 716 filings from 148 filers ([docs/data.md](docs/data.md)). The
200-item hand audit is drawn and waiting to be read.

The three frontier baselines are chosen and the gateway is configured: `gpt-5.6-sol` as
the quality ceiling, `claude-sonnet-5` as a second vendor, and `gpt-5.6-luna` as the cost
anchor the break-even has to beat ([docs/baseline.md](docs/baseline.md)).

**Nothing has been trained, quantised or served, no model has been run over the corpus,
and nothing has been spent.** The plan is in [PLAN.md](PLAN.md); the dataset design and the
open questions are in [docs/data.md](docs/data.md), the prompt and the runner in
[docs/baseline.md](docs/baseline.md).

## Result

Not yet measured. The build fills these tables.

**Quality, on held-out filings published after the base models' training cutoff**

| Model | Size | Field accuracy (95% CI) | Non-inferior to best API (delta, CI) | Pre-cutoff gap |
|---|---|---|---|---|
| _not yet_ | | | | |

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
