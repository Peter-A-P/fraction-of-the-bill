# Plan: Frontier Quality at a Fraction of the Bill

**Written:** 2026-09-07. **Revised:** 2026-09-14 (rev. 2, section 3).
**Status:** week 1 started 2026-09-14, about seven months early. The schema, the grader, the
EDGAR fair-access client, the XBRL fact selection and the splits are built and tested; nothing
has been fetched, trained, quantised, served or spent.

**Build:** three weeks, Apr 5 to Apr 25 2027, slack to Apr 30. **Package:** `smallprint`.
**Fed by:** 03 (the gate runs the comparison and supplies intervals and the power function
from 02), 04 (every frontier call and every call to the served small models goes through
the gateway, so cost per call lands in one ledger). **Feeds:** 05 (June 2027) inherits the
filing-extraction dataset and the fine-tuned extractor as a cheap first-pass verifier.

This is the project that closes the deep-learning gap. It is the only one in the portfolio
that touches PyTorch training internals, LoRA adapters, quantisation error, KV-cache
behaviour and serving throughput. It is also the one that answers the question a US
platform team asks first: at what volume does owning the model beat renting the API?

---

## 1. What this produces

Proof, with measurements, that a small model an organisation owns and hosts matches a
frontier API on a high-volume structured-extraction task at a small fraction of the
per-call cost, and the exact monthly volume at which the switch pays for itself, as a curve
over utilisation rather than a single number.

The numbers a stranger can check:

| Number | What it shows |
|---|---|
| Field-level extraction accuracy per model (three fine-tuned sizes, their untuned bases, two or three frontier APIs) on the same held-out filings, with 95% bootstrap CIs, paired | The quality axis of the frontier, on identical items |
| Non-inferiority of each fine-tuned size against the best frontier model, run through the 03 gate with a stated delta and the interval shown | Whether "matches the API" is true, and for which size |
| Accuracy on filings published after the base models' training cutoff against filings from before it | The contamination gap: what the pre-training already knew |
| Quality cost of quantisation: accuracy delta of AWQ, GPTQ and GGUF against bf16 merged weights, per size, paired CIs | Whether "quantisation is free" survives measurement |
| Data-scaling curve: accuracy against training examples (1k, 5k, 20k) per size, three seeds | How much data the result actually needs, with training variance shown |
| Distillation against ground truth: accuracy when trained on frontier-model outputs against XBRL truth at equal data volume, and the share of frontier errors the distilled model inherits | When distillation is the right tool and what it costs |
| Serving: requests per second, output tokens per second, time to first token p50 and p99, end-to-end p99, at concurrency 1, 8, 32 and 64, per size and format, GPU stated | The latency axis, on real hardware |
| Cost per 1,000 extractions: frontier from the gateway ledger, self-hosted from the GPU-hour price at measured throughput and stated utilisation | The cost axis, in dollars |
| Break-even monthly volume as a function of utilisation, with the inputs published | The one number a director wants, with the assumptions a reader can replace |

## 2. Design decisions

### 2.1 One task, with machine-checkable truth

The task is structured extraction from the primary financial statements of SEC filings
(10-K and 10-Q) into a fixed JSON schema of about fifteen fields: period end, revenue, net
income, diluted EPS, total assets, cash, equity, diluted share count, auditor, state of
incorporation and a few more. The truth for every field is the XBRL fact the company itself
filed, so grading is programmatic (numeric tolerance after unit scaling, exact match for
categorical fields) with no judge. This is the same principle as the 03 drift record: no
model grades another model. It also means the training labels are free and unlimited,
which is why the distillation question can be asked as an ablation rather than assumed.

Why not a clinical task, given the brief lists MIMIC-IV and Synthea: the MIMIC-IV demo has
no free-text notes at useful scale, the full corpus needs credentialing that cannot be
published against, and Synthea produces structured records, not notes. A health-shaped
extension is kept in Deferred with the door open.

### 2.2 The frontier models are baselines, not label sources

Two or three API models run the same task on the same held-out items through the 04
gateway. Their accuracy is the bar the small models are measured against; their cost per
call comes from the gateway ledger. A separate, smaller training set is built from a
frontier model's outputs for the distillation ablation only.

### 2.3 Held-out filings from after the training cutoff

Filings are public and are in every pre-training corpus. The test set is split into
filings published before the base models' documented training cutoffs and filings
published after (the first quarter of 2027 gives thousands). Headline numbers are reported
on the post-cutoff set; the gap between the two is reported as its own number.

### 2.4 Three sizes, QLoRA, checkpoint from the first step

Three base models at roughly 1.5B, 4B and 8B parameters, chosen in week 1 from the
open-weights families current in April 2027 with licences that permit publishing adapters
and merged weights (recorded in `docs/models.md`). QLoRA on 4-bit bases with PEFT and
Unsloth; ablations on rank (8, 16, 64), learning rate, and data volume; three seeds on the
final configuration so the intervals include training variance. Checkpoints every N steps
to object storage from the first run, because the GPUs are spot.

### 2.5 Quantisation is measured, not assumed

Each fine-tuned size is merged to bf16, then quantised to AWQ, GPTQ and GGUF (Q4_K_M and
Q8_0). Accuracy per format is measured on the same items with paired bootstrap intervals.
Numeric fields are where quantisation error is expected to show first; the per-field
breakdown is published.

### 2.6 Serving on real hardware, through the gateway

vLLM serves bf16, AWQ and GPTQ; llama.cpp serves GGUF. Each is registered with the
gateway as an OpenAI-compatible host whose price file entry is the GPU-hour rate divided by
measured throughput, so the ledger computes self-hosted cost per call the same way it
computes frontier cost. A closed-loop load client runs concurrency 1, 8, 32 and 64 with the
real prompt distribution. The laptop's 4 GB card runs the 1.5B GGUF as one clearly labelled
edge point; nothing else is measured on it.

### 2.7 GPU choice by price on the day

Rev. 4 rule: GCP or Azure spot if within 20% of the rental marketplaces, otherwise the
marketplace, and the README says which. A price table is written on the first day for L4,
A10G, RTX 4090 and A100 40 GB across the candidates; the 8B runs at 4k context take the
A100 class, the rest an L4 or 4090 class. Hard spend caps in the provider console before
the first job (plan repository actions 3 and 9).

### 2.8 Break-even as a curve

Self-hosted cost is GPU-hour price times hours per month divided by extractions per month
at measured throughput, so it depends on utilisation. The break-even is published as a
curve over utilisation from 10% to 90%, for spot and on-demand prices, against each
frontier model's ledger cost per call, with every input in a table so a reader substitutes
their own volume, price and utilisation.

### 2.9 Out of scope, on purpose

- Training anything above about 8B. The argument is that small models suffice; a 14B point
  is a different argument and a different budget.
- Full fine-tuning as the main method (it appears once, as a Rule C candidate).
- Pre-training, continued pre-training, RLHF or preference tuning. Supervised extraction
  only.
- A general benchmark. One task, done to the bottom.
- Retrieval. The input is the statement text, already located; finding it in a filing is
  05's problem, and 05 inherits this project's loaders for it.

## 3. Data

| Source | Size | What it gives | Access and terms |
|---|---|---|---|
| SEC EDGAR filings (10-K, 10-Q), 2022 to Q1 2027 | About 30k filings selected | The input text: primary financial statements extracted from the filing documents | Public domain; fair-access policy (declared User-Agent, request rate limit) followed and recorded |
| XBRL company facts and frames APIs | Every filed fact | Ground truth per field, with unit and period | Public domain |
| Financial Statement Data Sets (quarterly bulk) | Bulk | Cross-check and fast bulk join | Public domain |
| Frontier-model outputs on 5k training filings | 5k examples | The distillation ablation set | Generated through the gateway; vendor terms on output use recorded |

Dataset construction: pair each filing's statement text with the XBRL facts for the schema
fields; keep a filing only when every numeric fact is locatable in the text after unit
scaling (so the task is extraction, not inference); stratify by filer size and fiscal
period; split by filing date into train, validation, pre-cutoff test and post-cutoff test,
with company-level separation so no filer appears in both train and test. Published on
Hugging Face with a datasheet, checksums and the construction script. Nothing raw is
committed.

**Rev. 2, 2026-09-14, on building the splits.** The company-level rule above is made exact,
and one part of it changed. Companies are partitioned into three *pools*, train, validation
and test, and a company belongs to exactly one pool for ever. The cutoff then divides the
test pool's own filings in time, so a test filer contributes to both the pre-cutoff and the
post-cutoff set. That is deliberate and it is not a leak: neither set is trained on. Had
the two test sets held different companies, the gap between their scores would have mixed
contamination with whichever set happened to hold the easier filers, and the contamination
gap is a headline number in its own right. Sharing the filers makes it a within-filer
comparison. Two consequences follow. Training-pool filings dated after the cutoff are
dropped rather than used, and the count is published, so that no reader has to take on
trust that held-out-period filings were not the reason the score held up. And the leakage
check, which runs on every build rather than only in the test suite, is stated over pools
rather than over splits, because the two test splits are designed to share filers and a
check stated over splits would fire on the correct arrangement. `docs/data.md` carries the
reasoning; `smallprint/data/split.py` carries the rule.

**Rev. 2, 2026-09-14, on where the non-numeric truth comes from.** The `companyfacts` API
organises facts by unit, and three fields of the schema are not numeric: `auditor_name`,
`state_of_incorporation` and `fiscal_period`. The expectation is that the API does not
serve them and that they come instead from the Financial Statement Data Sets submission
table and the Notes Data Sets text table, which moves those bulk sets from the cross-check
role the table above gives them to a primary join. This is confirmed against the live API
on the first fetch and recorded in `docs/data.md` before any filing is paired. `Fact.value`
accepts a string for that reason. If the three fields prove unavailable at corpus scale
they are dropped and the task becomes twelve fields, which is a change to `SCHEMA`, to
`docs/data.md` and to this section in one commit.

## 4. Methods and architecture

```
smallprint/
  data/        edgar.py (filing index, document fetch, fair-access client), statements.py (locate and
               extract primary statements text), xbrl.py (facts, units, periods), pair.py (text to schema
               truth, locatability filter), split.py (date and company splits, cutoff sets), datasheet.py
  schema.py    the extraction schema (pydantic), field types, tolerances, unit scaling
  grade.py     field-level grader: numeric tolerance after scaling, exact for categoricals, per-field report
  prompts/     zero-shot and few-shot templates for the frontier baselines and the untuned bases
  train/       qlora.py (PEFT + Unsloth, checkpointing to object storage), ablate.py (rank, LR, volume,
               seeds), distill.py (frontier-output training set), merge.py (bf16 merge)
  quant/       awq.py, gptq.py, gguf.py; quality.py (paired deltas against bf16)
  serve/       vllm.py and llamacpp.py launchers, gateway host registration, price entry from GPU rate
  bench/       load.py (closed-loop client, concurrency sweep, TTFT and end-to-end percentiles),
               cost.py (cost per 1,000 from ledger and GPU rates), breakeven.py (utilisation curve)
  gate/        eval spec and item loader for the 03 gate; non-inferiority run
  report/      Pareto chart, tables, model cards from results
  cli.py       smallprint data build | baseline | train | quantise | serve | bench | breakeven | report
tests/         grader fixtures (units, negatives, restated values, missing fields); split leakage test
               (no company across splits); schema round-trip; price-entry arithmetic
docs/          models.md (bases, licences, cutoffs), gpu-prices.md (the day-one table and the decision),
               data.md, cards/ (one model card per published artefact, failure modes included)
```

### Tests that matter

The grader against adversarial fixtures (values in thousands against millions, negative
values in parentheses, restated prior periods, fields absent from the text); the split test
that fails if any filer appears in more than one split; the locatability filter; the
gateway price entry for a served model reproduces cost per call from GPU rate and measured
throughput; the load client's percentile arithmetic against a synthetic latency stream.

## 5. Week by week

| Dates | Built | Done when |
|---|---|---|
| Apr 5 to 11 | Data build: EDGAR client, statement extraction, XBRL pairing, filters, splits with cutoff sets; grader; gate eval spec; frontier baselines through the gateway (zero- and few-shot); GPU price table and provider decision; base models and licences chosen; first QLoRA smoke run with checkpointing | Dataset on Hugging Face with datasheet; frontier accuracies with CIs; `docs/gpu-prices.md` decided |
| Apr 12 to 18 | Fine-tune three sizes; rank, LR and data-volume ablations; distillation set and ablation; three seeds on final configurations; merge; AWQ, GPTQ and GGUF; quality cost per format | Accuracy tables with CIs per size, format and ablation |
| Apr 19 to 25 | Serving load tests at four concurrencies per size and format; cost per 1,000; break-even curve; Pareto chart; gate non-inferiority run; model cards; adapters and merged weights published; Rule C write-up; README; `v0.1.0` | Every table in section 1 filled; repository public |
| Apr 26 to 30 | Slack | |

First to drop if behind: GPTQ (AWQ and GGUF stay), then the third seed, then the
distillation ablation. None is in the project file's definition of done; the three sizes,
the quantisation quality cost, the serving numbers, the Pareto chart with break-even and the
identical-items comparison are not droppable.

## 6. Cost

Prices as of 2026-09-07; the GPU table is rewritten on Apr 5. GPU rates assumed: L4 or
4090 class about US$0.30 to 0.50 per hour spot or marketplace, A100 40 GB about US$0.90 to
1.30. Anthropic list prices per million tokens: Haiku 4.5 at $1 in and $5 out, Sonnet 5
at $2 in and $10 out.

| Item | Basis | US$ | CA$ |
|---|---|---:|---:|
| Fine-tuning, three sizes with ablations and seeds | About 55 GPU hours; 8B runs on the A100 class (about 20 h), the rest L4 or 4090 class | 40 | 54 |
| Quantisation and quality measurement | About 10 hours | 6 | 8 |
| Serving load tests | About 25 hours across sizes and formats, part on the A100 class | 20 | 27 |
| Frontier baselines | 2,000 test filings, 3 models, about 3.5k tokens in and 300 out each, through the gateway with the dev cache | 45 | 61 |
| Distillation set | 5k filings, one mid-tier model, batch pricing | 20 | 27 |
| Object storage for checkpoints, one month | | 3 | 4 |
| **Total** | | **134** | **181** |

That is about CA$180 of the CA$400 line, because 2026 GPU prices are well below the 2025
assumptions the line was written with. The line is not re-budgeted. The remainder buys, in
order: a fifth ablation axis (context length 2k against 4k), two more seeds, a longer load
test at sustained concurrency, and A100 hours if the 8B needs them at 4k context. Actuals go
in the plan repository's STATUS next to the estimate.

## 7. Handover

`smallprint` v0.1.0 on Apr 25 2027. 05 imports the EDGAR client, the statement locator,
the XBRL pairing and the grader, and can use the fine-tuned 1.5B or 4B extractor as a cheap
first-pass verifier before its own fact-level grounding. 03 gains a second real use of its
adapter. 04's ledger gains the first self-hosted price entries. The published dataset and
models on Hugging Face carry model cards with failure modes and the construction script.

## 8. Risks

| Risk | Handling |
|---|---|
| The dataset is the result; a sloppy pairing gives a sloppy curve | The whole first week; the locatability filter; adversarial grader fixtures; a hand audit of 200 pairs before training starts |
| The thesis fails: no size matches the frontier | Published either way. The non-inferiority test says by how much; the curve still gives the break-even for whatever accuracy a buyer accepts |
| Spot interruptions waste hours | Checkpoint every N steps to object storage from the first run; resumable training; cost of an interruption measured and reported |
| Base-model licences restrict publishing | Chosen in week 1 with licences recorded; only families whose terms allow adapters and merged weights |
| Contamination inflates the result | Post-cutoff test set is the headline; the gap is its own number |
| The laptop tempts a shortcut | Nothing but the 1.5B GGUF edge point is measured on it, labelled as such |
| Vendor terms on using outputs for training | Read and recorded before the distillation set is generated; if a vendor forbids it, that vendor is not the distillation source |
| Employer boundary | None. Public filings, public models, no counterpart at the employer |

## 9. Rule C candidates

1. **Prompting the untuned small base to match the frontier.** Zero-shot and few-shot on
   each base, same items. Expected: a gap of tens of points that no prompt closes,
   measured with intervals; the fine-tune is what moves it.
2. **Distilling from the frontier instead of training on the truth.** Equal data volume,
   same base, same recipe. Expected: equal or lower accuracy, with the frontier's own
   errors inherited at a measurable rate. The result says when distillation is the right
   tool: when there is no truth, and not otherwise.
3. **Full fine-tuning of the 1.5B instead of QLoRA.** One run. Expected: accuracy within
   the interval of QLoRA at several times the GPU hours and memory.

Whichever produces the clearest evidence becomes `docs/rejected.md`.

## 10. Definition of done

- [ ] Dataset published on Hugging Face with datasheet, checksums, construction script and company-level split test
- [ ] Three model sizes fine-tuned; adapters and merged weights published with model cards including failure modes
- [ ] Untuned bases and two or three frontier APIs measured on the same held-out items through the gateway
- [ ] Non-inferiority against the best frontier model through the 03 gate, delta stated, interval shown
- [ ] Pre-cutoff against post-cutoff accuracy reported
- [ ] Three quantisation formats measured for quality cost, paired CIs, per-field breakdown
- [ ] Data-scaling curve with three seeds
- [ ] Throughput, TTFT and p99 at four concurrencies on real hardware, GPU and prices stated
- [ ] Cost per 1,000 extractions from the gateway ledger for both frontier and self-hosted
- [ ] Pareto chart published with the break-even curve over utilisation and the inputs table
- [ ] GPU provider decision recorded with the day-one price table
- [ ] Reproducible training recipe and serving container
- [ ] One rejected approach documented with evidence (Rule C)
- [ ] Repository public, `v0.1.0` tagged

## 11. Deferred

| Deferred | Kept so the door stays open |
|---|---|
| A health-shaped extraction task (Synthea records rendered to notes by a model, extraction graded against the records) | The schema, grader and training recipe are task-agnostic; a second `schema.py` and pairing script is all it needs |
| A 14B point on the curve | Out by the budget rule against training above about 8B; the recipe would run unchanged on a larger card |
| Speculative decoding and prefix caching in serving | vLLM flags; a second load-test configuration |
| Multi-task adapters (extraction plus classification) | The training loop takes any supervised pairs |
| A reviewed write-up | Rule E names 02, 03 and 09; this project's curve feeds the 09 and 03 posts as a figure |
