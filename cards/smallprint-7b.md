---
license: apache-2.0
base_model: allenai/Olmo-3-1025-7B
tags:
- structured-extraction
- financial-statements
- sec-filings
- lora
---

# 7b-r64-lr1e-4-nall-s0-e1

Extracts fifteen figures from the primary financial statements of a US SEC 10-K or 10-Q,
as JSON. A LoRA fine-tune of [`allenai/Olmo-3-1025-7B`](https://huggingface.co/allenai/Olmo-3-1025-7B) at revision
`a81bae42db3975be1671e27b9c9a56da1a9f980f`, trained on filings with their XBRL facts as the labels. No model graded
another model anywhere in producing these numbers.

## Results, on filings published after the base model's training cutoff

| | This model | openai/gpt-5.6-luna, zero_shot |
|---|---|---|
| Fields correct | 97.4% (97.0% to 97.7%) | 96.2% (95.8% to 96.6%) |
| Every field right on a filing | 71.9% (68.5% to 75.2%) | 58.3% (54.6% to 62.0%) |
| Answers that are not valid JSON | 0.0% (0.0% to 0.0%) | 0.0% (0.0% to 0.0%) |
| Cost per 1,000 extractions | US$0.235 served (A40 on_demand at runpod, US$0.490/h, checked 2026-09-26; 1.16 requests a second at 32 in flight, 50% utilisation) | US$1.01 |

705 filings. On filings published before the base model's training cutoff it scores 96.0% (95.7% to 96.4%), against 97.4% (97.0% to 97.7%) after it. That difference is the contamination gap, reported and never used as the headline.

## The release gate

Non-inferiority field by field, paired on the same filings, a three-point margin per field
and Holm's adjustment across the fifteen, decided by the portfolio's release gate
(project 03) on this model's outcomes. A field blocks when the gate cannot rule out a
three-point loss on it.

| Against | Verdict | Fields that block | Record |
|---|---|---|---|
| `openai/gpt-5.6-sol` | block | `net_income` -2.7 (-4.1 to -1.3), `shares_diluted` -1.8 (-3.0 to -0.7) | `c9eb7d588c0cfcf8` |
| `openai/gpt-5.6-luna` | pass | none | `9acf6e24b0717058` |

## How it fails

| Failure | Count | Per 1,000 fields |
|---|---:|---:|
| Invents a figure the filing does not report | 146 | 13.8 |
| Reads the prior-period column | 15 | 1.4 |
| Drops or misapplies the reporting scale | 24 | 2.3 |
| Reads a loss as a profit, or the reverse | 4 | 0.4 |
| Leaves out a figure the filing does report | 55 | 5.2 |
| Takes a figure from the wrong line | 34 | 3.2 |
| Answer is not valid JSON of the schema | 0 | 0.0 |

## Per field

| Field | Accuracy (95% CI) | Misses by reason |
|---|---|---|
| `period_end` | 100.0% (100.0% to 100.0%) | none |
| `fiscal_period` | 100.0% (100.0% to 100.0%) | none |
| `revenue` | 98.7% (97.9% to 99.4%) | wrong_value 5, wrong_period 4 |
| `cost_of_revenue` | 91.3% (89.2% to 93.3%) | hallucinated 33, missing 28 |
| `operating_income` | 98.6% (97.6% to 99.4%) | hallucinated 9, wrong_period 1 |
| `net_income` | 96.0% (94.6% to 97.3%) | wrong_value 24, wrong_period 3, scale 1 |
| `eps_basic` | 92.2% (90.1% to 94.0%) | hallucinated 46, missing 5, wrong_period 2, wrong_value 1, sign 1 |
| `eps_diluted` | 94.0% (92.2% to 95.7%) | hallucinated 29, missing 7, sign 3, wrong_period 2, wrong_value 1 |
| `shares_diluted` | 91.6% (89.5% to 93.6%) | hallucinated 29, scale 22, missing 5, wrong_period 3 |
| `total_assets` | 100.0% (100.0% to 100.0%) | none |
| `total_liabilities` | 99.9% (99.6% to 100.0%) | missing 1 |
| `cash_and_equivalents` | 99.7% (99.3% to 100.0%) | wrong_value 1, missing 1 |
| `stockholders_equity` | 99.9% (99.6% to 100.0%) | scale 1 |
| `auditor_name` | 99.7% (99.3% to 100.0%) | wrong_value 2 |
| `state_of_incorporation` | 98.9% (98.0% to 99.6%) | missing 8 |

## Quantised formats

Each format is judged against this model's bf16 weights over the same filings, and is
published only if the lower end of its paired interval is above -1.0% of field accuracy.

| Format | Paired delta vs bf16 (95% CI) | Worst field | Published |
|---|---|---|---|
| awq | -0.2% (-0.4% to -0.0%) | `operating_income` -0.7% | yes |
| gptq | -0.1% (-0.2% to +0.1%) | `net_income` -0.6% | yes |
| gguf-q8_0 | -0.0% (-0.1% to +0.0%) | `shares_diluted` -0.4% | yes |
| gguf-q4_k_m | -0.1% (-0.3% to +0.0%) | `net_income` -0.6% | yes |

## Training

| | |
|---|---|
| Base | `allenai/Olmo-3-1025-7B` at `a81bae42db3975be1671e27b9c9a56da1a9f980f` |
| Method | QLoRA, rank 64, alpha 128, dropout 0.05 |
| Learning rate | 0.0001, cosine, warm-up 3% |
| Examples | 5,060 training filings, 713 validation |
| Epochs | 1 |
| Seed | 0 |
| Prompt fingerprint | `e015ec5e057be645` |
| Steps | 317 |
| Validation loss | 0.01583036407828331 |
| Libraries | accelerate 1.15.0, bitsandbytes 0.50.2, peft 0.21.0, python 3.13.15, torch 2.14.0+cu130, transformers 5.17.0, trl 1.13.0 |



## Limitations

- Trained and measured on US SEC 10-K and 10-Q filings under US GAAP. Nothing here says
  how it does on IFRS statements, on other regulators' filings, or on documents that are
  not financial statements.
- Banks and savings institutions are excluded from the corpus, and funds, trusts and
  pre-revenue companies are underrepresented. A score here says nothing about their
  statements. See the dataset's datasheet.
- The input is the located statements, not a whole filing. Given a whole filing, it has
  to find them first, which this model was not measured on.
- Every label is an XBRL fact the company filed, checked to be printed on the page. The
  label error rate from the hand audit is the ceiling on how far any of these numbers can
  be trusted.
