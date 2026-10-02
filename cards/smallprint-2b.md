---
license: apache-2.0
base_model: google/gemma-4-E2B
tags:
- structured-extraction
- financial-statements
- sec-filings
- lora
---

# 2b-r64-lr1e-4-nall-s0-e1

Extracts fifteen figures from the primary financial statements of a US SEC 10-K or 10-Q,
as JSON. A LoRA fine-tune of [`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B) at revision
`d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`, trained on filings with their XBRL facts as the labels. No model graded
another model anywhere in producing these numbers.

## Results, on filings published after the base model's training cutoff

| | This model | openai/gpt-5.6-luna, zero_shot |
|---|---|---|
| Fields correct | 96.9% (96.5% to 97.3%) | 96.2% (95.8% to 96.6%) |
| Every field right on a filing | 65.7% (62.1% to 69.1%) | 58.3% (54.6% to 62.0%) |
| Answers that are not valid JSON | 0.0% (0.0% to 0.0%) | 0.0% (0.0% to 0.0%) |
| Cost per 1,000 extractions | US$0.083 served (A40 on_demand at runpod, US$0.490/h, checked 2026-09-26; 3.29 requests a second at 32 in flight, 50% utilisation) | US$1.01 |

705 filings. On filings published before the base model's training cutoff it scores 95.2% (94.7% to 95.5%), against 96.9% (96.5% to 97.3%) after it. That difference is the contamination gap, reported and never used as the headline.

## The release gate

Non-inferiority field by field, paired on the same filings, a three-point margin per field
and Holm's adjustment across the fifteen, decided by the portfolio's release gate
(project 03) on this model's outcomes. A field blocks when the gate cannot rule out a
three-point loss on it.

| Against | Verdict | Fields that block | Record |
|---|---|---|---|
| `openai/gpt-5.6-sol` | block | `operating_income` -3.5 (-5.0 to -2.1), `net_income` -3.4 (-5.0 to -2.0) | `4bb6956e2ea1e120` |
| `openai/gpt-5.6-luna` | block | `operating_income` -3.0 (-4.7 to -1.6), `net_income` -2.0 (-3.8 to -0.3) | `85feabeb81103ba2` |

## How it fails

| Failure | Count | Per 1,000 fields |
|---|---:|---:|
| Invents a figure the filing does not report | 218 | 20.6 |
| Reads the prior-period column | 13 | 1.2 |
| Drops or misapplies the reporting scale | 29 | 2.7 |
| Reads a loss as a profit, or the reverse | 3 | 0.3 |
| Leaves out a figure the filing does report | 17 | 1.6 |
| Takes a figure from the wrong line | 46 | 4.3 |
| Answer is not valid JSON of the schema | 0 | 0.0 |

## Per field

| Field | Accuracy (95% CI) | Misses by reason |
|---|---|---|
| `period_end` | 100.0% (100.0% to 100.0%) | none |
| `fiscal_period` | 100.0% (100.0% to 100.0%) | none |
| `revenue` | 99.6% (99.0% to 100.0%) | wrong_value 2, wrong_period 1 |
| `cost_of_revenue` | 87.7% (85.1% to 90.1%) | hallucinated 70, missing 8, wrong_value 6, scale 3 |
| `operating_income` | 95.2% (93.6% to 96.7%) | hallucinated 33, wrong_period 1 |
| `net_income` | 95.3% (93.6% to 96.7%) | wrong_value 29, wrong_period 2, scale 1, sign 1 |
| `eps_basic` | 92.6% (90.6% to 94.5%) | hallucinated 48, missing 2, wrong_period 2 |
| `eps_diluted` | 94.5% (92.6% to 96.0%) | hallucinated 32, missing 3, sign 2, wrong_period 2 |
| `shares_diluted` | 92.5% (90.5% to 94.3%) | hallucinated 34, scale 18, wrong_period 1 |
| `total_assets` | 99.6% (99.0% to 100.0%) | scale 2, wrong_period 1 |
| `total_liabilities` | 99.4% (98.9% to 99.9%) | scale 2, hallucinated 1, wrong_period 1 |
| `cash_and_equivalents` | 99.6% (99.0% to 100.0%) | wrong_value 2, wrong_period 1 |
| `stockholders_equity` | 98.7% (97.9% to 99.4%) | wrong_value 5, scale 3, wrong_period 1 |
| `auditor_name` | 99.7% (99.3% to 100.0%) | wrong_value 2 |
| `state_of_incorporation` | 99.4% (98.9% to 99.9%) | missing 4 |

## Quantised formats

Each format is judged against this model's bf16 weights over the same filings, and is
published only if the lower end of its paired interval is above -1.0% of field accuracy.

| Format | Paired delta vs bf16 (95% CI) | Worst field | Published |
|---|---|---|---|
| gguf-q8_0 | -0.2% (-0.5% to +0.0%) | `fiscal_period` -0.4% | yes |
| gguf-q4_k_m | -0.7% (-1.4% to -0.2%) | `fiscal_period` -1.6% | no |

## Training

| | |
|---|---|
| Base | `google/gemma-4-E2B` at `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` |
| Method | QLoRA, rank 64, alpha 128, dropout 0.05 |
| Learning rate | 0.0001, cosine, warm-up 3% |
| Examples | 5,060 training filings, 713 validation |
| Epochs | 1 |
| Seed | 0 |
| Prompt fingerprint | `e015ec5e057be645` |
| Steps | 317 |
| Validation loss | 0.008599315769970417 |
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
