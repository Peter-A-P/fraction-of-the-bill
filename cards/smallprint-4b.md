---
license: apache-2.0
base_model: google/gemma-4-E4B
tags:
- structured-extraction
- financial-statements
- sec-filings
- lora
---

# 4b-r16-lr2e-4-nall-s0-e1

Extracts fifteen figures from the primary financial statements of a US SEC 10-K or 10-Q,
as JSON. A LoRA fine-tune of [`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) at revision
`411aa17b749aa952df1359d2dcea73917a544d9a`, trained on filings with their XBRL facts as the labels. No model graded
another model anywhere in producing these numbers.

## Results, on filings published after the base model's training cutoff

| | This model | openai/gpt-5.6-luna, zero_shot |
|---|---|---|
| Fields correct | 96.9% (96.5% to 97.3%) | 96.2% (95.8% to 96.6%) |
| Every field right on a filing | 66.5% (63.0% to 69.9%) | 58.3% (54.6% to 62.0%) |
| Answers that are not valid JSON | 0.0% (0.0% to 0.0%) | 0.0% (0.0% to 0.0%) |
| Cost per 1,000 extractions | not costed | US$1.01 |

705 filings. On filings published before the base model's training cutoff it scores 95.1% (94.7% to 95.4%), against 96.9% (96.5% to 97.3%) after it. That difference is the contamination gap, reported and never used as the headline.

## The release gate

Non-inferiority field by field, paired on the same filings, a three-point margin per field
and Holm's adjustment across the fifteen, decided by the portfolio's release gate
(project 03) on this model's outcomes. A field blocks when the gate cannot rule out a
three-point loss on it.

| Against | Verdict | Fields that block | Record |
|---|---|---|---|
| `openai/gpt-5.6-luna` | block | `net_income` -2.4 (-4.1 to -0.7) | `ab7d4c1689c8255d` |
| `openai/gpt-5.6-sol` | block | `cost_of_revenue` -1.8 (-3.1 to -0.6), `net_income` -3.8 (-5.5 to -2.3) | `0ec6a1a9abd68bd3` |

## How it fails

| Failure | Count | Per 1,000 fields |
|---|---:|---:|
| Invents a figure the filing does not report | 205 | 19.4 |
| Reads the prior-period column | 3 | 0.3 |
| Drops or misapplies the reporting scale | 35 | 3.3 |
| Reads a loss as a profit, or the reverse | 2 | 0.2 |
| Leaves out a figure the filing does report | 26 | 2.5 |
| Takes a figure from the wrong line | 58 | 5.5 |
| Answer is not valid JSON of the schema | 0 | 0.0 |

## Per field

| Field | Accuracy (95% CI) | Misses by reason |
|---|---|---|
| `period_end` | 100.0% (100.0% to 100.0%) | none |
| `fiscal_period` | 98.2% (97.2% to 99.0%) | wrong_value 13 |
| `revenue` | 98.6% (97.6% to 99.4%) | wrong_value 6, scale 4 |
| `cost_of_revenue` | 86.7% (84.1% to 89.1%) | hallucinated 76, missing 14, scale 4 |
| `operating_income` | 97.7% (96.6% to 98.7%) | hallucinated 11, scale 4, missing 1 |
| `net_income` | 94.9% (93.2% to 96.5%) | wrong_value 30, scale 6 |
| `eps_basic` | 92.6% (90.6% to 94.5%) | hallucinated 51, wrong_value 1 |
| `eps_diluted` | 94.3% (92.5% to 96.0%) | hallucinated 33, wrong_period 3, sign 2, wrong_value 1, missing 1 |
| `shares_diluted` | 92.8% (90.8% to 94.6%) | hallucinated 31, scale 16, wrong_value 3, missing 1 |
| `total_assets` | 100.0% (100.0% to 100.0%) | none |
| `total_liabilities` | 99.6% (99.0% to 100.0%) | hallucinated 3 |
| `cash_and_equivalents` | 99.7% (99.3% to 100.0%) | wrong_value 1, missing 1 |
| `stockholders_equity` | 99.6% (99.0% to 100.0%) | wrong_value 2, scale 1 |
| `auditor_name` | 99.9% (99.6% to 100.0%) | wrong_value 1 |
| `state_of_incorporation` | 98.9% (98.0% to 99.6%) | missing 8 |

## Quantised formats

Each format is judged against this model's bf16 weights over the same filings, and is
published only if the lower end of its paired interval is above -1.0% of field accuracy.

| Format | Paired delta vs bf16 (95% CI) | Worst field | Published |
|---|---|---|---|
| gguf-q8_0 | +0.0% (-0.1% to +0.1%) | `eps_diluted` -0.3% | yes |
| gguf-q4_k_m | -96.8% (-97.2% to -96.3%) | `period_end` -99.9% | no |

## Training

| | |
|---|---|
| Base | `google/gemma-4-E4B` at `411aa17b749aa952df1359d2dcea73917a544d9a` |
| Method | QLoRA, rank 16, alpha 32, dropout 0.05 |
| Learning rate | 0.0002, cosine, warm-up 3% |
| Examples | 5,060 training filings, 713 validation |
| Epochs | 1 |
| Seed | 0 |
| Prompt fingerprint | `e015ec5e057be645` |
| Steps | 317 |
| Validation loss | 0.009492110460996628 |
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
