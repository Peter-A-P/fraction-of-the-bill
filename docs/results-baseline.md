| Model | Prompt | Fields correct (95% CI) | Every field right | Paired delta vs openai/gpt-5.6-luna zero_shot | Cost per 1,000 | Latency p50 | p99 |
|---|---|---|---|---|---:|---:|---:|
| `openai/gpt-5.6-luna` | zero-shot | 96.2% (95.8% to 96.6%) | 58.3% (54.6% to 62.0%) | baseline | US$1.01 | 2.33 s | 4.95 s |
| `openai/gpt-5.6-luna` | few-shot | 96.2% (95.7% to 96.7%) | 60.4% (56.7% to 64.0%) | -0.0% (-0.3% to +0.3%) | US$1.95 | 2.17 s | 4.59 s |
| `anthropic/claude-sonnet-5` | zero-shot | 96.1% (95.7% to 96.5%) | 57.4% (53.8% to 61.0%) | -0.1% (-0.4% to +0.1%) | US$11.96 | 3.04 s | 8.95 s |
| `openai/gpt-5.6-sol` | zero-shot | 95.9% (95.3% to 96.5%) | 64.8% (61.3% to 68.4%) | -0.3% (-0.8% to +0.2%) | US$18.39 | 2.76 s | 6.49 s |
| `anthropic/claude-sonnet-5` | few-shot | 96.3% (95.8% to 96.8%) | 61.1% (57.4% to 64.7%) | +0.1% (-0.3% to +0.4%) | US$22.91 | 3.32 s | 9.86 s |
| `openai/gpt-5.6-sol` | few-shot | 96.9% (96.6% to 97.3%) | 67.0% (63.4% to 70.5%) | +0.7% (+0.5% to +1.0%) | US$37.07 | 2.92 s | 6.53 s |
| `selfhosted/untuned-2b-it` | few-shot | 73.6% (72.1% to 75.2%) | 5.2% (3.7% to 7.0%) | -22.6% (-24.1% to -21.1%) | uncosted | 2.70 s | 3.10 s |
| `selfhosted/untuned-2b-it` | zero-shot | 49.9% (48.2% to 51.6%) | 3.0% (1.8% to 4.3%) | -46.3% (-48.0% to -44.6%) | uncosted | 2.58 s | 3.01 s |
| `selfhosted/untuned-4b-it` | few-shot | 82.6% (81.1% to 84.1%) | 24.0% (20.9% to 27.2%) | -13.6% (-15.1% to -12.2%) | uncosted | 4.82 s | 5.36 s |
| `selfhosted/untuned-4b-it` | zero-shot | 87.5% (86.4% to 88.5%) | 23.1% (20.0% to 26.2%) | -8.8% (-9.7% to -7.8%) | uncosted | 4.56 s | 5.08 s |
| `selfhosted/untuned-7b-it` | few-shot | 57.7% (56.0% to 59.3%) | 1.7% (0.9% to 2.7%) | -38.6% (-40.2% to -36.9%) | uncosted | 4.98 s | 5.72 s |
| `selfhosted/untuned-7b-it` | zero-shot | 47.7% (46.3% to 49.1%) | 0.6% (0.1% to 1.1%) | -48.5% (-49.9% to -47.1%) | uncosted | 4.96 s | 5.64 s |

Measured on 705 test_post_cutoff filings, US$65.77 of calls through the gateway. Costs are recomputed from each run's response bytes by the pinned gateway at price list 2026-10-01, which bills a GPT-5.6 cache write at 1.25x input; the ledger as written priced writes as plain input and under-states OpenAI's runs by 16 to 22% (smallprint/recost.py).

**Where `openai/gpt-5.6-luna` zero_shot misses:**

| Field | Accuracy (95% CI) | How it missed |
|---|---|---|
| `cost_of_revenue` | 86.5% (84.0% to 89.1%) | hallucinated 82, missing 10, wrong_value 3 |
| `total_liabilities` | 90.5% (88.2% to 92.5%) | hallucinated 66, scale 1 |
| `eps_basic` | 91.9% (89.8% to 93.9%) | hallucinated 52, missing 4, wrong_period 1 |
| `shares_diluted` | 93.0% (91.1% to 94.9%) | hallucinated 31, scale 17, wrong_period 1 |
| `eps_diluted` | 94.5% (92.8% to 96.0%) | hallucinated 32, missing 4, sign 2, wrong_period 1 |
| `state_of_incorporation` | 94.9% (93.2% to 96.5%) | hallucinated 34, wrong_value 2 |
| `net_income` | 97.3% (96.0% to 98.4%) | wrong_value 16, wrong_period 2, scale 1 |
| `operating_income` | 98.2% (97.2% to 99.0%) | hallucinated 13 |
| `stockholders_equity` | 98.3% (97.3% to 99.1%) | wrong_value 5, missing 4, scale 3 |
| `fiscal_period` | 98.9% (98.0% to 99.6%) | wrong_value 8 |
| `cash_and_equivalents` | 99.7% (99.3% to 100.0%) | wrong_value 1, missing 1 |
| `auditor_name` | 99.7% (99.3% to 100.0%) | wrong_value 2 |
| `total_assets` | 99.9% (99.6% to 100.0%) | scale 1 |
| `period_end` | 100.0% (100.0% to 100.0%) | nothing |
| `revenue` | 100.0% (100.0% to 100.0%) | nothing |

Left out, as runs over part of the split rather than all of it: `luna-zero-smoke`.
