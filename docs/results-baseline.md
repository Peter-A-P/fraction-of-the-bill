| Model | Prompt | Fields correct (95% CI) | Every field right | Paired delta vs openai/gpt-5.6-luna zero_shot | Cost per 1,000 | Latency p50 | p99 |
|---|---|---|---|---|---:|---:|---:|
| `openai/gpt-5.6-luna` | zero-shot | 96.3% (95.9% to 96.6%) | 57.8% (54.2% to 61.5%) | baseline | US$0.87 | 2.33 s | 4.96 s |
| `openai/gpt-5.6-luna` | few-shot | 96.2% (95.8% to 96.7%) | 59.5% (55.9% to 63.1%) | -0.0% (-0.3% to +0.3%) | US$1.62 | 2.18 s | 4.59 s |
| `anthropic/claude-sonnet-5` | zero-shot | 96.1% (95.7% to 96.6%) | 57.4% (53.8% to 61.0%) | -0.1% (-0.4% to +0.1%) | US$11.96 | 3.04 s | 8.95 s |
| `openai/gpt-5.6-sol` | zero-shot | 95.9% (95.3% to 96.5%) | 64.1% (60.6% to 67.6%) | -0.3% (-0.9% to +0.2%) | US$15.48 | 2.76 s | 6.49 s |
| `anthropic/claude-sonnet-5` | few-shot | 96.4% (95.9% to 96.8%) | 61.2% (57.5% to 64.7%) | +0.1% (-0.3% to +0.4%) | US$22.92 | 3.32 s | 9.86 s |
| `openai/gpt-5.6-sol` | few-shot | 97.0% (96.6% to 97.3%) | 66.5% (63.0% to 69.8%) | +0.7% (+0.4% to +0.9%) | US$30.44 | 2.93 s | 6.53 s |

Measured on 716 test_post_cutoff filings, US$59.63 of calls through the gateway.

**Where `openai/gpt-5.6-luna` zero_shot misses:**

| Field | Accuracy (95% CI) | How it missed |
|---|---|---|
| `cost_of_revenue` | 85.8% (83.1% to 88.3%) | hallucinated 85, missing 10, wrong_value 7 |
| `total_liabilities` | 90.6% (88.4% to 92.7%) | hallucinated 66, scale 1 |
| `eps_basic` | 92.0% (89.9% to 94.0%) | hallucinated 52, missing 4, wrong_period 1 |
| `shares_diluted` | 94.1% (92.3% to 95.8%) | hallucinated 20, scale 18, missing 3, wrong_period 1 |
| `eps_diluted` | 94.6% (92.7% to 96.2%) | hallucinated 32, missing 4, sign 2, wrong_period 1 |
| `state_of_incorporation` | 95.0% (93.3% to 96.5%) | hallucinated 34, wrong_value 2 |
| `net_income` | 97.2% (95.9% to 98.3%) | wrong_value 17, wrong_period 2, scale 1 |
| `operating_income` | 98.2% (97.1% to 99.2%) | hallucinated 13 |
| `stockholders_equity` | 98.3% (97.3% to 99.2%) | wrong_value 5, missing 4, scale 3 |
| `fiscal_period` | 98.9% (98.0% to 99.6%) | wrong_value 8 |
| `cash_and_equivalents` | 99.7% (99.3% to 100.0%) | wrong_value 1, missing 1 |
| `auditor_name` | 99.7% (99.3% to 100.0%) | wrong_value 2 |
| `total_assets` | 99.9% (99.6% to 100.0%) | scale 1 |
| `period_end` | 100.0% (100.0% to 100.0%) | nothing |
| `revenue` | 100.0% (100.0% to 100.0%) | nothing |

Left out, as runs over part of the split rather than all of it: `luna-zero-smoke`.
