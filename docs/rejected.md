# Tried and rejected

PLAN.md section 9 named three approaches that might have replaced the one this project
took, and said the clearest evidence against one of them would be written up here. Two
were measured. One of them, distillation, gave the clearest evidence. Every number is on
the 705 post-cutoff filings, paired over the same filings, graded against the XBRL facts by
the same grader as everything else.

## Distilling from the frontier instead of training on the truth

**The idea.** Truth is expensive. This project has it for free, because the companies file
XBRL. Most extraction tasks do not, and the usual substitute is to have a frontier model
label the training set and fine-tune on its answers. If that worked as well here, the XBRL
pairing would be a convenience and not the point.

**The test.** `gpt-5.6-luna` zero-shot, the cost anchor, answered all 5,060 training
filings through the gateway, US$5.20 at OpenAI's rate. Seven answers would not parse and
those filings were left out. The chosen 2B recipe (`2b-r64-lr1e-4-nall-s0-e1`) was then
trained once more with the same filings, prompt, recipe and seed, towards luna's answers
instead of the facts. Validation kept the facts, so the run was selected and graded like
every other. Method, commands and the vendor's terms: [training.md](training.md#distillation-against-the-truth).

**The result.**

| On 705 post-cutoff filings | Trained on the facts | Trained on luna's answers | luna, the teacher |
|---|---:|---:|---:|
| Fields correct | 96.9% | 95.6% | 96.2% |
| Paired delta, distilled minus facts | | **-1.3 (-1.7 to -1.0)** | |
| Paired delta against the teacher | +0.7 (+0.3 to +1.0) | **-0.6 (-0.9 to -0.4)** | |
| Every field right | 65.7% | 52.1% | 58.3% |
| Of luna's 400 wrong fields, repeated with the identical answer | 52.5% (46.8% to 58.1%) | **69.8% (64.3% to 74.9%)** | |
| Validation, fields correct (713 filings) | 95.6% | 94.0% | |

The distilled model is worse than the one trained on the facts, and worse than its own
teacher. It gets 13.6 fewer filings in a hundred entirely right. The model trained on the
facts already shares half of luna's mistakes, which is the floor: some filings are hard for
every reader in the same place. The distilled model repeats 17 more of every hundred, and
the two intervals do not overlap.

**What it inherited is one behaviour.** luna's weakness on this task is abstention, not
reading. It is right on 99.0% of fields the filing reports and 73.7% of fields it does
not, where the right answer is null and luna answers anyway ([baseline.md](baseline.md)).
The distilled model learned exactly that:

| Field | Facts | Distilled | luna | The distilled model's misses |
|---|---:|---:|---:|---|
| `total_liabilities` | 99.4% | 89.4% | 90.5% | 71 invented, where luna invents 66 |
| `state_of_incorporation` | 99.4% | 94.9% | 94.9% | 34 invented, as luna's 34 |
| `cost_of_revenue` | 87.7% | 84.5% | 86.5% | 92 invented, against the control's 70 |
| `operating_income` | 95.2% | 94.0% | 98.2% | 41 invented, against 33 |

Where the filing prints no total liabilities line, the facts say null. luna sums the
balance sheet or carries a number from elsewhere, and the distilled model does the same,
71 times. Trained on the facts, the same base learned to leave the field empty, and invented a
value once. On the fields where luna reads correctly, the two fine-tunes are level.

**Why it is rejected here, and when it is not.** A student cannot learn what its teacher
does not know, and it learns what its teacher gets wrong as faithfully as what it gets
right. Here the truth costs nothing, so distilling buys nothing and costs 1.3 points of
fields and 13.6 points of whole filings, plus US$5.20 of API calls. Distillation is the tool
when there is no truth to train on. Even then, this result says to measure the teacher's
failure modes first, because the student will have them too: here a teacher that cannot
abstain produced a student that cannot abstain.

## Prompting the untuned base instead of fine-tuning

The other candidate, measured as part of the baseline table
([results-baseline.md](results-baseline.md)): each untuned instruction model, zero-shot,
two-shot, and for the Gemma bases with reasoning on.

| Base | Zero-shot | Two-shot | Reasoning on | Fine-tuned |
|---|---:|---:|---:|---:|
| 2B, Gemma 4 E2B | 49.9% | 73.6% | 79.8% | 96.9% |
| 4B, Gemma 4 E4B | 87.5% | 82.6% | 89.2% | 96.9% |
| 7B, OLMo 3 | 47.7% | 57.7% | (no reasoning mode) | 97.4% |

No prompt reaches luna's 96.2%. The best, the 4B with reasoning, is 7.0 points short
(-8.0 to -6.1) at 39 seconds a filing, against 4.6 seconds with reasoning off. The main
failure is scale: the 2B and 7B copy a figure printed "in thousands" as written, which the
prompt says not to do and a fine-tune learns in one epoch. This is the expected result, a
gap of tens of points for two of the three bases and one no prompt closes. It is less
interesting evidence than distillation, because nobody expected otherwise.

## Not tried

Full fine-tuning of the small base instead of QLoRA, the third candidate, was not run. The
QLoRA fine-tunes already sit above the best frontier result on fields, so the most a full
fine-tune could show is the same accuracy at several times the memory and GPU hours.
