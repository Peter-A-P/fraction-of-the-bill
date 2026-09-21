# GPU prices and the provider decision

**Not yet written. Nothing may be rented until it is.**

This document is a gate rather than a placeholder. The rule it implements is that every
price carries the date it was checked, and a rate copied weeks early carries a date that
lies about when it was checked. So the table below is written on the first morning of the
GPU work, from the providers' own pages on that day, and not before.

## What goes here, on the day

1. A price table for **L4, A10G, RTX 4090 and A100 40 GB**, spot and on-demand, across the
   candidate providers, each cell dated and each with the URL it was read from.
2. The decision, by the portfolio's rule: **GCP or Azure spot if within 20% of the rental
   marketplaces, otherwise the marketplace.** The README says which was chosen.
3. The region, because it moves the price and it is part of the price key.
4. Egress and storage rates for the checkpoint bucket.
5. The **hard spend cap set in the provider console**, with a screenshot reference and the
   date it was set. This happens before the first job, not after the first bill.

## Why it cannot be written early

The plan was costed on 2026-09-07 assuming roughly US$0.30 to 0.50 an hour for the L4 or
4090 class and US$0.90 to 1.30 for an A100 40 GB. Those numbers are an estimate for a
budget line, not prices. The break-even curve is computed from the rate actually paid, and
the whole point of publishing the inputs is that a reader can substitute their own; a made
up rate with a real-looking date would poison exactly the number this project exists to
produce.

## What depends on this

- Every self-hosted cost per call, which is the GPU-hour rate divided by measured
  throughput, registered as the price entry for that host in the gateway ledger.
- The break-even curve over utilisation, for both spot and on-demand prices.
- Which size runs on which card, and therefore how much of the budget the 8B at 4k context
  takes.

## A survey for the decision, 2026-09-21

**Not the day-one table.** This is what the budget decision was made against. The binding
table above is still written on the morning of the first rental, from the providers' pages
that day, and replaces these numbers for every cost this project publishes.

| Card | VRAM | Runpod Community | Runpod Secure | Lambda on-demand | GCP |
|---|---:|---:|---:|---:|---|
| RTX 4090 | 24 GB | US$0.34 | US$0.74 | not offered | not offered |
| L4 | 24 GB | US$0.44 | US$0.49 | not offered | about US$0.71 on-demand, spot not read |
| L40S | 48 GB | US$0.79 | US$1.09 | not offered | |
| A100 80 GB | 80 GB | US$1.19 | US$1.59 | US$1.99 (40 GB) | about US$3.67 on-demand |
| H100 SXM | 80 GB | US$2.69 | US$3.49 | US$4.29 | |

Per GPU hour. Runpod and Lambda from their own pricing pages on 2026-09-21. The GCP
figures are from a third-party index, because Google's pricing pages render their tables
in the browser and could not be read here; GCP spot prices in particular have to be read
from the console on the day, and the portfolio rule compares spot against the
marketplaces. Runpod network storage is US$0.07 per GB a month under a terabyte.

**All three bases fit a 24 GB card for QLoRA.** In 4-bit they load at roughly a quarter of
their bf16 size on disk (10, 16 and 15 GB, see [models.md](models.md)), which leaves room
for activations at a 4k-token sequence with gradient checkpointing. A 24 GB card is the
cheapest class that works; 48 and 80 GB buy speed and batch size, not feasibility.

**What the work needs, as a range.** The throughput is not measured and is not assumed
anywhere in the code. For budgeting only: each size's sweep and seeds are about 5,600
optimiser steps of about 50,000 tokens, roughly 280 million tokens a size. At a plausible
2,000 to 6,000 tokens a second of QLoRA on a 24 GB card, depending on the size, training
the three is **65 to 115 GPU hours**. Quantising, evaluating every format and the load
tests add perhaps 20 to 30. With a third again for the things that go wrong, **110 to 190
hours on a 24 GB card**, which is **US$40 to US$140** at the rates above, depending on the
card and the tier. An 80 GB card runs it in perhaps 60% of the time at two to four times
the rate.

**How the estimate becomes a number.** The first run on the rented card is a short smoke
run that measures seconds per step for each size. `train plan --seconds-per-step` then
prints the real schedule, and the full sweep does not start until that number has been
seen. The two-epoch default is itself a candidate for halving: if the validation loss has
flattened after one, the second epoch is the most expensive thing in the schedule.

**Two tiers, two jobs.** Training has no bearing on any published number except accuracy,
so it can run on the cheapest card that fits, on the cheaper tier, with checkpoints
uploaded from step one against the risk of losing the machine. Serving is different: the
throughput measured there is the denominator of every self-hosted cost per call, so it has
to be measured on a card and tier a business would actually deploy on, and the price in the
break-even has to be one a business would actually pay.
