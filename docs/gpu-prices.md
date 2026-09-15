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
