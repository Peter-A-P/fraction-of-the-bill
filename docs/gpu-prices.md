# GPU prices and the provider decision

**Written 2026-09-22, the morning of the first rental.** Every price this project
publishes for self-hosting comes from the table below.

This document is a gate rather than a placeholder. The rule it implements is that every
price carries the date it was checked, and a rate copied weeks early carries a date that
lies about when it was checked. So the table below is written on the first morning of the
GPU work, from the providers' own pages on that day, and not before.

## The day-one table, 2026-09-22

| Card | VRAM | Runpod Community | Runpod Secure | GCP spot, whole machine |
|---|---:|---:|---:|---:|
| RTX 4090 | 24 GB | US$0.34 | US$0.74 | not offered |
| L4 | 24 GB | US$0.44 | US$0.49 | US$0.42 (g2-standard-4: 4 vCPU, 16 GiB) |
| A10G | 24 GB | not offered | not offered | not offered |
| A100 40 GB | 40 GB | US$1.00 (SXM) | not listed | US$2.20 (a2-highgpu-1g) |
| A100 80 GB | 80 GB | US$1.19 (PCIe) | US$1.59 | |
| H100 SXM | 80 GB | US$2.69 | US$3.49 | |

Per GPU hour, on-demand; a Runpod pod's rate includes its CPU and memory. **Runpod** read
at 12:11 UTC from the Runpod API (`gpuTypes`, the source the console's deploy page prices
from); its spot rates equalled the on-demand ones on the day, and the RTX 4090 was marked
low stock. **GCP** read the same morning from the Spot VMs pricing page
(https://cloud.google.com/spot-vms/pricing), at the page's default region, which it did
not name beside the table; spot prices there change up to once every 30 days. GCP on-demand
and Azure were not read: the rule below turns on spot, and GCP's is already outside it. The
A10G is an AWS card, and no candidate here offers it.

**Storage and transfer.** Runpod network storage US$0.07 per GB a month under a terabyte,
US$0.05 over, US$0.14 for the high-performance tier; container disk US$0.10; volume disk
US$0.10 running and US$0.20 idle (https://www.runpod.io/pricing). No fees for ingress or
egress (https://docs.runpod.io/pods/pricing).

**The spend cap.** The account holds US$150 of prepaid credit with auto-reload off, set by
Peter on 2026-09-22 and read back from the API the same morning: balance US$150.00,
nothing running. When the credit is gone the pods stop, so the balance is the cap. The API
also reports the account's hourly spend limit, US$80. The API read-back stands in for the
screenshot the list below asked for: it is the provider's own record, and it is dated.

## The decision

**Runpod, Community Cloud, one RTX 4090 for training.** The portfolio's rule takes
hyperscaler spot when it is within 20% of the marketplace. The cheapest 24 GB machine on
GCP spot is an L4 at US$0.42, 24% above the 4090's US$0.34 and a slower card besides, so
the marketplace wins on the rule before the speed difference is counted. Serving is not
on this card; see "Two tiers, two jobs" below.

**Where checkpoints go.** The plan assumed a network volume mounted on the training pod.
Runpod's own documentation says network volumes mount only on Secure Cloud pods
(https://docs.runpod.io/storage/network-volumes), and Secure costs US$0.74 for the same
card, more than twice as much. So the pod stays on Community Cloud and pushes every
checkpoint to a network volume through Runpod's S3-compatible API, which exists for
volumes in thirteen named data centers (https://docs.runpod.io/storage/s3-api). That is
object storage off the machine, which is what the checkpointing rule asks for, at no
transfer cost. `smallprint train run --checkpoint-uri s3://<volume>/<path>
--s3-datacenter <DC>` does it.

**Region.** The volume's data center is the only fixed location; a Community pod is
wherever the card is free, and uploads over the internet. It is recorded with each run.

## What had to be on this page, and where it is

1. A price table for **L4, A10G, RTX 4090 and A100 40 GB**, spot and on-demand, across the
   candidate providers, each cell dated and each with the URL it was read from.
2. The decision, by the portfolio's rule: **GCP or Azure spot if within 20% of the rental
   marketplaces, otherwise the marketplace.** The README says which was chosen.
3. The region, because it moves the price and it is part of the price key.
4. Egress and storage rates for the checkpoint bucket.
5. The **hard spend cap set in the provider console**, with a screenshot reference and the
   date it was set. This happens before the first job, not after the first bill.

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

## The serving card, 2026-09-26

The load test runs on the card a business would rent to serve this, so its rate is the one
the cost per call and the break-even are built on. From the Runpod API at 16:35 UTC,
Secure Cloud, on-demand, per GPU hour:

| Card | VRAM | Secure | In stock |
|---|---:|---:|---|
| A40 | 48 GB | US$0.49 | low |
| L4 | 24 GB | US$0.49 | none listed |
| RTX A6000 | 48 GB | US$0.53 | low |
| RTX 4090 | 24 GB | US$0.74 | low |
| L40S | 48 GB | US$1.09 | low |

**An A40 on Secure Cloud at US$0.49**, the rate the pod was rented at, 16:46 UTC. A data
center card on the data center tier, the cheapest in stock at any size that holds every
format with room for a cache: the L4 is the same price and was not offered. Not the
RTX 4090 the training ran on, a consumer card whose driver terms exclude data center
deployment, and whose Community tier is the cheap one precisely because it carries no
guarantees a business would sign for.

## A cheaper serving card, 2026-10-03

The A40 carries 4.3 million extractions a month at 50% utilisation, and the break-even
against luna needs 353,000. While one card carries the volume, the break-even is a
month's rent over the API's price a call, so the cheapest card that holds the model and
keeps up sets it, and the A40's spare capacity is paid for and unused. From the Runpod API
at 06:20 UTC, Secure Cloud, on-demand, per GPU hour, cards of 16 to 24 GB:

| Card | VRAM | Secure | Listed in stock |
|---|---:|---:|---|
| RTX 2000 Ada | 16 GB | US$0.24 | no |
| RTX A4000 | 16 GB | US$0.25 | no |
| RTX A4500 | 20 GB | US$0.25 | no |
| RTX A5000 | 24 GB | US$0.27 | no |
| RTX 4000 Ada | 20 GB | US$0.28 | low |
| L4 | 24 GB | US$0.49 | low |
| A40 | 48 GB | US$0.49 | low |

None of the four cheapest could be rented at 06:20, and twelve attempts on the RTX 4000 Ada
found none free either. A watcher polling every five minutes rented **an RTX A5000 at
US$0.27** at 06:39, the first to come into stock: a data center card on the data center
tier, 24 GB, which holds the 2B in bf16 with room for its cache. The rate is the pod's own
`costPerHr` at rental, and it is in the load test's record beside the timings.

