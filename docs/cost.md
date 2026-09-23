# Serving numbers and the money

How every serving and cost number in the results is computed. The definitions are in
[`smallprint/bench/`](../smallprint/bench/) and tested against inputs whose answers can be
worked out by hand. **Nothing here has been measured yet**; no GPU has been rented.
Written 2026-09-19.

## Latency and throughput

A closed-loop load client keeps N requests in flight and records, for each, when it was
sent, when its first token arrived and when it finished
([`load.py`](../smallprint/bench/load.py)).

- **Percentiles are nearest-rank**: the p-th percentile of n values is the value at rank
  ceil(p / 100 * n) in sorted order. It is always a latency some request actually had, and a
  reader reproducing p99 from the published raw timings gets the same answer without asking
  which interpolation was used. Below 100 requests, p99 is the maximum.
- **Throughput** is completed requests (and output tokens) over the measured window, from
  the first request sent to the last finished, after the warm-up is discarded.
- **A failed request counts as an error and not as throughput.** A server that answers
  quickly by failing is not fast.
- **Every figure has a 95% bootstrap interval over requests.** The throughput interval holds
  the window fixed, so it reflects variation within a run, not between runs; the tables
  state the GPU and the run length.

## Cost per call

Frontier cost per call comes from the gateway ledger. Self-hosted cost has no bill, so it
is derived and registered with the gateway as that host's price entry, which keeps both in
one ledger computed one way ([`cost.py`](../smallprint/bench/cost.py)):

    cost per call = GPU-hour rate / (requests per second * 3600 * utilisation)

The entry also states cost per million output tokens, the unit frontier prices are quoted
in. Utilisation is the share of rented time spent serving at the measured rate. It is the
reader's input, not a measurement. Every GPU price carries the date it was read and its
source, and a price without a date is refused.

## The overlay: one cost, split into the gateway's two rates

The gateway costs every call per million input and per million output tokens, because that
is how vendors bill. A GPU bills by the hour. So a self-hosted host's rates are derived and
written as a dated file in the gateway's own price format, in the directory
`self_hosted_prices` names ([`overlay.py`](../smallprint/serve/overlay.py)).

The cost of a call is fixed by the rent and the throughput; how it divides between input
and output tokens is not, and it matters, because a long prompt with a short answer costs a
GPU something different from the reverse. The split charges each kind of token for the GPU
time it used. Reading the prompt is the time to first token, writing the answer is the
rest, so:

    prefill share = TTFT p50 / end-to-end p50, from the load test
    input  rate   = cost per call * prefill share       / mean input tokens  * 1e6
    output rate   = cost per call * (1 - prefill share) / mean output tokens * 1e6

One invariant holds by construction and is tested: a request with the mean token counts
costs exactly the rent per call. Decoding is where the time goes, so an output token comes
out priced many times an input one, as it does on every vendor's list.

The file's `source` field carries the derivation: card, provider, hourly rate, the day it
was read and where, the throughput, the utilisation, the prefill share. The tests build a
real `Gateway` on the written file and let its own validation decide, so a file the gateway
would refuse fails here first. It refuses an overlay naming a provider not flagged
`self_hosted`, which is the gateway keeping one copy of every vendor rate. A dated file is
never rewritten under the same date, because a ledger row already costed from it would
then cite a date whose rates had changed underneath it.

## The break-even

GPUs are rented whole, by the hour, so self-hosted monthly cost is a staircase
([`breakeven.py`](../smallprint/bench/breakeven.py)):

    GPUs needed    k(V) = max(1, ceil(V / capacity)),  capacity = rps * 3600 * H * u
    self-hosted    S(V) = k(V) * rate * H + fixed
    API            A(V) = V * API cost per call

The break-even is the smallest monthly volume V with A(V) >= S(V). A test checks the closed
form against the staircase evaluated at every whole volume up to five million.

**What utilisation does and does not change.** With no fixed costs, the break-even is one
GPU-month of rent expressed in API calls, rate * H / API cost per call, **at every
utilisation**, provided one GPU can carry that volume at the chosen utilisation. If it
cannot, because self-hosted cost per call at that utilisation is above the API's, there is
no break-even at all. Utilisation moves the break-even volume only through fixed costs,
which need enough volume to amortise them and so can push the answer out to many GPUs.
The plan's phrase "break-even as a function of utilisation" is made exact in that form: the
published curve has, for each utilisation, the self-hosted cost per call, whether it breaks
even, the volume, and the number of GPUs, beside every input.

Past the break-even the staircase can briefly put self-hosting back above the API, when the
first calls that need another GPU pay its whole month. The published point is where
self-hosting first stops costing more.

`smallprint breakeven` prints the curve from inputs the reader supplies: the GPU, provider,
spot or on-demand, the rate and the date it was read, measured throughput, the API's cost
per call and any fixed monthly cost. Nothing is defaulted except the fixed cost, which is
zero and printed either way.

## From the load test to the tables

`smallprint bench overlay` writes the dated price file from one or more load tests, at the
utilisation the ledger's costs are to assume, reading each model's throughput, prefill share
and mean token counts at 32 in flight. A level the gateway retried anything in is refused.

`smallprint report --finetuned ... --bench ... --chart docs/pareto.svg` then writes, from
the same records:

- **Serving and money**, one row per load-tested model: its paired accuracy delta against
  the bf16 build of the same fine-tune, requests per second and TTFT p99 at 32 in flight,
  cost per 1,000 at 50% utilisation, and the break-even volume. The cost's interval is the
  throughput's, turned over: the fast end of the rate is the cheap end of the cost.
- **Break-even across utilisation**, 10% to 90%, one row per model and card.
- **The Pareto chart**: field accuracy against cost per 1,000 on a log axis, every frontier
  run from the ledger and every load-tested fine-tune at its quoted concurrency and
  utilisation, with 95% intervals. The runs no other run beats on both axes are joined and
  labelled; the rest carry their figures as tooltips. Frontier runs are circles, fine-tunes
  diamonds, so identity does not rest on colour alone.

**Against the cheapest frontier run.** The tables break even against the frontier run with
the lowest ledger cost per call, which is the price a buyer who does not need the dearest
model would pay, and a fine-tune that beats it on cost has beaten them all. The plan asks
for the curve against each frontier model, on spot and on-demand prices: that is
`smallprint breakeven` with the other model's ledger cost or the other price, and the
tables name the anchor and the card so a reader knows what to substitute.

**32 in flight, 50% utilisation.** The concurrency is a busy production service on one
card short of saturation, where latency stops being worth quoting; the utilisation is the
middle of the curve. Both are named in the caption under the tables and inside the chart,
because a self-hosted cost without them is not a number.
