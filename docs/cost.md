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
