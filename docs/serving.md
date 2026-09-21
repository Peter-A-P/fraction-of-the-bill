# Quantisation, serving and the load test

The code is [`smallprint/quant`](../smallprint/quant), [`smallprint/serve`](../smallprint/serve)
and [`smallprint/bench/client.py`](../smallprint/bench/client.py). All of it is tested here
without a GPU or a server; on the rented card it is a command line and a loop.

## The quantisation gate

No format ships without its paired delta against the bf16 build of the same fine-tune.
Every fine-tuned size is published as bf16, AWQ, GPTQ, GGUF Q4_K_M and GGUF Q8_0, and each
quantised format is judged against bf16 over the same filings.

**Paired, because it is a paired question.** A quantised model is the same model with its
weights rounded, so what matters is how much the rounding cost, not how accurate the result
is in isolation. Both runs grade the identical filings, and an unpaired comparison of two
accuracies near 96% would be several times wider than the evidence warrants, wide enough to
wave through a format that has lost a point. The judge refuses two runs over different
filings.

**The margin is one point of field accuracy, on the lower bound.** A format ships when the
lower end of its 95% paired interval is above -1.0%. One point, because the frontier runs
put the whole spread between the cheapest API and the dearest at 0.7 points: a build that
loses more than a point has thrown away more than everything money buys at the frontier,
and the argument for owning the model goes with it. On the lower bound rather than the
point estimate, so a format does not ship on a lucky average from too few items.

**Per field, because the loss is not even.** Rounding is expected to hurt numeric fields
first while dates and names survive, and an average across fifteen fields can hide one that
broke outright. Every verdict carries every field's paired delta and names the worst.

## The servers

vLLM serves bf16, AWQ and GPTQ; llama.cpp serves GGUF, as the plan says. `launch.py` builds
their command lines rather than starting them, so the part that decides what is measured is
a pure, tested function and the part that needs a GPU is one call on the card. Every flag
that affects a measurement is set explicitly, because a default is whatever the installed
version says and a throughput is only reproducible from its command line.

| Setting | Value | Why |
|---|---|---|
| Context | 8,192 tokens | The longest post-cutoff prompts are about 4,300 tokens on the most generous tokeniser measured, plus about 350 out. 4,096 would cut the tail, and a truncated prompt is a wrong answer the model did not choose. Not higher, because KV cache is throughput and throughput is the denominator of the cost per call |
| vLLM quantisation | named, `--quantization awq` or `gptq` | Not detected from the weights |
| vLLM prefix caching | on | Every call carries the same system prompt; a real deployment would cache it |
| GPU memory | 90% | Headroom so an out-of-memory error does not land mid-run |
| llama.cpp slots | 64 | The highest concurrency tested. Fewer slots would measure llama.cpp's queue rather than the model |
| llama.cpp context | 8,192 times the slots | llama.cpp divides its context between slots |

Each served model is registered with the gateway as an `openai_compat` provider flagged
`self_hosted`, priced from the overlay described in [cost.md](cost.md).

## The load client

Closed loop: N requests in flight, and the next sent as each returns, at concurrency 1, 8,
32 and 64. Closed rather than open because the question is what one card sustains, and a
pool of N workers behaves this way; an open loop past capacity measures a queue whose
length is a property of the test. The test that it really is closed sends sixteen 50 ms
requests at concurrency 4 and asserts it took about four rounds rather than sixteen, and
that exactly four were ever in flight.

**Through the gateway, streamed, in standard mode, with retries off.** Every load-test call
goes through the gateway so its cost lands in the ledger with every other call, and it is
streamed because that is where the gateway records time to first token. The gateway refuses
to stream in pass-through mode, so the load test runs in standard mode, and its
configuration sets `retry.max_attempts: 1`: a retried request's latency includes the retry,
and a load test with retries on measures the retry policy. The client counts retries in
every result, so a run where that setting was wrong says so instead of hiding it in p99.

**The real prompt distribution.** Requests are built from test filings with the same
prompt every accuracy number uses, cycled in order.

## Not done yet

Nothing here has run against a real server. The first time it does is on the rented card,
and the first thing that run produces is the seconds-per-step and requests-per-second that
every cost in this project is divided by.
