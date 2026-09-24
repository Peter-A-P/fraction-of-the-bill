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
| vLLM quantisation | named, `--quantization compressed-tensors` | Not detected from the weights. AWQ and GPTQ are both written by llm-compressor in its compressed-tensors format; `awq` and `gptq` would select the kernels for the AutoAWQ and AutoGPTQ layouts, which these files are not |
| vLLM sampling defaults | `--generation-config vllm` | Not the checkpoint's `generation_config.json`, which the merge copies from the base and nobody chose |
| llama.cpp chat template | `--jinja` | Renders the template embedded in the GGUF, the one the model was trained on. Without it llama-server guesses a built-in format from the template |
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
prompt every accuracy number uses, cycled in order, at temperature 0 as the fine-tunes'
accuracy runs are: sampling changes how long an answer is, and the answers timed should be
the answers graded.

**On disk, raw.** `smallprint bench run` runs the levels lowest first and writes one
`bench.json` per model and card: the card and its dated price, the warm-up, and for every
level the raw timings of every request, prompt and answer token counts included, beside
their summary. The file is rewritten after each level, so a pod lost at 64 in flight keeps
the three levels it finished. Each level is at least 200 requests, for a p99 that is not
simply the maximum, and at least ten rounds of the loop. It refuses a gateway configuration
with retries on; the one `smallprint serve config` writes has them off, because the same
file serves the accuracy runs, which never retry in pass-through mode anyway.

## From a finished run to its formats

`smallprint quantise merge | format | gguf | judge`, and `scripts/formats.sh` to run them
on a pod. The code is [`train/merge.py`](../smallprint/train/merge.py) and
[`quant/formats.py`](../smallprint/quant/formats.py).

**The merge is the reference.** Each fine-tune's adapter is merged once into its base, at
the revision it was trained on, in bf16. Those weights are the bf16 row of the results and
the source of every other format, so a format's delta against them measures the rounding
and nothing else. The command refuses a run whose recorded base revision is not the one
pinned for its size.

**The merge is checked.** Folding the adapter in should change nothing but rounding, and a
merge that went wrong loads and answers and is subtly not the fine-tune. Before saving,
one validation filing's chat, prompt and answer, goes through the adapter on the base and
through the merged model, and the next-token predictions are compared over the last 512
positions, the answer among them. The share that agree and the largest logit difference go
in `merge.json`; below 99% agreement the merge is refused. The adapter was trained against
the 4-bit base and is merged into the 16-bit one, the usual QLoRA practice and the only way
to publish weights that are not themselves NF4, so the merged model's accuracy is measured
on the test set rather than carried over from the run.

**The merge carries the processor.** The Gemma bases load as vision-language models, and
vLLM will not serve one without the base's `processor_config.json`, even for text alone;
`save_pretrained` does not write it. The merge copies it from the base at its pinned
revision. The first merged 2B, 2026-09-24, would not start without it.

**AWQ and GPTQ come from one library.** llm-compressor, which took over AWQ when AutoAWQ
was archived in 2025, writes both algorithms in one format, compressed-tensors, that vLLM
serves with one flag. Both are 4-bit weights and 16-bit activations (W4A16). One library
means the two formats differ in the algorithm that chose the weights and in nothing else.

**Calibration is from the training pool only.** AWQ and GPTQ choose their rounding by
running real inputs through the model, so those inputs are data the format has seen. They
are 256 training filings, drawn by keyed hash so every size and both algorithms calibrate
on the same ones, and the code refuses any input holding a test filing. Validation filings
are left out too: they steered the checkpoints.

**Rounded where the adapter went.** The language model's linear layers. Not `lm_head`,
which published schemes keep at 16 bits because the output distribution is where rounding
shows first, and not the Gemma vision and audio towers, which this task never feeds.

**GGUF from llama.cpp's own tools.** The converter writes one bf16 GGUF, and
`llama-quantize` writes Q8_0 and Q4_K_M from it. No importance matrix: it is the llama.cpp
default and how most published GGUFs are made, and an importance matrix is calibration
data by another name. It is a candidate ablation, not a default.

Every format directory carries a `quant.json` (a `.json` beside each GGUF) naming the
digest of the merged weights it was made from, the scheme, the calibration filings by id,
and the library versions. `formats.sh` uploads each format's weights before its record, so
a record on the volume means the weights beside it are whole, and a format already there
is skipped.

## Serving a fine-tune through the gateway

`scripts/evaluate.sh NAME FORMAT [SPLIT...]`. It starts the server with the command line
`smallprint serve argv` prints, waits for its `/health`, and writes `boundary-served.yaml`
with `smallprint serve config`: the project's own gateway configuration, the same caps,
ledger and vendor prices, plus one provider, `selfhosted`, flagged `self_hosted`. Then it
runs the same `baseline run` every frontier number came from, with the zero-shot prompt the
training file was written with, at temperature 0, against `selfhosted/<run>-<format>`.

The served name is the training run's name and the format, so every row of every table
can be traced to both, and the report reads the size and the format back out of it. Until
the load test has measured a throughput and written a price file, the overlay directory is
left out of the configuration and the accuracy runs are written to the ledger uncosted,
which is the truth: nothing has yet said what a call costs.

## The fine-tuned tables

`smallprint report --finetuned data/finetuned` adds two tables to the frontier one, both
from the runs:

- **Quality.** Each fine-tune on the post-cutoff filings, with its paired delta against the
  most accurate frontier run over the same filings, and its pre-cutoff accuracy minus its
  post-cutoff accuracy. That last is the contamination gap, and the one difference in the
  project that cannot be paired: the two sets are different filings by construction, so it
  is resampled on each side independently and its interval is as wide as two samples make
  it. Positive would be a premium for filings the base model was trained on.
- **Quantisation cost.** Each quantised format judged against the bf16 run of the same
  fine-tune: the paired delta, the worst field, and whether it ships under the one-point
  margin. A format measured without its bf16 run is named and left out of the table.

## The release gate

`smallprint gate export` writes what the release gate (project 03) is to read: an eval spec
and both sides' outcomes, in the shapes the gate's own `Side` and `SuiteOutcomes` hold. One
suite per field, because the gate's paired test takes one right-or-wrong outcome per item
and a filing is fifteen; averaging them would need a test the gate does not have, and
pooling them would treat correlated fields as independent. Fifteen suites is what Holm's
adjustment is for, and a block names the field. The margin is the gate's default, three
points per field: on 705 filings a field near 97% has a paired standard error of about 0.75
points, so a one-point margin would leave every suite under the gate's power screen.

**Not runnable end to end yet.** The gate reads one source kind today, blocks of its own
drift record. Reading these files is its downstream adapter, stage 8 of its plan, which is
not built, and its spec model refuses the source kind these files name until it is.

## Not done yet

Nothing here has run against a real server or a real merge. The first time it does is on
the rented card, and the first thing that run produces is the seconds-per-step and
requests-per-second that every cost in this project is divided by. The llm-compressor and
llama.cpp versions are not pinned until the first format is made on the card, as the
training stack was not pinned until the first fine-tune, and whether both handle the Gemma
4 and OLMo 3 architectures is the first thing that run finds out.
