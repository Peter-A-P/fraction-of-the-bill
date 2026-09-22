# Fine-tuning: the dataset, the recipe, and what a run has to survive

The code is [`smallprint/train`](../smallprint/train); the commands are
`smallprint train dataset | plan | run`. Nothing here trains on the laptop, by rule, and
everything except the gradient step is tested without a GPU.

## The training file

`smallprint train dataset` writes `examples.jsonl`: one line per filing, as a system, user
and assistant turn, so each base model's own chat template applies and the loss lands on
the assistant turn alone.

**The target is exactly what the grader marks right.** The assistant turn is the same
`prompts.answer(truth)` string that writes the worked examples in a few-shot prompt,
character for character. Train towards anything else and the model learns a format that is
then marked wrong for being that format, and it surfaces as a `malformed` rate nobody can
account for.

**The prompt is the zero-shot one.** The frontier runs measured what worked examples buy:
nothing for two models of three, at twice the input cost. A fine-tune that needs examples
in its prompt pays that on every call for the life of the deployment, and per-call cost is
the entire argument of this project. The dataset manifest records the prompt fingerprint,
so a model trained against one prompt and measured against another can be caught.

**Nothing from a test pool can be written.** `build` refuses any split but train and
validation. A training file is the one artefact where a leak cannot be undone afterwards:
the model has seen it, and no later check unsees it.

**Volume subsets are nested.** The 1,000-filing set is inside the 2,500, which is inside
the 5,000, chosen by keyed hash. Two independent draws would differ in which filings they
hold as well as how many, and the scaling curve is meant to vary one thing.

On the full corpus: **5,060 training and 713 validation examples**, median 10,745
characters, p95 13,437.

## The recipe

A `TrainConfig` is the whole of a run, and its identifier is derived from its own
contents, so the same recipe is the same run on any machine, a checkpoint directory is
addressable without a registry, and a rerun resumes rather than starting a second copy
beside the first. Change any field and it is a different run. Same rule as the prompt
fingerprint, for the same reason.

**Where the adapter goes.** Every linear projection of the language model, attention and
MLP: 205 layers on the 2B, 258 on the 4B, 224 on the 7B, checked against each base's
module names. The QLoRA paper found attention-only adapters fall short of full
fine-tuning, and PEFT's default for these families is exactly that, `q_proj` and `v_proj`,
which is what the first real smoke run got before this was set. The Gemma bases load with
vision and audio towers attached; nothing in this task reaches them, so no adapter goes
there, and the run stops if one does.

**What the first day on the card found**, 2026-09-22, each before a single optimiser step,
and each fixed with a test: none of the three bases ships a chat template (the run takes
its instruction model's, pinned); TRL 1.x wants a `Dataset` and a float `warmup_steps`; the
examples were one `messages` field, on which TRL applies no completion mask, so the loss
would have fallen on the prompt and the filing too (they are now prompt and completion, and
the run reads the mask back from the processed data and refuses to train without one); and
`--volume` sized the step count but not the data, so every volume run would have trained on
everything.

`alpha` must not be below `rank`. The adapter's contribution scales as alpha over rank, so
holding alpha fixed while sweeping rank sweeps the scaling factor at the same time and the
result says nothing about rank. The sweep sets alpha to twice the rank.

**The ablations are one factor at a time, not a grid.** Rank, learning rate and volume at
three values each is 27 runs as a grid and 8 as a sweep, on each of three model sizes.
What a grid buys is interactions between the factors; what it would cost is the three
seeds on the chosen configuration, which is the part that says whether any difference is
real at all. The seeds are worth more than the interactions.

The sweep for one size is 8 runs: five at the full pool, 646 steps each at the default two
epochs, and the three volume points at 126, 314 and 626 steps, **4,296 steps** in all, 13
checkpoints on a full run. Two more seeds on the chosen recipe add 1,292. A step is 16
sequences of about 3,100 tokens, roughly 50,000 tokens.

**Correction.** An earlier version of this page said the 2B sweep was 4.2 GPU hours, at 3.5
seconds a step. That rate was an example typed into `train plan` to show the arithmetic,
not a measurement, and it is not plausible: it is about 14,000 tokens a second of QLoRA on
one card. The throughput has not been measured and will not be until the first run on the
rented card; `gpu_hours` refuses to invent it for exactly this reason. The budgeting range
in [gpu-prices.md](gpu-prices.md) is stated as a range and labelled as an estimate.

**Plan amendment.** `PLAN.md` asked for the data-scaling curve at 1k, 5k and 20k. The
training pool holds 5,060 filings, so the 20k point needs a corpus expansion of about
4,800 more companies: five to seven hours of fetching within the SEC's rate limit and
another 8 GB of cache. The curve now runs at 1k, 2.5k and 5k, and the expansion happens
only if accuracy is still climbing at 5k. If it has flattened, the 20k point buys nothing,
and "this needs five thousand examples, not twenty" is a more useful finding than the
curve the plan imagined. Amended in the commit that added this file.

## Checkpointing, which is a money problem

The rule is that a run checkpoints from the first step and resumes from object storage.
The reason is that the GPU work is costed on spot instances, a spot instance is reclaimed
with about thirty seconds of warning, and a run keeping its only copy on the instance's own
disk loses everything it has done. A run that uploads from step one loses at most
`save_steps` steps.

`CheckpointStore` knows where checkpoints go and which is newest; a `Syncer` moves them.
`LocalSyncer` copies to another directory, which is what the tests use and what a mounted
volume gives. `CommandSyncer` shells out to whatever the provider offers: `aws s3 sync`,
`gcloud storage rsync` and `rclone sync` all take source then destination, so the provider
decision in [gpu-prices.md](gpu-prices.md) stays configuration rather than a code change.

A checkpoint is resumable only once a `.complete` marker is written beside it. Resuming
from a half-uploaded directory is worse than resuming from the step before it, and an
instance reclaimed mid-upload is exactly when that happens. The test writes a
`checkpoint-99` with no marker next to a complete `checkpoint-10` and asserts the run
resumes from 10.

Over an object store the marker has to go last, and it did not at first. `CommandSyncer`
wrote the marker into the directory and synced the lot, and a sync promises no order: the
marker could land before the weights, so a pod lost mid-upload left a checkpoint that
claimed to be complete. It now syncs the weights, then writes the marker and syncs again,
which sends only the marker; and a remote checkpoint counts only if its marker is listed.
Found on 2026-09-22, before the first rental, by a test whose stand-in for `aws s3` copies
files in name order, which puts `.complete` first and cuts the upload off after one file:
the old code offered the broken checkpoint for a resume, the new one falls back to the
last whole one.

**On Runpod** the store is a network volume reached through its S3-compatible API,
`train run --checkpoint-uri s3://<volume id>/<path> --s3-datacenter <DC>`, because the
training pod is on Community Cloud and volumes mount only on Secure Cloud
([gpu-prices.md](gpu-prices.md#the-decision)).

`latest` looks locally first and remotely second. A live instance has the weights on disk
and a download is minutes of rent; a fresh instance after a reclaim has a bare disk, and
that is the case the remote copy exists for.

## What the run records

A `RunRecord` beside the weights: the recipe, the dataset manifest with its prompt
fingerprint, the base model's revision as resolved on the day, the library versions, the
step it resumed from, the losses, and the measured seconds per step. Model cards are
written from that file, so a card cannot claim something the run did not do.

## Not done yet

The `train` extra in `pyproject.toml` is deliberately empty until the card is rented: the
versions that matter are the ones agreeing with the CUDA build on the machine. The
packages the code imports are torch, transformers, trl, peft, bitsandbytes and accelerate.
No base model revision has been resolved, which `docs/models.md` requires before any run
starts. Nothing has been trained.
