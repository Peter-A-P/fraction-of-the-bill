# The first day on a rented GPU

Written before the card is rented, so the hours paid for are spent running things rather
than working out what to run. Every step names what it produces and what has to be true
before the next one starts. Two of the steps are gates that stop and report rather than
carrying on, because they are where a mistake would otherwise be multiplied by the
schedule.

## Before renting

- `RUNPOD_API_KEY` is in `.env`, and the account holds prepaid credit with auto-reload
  off. The credit is the hard spend cap: pods stop when it runs out, which is the console
  cap the plan requires, and the date it was set goes in [gpu-prices.md](gpu-prices.md).
- The public half of `~/.ssh/runpod_smallprint` is registered under the account's SSH
  public keys, so pods accept it. The private half never leaves the machine that made it.
- No Hugging Face token: all six base and baseline repositories are ungated Apache 2.0.
  One is needed only to publish.
- An S3 API key for the account, created in the console under Settings, S3 API Keys. Its
  access key (the `user_...` id) and secret go in `.env` as `RUNPOD_S3_ACCESS_KEY_ID` and
  `RUNPOD_S3_SECRET_ACCESS_KEY`, and reach the pod as `AWS_ACCESS_KEY_ID` and
  `AWS_SECRET_ACCESS_KEY`. Checkpoints need it; see step 1.
- The day-one price table in [gpu-prices.md](gpu-prices.md), from the provider's pages that
  morning. Written 2026-09-22.

## 1. The pod and the volume

One RTX 4090 on Community Cloud for training: 24 GB holds all three bases in 4-bit, and
training has no bearing on any published number except accuracy, so the cheapest card that
fits is the right one.

A network volume of about 50 GB in one of the data centers with Runpod's S3-compatible API,
which survives any pod: that is where checkpoints go, so a lost pod loses at most
`save_steps` steps. It is not mounted. Network volumes mount only on Secure Cloud pods, at
more than twice the price for the same card, so the Community pod pushes each checkpoint
to it over the S3 API instead ([gpu-prices.md](gpu-prices.md#the-decision)). The pod needs
the AWS CLI (`pip install awscli`) and the S3 key in its environment.

**Renting, as learned on 2026-09-22.** Ask for a public IP (`supportPublicIp`): without one
a Community pod offers SSH only through Runpod's proxy, which carries neither file copies
nor remote commands, and three pods were paid for and let go before this was understood.
A pod whose image is not cached on its host takes minutes to start, so wait on the pod's
runtime appearing, not on a fixed timeout. And check CUDA before installing anything: the
first 4090 rented listed its card in `nvidia-smi` and failed `torch.cuda` with "CUDA
unknown error" under the image's own PyTorch, a host fault nothing inside the container
fixes. A matrix multiply on the card is the check; a host that fails it is terminated and
replaced. **Check the driver, not just the card.** The pinned torch is a CUDA 13 build and
needs a 580 driver; Community hosts still run 575, and such a host passes a matmul under
the image's own PyTorch and then refuses ours with "driver is too old". One was rented and
let go for it on 2026-09-22. The rental check now reads
`nvidia-smi --query-gpu=driver_version` and takes nothing below 580, because the
alternative, an older torch on that pod, would train one size on a different stack from the
other two. Put the model cache on the container disk (`HF_HOME=/root/hf`, 80 GB), not the
20 GB pod volume: the Python environment already takes most of the volume, and the 4B
download failed there with the disk full. The cache is re-downloadable and nothing is lost
with the pod. The code, the corpus and the S3 key go over `scp`; the repository is private and
no GitHub token goes on a rented machine.

Serving is different and is not on this card. Its throughput is the denominator of every
self-hosted cost per call, so it is measured on a card and tier a business would deploy on.

## 2. The code and the stack

    git clone https://github.com/Peter-A-P/fraction-of-the-bill && cd fraction-of-the-bill
    uv sync --extra gateway
    uv pip install torch transformers trl peft bitsandbytes accelerate

Then pin what was installed in the `train` extra of `pyproject.toml`, in a commit made
from the pod, because the versions that matter are the ones that agree with that machine's
CUDA. Every `RunRecord` also records them.

## 3. The corpus

The build is not in git. Copy `data/build/full` to the pod (65 MB; `runpodctl send` or
`scp`), then:

    smallprint data verify --build-dir data/build/full
    smallprint train dataset --build-dir data/build/full --out data/train/full

`verify` checks every file against the `SHA256SUMS` the datasheet wrote, so the pod trains
on byte-for-byte the corpus the baselines were measured on. The dataset step prints 5,060
training and 713 validation examples and the prompt fingerprint `e015ec5e057be645`. Any
other numbers mean a different corpus, and nothing goes further until that is explained.

## 4. A smoke run per size, and a deliberate interruption

    smallprint train run --size 2b --out runs/smoke-2b --volume 320 --epochs 1 \
      --save-steps 5 --checkpoint-uri s3://<volume id>/checkpoints/smoke-2b \
      --s3-datacenter <DC>

Twenty optimiser steps. Partway through, stop it, **delete `runs/smoke-2b`**, then run the
same command again. Deleting the local copy is what makes this the test that matters: the
pod is now in the state a replacement pod would be, and the run has to fetch the newest
complete checkpoint from the volume and resume from it rather than from step zero.
`run.json` has to say which step it resumed from. That is the checkpointing the whole
spot-instance budget rests on, proved on real hardware before anything depends on it.

`run.json` then holds `seconds_per_step`, measured on the trainer's own clock over the
steps that session ran, not wall time, which on a run this short would be mostly model
loading. Repeat for `4b` and `7b`.

## 5. Gate: the real schedule, before the real spend

    smallprint train plan --size 2b --seconds-per-step <measured>
    smallprint train plan --size 4b --seconds-per-step <measured>
    smallprint train plan --size 7b --seconds-per-step <measured>

This turns the budgeting range in [gpu-prices.md](gpu-prices.md) into a number. **Stop
here and report it**: the GPU hours and the dollars for the three sweeps and the seeds,
against the credit remaining. The full sweep starts only after that is agreed. If one
epoch looks enough on the smoke runs' validation loss, this is also where the two-epoch
default is reconsidered, since the second epoch is the most expensive thing in the plan.

## 6. The sweeps

Each size's eight runs from `train plan`, output on the pod's local disk and checkpoints
on the volume. Cheapest size first, so a problem in the recipe shows up on the run that
costs least to repeat.

    VOLUME_ID=<volume id> DATACENTER=<DC> scripts/sweep.sh 2b 2 16:1e-4:all:0 8:1e-4:all:0 ...

A recipe is `rank:learning_rate:volume:seed`. Each finished run's adapter and `run.json` go
to the volume under `adapters/<name>`, and a run already there is skipped, so the same
command on a replacement pod picks up where the lost one stopped. One pod a size, once the
first run has shown the recipe works: the cost is the same and the wait is a third.

**A host reboot does not restart the run.** On 2026-09-23 the host under both the 4B and 7B
pods dropped its heartbeat for four minutes and came back rebooted: `/workspace` and the
Python environment on it survived, the container disk and with it the model cache did not,
and nothing was running. The same `sweep.sh` command, started again, carries on from the
newest checkpoint on the volume; the base model downloads again. Two pods on one host are
one failure, not two, so where there is a choice, rent them on different hosts.

## 7. Gate: the first fine-tune measured

The best 2B run is evaluated on the post-cutoff test set before the other two sizes are
trained to completion. If a 2B fine-tune does not clear the frontier cost anchor's field
accuracy, that is the finding to report before spending on the larger sizes, not after.

    uv sync --extra gateway --extra train
    scripts/formats.sh 2b-r16-lr1e-4-nall-s0-e1 bf16
    uv pip install vllm
    scripts/evaluate.sh 2b-r16-lr1e-4-nall-s0-e1 bf16

vLLM is installed after the merge because it brings its own torch, which replaces the pinned
training stack in that environment; the merge is made on the stack the run trained on.

`formats.sh ... bf16` merges the adapter into its base and checks the merge
([serving.md](serving.md#from-a-finished-run-to-its-formats)); `evaluate.sh` serves the
merged weights with vLLM, adds them to the gateway as a self-hosted provider, and runs the
same `baseline run` the frontier numbers came from, at temperature 0, into
`data/finetuned/`. Then, on the laptop, with that directory copied back:

    smallprint report --finetuned data/finetuned

The two-epoch probe is evaluated the same way beside the one-epoch run, which is what
decides whether the loss curve missed anything ([training.md](training.md)).

## 8. The formats, and what each one cost

For each size's chosen run, after its seeds:

    scripts/formats.sh <run>
    for f in bf16 awq gptq gguf-q8_0 gguf-q4_k_m; do scripts/evaluate.sh <run> "$f"; done
    scripts/evaluate.sh <run> bf16 test_pre_cutoff

AWQ and GPTQ need the `quant` extra; GGUF needs a llama.cpp checkout built on the pod, in
`LLAMA_CPP`, and GGUF is served with its `llama-server`. Pin what was installed, the
llm-compressor version and the llama.cpp commit, in a commit made from the pod, as the train
extra was. `smallprint report --finetuned` then fills the quantisation table: every format's
paired delta against bf16, and whether it ships.
