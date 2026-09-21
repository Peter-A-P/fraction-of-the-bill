# The first day on a rented GPU

Written before the card is rented, so the hours paid for are spent running things rather
than working out what to run. Every step names what it produces and what has to be true
before the next one starts. Two of the steps are gates that stop and report rather than
carrying on, because they are where a mistake would otherwise be multiplied by the
schedule.

## Before renting

- `RUNPOD_API_KEY` is in `.env`, and the account holds prepaid credit. The credit is the
  hard spend cap: pods stop when it runs out, which is the console cap the plan requires.
- Write the day-one price table in [gpu-prices.md](gpu-prices.md) from the provider's page
  that morning: the card, the tier, the rate, the region, the storage rate, the date. The
  survey in that file was for the budget decision and is replaced by this.

## 1. The pod and the volume

One RTX 4090 on Community Cloud for training: 24 GB holds all three bases in 4-bit, and
training has no bearing on any published number except accuracy, so the cheapest card that
fits is the right one. A network volume of about 50 GB mounted at `/workspace`, which
survives the pod: that is where checkpoints go, so a reclaimed pod loses at most
`save_steps` steps.

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
on byte-for-byte the corpus the baselines were measured on. The dataset step prints 5,154
training and 736 validation examples and the prompt fingerprint `e015ec5e057be645`. Any
other numbers mean a different corpus, and nothing goes further until that is explained.

## 4. A smoke run per size, and a deliberate interruption

    smallprint train run --size 2b --out runs/smoke-2b --volume 320 --epochs 1 \
      --save-steps 5 --checkpoint-dir /workspace/checkpoints/smoke-2b

Twenty optimiser steps. Partway through, stop it, then run the same command again. It has
to resume from the newest complete checkpoint on the volume rather than from step zero,
and `run.json` has to say which step it resumed from. That is the checkpointing the whole
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

## 7. Gate: the first fine-tune measured

The best 2B run is evaluated on the post-cutoff test set with `smallprint baseline run`,
served through the gateway, before the other two sizes are trained to completion. If a 2B
fine-tune does not clear the frontier cost anchor's field accuracy, that is the finding to
report before spending on the larger sizes, not after.
