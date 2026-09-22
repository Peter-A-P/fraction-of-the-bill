"""The training recipe: what a run is, and what a set of runs varies.

A `TrainConfig` is the whole of a run. It has a deterministic identifier derived from its
own contents, so two people who write the same recipe get the same run id, a checkpoint
directory is addressable without a registry, and a rerun of the same recipe resumes rather
than starting a second copy beside the first. Change any field and it is a different run,
which is the same rule the prompt fingerprint follows and for the same reason.

The ablations are one factor at a time from a base recipe rather than a full grid. Rank,
learning rate and data volume across three values each is 27 runs as a grid and 8 as a
sweep, and a grid of 27 on three model sizes is not a week of GPU time this project has.
What a grid would buy is interactions between the factors; what it would cost is the three
seeds on the chosen configuration, which is the part that says whether a difference is
real at all. The seeds are worth more.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: LoRA ranks to sweep. 8 is the smallest that is usually enough for a format-following
#: task; 64 is where the adapter stops being small relative to the base.
RANKS: Final[tuple[int, ...]] = (8, 16, 64)
LEARNING_RATES: Final[tuple[float, ...]] = (5e-5, 1e-4, 2e-4)
#: The data-scaling curve. The plan asked for 1k, 5k and 20k; the corpus holds 5,154
#: training filings, so the third point needs a corpus expansion that is deferred until
#: this curve says more data would buy anything. See docs/training.md.
VOLUMES: Final[tuple[int, ...]] = (1_000, 2_500, 5_000)


class Base(BaseModel):
    """One of the three sizes: the checkpoint that is fine-tuned, and the untuned
    instruction model measured beside it, each pinned to the revision it was resolved at."""

    model_config = ConfigDict(frozen=True)

    repo: str
    revision: str
    baseline_repo: str
    baseline_revision: str


#: The three bases, resolved from the Hugging Face API on 2026-09-21 and recorded with
#: their weight digests in docs/models.md. A base named without its revision is not
#: reproducible: a later push to the repository would change what a recorded run trained
#: on. Keyed by the label the tables print.
BASES: Final[dict[str, Base]] = {
    "2b": Base(
        repo="google/gemma-4-E2B",
        revision="d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        baseline_repo="google/gemma-4-E2B-it",
        baseline_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    ),
    "4b": Base(
        repo="google/gemma-4-E4B",
        revision="411aa17b749aa952df1359d2dcea73917a544d9a",
        baseline_repo="google/gemma-4-E4B-it",
        baseline_revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
    ),
    "7b": Base(
        repo="allenai/Olmo-3-1025-7B",
        revision="a81bae42db3975be1671e27b9c9a56da1a9f980f",
        baseline_repo="allenai/Olmo-3-7B-Instruct",
        baseline_revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
    ),
}


class TrainConfig(BaseModel):
    """One QLoRA run, entire."""

    model_config = ConfigDict(frozen=True)

    #: The base checkpoint, as docs/models.md records it.
    base: str
    #: Which of the three sizes this is, for the tables. Not derived from `base`: the
    #: label is what the Pareto chart prints and a repository path is not a label.
    size: str

    rank: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    #: Which layers carry an adapter, as a full-match pattern on module names: every linear
    #: projection of the language model, attention and MLP, as the QLoRA paper found an
    #: attention-only adapter does not match full fine-tuning. Not the Gemma vision and
    #: audio towers, which this task never feeds: adapters there are parameters trained on
    #: nothing. Checked against all three bases' module names on 2026-09-22: 205, 258 and
    #: 224 layers, every one in the language model. PEFT's default for these families is
    #: q_proj and v_proj alone, which is what the first real smoke run got.
    target_modules: str = (
        r"^(?!.*(?:vision|audio)).*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    )
    learning_rate: float = Field(default=1e-4, gt=0)
    epochs: int = Field(default=2, ge=1)
    #: Examples per optimiser step is batch_size * grad_accum. Both are here because the
    #: first is bounded by the card and the second is not, and the tables report the
    #: effective size rather than either.
    batch_size: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=16, ge=1)
    max_seq_len: int = Field(default=8192, ge=512)
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    seed: int = 0
    #: Training filings to use. None means every one in the pool.
    volume: int | None = Field(default=None, ge=1)
    #: Every this many optimiser steps a checkpoint is written and uploaded. Small on
    #: purpose: the repository's rule is that a run checkpoints from the first step,
    #: because a spot instance can be reclaimed during the first one.
    save_steps: int = Field(default=50, ge=1)
    #: 4-bit base weights. The Q in QLoRA, and the reason a 7B trains on a 24 GB card.
    load_in_4bit: bool = True

    @model_validator(mode="after")
    def _alpha_follows_rank(self) -> Self:
        if self.alpha < self.rank:
            raise ValueError(
                f"alpha {self.alpha} below rank {self.rank}: the scaling factor alpha/rank "
                "would shrink the adapter's contribution as rank grows, which makes a rank "
                "sweep measure two things at once"
            )
        return self

    @property
    def examples_per_step(self) -> int:
        return self.batch_size * self.grad_accum

    def steps(self, n_examples: int) -> int:
        """Optimiser steps for a pool of this size. Partial final batches count."""
        if n_examples <= 0:
            raise ValueError("no examples")
        per_epoch = -(-min(n_examples, self.volume or n_examples) // self.examples_per_step)
        return per_epoch * self.epochs

    def checkpoints(self, n_examples: int) -> int:
        """How many checkpoints a run writes, the one at step 1 included."""
        return 1 + self.steps(n_examples) // self.save_steps

    @property
    def run_id(self) -> str:
        """A name derived from the recipe, so the same recipe is the same run everywhere."""
        payload = self.model_dump_json()
        digest = hashlib.blake2b(payload.encode(), digest_size=4).hexdigest()
        volume = "all" if self.volume is None else f"{self.volume}"
        return f"{self.size}-r{self.rank}-lr{self.learning_rate:g}-n{volume}-s{self.seed}-{digest}"


def sweep(base: TrainConfig) -> list[TrainConfig]:
    """One factor at a time from `base`: rank, then learning rate, then volume.

    The base recipe appears once rather than once per factor. With the default base, which
    is already rank 16 and 1e-4 and uses every filing, that is eight runs: the base, two
    more ranks, two more learning rates and three volumes.
    """
    runs = [base]
    seen = {base.run_id}
    for field, values in (
        ("rank", RANKS),
        ("learning_rate", LEARNING_RATES),
        ("volume", VOLUMES),
    ):
        for value in values:
            candidate = base.model_copy(update={field: value})
            if field == "rank":
                candidate = candidate.model_copy(update={"alpha": 2 * int(value)})
            if candidate.run_id not in seen:
                runs.append(candidate)
                seen.add(candidate.run_id)
    return runs


def seeds(config: TrainConfig, n: int = 3) -> list[TrainConfig]:
    """The same recipe `n` times. Without this a difference between two recipes cannot be
    told from the difference between two runs of one recipe."""
    if n < 1:
        raise ValueError("at least one seed")
    return [config.model_copy(update={"seed": s}) for s in range(n)]


def gpu_hours(configs: Sequence[TrainConfig], n_examples: int, seconds_per_step: float) -> float:
    """A schedule's length in GPU hours, from a measured seconds per step.

    The rate is measured by the first smoke run on the rented card and passed in; this
    function does arithmetic and never guesses a throughput.
    """
    if seconds_per_step <= 0:
        raise ValueError("seconds per step must be positive and measured, not assumed")
    return sum(c.steps(n_examples) for c in configs) * seconds_per_step / 3600
