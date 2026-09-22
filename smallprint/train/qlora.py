"""The QLoRA run itself.

Everything that needs a GPU lives behind a function call rather than an import, so the
rest of this package, its tests and `smallprint data build` all work on a laptop with no
torch installed. The heavy imports happen inside `train`, and the error a laptop gets says
which extra to install rather than a bare ImportError from four frames down.

What this file is responsible for, beyond calling a trainer:

**Checkpointing from the first step.** `SaveToStore` uploads every checkpoint the trainer
writes, and the run asks the store where to resume before it builds anything. A reclaimed
spot instance costs `save_steps` of work and no more.

**The chat format.** A base checkpoint ships no chat template, and the examples are chats.
The run adopts the template of the same family's instruction model, at its pinned
revision: the fine-tune is then prompted exactly as the untuned model it is compared with,
and the template is saved with the adapter, so the server uses it too. Checked on the pod
on 2026-09-22 that for all three sizes the templated text tokenises identically under the
base's own tokenizer, so the turn markers are the tokens the base already has.

**Recording what was run.** A `RunRecord` beside the weights: the recipe, the dataset
manifest with its prompt fingerprint, the base model's revision as resolved on the day,
and the library versions. A model card is written from that file, so a card cannot claim
something the run did not do.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from smallprint.train.checkpoint import CheckpointStore, step_of
from smallprint.train.dataset import DatasetManifest, Example, as_chat, take_examples
from smallprint.train.recipe import TrainConfig

#: Validation filings scored at every checkpoint. The loss curve steers decisions, not the
#: headline, and the full 713 at every checkpoint was 26 of the schedule's 154 GPU hours on
#: 2026-09-22; 200 draws the same curve for about a quarter of that. The full set is still
#: scored once, at the end, and that is the run's validation loss.
MONITOR_FILINGS: Final = 200

INSTALL_HINT = (
    "training needs the GPU extra, which is not installed on this machine: "
    "`uv sync --extra train` on the rented card. Nothing in this project trains on a laptop."
)


class RunRecord(BaseModel):
    """What a training run was. Written before the first step and updated at the end."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    config: TrainConfig
    dataset: DatasetManifest
    #: The exact weights. A base named without its revision is not reproducible, and
    #: docs/models.md says a revision is recorded before any run starts.
    base_revision: str | None = None
    #: Where the chat template came from: "base" when the checkpoint has its own, otherwise
    #: the instruction model's repository and revision it was taken from, with the digest
    #: of the template text, so the served format can be checked against the trained one.
    chat_template: str | None = None
    chat_template_sha256: str | None = None
    started_at: dt.datetime
    finished_at: dt.datetime | None = None
    resumed_from_step: int = 0
    steps: int = 0
    train_loss: float | None = None
    validation_loss: float | None = None
    seconds_per_step: float | None = None
    #: One pass over the whole validation set, as the run makes at its end.
    evaluation_seconds: float | None = None
    #: How many validation filings were scored at each checkpoint, and the loss each time,
    #: as (step, loss). Where this flattens is where more steps stop buying anything, which
    #: is how the second epoch is judged.
    monitored_on: int | None = None
    monitor_losses: list[tuple[int, float]] = []
    versions: dict[str, str] = {}

    def write(self, out_dir: Path) -> Path:
        path = out_dir / "run.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        return path


def _versions() -> dict[str, str]:
    """Library versions, for the record. Absent ones are recorded as absent, not guessed."""
    found = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        found[name] = str(getattr(module, "__version__", "unknown"))
    return found


class Callback:
    """A trainer callback without the trainer: every hook it does not define does nothing.

    The trainer calls a dozen hooks on every callback. Defined here rather than taken from
    transformers so this module imports without the GPU stack. The timing runs of
    2026-09-22 stopped at step 0 on a callback that defined only the two hooks it used.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


class StepTimer(Callback):
    """Seconds per optimiser step, timed around the steps and nothing else.

    Not the trainer's `train_runtime`: that counts the evaluations and checkpoint saves in
    between, and on the 2B smoke run, 2026-09-22, it gave 84.7 seconds a step against about
    25 for the steps themselves, two four-minute evaluations spread over ten steps. The
    smoke run's figure is what every GPU-hour estimate is multiplied out from, so it has to
    be the steps. `on_step_begin` fires at the start of an optimiser step, gradient
    accumulation included, and `on_step_end` at its end, before any save or evaluation.
    A callback in the trainer's sense, defined here so the module imports without the GPU
    stack, and with the clock a parameter so it is tested without one.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        import time

        self.clock = clock or time.perf_counter
        self.started: float | None = None
        self.total = 0.0
        self.steps = 0

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.started = self.clock()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.started is not None:
            self.total += self.clock() - self.started
            self.steps += 1
            self.started = None

    @property
    def seconds_per_step(self) -> float | None:
        return self.total / self.steps if self.steps else None


def adopt_chat_template(
    tokenizer: Any, source: tuple[str, str] | None, load: Callable[[str, str], str | None]
) -> tuple[str, str]:
    """Give a tokenizer a chat template if it has none, and say where it came from.

    `source` is the instruction model's repository and revision; `load` fetches its
    template, and is a parameter so that this runs in a test without a download. A base
    without a template and nowhere named to take one from is refused, rather than trained
    on a format nothing will serve.
    """
    if tokenizer.chat_template is not None:
        origin = "base"
    else:
        if source is None:
            raise ValueError("the base has no chat template and no instruction model was named")
        template = load(*source)
        if template is None:
            raise ValueError(f"{source[0]} has no chat template either")
        tokenizer.chat_template = template
        origin = f"{source[0]}@{source[1]}"
    digest = hashlib.sha256(str(tokenizer.chat_template).encode("utf-8")).hexdigest()
    return origin, digest


def monitor_subset(validation: list[Example], n: int = MONITOR_FILINGS) -> list[Example]:
    """The validation filings scored at each checkpoint: the same ones on every run, chosen
    by keyed hash, so two runs' curves are over the same filings."""
    return take_examples(validation, min(n, len(validation)))


def monitor_losses(log_history: list[dict[str, Any]]) -> list[tuple[int, float]]:
    """(step, eval loss) from the trainer's log, in step order."""
    return sorted(
        (int(entry["step"]), float(entry["eval_loss"]))
        for entry in log_history
        if "eval_loss" in entry and "step" in entry
    )


def loss_tokens(row: dict[str, Any]) -> tuple[int, int]:
    """How many tokens of a processed example carry the loss, and how many there are.

    Libraries say it differently: TRL 1.x writes `labels`, -100 where the loss is off; others
    carry a `completion_mask`. A row that says neither, or whose every token counts, is a
    format the library did not split, and training on it would teach the prompt as well as
    the answer: the task specification and a filing, as if they were output.
    """
    if "labels" in row:
        labels = list(row["labels"])
        counted, total = sum(1 for t in labels if t != -100), len(labels)
    elif "completion_mask" in row:
        mask = list(row["completion_mask"])
        counted, total = sum(mask), len(mask)
    else:
        raise RuntimeError(
            f"no labels and no completion mask in the processed data ({sorted(row)})"
        )
    if counted == 0 or counted == total:
        raise RuntimeError(
            f"the loss falls on {counted} of {total} tokens: the answer is not separated "
            "from the prompt"
        )
    return counted, total


def require_gpu_stack() -> None:
    """Fail early and legibly, rather than four frames into a trainer."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as missing:
        raise RuntimeError(f"{INSTALL_HINT} ({missing.name} is missing)") from missing


def sequence_length(config: TrainConfig, examples: list[Example], tokenizer: Any) -> int:
    """Check the configured sequence length against the data before spending an hour.

    A truncated example teaches the model to stop in the middle of its JSON, and the
    symptom is a malformed rate that looks like the model cannot produce JSON at all. One
    per cent is the most this will pass silently.
    """
    from smallprint.train.dataset import token_lengths, truncated

    lengths = token_lengths(
        examples, lambda messages: tokenizer.apply_chat_template(messages, tokenize=True)
    )
    cut, share = truncated(lengths, config.max_seq_len)
    if share > 0.01:
        longest = max(lengths)
        raise ValueError(
            f"max_seq_len {config.max_seq_len} would truncate {cut} of {len(lengths)} "
            f"examples ({share:.1%}); the longest is {longest} tokens. Raise it, or drop "
            "the filings that do not fit and say so in the datasheet"
        )
    return config.max_seq_len


def train(
    config: TrainConfig,
    examples: list[Example],
    dataset: DatasetManifest,
    *,
    out_dir: Path,
    store: CheckpointStore | None = None,
    base_revision: str | None = None,
    chat_template_from: tuple[str, str] | None = None,
) -> RunRecord:
    """Fine-tune one base on these examples, resuming from the newest checkpoint.

    The GPU libraries are imported here, not at the top of the file, so that importing
    `smallprint.train` costs nothing on a machine that will never train.
    """
    require_gpu_stack()
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = store if store is not None else CheckpointStore(out_dir)
    resume_from = checkpoints.latest()

    record = RunRecord(
        run_id=config.run_id,
        config=config,
        dataset=dataset,
        base_revision=base_revision,
        started_at=dt.datetime.now(dt.UTC),
        resumed_from_step=step_of(resume_from) or 0 if resume_from else 0,
        versions=_versions(),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base, revision=base_revision)

    def template_of(repo: str, revision: str) -> str | None:
        found = AutoTokenizer.from_pretrained(repo, revision=revision).chat_template
        return None if found is None else str(found)

    origin, digest = adopt_chat_template(tokenizer, chat_template_from, template_of)
    record = record.model_copy(update={"chat_template": origin, "chat_template_sha256": digest})
    record.write(out_dir)
    sequence_length(config, examples, tokenizer)

    quantisation = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if config.load_in_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base,
        revision=base_revision,
        quantization_config=quantisation,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    train_examples = take_examples([e for e in examples if e.split.value == "train"], config.volume)
    print(f"training on {len(train_examples):,} filings", flush=True)
    validation = [e for e in examples if e.split.value == "validation"]
    monitored = monitor_subset(validation)

    arguments = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        # Not the default of 8: with a 262k-token vocabulary, eight 3,000-token sequences are
        # about 12 GB of logits at once, which the 2B smoke run failed to allocate mid-eval.
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        # A float below one is a ratio of the total steps; TRL 1.x dropped warmup_ratio.
        warmup_steps=config.warmup_ratio,
        lr_scheduler_type="cosine",
        max_length=config.max_seq_len,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=config.save_steps,
        save_strategy="steps",
        # The repository's rule, made mechanical: a checkpoint exists before the run has
        # anything to lose, so a reclaim in the first minutes costs the first minutes.
        save_on_each_node=False,
        # On the pod's disk, only the newest two; the store has what a resume needs.
        save_total_limit=2,
        eval_strategy="steps" if validation else "no",
        eval_steps=config.save_steps,
        seed=config.seed,
        data_seed=config.seed,
        report_to=[],
        # The completion only: the prompt is identical across every example and training
        # on it teaches the model to recite the task specification.
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_list(as_chat(train_examples)),
        eval_dataset=Dataset.from_list(as_chat(monitored)) if monitored else None,
        peft_config=LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        processing_class=tokenizer,
    )
    trainer.add_callback(SaveToStore(checkpoints, out_dir))
    timer = StepTimer()
    trainer.add_callback(timer)
    # Which layers the adapter wraps, in the log: the Gemma bases load with their vision and
    # audio towers attached, and an adapter on those is parameters trained on nothing.
    adapted = [
        n.split(".lora_A")[0] for n, _ in trainer.model.named_parameters() if ".lora_A." in n
    ]
    places = Counter(
        next((t for t in ("vision", "audio", "language_model") if t in n), "layers")
        for n in adapted
    )
    kinds = sorted({n.rsplit(".", 1)[-1] for n in adapted})
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(
        f"LoRA on {len(adapted)} layers {dict(places)}, kinds {kinds}; "
        f"{trainable:,} trainable parameters",
        flush=True,
    )
    if places.get("vision") or places.get("audio"):
        raise RuntimeError(f"adapters on the vision or audio tower: {dict(places)}")
    # The loss must fall on the answer alone. Read back from the trainer's own processed
    # data, so a format the library silently ignores shows up here rather than as a model
    # that learned to recite filings.
    counted, total = loss_tokens(trainer.train_dataset[0])
    print(f"loss on {counted:,} of {total:,} tokens of the first example", flush=True)

    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    # The adapter, not a checkpoint: what gets published and merged, written once at the end.
    trainer.save_model(str(out_dir / "adapter"))

    history = monitor_losses(list(trainer.state.log_history))
    metrics = (
        trainer.evaluate(eval_dataset=Dataset.from_list(as_chat(validation))) if validation else {}
    )
    finished = record.model_copy(
        update={
            "finished_at": dt.datetime.now(dt.UTC),
            "steps": int(result.global_step),
            "train_loss": float(result.training_loss),
            "validation_loss": metrics.get("eval_loss"),
            "seconds_per_step": timer.seconds_per_step,
            "evaluation_seconds": metrics.get("eval_runtime"),
            "monitored_on": len(monitored),
            "monitor_losses": history,
        }
    )
    finished.write(out_dir)
    return finished


class SaveToStore(Callback):
    """A trainer callback that uploads each checkpoint as it is written.

    Defined here rather than imported from transformers so that this module still imports
    without the GPU stack; the trainer only needs the methods it calls.
    """

    def __init__(self, store: CheckpointStore, out_dir: Path) -> None:
        self.store = store
        self.out_dir = out_dir

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        directory = self.out_dir / f"checkpoint-{int(state.global_step)}"
        if directory.is_dir():
            self.store.save(directory)


def read_record(out_dir: Path) -> RunRecord:
    return RunRecord.model_validate_json((out_dir / "run.json").read_text(encoding="utf-8"))


def describe(record: RunRecord) -> str:
    return json.dumps(
        {
            "run": record.run_id,
            "base": record.config.base,
            "revision": record.base_revision,
            "examples": record.dataset.train,
            "steps": record.steps,
            "train_loss": record.train_loss,
            "validation_loss": record.validation_loss,
        },
        indent=2,
    )
