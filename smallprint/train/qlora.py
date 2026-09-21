"""The QLoRA run itself.

Everything that needs a GPU lives behind a function call rather than an import, so the
rest of this package, its tests and `smallprint data build` all work on a laptop with no
torch installed. The heavy imports happen inside `train`, and the error a laptop gets says
which extra to install rather than a bare ImportError from four frames down.

What this file is responsible for, beyond calling a trainer:

**Checkpointing from the first step.** `SaveToStore` uploads every checkpoint the trainer
writes, and the run asks the store where to resume before it builds anything. A reclaimed
spot instance costs `save_steps` of work and no more.

**Recording what was run.** A `RunRecord` beside the weights: the recipe, the dataset
manifest with its prompt fingerprint, the base model's revision as resolved on the day,
and the library versions. A model card is written from that file, so a card cannot claim
something the run did not do.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from smallprint.train.checkpoint import CheckpointStore, step_of
from smallprint.train.dataset import DatasetManifest, Example, as_chat
from smallprint.train.recipe import TrainConfig

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
    started_at: dt.datetime
    finished_at: dt.datetime | None = None
    resumed_from_step: int = 0
    steps: int = 0
    train_loss: float | None = None
    validation_loss: float | None = None
    seconds_per_step: float | None = None
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


def step_seconds(train_runtime: float | None, global_step: int, resumed_from: int) -> float | None:
    """Seconds per optimiser step, on the trainer's own clock, over the steps this session ran.

    Not wall time since the run started: that includes downloading and loading the model,
    which on a twenty-step smoke run is most of it, and the smoke run's figure is what every
    GPU-hour estimate is then multiplied out from. And over this session's steps only, since
    a resumed run's global step counts the steps a previous machine did.
    """
    steps = global_step - resumed_from
    if train_runtime is None or steps <= 0:
        return None
    return train_runtime / steps


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
) -> RunRecord:
    """Fine-tune one base on these examples, resuming from the newest checkpoint.

    The GPU libraries are imported here, not at the top of the file, so that importing
    `smallprint.train` costs nothing on a machine that will never train.
    """
    require_gpu_stack()
    import torch
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
    record.write(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(config.base, revision=base_revision)
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

    train_examples = [e for e in examples if e.split.value == "train"]
    validation = [e for e in examples if e.split.value == "validation"]

    arguments = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
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
        train_dataset=as_chat(train_examples),
        eval_dataset=as_chat(validation) if validation else None,
        peft_config=LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        processing_class=tokenizer,
    )
    trainer.add_callback(SaveToStore(checkpoints, out_dir))

    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    # The adapter, not a checkpoint: what gets published and merged, written once at the end.
    trainer.save_model(str(out_dir / "adapter"))

    metrics = trainer.evaluate() if validation else {}
    finished = record.model_copy(
        update={
            "finished_at": dt.datetime.now(dt.UTC),
            "steps": int(result.global_step),
            "train_loss": float(result.training_loss),
            "validation_loss": metrics.get("eval_loss"),
            "seconds_per_step": step_seconds(
                result.metrics.get("train_runtime"),
                int(result.global_step),
                record.resumed_from_step,
            ),
        }
    )
    finished.write(out_dir)
    return finished


class SaveToStore:
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

    def __getattr__(self, name: str) -> Any:
        """Every other callback hook does nothing. The trainer calls a dozen of them."""
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


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
