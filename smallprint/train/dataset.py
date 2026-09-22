"""Turning the corpus into training examples.

The target a model is trained to emit is exactly the string the grader parses: the same
`prompts.answer` that writes the worked examples, character for character. Anything else
trains the model towards a format that is then marked wrong for being that format, and the
gap shows up as a `malformed` rate nobody can explain.

The prompt is the zero-shot one, with no worked examples in it. The frontier runs measured
what examples buy: nothing for two models of three, at twice the input cost. A fine-tune
that needs examples in its prompt is paying that on every call forever, and the whole
argument of this project is per-call cost.

Two rules this module enforces rather than assumes:

**Nothing from a test pool is ever written.** `build` refuses any split other than train
and validation, because a training file is the one artefact where a leak is invisible
afterwards: the model has already seen it and no later check can unsee it.

**Volume subsets are nested.** The 1,000-example set is a subset of the 2,500, which is a
subset of the 5,000. The data-scaling curve is supposed to vary one thing, and two
independent draws differ in which filings they contain as well as how many.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from smallprint.data.build import SplitItem
from smallprint.data.split import Split
from smallprint.prompts import Prompt, PromptStyle, answer, build_prompt, user_message

#: The splits a training file may be written from. The test pools are not here on purpose.
TRAINABLE: Final[frozenset[Split]] = frozenset({Split.TRAIN, Split.VALIDATION})

EXAMPLES: Final = "examples.jsonl"
DATASET_MANIFEST: Final = "dataset.json"


class Example(BaseModel):
    """One training example, in the shape a chat fine-tune expects.

    Messages rather than a single string, so the tokeniser applies each base model's own
    chat template and the loss lands on the assistant turn alone. The item id travels with
    it so a training loss can be traced back to the filing that produced it.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str
    split: Split
    system: str
    user: str
    assistant: str

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.assistant},
        ]


class DatasetManifest(BaseModel):
    """What was written, from what, and with which prompt.

    The fingerprint is the load-bearing field: a fine-tune trained against one prompt and
    measured against another is measuring the difference between two prompts. The training
    run records this, the evaluation records the same thing, and they have to match.
    """

    model_config = ConfigDict(frozen=True)

    build_dir: str
    prompt_fingerprint: str
    style: PromptStyle
    train: int
    validation: int
    seed: int
    #: Character length of the whole example, for choosing a sequence length.
    median_chars: int
    p95_chars: int


#: The seed the volume subsets are drawn with. One constant, because the subsets have to
#: be the same filings whether they are taken when the file is written or when a run reads it.
VOLUME_SEED: Final = 20260920


def _rank(seed: int, item_id: str) -> bytes:
    return hashlib.blake2b(f"train:{seed}:{item_id}".encode(), digest_size=8).digest()


def example(item: SplitItem, prompt: Prompt) -> Example:
    return Example(
        item_id=item.item.item_id,
        split=item.split,
        system=prompt.system,
        user=user_message(item.item.text),
        assistant=answer(item.item.truth),
    )


def take(items: Sequence[SplitItem], volume: int | None, *, seed: int) -> list[SplitItem]:
    """The first `volume` filings by keyed hash, or all of them.

    By hash rather than by file order so the subsets are nested and stable: raising the
    volume adds filings and removes none, which is what makes the scaling curve a curve
    over volume rather than over two different corpora.
    """
    ordered = sorted(items, key=lambda s: _rank(seed, s.item.item_id))
    return ordered if volume is None else ordered[:volume]


def take_examples(
    examples: Sequence[Example], volume: int | None, *, seed: int = VOLUME_SEED
) -> list[Example]:
    """The same subset as `take`, from examples already written.

    The sweep's three volume runs read one dataset file, so the run itself takes its
    subset. The first real smoke run on 2026-09-22 asked for 320 filings and was handed all
    5,060: the step count honoured the volume and the data did not, and the scaling curve
    would have been flat by construction. Asking for more filings than there are is
    refused rather than quietly trained on fewer.
    """
    if volume is not None and volume > len(examples):
        raise ValueError(f"volume {volume} is more than the {len(examples)} examples there are")
    ordered = sorted(examples, key=lambda e: _rank(seed, e.item_id))
    return ordered if volume is None else ordered[:volume]


def build(
    items: Iterable[SplitItem],
    *,
    volume: int | None = None,
    seed: int = VOLUME_SEED,
    prompt: Prompt | None = None,
) -> tuple[list[Example], DatasetManifest]:
    """Training and validation examples, and the manifest that describes them."""
    pool = list(items)
    wrong = {s.split for s in pool} - TRAINABLE
    if wrong:
        raise ValueError(
            "a training file may only hold train and validation filings; got "
            + ", ".join(sorted(s.value for s in wrong))
        )
    chosen_prompt = prompt if prompt is not None else build_prompt(PromptStyle.ZERO_SHOT)
    train = take([s for s in pool if s.split is Split.TRAIN], volume, seed=seed)
    validation = [s for s in pool if s.split is Split.VALIDATION]
    if not train:
        raise ValueError("no training filings")

    examples = [example(s, chosen_prompt) for s in (*train, *validation)]
    lengths = sorted(len(e.system) + len(e.user) + len(e.assistant) for e in examples)
    return examples, DatasetManifest(
        build_dir="",
        prompt_fingerprint=chosen_prompt.fingerprint,
        style=chosen_prompt.style,
        train=len(train),
        validation=len(validation),
        seed=seed,
        median_chars=lengths[len(lengths) // 2],
        p95_chars=lengths[int(len(lengths) * 0.95)],
    )


def write(
    out_dir: Path, examples: Sequence[Example], manifest: DatasetManifest, *, build_dir: Path
) -> Path:
    """Write `examples.jsonl` and its manifest. Returns the JSONL path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / EXAMPLES
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in examples:
            handle.write(item.model_dump_json() + "\n")
    (out_dir / DATASET_MANIFEST).write_text(
        manifest.model_copy(update={"build_dir": str(build_dir)}).model_dump_json(indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return path


def read(out_dir: Path) -> tuple[list[Example], DatasetManifest]:
    with (out_dir / EXAMPLES).open(encoding="utf-8") as handle:
        examples = [Example.model_validate_json(line) for line in handle if line.strip()]
    manifest = DatasetManifest.model_validate_json(
        (out_dir / DATASET_MANIFEST).read_text(encoding="utf-8")
    )
    return examples, manifest


def as_chat(examples: Iterable[Example]) -> list[dict[str, object]]:
    """One row per example, split into the conversation so far and the answer.

    The split is what makes the loss fall on the answer alone. With the whole chat as one
    `messages` field, TRL applies `completion_only_loss` to nothing and trains on the prompt
    too: the task specification and ten thousand characters of filing, identical in intent
    across every example, taught as if they were output. Found on the pod on 2026-09-22,
    before a step had run.
    """
    return [{"prompt": e.messages()[:-1], "completion": e.messages()[-1:]} for e in examples]


def token_lengths(examples: Sequence[Example], tokenise: object) -> list[int]:
    """Tokenised length of each example, using the base model's own tokeniser.

    Takes the callable rather than a model name so the sequence-length decision can be
    made, and tested, without loading anything: the trainer passes
    `tokenizer.apply_chat_template`.
    """
    if not callable(tokenise):
        raise TypeError("tokenise must be callable, for example tokenizer.apply_chat_template")
    return [len(tokenise(e.messages())) for e in examples]


def truncated(lengths: Sequence[int], max_seq_len: int) -> tuple[int, float]:
    """How many examples a sequence length would cut, and what share that is.

    A truncated example teaches the model to stop mid-JSON, so this is checked before a
    run rather than discovered in the malformed rate afterwards.
    """
    cut = sum(1 for n in lengths if n > max_seq_len)
    return cut, (cut / len(lengths) if lengths else 0.0)


def summary(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    return (
        f"{first['item_id']}: {len(first['user']):,} characters in, {len(first['assistant'])} out"
    )
