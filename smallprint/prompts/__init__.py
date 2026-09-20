"""The prompt: one task specification, rendered identically for every model measured.

Every system in this project answers the same question, so there is one prompt and every
model gets it: the frontier APIs, the untuned bases, the fine-tunes, the quantised builds.
If the frontier baseline ran on a better-written prompt than the small models, the gap this
project publishes would be a gap between two prompts.

The field list is generated from `SCHEMA` rather than written out here, so the prompt, the
training target and the grader cannot drift apart. What is written out here is the part a
schema cannot express: which of several numbers on the page is the answer. Reading the
current period rather than the comparative column beside it, applying the scale printed
above the statement, and returning null rather than a plausible figure are the three things
models get wrong on this task, so each is a rule in its own right rather than a clause.

Few-shot examples come from the training pool only, never from a test or validation filing.
An example drawn from the test set would put the answer to a held-out item in the prompt of
its neighbours and quietly inflate every number measured afterwards, so `examples_from`
refuses any item that is not from the training split rather than filtering it out silently.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
from statistics import median
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from smallprint.data.build import SplitItem
from smallprint.data.split import Split
from smallprint.schema import SCHEMA, Extraction, json_schema_for_prompt

#: Gemma 4 turns reasoning on when this token opens the system prompt. The fine-tunes are
#: trained to answer without it; the untuned baselines are measured both ways, which is
#: what says whether reasoning buys anything on this task before any training.
THINK_TOKEN: Final = "<|think|>"

_TASK: Final = """You read the primary financial statements of a filing made to the US
Securities and Exchange Commission and return a fixed set of figures as JSON.

The input is an excerpt of one filing: its cover page, its income statement, its balance
sheet, and, in an annual report, the signature block of the audit report.

Return one JSON object with exactly these fifteen keys and no others:

{fields}

Rules:

1. Values are in whole units. A statement is printed in thousands or in millions and says
   which in a heading above it or beside a line; apply it. A line reading 1,234 under a
   heading that says "in thousands" is 1234000. Per-share amounts are printed as they are
   meant and are never scaled. Share counts follow the scale stated for share counts,
   which is often not the one stated for the money columns.
2. Report the period the filing covers. Every statement prints earlier periods in the
   columns beside it, and the balance sheet prints the prior year end; those are not the
   answer.
3. A loss, a deficit or an outflow is negative. Figures printed in parentheses are
   negative. Cost of revenue is positive.
4. If the filing does not report a field, its value is null. Do not derive a figure that
   is not printed, and do not take one from a line that means something else.
5. Answer with the JSON object alone: no explanation, no code fence, no other text."""

#: What comes after the excerpt. Short, and last, because the instruction furthest from the
#: start of a long input is the one a model is most likely to still be following.
_CLOSING: Final = "Return the JSON object for the filing above."


class PromptStyle(StrEnum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


def system_prompt(*, thinking: bool = False) -> str:
    """The task specification, with the field list generated from `SCHEMA`."""
    text = _TASK.format(fields=json_schema_for_prompt())
    return f"{THINK_TOKEN}\n{text}" if thinking else text


def _plain(value: object) -> Any:
    """A label as JSON. Whole floats are written as integers, which is how a filing prints
    them and so how a model is likely to answer; both parse back to the same float."""
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def answer(truth: Extraction) -> str:
    """One extraction as the model is asked to return it: every key, in schema order."""
    return json.dumps({spec.name: _plain(getattr(truth, spec.name)) for spec in SCHEMA})


def user_message(text: str) -> str:
    return f"{text}\n\n{_CLOSING}"


class Prompt(BaseModel):
    """A rendered prompt: the system text, and the examples that go in front of the input."""

    model_config = ConfigDict(frozen=True)

    style: PromptStyle
    thinking: bool
    system: str
    #: Example (input, answer) pairs, in the order they are shown.
    examples: tuple[tuple[str, str], ...] = ()
    #: Which items the examples came from, so a run can be traced back to them.
    example_ids: tuple[str, ...] = ()

    def messages(self, text: str) -> list[dict[str, str]]:
        """The chat messages for one filing: examples as completed turns, then the input.

        Examples go in as user and assistant turns rather than inside the system prompt
        because that is the shape every chat template in the three families renders
        natively, and a model that has just produced two bare JSON objects is markedly
        less likely to open the third with prose.
        """
        out: list[dict[str, str]] = []
        for example_text, example_answer in self.examples:
            out.append({"role": "user", "content": user_message(example_text)})
            out.append({"role": "assistant", "content": example_answer})
        out.append({"role": "user", "content": user_message(text)})
        return out

    @property
    def fingerprint(self) -> str:
        """A digest over everything the model is shown except the filing.

        Recorded with every run. Two accuracy numbers are comparable only if this matches,
        and an edit to a rule that looks cosmetic changes it, which is the point.
        """
        digest = hashlib.sha256(self.system.encode())
        for example_text, example_answer in self.examples:
            digest.update(example_text.encode())
            digest.update(example_answer.encode())
        return digest.hexdigest()


def _rank(seed: int, item_id: str) -> bytes:
    return hashlib.blake2b(f"example:{seed}:{item_id}".encode(), digest_size=8).digest()


def examples_from(pool: Sequence[SplitItem], k: int, *, seed: int = 20260919) -> list[SplitItem]:
    """Choose `k` training filings to show as worked examples.

    Forms take turns, so a two-shot prompt shows one annual report and one quarterly one.
    That is not only for coverage of the wording: an annual report has an auditor to name
    and a quarterly one does not, so the pair demonstrates both a filled field and a null
    one without a rule having to describe what a null looks like.

    Drawn from the shorter half of the pool by input length, so that a few-shot prompt
    costs about what the zero-shot one does per example rather than being decided by one
    very long filing, and by a keyed hash, so every machine builds the same prompt.
    """
    if k <= 0:
        return []
    wrong = {s.split for s in pool} - {Split.TRAIN}
    if wrong:
        raise ValueError(
            "few-shot examples may only come from the training pool; "
            f"got {', '.join(sorted(s.value for s in wrong))}"
        )
    if not pool:
        raise ValueError("no training items to draw examples from")
    cutoff = median(len(s.item.text) for s in pool)
    short = [s for s in pool if len(s.item.text) <= cutoff] or list(pool)
    by_form: dict[str, list[SplitItem]] = defaultdict(list)
    for s in short:
        by_form[s.item.form].append(s)
    queues = [
        sorted(group, key=lambda s: _rank(seed, s.item.item_id))
        for _, group in sorted(by_form.items())
    ]
    chosen: list[SplitItem] = []
    depth = 0
    while len(chosen) < k and any(depth < len(q) for q in queues):
        for queue in queues:
            if depth < len(queue) and len(chosen) < k:
                chosen.append(queue[depth])
        depth += 1
    if len(chosen) < k:
        raise ValueError(f"asked for {k} examples, the training pool has {len(chosen)}")
    return chosen


def build_prompt(
    style: PromptStyle,
    *,
    pool: Sequence[SplitItem] = (),
    k: int = 2,
    thinking: bool = False,
    seed: int = 20260919,
) -> Prompt:
    """The prompt one run uses. `pool` is the training split, and is unused when zero-shot."""
    if style is PromptStyle.ZERO_SHOT:
        return Prompt(style=style, thinking=thinking, system=system_prompt(thinking=thinking))
    chosen = examples_from(pool, k, seed=seed)
    return Prompt(
        style=style,
        thinking=thinking,
        system=system_prompt(thinking=thinking),
        examples=tuple((s.item.text, answer(s.item.truth)) for s in chosen),
        example_ids=tuple(s.item.item_id for s in chosen),
    )
