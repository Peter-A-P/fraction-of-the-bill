"""The quantisation gate: no format ships without its paired delta.

A quantised model is the same model with its weights rounded, so the question is never
"how accurate is it" in isolation, it is "how much did the rounding cost". That is a paired
question, the two runs grade the same filings, and the paired interval is the honest one:
an unpaired comparison of two accuracies near 96% would be several times wider than the
evidence warrants, wide enough to wave through a format that has lost a point.

**The margin, and where it comes from.** A format ships when the lower end of its paired
delta against bf16 is above minus one point of field accuracy. One point, because the
frontier measurement put the whole spread between the cheapest API and the dearest at 0.7
points of field accuracy: a quantised build that loses more than a point has thrown away
more than the entire difference money buys at the frontier, and the argument for owning
the model goes with it. The margin is on the lower bound, not the point estimate, so a
format with too few items to tell does not ship on the strength of a lucky average.

**Per field, because the loss is not spread evenly.** Rounding weights is expected to hurt
numeric fields first, a digit dropped or a scale misread, while dates and names survive.
An average across fifteen fields can hide one field that broke outright, so the verdict
carries every field's paired delta and names the worst.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

from smallprint.grade import Interval, ItemGrade, mean_ci, paired_delta_ci
from smallprint.schema import FIELDS

#: Field accuracy a format may lose against bf16, at the lower end of its 95% interval.
NON_INFERIORITY_MARGIN: Final = 0.01


class Format(StrEnum):
    """The formats every fine-tuned size is published in, and what serves each."""

    BF16 = "bf16"
    AWQ = "awq"
    GPTQ = "gptq"
    GGUF_Q4_K_M = "gguf-q4_k_m"
    GGUF_Q8_0 = "gguf-q8_0"

    @property
    def server(self) -> str:
        """vLLM for the safetensors formats, llama.cpp for GGUF, as PLAN.md section 2.6 says."""
        return "llamacpp" if self.value.startswith("gguf") else "vllm"

    @property
    def is_reference(self) -> bool:
        return self is Format.BF16


class FieldDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    delta: Interval


class Verdict(BaseModel):
    """Whether one format of one size may be published, and the evidence for it."""

    model_config = ConfigDict(frozen=True)

    format: Format
    delta: Interval
    fields: tuple[FieldDelta, ...]
    ships: bool
    reason: str

    @property
    def worst_field(self) -> FieldDelta:
        return min(self.fields, key=lambda f: f.delta.point)


def _by_id(grades: Sequence[ItemGrade]) -> dict[str, ItemGrade]:
    return {g.item_id: g for g in grades}


def field_deltas(
    quantised: Sequence[ItemGrade], reference: Sequence[ItemGrade], *, seed: int = 0
) -> tuple[FieldDelta, ...]:
    """Every field's paired delta, quantised minus reference, over the same filings."""
    q, r = _by_id(quantised), _by_id(reference)
    if q.keys() != r.keys():
        raise ValueError("a per-field comparison needs the same filings in both runs")
    out = []
    for name in FIELDS:
        differences = []
        for item_id in sorted(q):
            correct_q = next(o.correct for o in q[item_id].outcomes if o.field == name)
            correct_r = next(o.correct for o in r[item_id].outcomes if o.field == name)
            differences.append(float(correct_q) - float(correct_r))
        out.append(FieldDelta(field=name, delta=mean_ci(differences, seed=seed)))
    return tuple(out)


def judge(
    format: Format,
    quantised: Sequence[ItemGrade],
    reference: Sequence[ItemGrade],
    *,
    margin: float = NON_INFERIORITY_MARGIN,
    seed: int = 0,
) -> Verdict:
    """The verdict on one format against the bf16 build of the same fine-tune.

    Raises if the two runs graded different filings: `paired_delta_ci` refuses, and that
    refusal is the point of pairing.
    """
    if format.is_reference:
        raise ValueError("bf16 is the reference; it is not judged against itself")
    delta = paired_delta_ci(quantised, reference, seed=seed)
    fields = field_deltas(quantised, reference, seed=seed)
    ships = delta.low > -margin
    worst = min(fields, key=lambda f: f.delta.point)
    reason = (
        f"paired delta {delta.point:+.1%} ({delta.low:+.1%} to {delta.high:+.1%}); "
        + (
            f"the lower bound is inside the {margin:.0%} margin"
            if ships
            else f"the lower bound is below -{margin:.0%}, so it may have lost more than "
            "the whole spread between frontier tiers"
        )
        + f". Worst field: {worst.field} {worst.delta.point:+.1%}."
    )
    return Verdict(format=format, delta=delta, fields=fields, ships=ships, reason=reason)
