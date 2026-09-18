"""Field-level grading against XBRL truth, and the intervals every reported number carries.

Grading is a program. No model is asked whether another model was right, here or anywhere
else in this project, because a judge that drifts makes every number downstream of it
unfalsifiable.

A wrong field is not just wrong, it is wrong in one of a small number of ways, and which
way matters more than the accuracy figure does. A model that reads the prior-year column is
broken differently from one that drops three zeros, and the two are fixed differently. So
every miss is classified, the classification is reported per field, and the model cards are
written from that table rather than from a single number.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from smallprint.schema import (
    FIELDS,
    PER_SHARE_TOLERANCE,
    RELATIVE_TOLERANCE,
    Extraction,
    FieldKind,
    FieldSpec,
)


class MissReason(StrEnum):
    """Why a field was wrong. Ordered from most diagnostic to least."""

    #: The model returned nothing for a field the filing does report.
    MISSING = "missing"
    #: The model returned a value for a field the filing does not report. The expensive
    #: failure in production: an abstention can be routed to a human, an invention cannot.
    HALLUCINATED = "hallucinated"
    #: The value belongs to a different period or a different column of the same statement,
    #: usually the prior-year comparative sitting next to the one that was asked for.
    WRONG_PERIOD = "wrong_period"
    #: Right magnitude, wrong sign. A loss printed in parentheses, read as a profit.
    SIGN = "sign"
    #: Off by a power of ten, nearly always the reporting scale: the statement says "in
    #: thousands" and the model copied the printed digits.
    SCALE = "scale"
    #: The output was not JSON, or the field did not hold a value of the declared type.
    MALFORMED = "malformed"
    #: Wrong, and not in any of the ways above.
    WRONG_VALUE = "wrong_value"


#: Ratios treated as a scale error rather than as an unrelated wrong number. Powers of a
#: thousand are the reporting scales a statement actually uses; ten and a hundred are here
#: because a decimal point in the wrong place produces them and it is worth telling that
#: apart from a number read off the wrong line.
_SCALE_RATIOS: Final[tuple[float, ...]] = (1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e6, 1e9)

#: How close a ratio has to be to one of the above to count as that scale error.
_SCALE_RATIO_TOLERANCE: Final = 0.01

#: The straight apostrophe and the curly one, named by code point so that the source stays
#: ASCII and the two cannot be told apart by eye in a diff. Company names on EDGAR carry
#: both, sometimes in the same filing.
_APOSTROPHES: Final = chr(0x27) + chr(0x2019)

#: Periods and apostrophes are deleted rather than turned into spaces, so that `L.L.P.`
#: folds to one token and matches `LLP` instead of becoming three single letters.
_ELIDED = re.compile(f"[.{_APOSTROPHES}]")
_PUNCTUATION = re.compile(r"[^\w&]+")
_LEGAL_SUFFIXES: Final[frozenset[str]] = frozenset(
    {"llp", "llc", "lp", "pc", "pllc", "inc", "ltd", "plc", "sc", "ag", "gmbh"}
)


class FieldOutcome(BaseModel):
    """One field of one item, graded."""

    model_config = ConfigDict(frozen=True)

    field: str
    correct: bool
    reason: MissReason | None = None
    predicted: str | None = None
    truth: str | None = None


class ItemGrade(BaseModel):
    """One filing, graded across every field of the schema."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    outcomes: tuple[FieldOutcome, ...]

    @property
    def n_correct(self) -> int:
        return sum(1 for o in self.outcomes if o.correct)

    @property
    def n_fields(self) -> int:
        return len(self.outcomes)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_fields if self.n_fields else 0.0

    @property
    def exact(self) -> bool:
        """Every field right. The number a user of the extractor actually feels."""
        return self.n_correct == self.n_fields


class GradeContext(BaseModel):
    """What the grader needs about an item beyond its truth.

    `scale` and `share_scale` are the reporting units the statement was printed in, taken
    from the pairing step rather than guessed here, and they set the floor on how precise a
    correct reading can be: a statement in thousands cannot distinguish two values that are
    four hundred apart, so neither can the grader.

    `distractors` are the other values printed on the same statement for the same field,
    the prior-year column above all. They are never treated as correct. They exist so that a
    miss can be named `wrong_period` instead of disappearing into `wrong_value`.
    """

    model_config = ConfigDict(frozen=True)

    scale: float = 1.0
    share_scale: float | None = None
    distractors: Mapping[str, tuple[float | str, ...]] = {}


def tolerance(spec: FieldSpec, truth: float, ctx: GradeContext) -> float:
    """How far a numeric reading may be from its truth and still be correct.

    Public because the locatability filter in `pair.py` uses the same rule: a filing is kept
    only if a reading the grader would accept is actually printed on the page. Two rules
    would let a filing in whose label no correct reading of the page could reach.
    """
    match spec.kind:
        case FieldKind.PER_SHARE:
            return PER_SHARE_TOLERANCE
        case FieldKind.SHARE_COUNT:
            scale = ctx.share_scale if ctx.share_scale is not None else ctx.scale
            return max(RELATIVE_TOLERANCE * abs(truth), scale / 2)
        case _:
            return max(RELATIVE_TOLERANCE * abs(truth), ctx.scale / 2)


def _numbers_agree(predicted: float, truth: float, tolerance: float) -> bool:
    return abs(predicted - truth) <= tolerance


def normalise_categorical(value: str) -> str:
    """Fold a categorical to the form two correct readings of the same page share.

    Case, punctuation and the legal suffix on an audit firm are printing conventions, not
    facts. `Deloitte & Touche LLP` and `DELOITTE AND TOUCHE, L.L.P.` are one answer.
    """
    text = value.strip().lower().replace(" and ", " & ")
    text = _ELIDED.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    words = [w for w in text.split() if w]
    while words and words[-1] in _LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def _classify_number(predicted: float, truth: float, tolerance: float) -> MissReason:
    """Name a numeric miss. Only called once the value is known to be wrong."""
    if truth != 0.0:
        if _numbers_agree(predicted, -truth, tolerance):
            return MissReason.SIGN
        ratio = predicted / truth
        for candidate in _SCALE_RATIOS:
            if abs(ratio - candidate) <= _SCALE_RATIO_TOLERANCE * candidate:
                return MissReason.SCALE
    return MissReason.WRONG_VALUE


def _matches_distractor(
    spec: FieldSpec, predicted: object, ctx: GradeContext, tolerance: float
) -> bool:
    for other in ctx.distractors.get(spec.name, ()):
        match predicted, other:
            case (float() | int(), float()):
                if _numbers_agree(float(predicted), other, tolerance):
                    return True
            case (dt.date(), str()):
                if predicted.isoformat() == other:
                    return True
            case (str(), str()):
                if normalise_categorical(predicted) == normalise_categorical(other):
                    return True
    return False


def _render(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


def _grade_field(
    spec: FieldSpec, predicted: object, truth: object, ctx: GradeContext
) -> FieldOutcome:
    rendered = FieldOutcome(
        field=spec.name, correct=False, predicted=_render(predicted), truth=_render(truth)
    )

    if truth is None and predicted is None:
        return rendered.model_copy(update={"correct": True})
    if predicted is None:
        return rendered.model_copy(update={"reason": MissReason.MISSING})
    if truth is None:
        return rendered.model_copy(update={"reason": MissReason.HALLUCINATED})

    match spec.kind:
        case FieldKind.DATE:
            assert isinstance(truth, dt.date)
            if isinstance(predicted, dt.date) and predicted == truth:
                return rendered.model_copy(update={"correct": True})
            if _matches_distractor(spec, predicted, ctx, 0.0):
                return rendered.model_copy(update={"reason": MissReason.WRONG_PERIOD})
            return rendered.model_copy(update={"reason": MissReason.WRONG_VALUE})

        case FieldKind.CATEGORICAL:
            assert isinstance(truth, str)
            if isinstance(predicted, str) and normalise_categorical(
                predicted
            ) == normalise_categorical(truth):
                return rendered.model_copy(update={"correct": True})
            if _matches_distractor(spec, predicted, ctx, 0.0):
                return rendered.model_copy(update={"reason": MissReason.WRONG_PERIOD})
            return rendered.model_copy(update={"reason": MissReason.WRONG_VALUE})

        case _:
            assert isinstance(truth, float | int)
            if not isinstance(predicted, float | int) or isinstance(predicted, bool):
                return rendered.model_copy(update={"reason": MissReason.MALFORMED})
            truth_value = float(truth)
            predicted_value = float(predicted)
            if not math.isfinite(predicted_value):
                return rendered.model_copy(update={"reason": MissReason.MALFORMED})
            allowed = tolerance(spec, truth_value, ctx)
            if _numbers_agree(predicted_value, truth_value, allowed):
                return rendered.model_copy(update={"correct": True})
            if _matches_distractor(spec, predicted_value, ctx, allowed):
                return rendered.model_copy(update={"reason": MissReason.WRONG_PERIOD})
            reason = _classify_number(predicted_value, truth_value, allowed)
            return rendered.model_copy(update={"reason": reason})


def grade_item(
    item_id: str,
    predicted: Extraction | str | None,
    truth: Extraction,
    ctx: GradeContext | None = None,
) -> ItemGrade:
    """Grade one filing.

    `predicted` takes the model's raw text as well as a parsed `Extraction`, because how
    often a model fails to produce parseable JSON at all is part of the result rather than
    an error to be retried away. A raw string that will not parse scores zero with every
    field marked `malformed`.
    """
    context = ctx if ctx is not None else GradeContext()

    parsed: Extraction | None
    if isinstance(predicted, Extraction):
        parsed = predicted
    elif predicted is None:
        parsed = None
    else:
        parsed = parse_extraction(predicted)

    if parsed is None:
        return ItemGrade(
            item_id=item_id,
            outcomes=tuple(
                FieldOutcome(
                    field=spec.name,
                    correct=False,
                    reason=MissReason.MALFORMED,
                    predicted=None,
                    truth=_render(getattr(truth, spec.name)),
                )
                for spec in FIELDS.values()
            ),
        )

    return ItemGrade(
        item_id=item_id,
        outcomes=tuple(
            _grade_field(spec, getattr(parsed, spec.name), getattr(truth, spec.name), context)
            for spec in FIELDS.values()
        ),
    )


def parse_extraction(text: str) -> Extraction | None:
    """Parse a model's answer, tolerating the wrapping models put around JSON.

    Tolerant about fences and prose, strict about the object itself: an unknown field or a
    value of the wrong type is a malformed answer, not something to coerce. Coercion here
    would silently repair exactly the failure mode the schema exists to measure.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return Extraction.model_validate(payload)
    except ValidationError:
        return None


class Interval(BaseModel):
    """A point estimate and its 95% bootstrap interval. Every reported number is one."""

    model_config = ConfigDict(frozen=True)

    point: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.1%} ({self.low:.1%} to {self.high:.1%}, n = {self.n:,})"


def _bootstrap(samples: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = samples.shape[0]
    if n == 0:
        return (float("nan"), float("nan"))
    draws = rng.integers(0, n, size=(resamples, n))
    means = samples[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(low), float(high))


def accuracy_ci(
    grades: Sequence[ItemGrade],
    field: str | None = None,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Accuracy with a 95% interval, resampled over filings rather than over fields.

    The unit of resampling is the filing, not the field, because the fifteen fields of one
    filing are not independent: a badly rendered statement gets several of them wrong at
    once. Resampling fields would treat fifteen correlated draws as fifteen independent
    ones and report an interval roughly a third too narrow.
    """
    if field is None:
        per_item = np.array([g.accuracy for g in grades], dtype=float)
        n = len(grades)
    else:
        values = [float(o.correct) for g in grades for o in g.outcomes if o.field == field]
        per_item = np.array(values, dtype=float)
        n = len(values)
    if per_item.size == 0:
        return Interval(point=float("nan"), low=float("nan"), high=float("nan"), n=0)
    low, high = _bootstrap(per_item, resamples, seed)
    return Interval(point=float(per_item.mean()), low=low, high=high, n=n)


def paired_delta_ci(
    a: Sequence[ItemGrade],
    b: Sequence[ItemGrade],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """The paired accuracy difference `a - b` with a 95% interval.

    Paired because every comparison in this project runs the two systems over the same
    filings: bf16 against its own AWQ build, a fine-tune against the frontier API. An
    unpaired interval on the same data is wider than the evidence warrants, and the
    quantisation deltas are small enough that the difference decides whether a format ships.

    Raises if the two runs did not grade the same items, which is the mistake this function
    exists to make impossible.
    """
    by_id_a = {g.item_id: g for g in a}
    by_id_b = {g.item_id: g for g in b}
    if by_id_a.keys() != by_id_b.keys():
        only_a = sorted(by_id_a.keys() - by_id_b.keys())[:3]
        only_b = sorted(by_id_b.keys() - by_id_a.keys())[:3]
        raise ValueError(
            "paired comparison needs identical items: "
            f"{len(by_id_a)} vs {len(by_id_b)}, first differences {only_a} / {only_b}"
        )
    deltas = np.array(
        [by_id_a[k].accuracy - by_id_b[k].accuracy for k in sorted(by_id_a)], dtype=float
    )
    low, high = _bootstrap(deltas, resamples, seed)
    return Interval(point=float(deltas.mean()), low=low, high=high, n=deltas.size)


class FieldReport(BaseModel):
    """Per-field accuracy and the named reasons behind its misses."""

    model_config = ConfigDict(frozen=True)

    field: str
    accuracy: Interval
    reasons: Mapping[str, int]


def field_report(
    grades: Sequence[ItemGrade], *, resamples: int = 10_000, seed: int = 0
) -> tuple[FieldReport, ...]:
    """The table the model cards are written from: every field, its accuracy, its failures."""
    reports = []
    for name in FIELDS:
        reasons: dict[str, int] = {}
        for grade in grades:
            for outcome in grade.outcomes:
                if outcome.field == name and outcome.reason is not None:
                    reasons[outcome.reason.value] = reasons.get(outcome.reason.value, 0) + 1
        reports.append(
            FieldReport(
                field=name,
                accuracy=accuracy_ci(grades, name, resamples=resamples, seed=seed),
                reasons=dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            )
        )
    return tuple(reports)
