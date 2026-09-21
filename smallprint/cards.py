"""Model cards, written from the records of what was run.

Every published adapter and weight file gets a card in the same commit as the upload, and
the card is generated rather than typed: from the training `RunRecord`, the evaluation
summary on the post-cutoff test set, and the quantisation verdicts. A typed card can claim
something the run did not do; a generated one can only say what the records say.

A card with only a headline number is not a card. `render` refuses to produce one without
the per-field table and the named failure modes, because the failure modes are the part a
user of an extractor needs: how often it reads the prior-year column, drops the reporting
scale, flips a loss into a profit, invents a figure the filing does not report, or fails to
produce JSON at all. Those are different faults with different fixes, and one accuracy
number averages them into something no one can act on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from smallprint.baseline import BaselineSummary
from smallprint.grade import Interval, MissReason
from smallprint.quant.quality import Verdict
from smallprint.train.qlora import RunRecord
from smallprint.train.recipe import Base

#: The failure modes a card must report, in the words a reader of the card will use.
FAILURE_MODES: Final[tuple[tuple[MissReason, str], ...]] = (
    (MissReason.HALLUCINATED, "Invents a figure the filing does not report"),
    (MissReason.WRONG_PERIOD, "Reads the prior-period column"),
    (MissReason.SCALE, "Drops or misapplies the reporting scale"),
    (MissReason.SIGN, "Reads a loss as a profit, or the reverse"),
    (MissReason.MISSING, "Leaves out a figure the filing does report"),
    (MissReason.WRONG_VALUE, "Takes a figure from the wrong line"),
    (MissReason.MALFORMED, "Answer is not valid JSON of the schema"),
)

LIMITATIONS: Final = """- Trained and measured on US SEC 10-K and 10-Q filings under US GAAP. Nothing here says
  how it does on IFRS statements, on other regulators' filings, or on documents that are
  not financial statements.
- Banks and savings institutions are excluded from the corpus, and funds, trusts and
  pre-revenue companies are underrepresented. A score here says nothing about their
  statements. See the dataset's datasheet.
- The input is the located statements, not a whole filing. Given a whole filing, it has
  to find them first, which this model was not measured on.
- Every label is an XBRL fact the company filed, checked to be printed on the page. The
  label error rate from the hand audit is the ceiling on how far any of these numbers can
  be trusted."""


def _pct(interval: Interval) -> str:
    return f"{interval.point:.1%} ({interval.low:.1%} to {interval.high:.1%})"


def _signed(interval: Interval) -> str:
    return f"{interval.point:+.1%} ({interval.low:+.1%} to {interval.high:+.1%})"


def failure_table(summary: BaselineSummary) -> str:
    """Each named failure mode, as a count and as a rate per thousand fields."""
    counts: dict[str, int] = {}
    for report in summary.fields:
        for reason, count in report.reasons.items():
            counts[reason] = counts.get(reason, 0) + count
    fields_graded = summary.graded * len(summary.fields)
    lines = ["| Failure | Count | Per 1,000 fields |", "|---|---:|---:|"]
    for reason, words in FAILURE_MODES:
        count = counts.get(reason.value, 0)
        lines.append(f"| {words} | {count:,} | {1000 * count / fields_graded:.1f} |")
    return "\n".join(lines)


def field_table(summary: BaselineSummary) -> str:
    lines = ["| Field | Accuracy (95% CI) | Misses by reason |", "|---|---|---|"]
    for report in summary.fields:
        reasons = ", ".join(f"{k} {v}" for k, v in report.reasons.items()) or "none"
        lines.append(f"| `{report.field}` | {_pct(report.accuracy)} | {reasons} |")
    return "\n".join(lines)


def quant_table(verdicts: Sequence[Verdict]) -> str:
    if not verdicts:
        return "No quantised formats were published for this model."
    lines = [
        "| Format | Paired delta vs bf16 (95% CI) | Worst field | Published |",
        "|---|---|---|---|",
    ]
    for v in verdicts:
        worst = v.worst_field
        lines.append(
            f"| {v.format.value} | {_signed(v.delta)} | `{worst.field}` "
            f"{worst.delta.point:+.1%} | {'yes' if v.ships else 'no'} |"
        )
    return "\n".join(lines)


def render(
    *,
    name: str,
    base: Base,
    record: RunRecord,
    evaluation: BaselineSummary,
    evaluation_prompt: str,
    anchor: BaselineSummary,
    verdicts: Sequence[Verdict] = (),
    pre_cutoff: BaselineSummary | None = None,
    dataset_url: str = "",
) -> str:
    """The card, as Markdown with Hugging Face front matter.

    `evaluation_prompt` is the prompt fingerprint from the evaluation run's manifest. It
    must match the one the training file was written with.
    """
    if not evaluation.fields:
        raise ValueError("a card needs the per-field table; a headline number is not a card")
    if evaluation.split != "test_post_cutoff":
        raise ValueError(
            f"the headline is the post-cutoff test set, not {evaluation.split}; the "
            "pre-cutoff set is reported as the contamination gap and never as the headline"
        )
    if evaluation_prompt != record.dataset.prompt_fingerprint:
        raise ValueError(
            "trained with prompt "
            f"{record.dataset.prompt_fingerprint[:16]} and measured with {evaluation_prompt[:16]}: "
            "the result would be a comparison of two prompts, not a measure of the model"
        )
    cost = (
        f"US${evaluation.usd_per_1000.point:,.2f}"
        if evaluation.usd_per_1000 is not None
        else "not costed"
    )
    anchor_cost = (
        f"US${anchor.usd_per_1000.point:,.2f}" if anchor.usd_per_1000 is not None else "not costed"
    )
    gap = (
        f"On filings published before the base model's training cutoff it scores "
        f"{_pct(pre_cutoff.accuracy)}, against {_pct(evaluation.accuracy)} after it. That "
        "difference is the contamination gap, reported and never used as the headline."
        if pre_cutoff is not None
        else "The contamination gap was not measured for this model."
    )
    config = record.config
    return f"""---
license: apache-2.0
base_model: {base.repo}
tags:
- structured-extraction
- financial-statements
- sec-filings
- lora
---

# {name}

Extracts fifteen figures from the primary financial statements of a US SEC 10-K or 10-Q,
as JSON. A LoRA fine-tune of [`{base.repo}`](https://huggingface.co/{base.repo}) at revision
`{base.revision}`, trained on filings with their XBRL facts as the labels. No model graded
another model anywhere in producing these numbers.

## Results, on filings published after the base model's training cutoff

| | This model | {anchor.model}, {anchor.style.value} |
|---|---|---|
| Fields correct | {_pct(evaluation.accuracy)} | {_pct(anchor.accuracy)} |
| Every field right on a filing | {_pct(evaluation.exact_match)} | {_pct(anchor.exact_match)} |
| Answers that are not valid JSON | {_pct(evaluation.unparseable)} | {_pct(anchor.unparseable)} |
| Cost per 1,000 extractions | {cost} | {anchor_cost} |

{evaluation.graded:,} filings. {gap}

## How it fails

{failure_table(evaluation)}

## Per field

{field_table(evaluation)}

## Quantised formats

Each format is judged against this model's bf16 weights over the same filings, and is
published only if the lower end of its paired interval is above -1.0% of field accuracy.

{quant_table(verdicts)}

## Training

| | |
|---|---|
| Base | `{base.repo}` at `{base.revision}` |
| Method | QLoRA, rank {config.rank}, alpha {config.alpha}, dropout {config.dropout} |
| Learning rate | {config.learning_rate:g}, cosine, warm-up {config.warmup_ratio:.0%} |
| Examples | {record.dataset.train:,} training filings, {record.dataset.validation:,} validation |
| Epochs | {config.epochs} |
| Seed | {config.seed} |
| Prompt fingerprint | `{record.dataset.prompt_fingerprint[:16]}` |
| Steps | {record.steps:,} |
| Validation loss | {record.validation_loss if record.validation_loss is not None else "not recorded"} |
| Libraries | {", ".join(f"{k} {v}" for k, v in sorted(record.versions.items()))} |

{f"Dataset: {dataset_url}" if dataset_url else ""}

## Limitations

{LIMITATIONS}
"""
