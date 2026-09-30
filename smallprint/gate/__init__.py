"""What the release gate (project 03) is handed: an eval spec and both sides' outcomes.

The plan says the non-inferiority of each fine-tune against the best frontier model is run
through the 03 gate, not computed here, so that the statistics, the power screen and Holm's
adjustment are one implementation used across the portfolio rather than a second copy of
them. The gate's reuse plan (its PLAN.md, B9) is that a project provides an eval spec and an
item loader and the gate does the rest. This module is this project's half of that.

**One suite per field.** The gate's paired test works on one right-or-wrong outcome per item,
and a filing here is fifteen outcomes. Averaging them into a share per filing would need a
test the gate does not have; treating the fifteen as fifteen items of one suite would treat
correlated fields as independent, the mistake `grade.accuracy_ci` exists to avoid. A suite
per field keeps the unit the filing, gives the gate fifteen tests to adjust across, which is
what Holm is for, and names the field that blocks, which is what a reader of a verdict
wants to know.

**The margin is the gate's default, three points, per field.** Stated here so the
spec carries it. A tighter one could not be decided on 705 filings: a field near 97%
accuracy has a paired standard error of about 0.75 points, so a one-point margin would put
every suite under the power screen and the gate would warn rather than decide.

**Runnable since 2026-09-30.** The gate's adapter (its `gate/adapter.py` and
`docs/gate-adapter.md`) reads these files as written: a spec whose suites are source kind
`outcomes_file`, and one side file per side in the shape of its `Side` and `SuiteOutcomes`.
It refuses a file it cannot read exactly (an unknown key, a grade that is not a boolean, a
missing suite) rather than guessing, and names each file in its record by the hash of its
bytes. The decision goes to this project's own ledger:

    uv run gate compare --spec spec.yaml --baseline baseline.json \
        --candidate candidate.json --ledger <this repository>/gate-ledger.jsonl
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from smallprint.baseline import Prediction, grade_run, within
from smallprint.bench.load import percentile
from smallprint.data.build import SplitItem
from smallprint.schema import FIELDS

#: The non-inferiority margin, in accuracy points, per field. See the module docstring.
DELTA_POINTS: Final = 3.0

SPEC_NAME: Final = "fraction-of-the-bill"


def suite_key(field: str) -> str:
    """The gate's suite keys are lower case with dashes or underscores; field names already are."""
    return field.lower()


def side(
    label: str,
    predictions: Sequence[Prediction],
    items: Sequence[SplitItem],
    *,
    source: dict[str, str],
) -> dict[str, Any]:
    """One side of the comparison, as the gate's `Side`: per field, a verdict per filing.

    A call that failed is not an answer, so its filing is counted as ungradeable in every
    suite rather than marked wrong, which is the gate's own rule for its drift record.
    """
    kept, _ = within(predictions, items)
    grades, failed = grade_run(kept, items)
    costed = [p.cost_usd for p in kept if p.costed and p.cost_usd is not None]
    latency = percentile([p.latency_ms for p in kept if p.ok], 50) if grades else 0.0
    suites = {}
    for name in FIELDS:
        suites[suite_key(name)] = {
            "suite": suite_key(name),
            "outcomes": {
                g.item_id: next(o.correct for o in g.outcomes if o.field == name) for g in grades
            },
            "ungradeable_items": len(failed),
            "calls": len(kept),
            "latency_p50_ms": latency,
            "cost_usd": sum(costed),
            "uncosted_calls": sum(1 for p in kept if p.ok and not p.costed),
        }
    return {"label": label, "source": dict(source), "suites": suites}


def spec(*, delta_points: float = DELTA_POINTS) -> dict[str, Any]:
    """The eval spec, in the gate's `EvalSpec` shape, one suite per field.

    The source kind is the gate's `outcomes_file`: outcomes graded here, read from a side
    file, and `block` names the suite in that file. The gate's spec model forbids unknown
    kinds, so a misspelt kind is refused loudly rather than read as something it is not.
    """
    return {
        "version": 1,
        "name": SPEC_NAME,
        "delta_points": delta_points,
        "alpha": 0.05,
        "resamples": 2000,
        "seed": 0,
        "suites": [
            {"key": suite_key(name), "source": {"kind": "outcomes_file", "block": suite_key(name)}}
            for name in FIELDS
        ],
    }
