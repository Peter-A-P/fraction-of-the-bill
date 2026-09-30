"""Model cards: generated from records, and refused when the records cannot support them."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from items import training_pool
from test_baseline import RIGHT, WRONG, StubGateway, items, reply, zero_shot
from test_serving import grades

from smallprint import baseline, cards
from smallprint.quant.quality import Format, judge
from smallprint.train import dataset
from smallprint.train.qlora import RunRecord
from smallprint.train.recipe import BASES, TrainConfig


def evaluated(tmp_path: Path, answers: dict[int, str], *, cost: float) -> baseline.BaselineSummary:
    gateway = StubGateway(lambda i: reply(answers.get(i, RIGHT), cost=cost))
    manifest = baseline.run(
        gateway,
        model="selfhosted/smallprint-4b",
        items=items(),
        prompt=zero_shot(),
        out_dir=tmp_path,
        build_dir=Path("data/build/full"),
        run_id=f"run-{cost}",
    )
    return baseline.summarise(manifest, baseline.read_predictions(tmp_path), items())


def record() -> RunRecord:
    _, manifest = dataset.build(training_pool())
    config = TrainConfig(base=BASES["4b"].repo, size="4b")
    return RunRecord(
        run_id=config.run_id,
        config=config,
        dataset=manifest,
        base_revision=BASES["4b"].revision,
        started_at=dt.datetime(2026, 9, 22, tzinfo=dt.UTC),
        steps=646,
        versions={"python": "3.13.1", "peft": "0.x"},
    )


def card(
    tmp_path: Path,
    *,
    evaluation: baseline.BaselineSummary | None = None,
    evaluation_prompt: str | None = None,
) -> str:
    return cards.render(
        name="smallprint-4b",
        base=BASES["4b"],
        record=record(),
        evaluation=evaluation or evaluated(tmp_path / "model", {1: WRONG}, cost=0.0002),
        evaluation_prompt=evaluation_prompt or zero_shot().fingerprint,
        anchor=evaluated(tmp_path / "anchor", {}, cost=0.001),
        verdicts=[judge(Format.AWQ, grades([RIGHT] * 20), grades([RIGHT] * 20))],
    )


def test_the_card_names_every_failure_mode_not_just_a_headline(tmp_path: Path) -> None:
    text = card(tmp_path)
    for _, words in cards.FAILURE_MODES:
        assert words in text
    assert "| `revenue` |" in text  # the per-field table
    assert "base_model: google/gemma-4-E4B" in text
    assert BASES["4b"].revision in text
    assert "US$0.20" in text and "US$1.00" in text  # this model beside the cost anchor


def test_the_quantised_formats_carry_their_paired_delta_and_whether_they_shipped(
    tmp_path: Path,
) -> None:
    text = card(tmp_path)
    assert "| awq | +0.0% (+0.0% to +0.0%) |" in text
    assert "| yes |" in text


def test_a_card_is_refused_for_a_model_measured_with_a_different_prompt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two prompts"):
        card(tmp_path, evaluation_prompt="f" * 64)


def test_the_pre_cutoff_set_can_never_be_the_headline(tmp_path: Path) -> None:
    summary = evaluated(tmp_path / "pre", {}, cost=0.0002)
    pre = summary.model_copy(update={"split": "test_pre_cutoff"})
    with pytest.raises(ValueError, match="contamination gap"):
        card(tmp_path, evaluation=pre)


def test_a_headline_number_alone_is_not_a_card(tmp_path: Path) -> None:
    summary = evaluated(tmp_path / "bare", {}, cost=0.0002)
    with pytest.raises(ValueError, match="not a card"):
        card(tmp_path, evaluation=summary.model_copy(update={"fields": ()}))


def test_a_model_costed_by_its_load_test_says_so_on_the_card(tmp_path: Path) -> None:
    """The accuracy runs came before any price for the served model existed, so the ledger
    has them uncosted; the card carries the cost the load test measured, with its terms."""
    uncosted = evaluated(tmp_path / "model", {}, cost=0.0)
    uncosted = uncosted.model_copy(update={"usd_per_1000": None})
    text = cards.render(
        name="smallprint-2b",
        base=BASES["2b"],
        record=record(),
        evaluation=uncosted,
        evaluation_prompt=zero_shot().fingerprint,
        anchor=evaluated(tmp_path / "anchor", {}, cost=0.001),
        served_cost="US$0.083 served (A40 at US$0.49/h, 50% utilisation)",
    )
    assert "US$0.083 served (A40 at US$0.49/h, 50% utilisation)" in text
    assert "not costed" not in text.split("## How it fails")[0].split("| Cost per 1,000")[1]


def test_the_card_carries_the_gate_s_decisions_and_the_fields_that_block(tmp_path: Path) -> None:
    decision = {
        "record_id": "c9eb7d588c0cfcf8",
        "passed": False,
        "baseline": {"label": "openai/gpt-5.6-sol"},
        "suites": [
            {"suite": "revenue", "difference": 0.0, "lo": -0.004, "hi": 0.004, "verdict": "pass"},
            {
                "suite": "net_income",
                "difference": -0.027,
                "lo": -0.041,
                "hi": -0.013,
                "verdict": "block",
            },
        ],
    }
    text = cards.render(
        name="smallprint-4b",
        base=BASES["4b"],
        record=record(),
        evaluation=evaluated(tmp_path / "model", {}, cost=0.0002),
        evaluation_prompt=zero_shot().fingerprint,
        anchor=evaluated(tmp_path / "anchor", {}, cost=0.001),
        gate=[decision],
    )
    row = "| `openai/gpt-5.6-sol` | block | `net_income` -2.7 (-4.1 to -1.3) | `c9eb7d588c0cfcf8` |"
    assert row in text
    assert "not been run" in cards.gate_table([])
