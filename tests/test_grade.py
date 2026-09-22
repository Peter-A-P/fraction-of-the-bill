"""Adversarial fixtures for the grader.

These are the readings a model actually gets wrong on a financial statement, and each one
has a name the grader has to produce. A grader that scored them all as simply "wrong" would
still give the same headline accuracy and would be useless for writing a model card, so the
assertions here are on the reason as much as on the verdict.
"""

from __future__ import annotations

import datetime as dt

import pytest

from smallprint.grade import (
    GradeContext,
    MissReason,
    accuracy_ci,
    field_report,
    grade_item,
    normalise_categorical,
    paired_delta_ci,
    parse_extraction,
)
from smallprint.schema import Extraction

TRUTH = Extraction(
    period_end=dt.date(2026, 12, 31),
    fiscal_period="FY",
    revenue=1_234_567_000,
    cost_of_revenue=800_000_000,
    operating_income=180_000_000,
    net_income=-45_200_000,
    eps_basic=-0.31,
    eps_diluted=-0.31,
    shares_diluted=145_800_000,
    total_assets=2_500_000_000,
    total_liabilities=1_400_000_000,
    cash_and_equivalents=310_000_000,
    stockholders_equity=1_100_000_000,
    auditor_name="Deloitte & Touche LLP",
    state_of_incorporation="DE",
)

#: The statement was printed in thousands, and the prior-year column sits beside every
#: current-year figure.
CTX = GradeContext(
    scale=1000.0,
    distractors={
        "revenue": (1_100_400_000.0,),
        "net_income": (22_700_000.0,),
        "total_assets": (2_310_000_000.0,),
        "period_end": ("2025-12-31",),
    },
)


def reason_for(prediction: Extraction, field: str, ctx: GradeContext = CTX) -> MissReason | None:
    grade = grade_item("test", prediction, TRUTH, ctx)
    outcome = next(o for o in grade.outcomes if o.field == field)
    return outcome.reason


def correct_for(prediction: Extraction, field: str, ctx: GradeContext = CTX) -> bool:
    grade = grade_item("test", prediction, TRUTH, ctx)
    return next(o for o in grade.outcomes if o.field == field).correct


def test_the_truth_grades_itself_perfectly() -> None:
    grade = grade_item("test", TRUTH, TRUTH, CTX)
    assert grade.exact
    assert grade.accuracy == 1.0
    assert grade.n_fields == 15


def test_thousands_read_as_units_is_a_scale_error() -> None:
    """The statement says "in thousands"; the model copied the printed digits."""
    prediction = TRUTH.model_copy(update={"revenue": 1_234_567.0})
    assert reason_for(prediction, "revenue") is MissReason.SCALE


def test_millions_read_as_thousands_is_a_scale_error() -> None:
    prediction = TRUTH.model_copy(update={"total_assets": 2_500_000.0})
    assert reason_for(prediction, "total_assets") is MissReason.SCALE


def test_a_loss_in_parentheses_read_as_a_profit_is_a_sign_error() -> None:
    """(45,200) on the page is negative. A model that drops the brackets inverts the year."""
    prediction = TRUTH.model_copy(update={"net_income": 45_200_000.0})
    assert reason_for(prediction, "net_income") is MissReason.SIGN


def test_a_negative_eps_read_as_positive_is_a_sign_error() -> None:
    prediction = TRUTH.model_copy(update={"eps_diluted": 0.31})
    assert reason_for(prediction, "eps_diluted") is MissReason.SIGN


def test_the_prior_year_column_is_named_not_lumped_in() -> None:
    prediction = TRUTH.model_copy(update={"revenue": 1_100_400_000.0})
    assert reason_for(prediction, "revenue") is MissReason.WRONG_PERIOD


def test_the_prior_period_end_date_is_named_too() -> None:
    prediction = TRUTH.model_copy(update={"period_end": dt.date(2025, 12, 31)})
    assert reason_for(prediction, "period_end") is MissReason.WRONG_PERIOD


def test_a_field_the_filing_does_not_report_and_the_model_invents() -> None:
    """Quarterly reports are reviewed, not audited, so there is no auditor to name."""
    truth = TRUTH.model_copy(update={"auditor_name": None, "fiscal_period": "Q2"})
    prediction = truth.model_copy(update={"auditor_name": "Ernst & Young LLP"})
    grade = grade_item("test", prediction, truth, CTX)
    outcome = next(o for o in grade.outcomes if o.field == "auditor_name")
    assert outcome.reason is MissReason.HALLUCINATED


def test_a_field_the_filing_reports_and_the_model_abstains() -> None:
    prediction = TRUTH.model_copy(update={"cash_and_equivalents": None})
    assert reason_for(prediction, "cash_and_equivalents") is MissReason.MISSING


def test_both_absent_is_correct() -> None:
    truth = TRUTH.model_copy(update={"operating_income": None})
    prediction = truth.model_copy(update={"operating_income": None})
    grade = grade_item("test", prediction, truth, CTX)
    assert next(o for o in grade.outcomes if o.field == "operating_income").correct


def test_output_that_is_not_json_scores_zero_with_every_field_named_malformed() -> None:
    grade = grade_item("test", "I could not find a balance sheet in this document.", TRUTH)
    assert grade.n_correct == 0
    assert all(o.reason is MissReason.MALFORMED for o in grade.outcomes)


def test_fenced_json_parses_because_models_wrap_their_answers() -> None:
    body = '```json\n{"revenue": 1234567000, "state_of_incorporation": "DE"}\n```'
    parsed = parse_extraction(body)
    assert parsed is not None
    assert parsed.revenue == 1_234_567_000
    assert parsed.net_income is None


def test_an_unknown_field_is_malformed_rather_than_quietly_dropped() -> None:
    """Coercing here would repair the failure the schema exists to measure."""
    assert parse_extraction('{"revenue": 1, "gross_margin_pct": 42}') is None


def test_a_string_where_a_number_belongs_is_malformed() -> None:
    assert parse_extraction('{"revenue": "1,234,567"}') is None


def test_rounding_at_the_reporting_scale_is_not_an_error() -> None:
    """A statement in millions cannot express the fact more precisely than it prints it."""
    truth = TRUTH.model_copy(update={"revenue": 1_234_600_000.0})
    ctx = GradeContext(scale=1_000_000.0)
    prediction = truth.model_copy(update={"revenue": 1_235_000_000.0})
    grade = grade_item("test", prediction, truth, ctx)
    assert next(o for o in grade.outcomes if o.field == "revenue").correct


def test_a_small_value_under_the_reporting_scale_is_forgiven_to_half_a_unit() -> None:
    """Printed in thousands, a balance of 300 dollars appears on the page as a zero."""
    truth = TRUTH.model_copy(update={"cash_and_equivalents": 300.0})
    prediction = truth.model_copy(update={"cash_and_equivalents": 0.0})
    grade = grade_item("test", prediction, truth, GradeContext(scale=1000.0))
    assert next(o for o in grade.outcomes if o.field == "cash_and_equivalents").correct


def test_an_unrelated_line_item_is_a_plain_wrong_value() -> None:
    """Not a scale error, not the prior year: a different number off the same page."""
    prediction = TRUTH.model_copy(update={"total_liabilities": 913_400_000.0})
    assert reason_for(prediction, "total_liabilities") is MissReason.WRONG_VALUE


def test_eps_is_compared_to_the_cent_not_to_a_percentage_of_itself() -> None:
    """Half a cent out is rounding; a cent and a half out is a different printed number."""
    assert correct_for(TRUTH.model_copy(update={"eps_basic": -0.3055}), "eps_basic")
    assert not correct_for(TRUTH.model_copy(update={"eps_basic": -0.32}), "eps_basic")


def test_share_counts_use_their_own_scale_when_it_differs_from_the_dollars() -> None:
    """Dollars in thousands, shares in millions, on the same page. It happens constantly."""
    ctx = GradeContext(scale=1000.0, share_scale=1_000_000.0)
    prediction = TRUTH.model_copy(update={"shares_diluted": 146_000_000.0})
    assert correct_for(prediction, "shares_diluted", ctx)


def test_a_column_within_the_tolerance_but_nearer_another_period_is_that_period() -> None:
    """729,341 thousand diluted shares for the quarter, 729,950 for the nine months: 0.08%
    apart, well inside half a percent. Reading the nine-month column is still a misread."""
    ctx = GradeContext(scale=1000.0, distractors={"shares_diluted": (729_950_000.0,)})
    truth = TRUTH.model_copy(update={"shares_diluted": 729_341_000.0})
    misread = truth.model_copy(update={"shares_diluted": 729_950_000.0})
    grade = grade_item("test", misread, truth, ctx)
    outcome = next(o for o in grade.outcomes if o.field == "shares_diluted")
    assert outcome.reason is MissReason.WRONG_PERIOD
    assert grade_item("test", truth, truth, ctx).exact


def test_a_distractor_printed_as_the_same_figure_cannot_count_against_a_reading() -> None:
    """Both columns print 146 in millions. The page cannot tell them apart, so neither can
    the grader, and the reading nearer the distractor is still right."""
    ctx = GradeContext(
        scale=1000.0, share_scale=1_000_000.0, distractors={"shares_diluted": (146_100_000.0,)}
    )
    prediction = TRUTH.model_copy(update={"shares_diluted": 146_000_000.0})
    assert correct_for(prediction, "shares_diluted", ctx)


def test_a_restated_prior_period_that_matches_nothing_current_is_still_wrong() -> None:
    """Restatements move the comparative column. They never make the current column wrong."""
    prediction = TRUTH.model_copy(update={"total_assets": 2_310_000_000.0})
    assert reason_for(prediction, "total_assets") is MissReason.WRONG_PERIOD


def test_auditor_names_survive_punctuation_and_the_legal_suffix() -> None:
    assert normalise_categorical("DELOITTE AND TOUCHE, L.L.P.") == normalise_categorical(
        "Deloitte & Touche LLP"
    )
    assert correct_for(
        TRUTH.model_copy(update={"auditor_name": "deloitte & touche"}), "auditor_name"
    )


def test_a_different_audit_firm_is_not_normalised_into_agreement() -> None:
    assert not correct_for(
        TRUTH.model_copy(update={"auditor_name": "PricewaterhouseCoopers LLP"}), "auditor_name"
    )


def test_state_codes_ignore_case_only() -> None:
    assert correct_for(
        TRUTH.model_copy(update={"state_of_incorporation": "de"}), "state_of_incorporation"
    )
    assert not correct_for(
        TRUTH.model_copy(update={"state_of_incorporation": "NY"}), "state_of_incorporation"
    )


def test_a_zero_truth_is_graded_without_dividing_by_it() -> None:
    truth = TRUTH.model_copy(update={"operating_income": 0.0})
    grade = grade_item("test", truth.model_copy(update={"operating_income": 5_000_000.0}), truth)
    assert next(o for o in grade.outcomes if o.field == "operating_income").reason is (
        MissReason.WRONG_VALUE
    )


def test_intervals_resample_over_filings_so_n_is_the_number_of_filings() -> None:
    grades = [grade_item(f"item-{i}", TRUTH, TRUTH, CTX) for i in range(40)]
    overall = accuracy_ci(grades, resamples=200)
    assert overall.n == 40
    assert overall.point == 1.0


def test_a_field_interval_counts_fields_of_that_name() -> None:
    grades = [grade_item(f"item-{i}", TRUTH, TRUTH, CTX) for i in range(40)]
    assert accuracy_ci(grades, "revenue", resamples=200).n == 40


def test_a_paired_delta_refuses_runs_that_did_not_see_the_same_filings() -> None:
    """Identical items is the whole basis of the comparison, so it is enforced, not trusted."""
    a = [grade_item("item-1", TRUTH, TRUTH, CTX)]
    b = [grade_item("item-2", TRUTH, TRUTH, CTX)]
    with pytest.raises(ValueError, match="identical items"):
        paired_delta_ci(a, b)


def test_a_paired_delta_is_zero_when_two_runs_agree() -> None:
    a = [grade_item(f"item-{i}", TRUTH, TRUTH, CTX) for i in range(30)]
    delta = paired_delta_ci(a, list(a), resamples=200)
    assert delta.point == 0.0
    assert delta.low == 0.0 and delta.high == 0.0


def test_a_paired_delta_recovers_a_known_one_field_regression() -> None:
    """One field of fifteen lost on every filing is exactly 1/15 of the accuracy."""
    good = [grade_item(f"item-{i}", TRUTH, TRUTH, CTX) for i in range(30)]
    worse = [
        grade_item(f"item-{i}", TRUTH.model_copy(update={"revenue": 1_234_567.0}), TRUTH, CTX)
        for i in range(30)
    ]
    delta = paired_delta_ci(good, worse, resamples=200)
    assert delta.point == pytest.approx(1 / 15)


def test_the_field_report_names_the_failure_modes_a_model_card_has_to_carry() -> None:
    grades = [
        grade_item(f"item-{i}", TRUTH.model_copy(update={"revenue": 1_234_567.0}), TRUTH, CTX)
        for i in range(10)
    ]
    report = {r.field: r for r in field_report(grades, resamples=200)}
    assert report["revenue"].accuracy.point == 0.0
    assert report["revenue"].reasons == {"scale": 10}
    assert report["net_income"].reasons == {}
