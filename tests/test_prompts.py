"""The prompt: that it states the whole task, that its examples are answers the grader
accepts, and that no example ever comes from an item the model is measured on."""

from __future__ import annotations

import pytest
from items import TRUTH, make_item, training_pool

from smallprint import prompts
from smallprint.data.split import Split
from smallprint.grade import grade_item, parse_extraction
from smallprint.schema import SCHEMA


def test_the_prompt_names_every_field_and_says_what_to_do_with_the_scale() -> None:
    system = prompts.system_prompt()
    for spec in SCHEMA:
        assert system.count(f"  {spec.name} (") == 1, spec.name
    assert "1234000" in system  # the scale rule is worked, not asserted
    assert "null" in system


def test_the_example_answer_is_one_the_grader_marks_right() -> None:
    """An example that would score less than full marks teaches the model to be wrong."""
    text = prompts.answer(TRUTH)
    parsed = parse_extraction(text)
    assert parsed is not None
    grade = grade_item("example", parsed, TRUTH)
    assert grade.exact, [o for o in grade.outcomes if not o.correct]
    assert '"revenue": 1200000' in text  # whole units, written as a filing prints them


def test_thinking_is_off_unless_asked_for() -> None:
    assert not prompts.system_prompt().startswith(prompts.THINK_TOKEN)
    assert prompts.system_prompt(thinking=True).startswith(prompts.THINK_TOKEN)


def test_a_few_shot_prompt_shows_both_forms_and_is_the_same_every_time() -> None:
    pool = training_pool()
    first = prompts.build_prompt(prompts.PromptStyle.FEW_SHOT, pool=pool, k=2)
    second = prompts.build_prompt(prompts.PromptStyle.FEW_SHOT, pool=list(reversed(pool)), k=2)
    assert first.example_ids == second.example_ids
    forms = {s.item.form for s in prompts.examples_from(pool, 2)}
    assert forms == {"10-K", "10-Q"}


def test_examples_come_from_the_shorter_half_so_the_prompt_stays_affordable() -> None:
    pool = training_pool()
    longest = max(len(s.item.text) for s in pool)
    assert all(len(s.item.text) < longest for s in prompts.examples_from(pool, 2))


def test_an_example_may_never_come_from_a_pool_holding_a_test_filing() -> None:
    pool = [*training_pool(4), make_item("test-1", split=Split.TEST_POST_CUTOFF)]
    with pytest.raises(ValueError, match="training pool"):
        prompts.examples_from(pool, 2)


def test_asking_for_more_examples_than_exist_is_an_error_not_a_short_prompt() -> None:
    with pytest.raises(ValueError, match="asked for 4"):
        prompts.examples_from(training_pool(2), 4)


def test_the_messages_end_with_the_filing_and_hold_the_examples_as_finished_turns() -> None:
    prompt = prompts.build_prompt(prompts.PromptStyle.FEW_SHOT, pool=training_pool(), k=2)
    messages = prompt.messages("THE FILING UNDER TEST")
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert "THE FILING UNDER TEST" in messages[-1]["content"]
    assert all(parse_extraction(m["content"]) is not None for m in messages[1::2])


def test_the_fingerprint_changes_when_anything_the_model_sees_changes() -> None:
    zero = prompts.build_prompt(prompts.PromptStyle.ZERO_SHOT)
    few = prompts.build_prompt(prompts.PromptStyle.FEW_SHOT, pool=training_pool(), k=2)
    other = prompts.build_prompt(prompts.PromptStyle.FEW_SHOT, pool=training_pool(), k=1)
    thinking = prompts.build_prompt(prompts.PromptStyle.ZERO_SHOT, thinking=True)
    assert len({zero.fingerprint, few.fingerprint, other.fingerprint, thinking.fingerprint}) == 4
    assert zero.fingerprint == prompts.build_prompt(prompts.PromptStyle.ZERO_SHOT).fingerprint
