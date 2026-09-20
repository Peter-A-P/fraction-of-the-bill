"""The baseline runner, against a stub gateway: what it records, what it refuses, and that
an interrupted run costs nothing to finish."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from boundary import ChatRequest, ChatResponse, Mode, Usage
from items import TRUTH, make_item

from smallprint import baseline, prompts
from smallprint.data.build import SplitItem
from smallprint.data.split import Split
from smallprint.grade import parse_extraction
from smallprint.schema import Extraction

RIGHT = prompts.answer(TRUTH)
WRONG = prompts.answer(TRUTH.model_copy(update={"revenue": 999.0, "net_income": None}))


def reply(
    text: str | None,
    *,
    cost: float | None = 0.002,
    status: int | str = 200,
    latency: float = 1200.0,
    ttft: float | None = 300.0,
    cached: bool = False,
) -> ChatResponse:
    return ChatResponse(
        text=text,
        finish_reason="stop" if text is not None else None,
        usage=Usage(input_tokens=1800, output_tokens=210),
        cost_usd=cost,
        costed=cost is not None,
        model_requested="anthropic/claude-haiku-4-5-20251001",
        model_returned="claude-haiku-4-5-20251001",
        provider="anthropic",
        latency_ms=latency,
        status=status,
        headers={},
        raw=None,
        ledger_id=1,
        mode=Mode.PASSTHROUGH,
        price_list="2026-09-01",
        price_sha256="f" * 64,
        ttft_ms=ttft,
        cached=cached,
    )


class StubGateway:
    """Answers in a fixed order, and keeps every request it was given."""

    def __init__(self, answers: Callable[[int], ChatResponse]) -> None:
        self.answers = answers
        self.requests: list[ChatRequest] = []

    def chat(
        self,
        request: ChatRequest,
        *,
        purpose: str,
        run_id: str | None = None,
        mode: Mode = Mode.STANDARD,
    ) -> ChatResponse:
        self.requests.append(request)
        self.purpose = purpose
        self.run_id = run_id
        self.mode = mode
        return self.answers(len(self.requests) - 1)


def items(n: int = 4) -> list[SplitItem]:
    return [make_item(f"test-{i}", split=Split.TEST_POST_CUTOFF) for i in range(n)]


def zero_shot() -> prompts.Prompt:
    return prompts.build_prompt(prompts.PromptStyle.ZERO_SHOT)


def go(
    gateway: StubGateway, out: Path, *, pool: list[SplitItem] | None = None
) -> baseline.RunManifest:
    return baseline.run(
        gateway,
        model="anthropic/claude-haiku-4-5-20251001",
        items=pool if pool is not None else items(),
        prompt=zero_shot(),
        out_dir=out,
        build_dir=Path("data/build/full"),
        run_id="run-1",
    )


def test_a_run_records_one_prediction_a_filing_and_the_manifest_says_how(tmp_path: Path) -> None:
    gateway = StubGateway(lambda i: reply(RIGHT))
    manifest = go(gateway, tmp_path)

    assert len(gateway.requests) == 4
    assert gateway.mode is Mode.PASSTHROUGH  # a measurement, so no retries and no cache
    assert gateway.requests[0].temperature == 0.0
    assert gateway.requests[0].system == zero_shot().system
    assert manifest.finished_at is not None
    assert manifest.prompt_fingerprint == zero_shot().fingerprint

    predictions = baseline.read_predictions(tmp_path)
    assert [p.item_id for p in predictions] == [f"test-{i}" for i in range(4)]
    assert predictions[0].ledger_id == 1  # the row that says what it was billed


def test_an_interrupted_run_is_finished_rather_than_paid_for_twice(tmp_path: Path) -> None:
    class Stop(RuntimeError):
        pass

    def two_then_fail(i: int) -> ChatResponse:
        if i >= 2:
            raise Stop("the vendor went away")
        return reply(RIGHT)

    with pytest.raises(Stop):
        go(StubGateway(two_then_fail), tmp_path)
    assert len(baseline.read_predictions(tmp_path)) == 2
    assert baseline.read_manifest(tmp_path).finished_at is None

    second = StubGateway(lambda i: reply(RIGHT))
    manifest = go(second, tmp_path)
    assert len(second.requests) == 2  # only the two that were missing
    assert len(baseline.read_predictions(tmp_path)) == 4
    assert manifest.finished_at is not None


def test_resuming_with_a_different_prompt_is_refused(tmp_path: Path) -> None:
    go(StubGateway(lambda i: reply(RIGHT)), tmp_path)
    with pytest.raises(ValueError, match="prompt_fingerprint"):
        baseline.run(
            StubGateway(lambda i: reply(RIGHT)),
            model="anthropic/claude-haiku-4-5-20251001",
            items=items(),
            prompt=prompts.build_prompt(prompts.PromptStyle.ZERO_SHOT, thinking=True),
            out_dir=tmp_path,
            build_dir=Path("data/build/full"),
            run_id="run-1",
        )


def test_one_run_covers_one_split(tmp_path: Path) -> None:
    mixed = [*items(2), make_item("train-1", split=Split.TRAIN)]
    with pytest.raises(ValueError, match="one run covers one split"):
        go(StubGateway(lambda i: reply(RIGHT)), tmp_path, pool=mixed)


def test_a_call_that_never_answered_is_reported_and_not_graded_as_a_wrong_answer(
    tmp_path: Path,
) -> None:
    def one_error(i: int) -> ChatResponse:
        return reply(None, status="ReadTimeout", cost=None) if i == 1 else reply(RIGHT)

    go(StubGateway(one_error), tmp_path)
    grades, failed = baseline.grade_run(baseline.read_predictions(tmp_path), items())
    assert failed == ["test-1"]
    assert len(grades) == 3
    assert all(g.exact for g in grades)


def test_the_summary_carries_the_accuracy_the_cost_and_what_would_not_parse(
    tmp_path: Path,
) -> None:
    def mixed(i: int) -> ChatResponse:
        return {
            0: reply(RIGHT),
            1: reply(WRONG),
            2: reply("I cannot help with that"),
            3: reply(RIGHT),
        }[i]

    manifest = go(StubGateway(mixed), tmp_path)
    summary = baseline.summarise(manifest, baseline.read_predictions(tmp_path), items())

    assert summary.graded == 4
    assert summary.exact_match.point == 0.5
    assert summary.unparseable.point == 0.25
    # 13 of 15 right on the wrong one, 0 of 15 on the unparseable one.
    assert summary.accuracy.point == pytest.approx((15 + 13 + 0 + 15) / 60)
    assert summary.accuracy.low < summary.accuracy.point < summary.accuracy.high
    assert summary.usd_per_1000 is not None
    assert summary.usd_per_1000.point == pytest.approx(2.0)  # US$0.002 a call
    assert summary.uncosted_calls == 0
    assert summary.latency_p50_ms.point == 1200.0
    assert summary.ttft_p50_ms is not None
    assert summary.input_tokens == 4 * 1800


def test_a_run_answered_entirely_from_the_cache_says_so(tmp_path: Path) -> None:
    manifest = go(StubGateway(lambda i: reply(RIGHT, cached=True)), tmp_path)
    summary = baseline.summarise(manifest, baseline.read_predictions(tmp_path), items())
    assert summary.cached_calls == 4
    assert "development cache" in str(summary)


def test_summarising_a_run_where_every_call_failed_is_an_error_not_a_zero(
    tmp_path: Path,
) -> None:
    manifest = go(StubGateway(lambda i: reply(None, status=503, cost=None)), tmp_path)
    with pytest.raises(ValueError, match="nothing to grade"):
        baseline.summarise(manifest, baseline.read_predictions(tmp_path), items())


def test_a_prediction_for_an_item_that_is_not_in_the_split_is_an_error(tmp_path: Path) -> None:
    go(StubGateway(lambda i: reply(RIGHT)), tmp_path)
    with pytest.raises(ValueError, match="not in these items"):
        baseline.grade_run(baseline.read_predictions(tmp_path), items(2))


def test_an_empty_extraction_still_round_trips_through_the_answer_format() -> None:
    assert parse_extraction(prompts.answer(Extraction())) == Extraction()
