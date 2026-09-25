"""Quantisation gate, server command lines and the load client, without a GPU or a server."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from boundary import ChatRequest, ChatResponse, Mode, Usage
from items import TRUTH, make_item
from test_baseline import RIGHT, WRONG

from smallprint.bench import client
from smallprint.data.split import Split
from smallprint.grade import ItemGrade, grade_item
from smallprint.prompts import PromptStyle, build_prompt
from smallprint.quant.quality import NON_INFERIORITY_MARGIN, Format, judge
from smallprint.serve import launch

# -- the quantisation gate -------------------------------------------------------------


def grades(answers: list[str]) -> list[ItemGrade]:
    return [grade_item(f"f-{i}", text, TRUTH) for i, text in enumerate(answers)]


def test_a_format_that_lost_nothing_ships() -> None:
    reference = grades([RIGHT] * 40)
    verdict = judge(Format.AWQ, grades([RIGHT] * 40), reference)
    assert verdict.ships
    assert verdict.delta.point == 0.0


def test_a_format_that_lost_more_than_the_margin_does_not_ship() -> None:
    reference = grades([RIGHT] * 40)
    quantised = grades([WRONG] * 10 + [RIGHT] * 30)  # 2 of 15 wrong on a quarter of filings
    verdict = judge(Format.GGUF_Q4_K_M, quantised, reference)
    assert not verdict.ships
    assert verdict.delta.low < -NON_INFERIORITY_MARGIN
    assert "below" in verdict.reason


def test_the_verdict_names_the_field_that_broke() -> None:
    """WRONG breaks revenue and net income; the average across fifteen fields hides that."""
    verdict = judge(Format.GPTQ, grades([WRONG] * 10 + [RIGHT] * 30), grades([RIGHT] * 40))
    assert verdict.worst_field.field in {"revenue", "net_income"}
    assert verdict.worst_field.delta.point == pytest.approx(-0.25)


def test_a_quantised_run_over_different_filings_cannot_be_judged() -> None:
    with pytest.raises(ValueError, match="identical items"):
        judge(Format.AWQ, grades([RIGHT] * 5), grades([RIGHT] * 6))


def test_bf16_is_the_reference_and_is_not_judged_against_itself() -> None:
    with pytest.raises(ValueError, match="reference"):
        judge(Format.BF16, grades([RIGHT]), grades([RIGHT]))


# -- server command lines -----------------------------------------------------------------


def test_each_format_goes_to_the_server_the_plan_names() -> None:
    assert {f for f in Format if f.server == "vllm"} == {Format.BF16, Format.AWQ, Format.GPTQ}
    assert {f for f in Format if f.server == "llamacpp"} == {
        Format.GGUF_Q4_K_M,
        Format.GGUF_Q8_0,
    }


def test_vllm_is_told_the_quantisation_rather_than_left_to_guess() -> None:
    """llm-compressor writes AWQ and GPTQ alike as compressed-tensors; "awq" would pick the
    kernels for AutoAWQ's layout, which these files are not."""
    for fmt in (Format.AWQ, Format.GPTQ):
        argv = launch.vllm_argv("merged/4b-q", "smallprint-4b-q", fmt)
        assert argv[argv.index("--quantization") + 1] == "compressed-tensors"
    awq = launch.vllm_argv("merged/4b-awq", "smallprint-4b-awq", Format.AWQ)
    assert awq[awq.index("--max-model-len") + 1] == str(launch.MAX_MODEL_LEN)
    bf16 = launch.vllm_argv("merged/4b", "smallprint-4b", Format.BF16)
    assert "--quantization" not in bf16
    assert bf16[bf16.index("--dtype") + 1] == "bfloat16"


def test_llamacpp_gives_every_slot_the_full_context_and_a_slot_per_concurrent_request() -> None:
    argv = launch.llamacpp_argv(Path("quant/4b-q4.gguf"), "smallprint-4b-q4", Format.GGUF_Q4_K_M)
    assert argv[argv.index("--parallel") + 1] == "64"
    assert argv[argv.index("--ctx-size") + 1] == str(launch.MAX_MODEL_LEN * 64)


def test_the_longest_test_prompt_and_the_output_cap_fit_in_the_served_context() -> None:
    """The first post-cutoff run of a fine-tune, 2026-09-25, lost one filing to a 400: a
    6,349-token prompt and a 2,048-token cap are 8,397, and the server held 8,192."""
    from smallprint.baseline import DEFAULT_MAX_TOKENS

    assert launch.LONGEST_PROMPT + DEFAULT_MAX_TOKENS <= launch.MAX_MODEL_LEN


def test_a_format_sent_to_the_wrong_server_is_refused() -> None:
    with pytest.raises(ValueError, match="served by llama"):
        launch.vllm_argv("x.gguf", "x", Format.GGUF_Q8_0)
    with pytest.raises(ValueError, match="vLLM"):
        launch.llamacpp_argv("merged/4b", "x", Format.AWQ)
    assert launch.argv_for("x.gguf", "x", Format.GGUF_Q8_0)[0] == "llama-server"


def test_the_gateway_entry_marks_the_host_self_hosted() -> None:
    entry = launch.provider_entry("http://gpu.internal:8000/v1/")
    assert entry == {
        "kind": "openai_compat",
        "base_url": "http://gpu.internal:8000/v1",
        "self_hosted": True,
    }
    with pytest.raises(ValueError, match="not a URL"):
        launch.provider_entry("gpu.internal:8000")


# -- the load client --------------------------------------------------------------------


class SlowServer:
    """Answers every streamed call after a fixed delay, first token a fifth of the way in."""

    def __init__(self, seconds: float, *, retries: int = 0) -> None:
        self.seconds = seconds
        self.retries = retries
        self.modes: list[Mode] = []

    async def achat_stream(
        self,
        request: ChatRequest,
        *,
        purpose: str,
        run_id: str | None = None,
        mode: Mode = Mode.STANDARD,
    ) -> ChatResponse:
        self.modes.append(mode)
        await asyncio.sleep(self.seconds)
        return ChatResponse(
            text=RIGHT,
            finish_reason="stop",
            usage=Usage(input_tokens=3000, output_tokens=300),
            cost_usd=0.0001,
            costed=True,
            model_requested="selfhosted/smallprint-4b",
            model_returned="smallprint-4b",
            provider="selfhosted",
            latency_ms=self.seconds * 1000,
            status=200,
            headers={},
            raw=None,
            ledger_id=1,
            mode=mode,
            retries=self.retries,
            ttft_ms=self.seconds * 200,
        )


def some_requests() -> list[ChatRequest]:
    items = [make_item(f"t-{i}", split=Split.TEST_POST_CUTOFF) for i in range(3)]
    return client.requests_for(items, build_prompt(PromptStyle.ZERO_SHOT), "selfhosted/x")


def test_the_loop_is_closed_and_keeps_exactly_n_requests_in_flight() -> None:
    server = SlowServer(0.05)
    started = time.monotonic()
    result = client.run_level(
        server, some_requests(), concurrency=4, total=16, purpose="t", run_id="r"
    )
    elapsed = time.monotonic() - started
    assert result.max_in_flight == 4
    assert len(result.timings) == 16
    # Four rounds of 50 ms, not sixteen: serial would take 0.8 s.
    assert elapsed < 0.5
    assert set(server.modes) == {Mode.STANDARD}  # streaming is refused in pass-through


def test_the_timings_carry_time_to_first_token_and_summarise() -> None:
    result = client.run_level(
        SlowServer(0.02), some_requests(), concurrency=2, total=10, purpose="t", run_id="r"
    )
    assert result.summary is not None
    assert result.summary.ttft_p50_ms.point == pytest.approx(4.0, abs=1.0)
    assert result.summary.errors == 0
    assert result.retried == 0


def test_a_run_where_the_gateway_retried_says_so() -> None:
    result = client.run_level(
        SlowServer(0.01, retries=1),
        some_requests(),
        concurrency=2,
        total=4,
        purpose="t",
        run_id="r",
    )
    assert result.retried == 4


def test_the_requests_are_the_prompt_every_accuracy_run_sends() -> None:
    request = some_requests()[0]
    prompt = build_prompt(PromptStyle.ZERO_SHOT)
    assert request.system == prompt.system
    assert "t-0" in request.messages[-1]["content"]


def test_a_load_test_with_nothing_to_send_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        client.requests_for([], build_prompt(PromptStyle.ZERO_SHOT), "m")
