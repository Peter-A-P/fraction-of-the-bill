"""The servers: the exact command line that serves each format, and the gateway entry for it.

vLLM serves the safetensors formats and llama.cpp serves GGUF, per PLAN.md section 2.6.
This module builds their command lines rather than starting them, so the part that decides
what is measured, which flags, which context length, which quantisation, is a pure
function with tests, and the part that needs a GPU is one `subprocess` call on the card.

The flags that matter to a measurement are fixed here and not left to defaults, because a
default is whatever the installed version says, and a throughput number is only
reproducible from its command line.

**Context length.** 8,192 tokens. The longest prompts in the post-cutoff set are about 4,300
tokens on the most generous tokeniser measured, with 350 or so out, so 4,096 would cut the
tail and a truncated prompt is a wrong answer the model did not choose. Longer than needed
costs KV cache and so throughput, which is the denominator of the cost per call, so it is
not set higher than the data needs either.

**Concurrency.** llama.cpp serves a fixed number of parallel slots and queues the rest, so
its slot count is set to the highest concurrency the load test runs. Fewer slots would
measure the queue rather than the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from smallprint.quant.quality import Format

#: Tokens of context each server is started with. See the module docstring.
MAX_MODEL_LEN: Final = 8192

#: The concurrency levels the load test runs, per PLAN.md section 2.6.
CONCURRENCY: Final[tuple[int, ...]] = (1, 8, 32, 64)

#: Share of the card's memory vLLM may claim. The rest is the headroom that keeps an
#: out-of-memory error from landing in the middle of a timed run.
GPU_MEMORY_UTILISATION: Final = 0.90


def vllm_argv(
    model: Path | str,
    served_name: str,
    fmt: Format,
    *,
    port: int = 8000,
    max_model_len: int = MAX_MODEL_LEN,
    seed: int = 0,
) -> list[str]:
    """`vllm serve` for bf16, AWQ or GPTQ weights."""
    if fmt.server != "vllm":
        raise ValueError(f"{fmt.value} is served by llama.cpp, not vLLM")
    argv = [
        "vllm",
        "serve",
        str(model),
        "--served-model-name",
        served_name,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        f"{GPU_MEMORY_UTILISATION}",
        "--seed",
        str(seed),
        # The load test sends the same system prompt on every call; caching it is what a
        # real deployment of this extractor would do, so it is on for the measurement.
        "--enable-prefix-caching",
    ]
    if fmt is Format.BF16:
        argv += ["--dtype", "bfloat16"]
    else:
        argv += ["--quantization", fmt.value]
    return argv


def llamacpp_argv(
    model: Path | str,
    served_name: str,
    fmt: Format,
    *,
    port: int = 8000,
    max_model_len: int = MAX_MODEL_LEN,
    parallel: int = max(CONCURRENCY),
    seed: int = 0,
) -> list[str]:
    """`llama-server` for a GGUF file, every layer on the GPU."""
    if fmt.server != "llamacpp":
        raise ValueError(f"{fmt.value} is served by vLLM, not llama.cpp")
    return [
        "llama-server",
        "--model",
        str(model),
        "--alias",
        served_name,
        "--port",
        str(port),
        # llama.cpp divides the context between its slots, so each slot gets the full
        # length only if the total is the per-request length times the slot count.
        "--ctx-size",
        str(max_model_len * parallel),
        "--parallel",
        str(parallel),
        "--n-gpu-layers",
        "999",
        "--cont-batching",
        "--seed",
        str(seed),
    ]


def argv_for(model: Path | str, served_name: str, fmt: Format, **options: Any) -> list[str]:
    """The command line for whichever server serves this format."""
    builder = vllm_argv if fmt.server == "vllm" else llamacpp_argv
    return builder(model, served_name, fmt, **options)


def provider_entry(base_url: str) -> dict[str, Any]:
    """The gateway provider entry for a served model.

    `self_hosted` is what makes the gateway price it from the overlay this project writes
    rather than from any vendor list, and what makes it refuse a vendor rate for it.
    Both servers speak the OpenAI chat completions API, and both accept `max_tokens`.
    """
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(f"not a URL: {base_url!r}")
    return {
        "kind": "openai_compat",
        "base_url": base_url.rstrip("/"),
        "self_hosted": True,
    }
