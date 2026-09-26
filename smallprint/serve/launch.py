"""The servers: the exact command line that serves each format, and the gateway entry for it.

vLLM serves the safetensors formats and llama.cpp serves GGUF, per PLAN.md section 2.6.
This module builds their command lines rather than starting them, so the part that decides
what is measured, which flags, which context length, which quantisation, is a pure
function with tests, and the part that needs a GPU is one `subprocess` call on the card.

The flags that matter to a measurement are fixed here and not left to defaults, because a
default is whatever the installed version says, and a throughput number is only
reproducible from its command line.

**Context length.** 10,240 tokens: the longest prompt in any test set plus the output cap
every run sends, which a server counts together and refuses a request over. The longest
post-cutoff prompt is 6,349 tokens on the Gemma tokeniser, measured on the pod on
2026-09-25 after the first test run lost that filing to a 400 at 8,192; OLMo's longest is
5,202. The cap is `baseline.DEFAULT_MAX_TOKENS`, 2,048, the same the frontier runs sent,
so the fine-tunes are asked on the same terms. A truncated or refused prompt is a wrong
answer the model did not choose. Longer than needed costs KV cache and so throughput, which
is the denominator of the cost per call, so it is not set higher than the data needs either.

**Concurrency.** llama.cpp serves a fixed number of parallel slots and queues the rest, so
its slot count is set to the highest concurrency the load test runs. Fewer slots would
measure the queue rather than the model.

**The chat format.** The fine-tunes were trained on one chat template, the instruction
model's, saved beside the weights. vLLM reads it from the tokenizer files. llama-server
renders the template embedded in the GGUF only when given `--jinja`; without it, it falls
back to a built-in format it guesses from the template, and a model served in a format it
was not trained on is measured on a task it was not trained for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from smallprint.quant.quality import Format

#: The provider name every served model is registered under in the gateway configuration,
#: so a self-hosted run's model is `selfhosted/<served name>` in the ledger and the tables.
PROVIDER: Final = "selfhosted"

#: Tokens of context each server is started with. See the module docstring.
MAX_MODEL_LEN: Final = 10_240

#: The longest prompt in any test set, in tokens on the tokeniser that makes it longest,
#: measured 2026-09-25. See the module docstring.
LONGEST_PROMPT: Final = 6349

#: The concurrency levels the load test runs, per PLAN.md section 2.6.
CONCURRENCY: Final[tuple[int, ...]] = (1, 8, 32, 64)

#: Sizes whose embedded chat template llama.cpp cannot run, and the built-in template it
#: uses instead. OLMo 3's is refused by llama.cpp's template engine ("Unable to generate
#: parser for this template", 2026-09-26). Its built-in ChatML renders a system and a user
#: turn and the generation prompt to exactly the string the trainer's template does,
#: checked through the server's /apply-template against transformers on the pod, and the
#: server adds no start token to it; the model's end-of-text token, which ends its answers,
#: is the GGUF's end-of-sequence. The Gemma template runs as it is.
BUILTIN_TEMPLATES: Final[dict[str, str]] = {"7b": "chatml"}

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
        # The checkpoint's generation_config.json, which the merge writes: the stop tokens
        # of the chat template the model was trained on, and no sampling defaults
        # (smallprint/train/merge.py). With vLLM's own defaults the only stop is
        # end-of-text, which a chat fine-tune does not emit, and every answer ran to the
        # token limit (2026-09-24). Every accuracy run sends its temperature explicitly.
        "--generation-config",
        "auto",
    ]
    if fmt is Format.BF16:
        argv += ["--dtype", "bfloat16"]
    else:
        # AWQ and GPTQ are both written by llm-compressor (smallprint/quant/formats.py),
        # whose on-disk format is compressed-tensors whichever algorithm chose the weights.
        # Named rather than detected, and not "awq" or "gptq": those select the kernels for
        # the AutoAWQ and AutoGPTQ layouts, which these files are not.
        argv += ["--quantization", "compressed-tensors"]
    return argv


def llamacpp_argv(
    model: Path | str,
    served_name: str,
    fmt: Format,
    *,
    port: int = 8000,
    max_model_len: int = MAX_MODEL_LEN,
    parallel: int = max(CONCURRENCY),
    kv_tokens: int | None = None,
    seed: int = 0,
) -> list[str]:
    """`llama-server` for a GGUF file, every layer on the GPU.

    One key-value cache shared by every slot (`--kv-unified`), `kv_tokens` long, rather than
    a slice of it reserved for each: that is how vLLM pages its cache, and the only way a
    7B serves 64 slots on a 24 GB card at all. Reserving the full context for every slot
    is 655,360 tokens, which the Gemma 2B's small cache holds and the OLMo 7B's, about
    384 KB a token, cannot: its first GGUF server asked for 35 GB of cache and stopped
    (2026-09-26). Unset, the pool is the full context for every slot, as before. A request
    still cannot exceed `max_model_len`, which the gateway's output cap and the longest
    prompt are sized against; concurrency is then bounded by the pool, as it is in vLLM.
    """
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
        "--ctx-size",
        str(kv_tokens if kv_tokens is not None else max_model_len * parallel),
        "--kv-unified",
        "--kv-unified-per-slot",
        str(max_model_len),
        "--parallel",
        str(parallel),
        "--n-gpu-layers",
        "999",
        "--cont-batching",
        # The template embedded in the GGUF, rendered as the trainer rendered it, or where
        # llama.cpp cannot run it, the built-in one that renders the same string. See the
        # module docstring and BUILTIN_TEMPLATES.
        *template_flags(served_name),
        "--seed",
        str(seed),
    ]


def template_flags(served_name: str) -> list[str]:
    """How llama-server renders the chat: the GGUF's own template, or a checked built-in."""
    builtin = BUILTIN_TEMPLATES.get(served_name.split("-", 1)[0])
    return ["--chat-template", builtin] if builtin else ["--jinja"]


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


def served_name(run: str, fmt: Format) -> str:
    """What a fine-tune in one format is called on the server, in the ledger and the tables:
    the training run's name and the format, so a row can be traced to both."""
    if not run or "/" in run:
        raise ValueError(f"not a run name: {run!r}")
    return f"{run}-{fmt.value}"


def parse_served_name(name: str) -> tuple[str, str, Format]:
    """(size, run, format) from a served name, with or without the provider in front.

    The size is the run name's first part, as `TrainConfig.run_id` and `scripts/sweep.sh`
    both write it. A name that does not end in a format is refused, rather than put in a
    table with a guessed one.
    """
    bare = name.split("/", 1)[1] if name.startswith(f"{PROVIDER}/") else name
    for fmt in sorted(Format, key=lambda f: -len(f.value)):
        suffix = f"-{fmt.value}"
        if bare.endswith(suffix) and len(bare) > len(suffix):
            run = bare[: -len(suffix)]
            return run.split("-", 1)[0], run, fmt
    raise ValueError(f"{name!r} does not end in a format ({', '.join(f.value for f in Format)})")


def served_config(
    base: dict[str, Any], base_url: str, *, self_hosted_prices: str | None = None
) -> dict[str, Any]:
    """The project's gateway configuration with the served model added as a provider.

    Everything else is the project's own file, retries apart: the same caps, the same ledger,
    the same vendor prices, so a self-hosted call lands in the ledger every frontier call is
    in. The price overlay is named only once a price file exists, because the gateway
    refuses a `self_hosted_prices` directory with nothing in it; until then the calls are
    written uncosted, which is what an accuracy run before the load test is.
    """
    providers = dict(base.get("providers", {}))
    if PROVIDER in providers:
        raise ValueError(f"the base configuration already has a {PROVIDER!r} provider")
    providers[PROVIDER] = provider_entry(base_url)
    # Retries off. Accuracy runs are in pass-through mode, which never retries anyway; the
    # load test streams, which the gateway allows only in standard mode, and a retried
    # request's latency includes its retry, so a load test with retries on is measuring the
    # retry policy. One setting for both, so the load test cannot be run with the wrong one.
    config = {**base, "providers": providers, "retry": {**base.get("retry", {}), "max_attempts": 1}}
    if self_hosted_prices is not None:
        config["self_hosted_prices"] = self_hosted_prices
    return config
