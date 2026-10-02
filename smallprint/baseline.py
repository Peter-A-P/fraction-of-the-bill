"""Running a model over held-out filings, and what that run measured.

This is how every accuracy number in the project is produced, frontier API and self-hosted
alike. One runner, one prompt, one grader, one ledger: the point of the comparison is that
the only difference between two rows of the results table is the model.

Three things this module insists on, each because the alternative produces a number that
looks fine and is not:

**Calls go through the gateway.** The runner takes something with the gateway's `chat`,
never an HTTP client of its own, so every call is priced and written to one ledger and the
cost axis of the frontier chart is read rather than estimated.

**Measurements run in pass-through mode by default.** No retries, no cache, no filled-in
defaults, and the request and response bytes kept. A retry turns an error rate into a
latency, and a cache hit turns a measurement into a memory of one. Standard mode is
available for development and the mode is recorded in the manifest, so a number measured
with the cache on cannot be published as if it were not.

**A run resumes.** Predictions are appended as they arrive and the manifest is written
before the first call, so an interrupted run over two thousand filings continues rather
than starting again and paying again. A filing that was answered is never called twice; a
filing whose last attempt failed is, because the usual reason a run fails is one the next
run has fixed. Resuming into a directory whose manifest describes a different model,
prompt or mode is refused: that is two runs, not one, and averaging them would be the
quiet kind of wrong.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Protocol

from boundary import ChatRequest, ChatResponse, Gateway, Mode
from pydantic import BaseModel, ConfigDict

from smallprint.bench.load import Measured, mean_measured, percentile_measured
from smallprint.data.build import SplitItem
from smallprint.data.datasheet import read_items as read_build
from smallprint.grade import (
    FieldReport,
    Interval,
    ItemGrade,
    accuracy_ci,
    field_report,
    grade_item,
    mean_ci,
    parse_extraction,
)
from smallprint.prompts import Prompt, PromptStyle
from smallprint.recost import Recost

#: The answer is fifteen fields of JSON, about 200 tokens. The ceiling is generous because
#: a truncated answer grades as malformed and would be indistinguishable from a model that
#: cannot produce JSON, and because a reasoning model spends its thinking inside it.
DEFAULT_MAX_TOKENS: Final = 2048

#: No temperature at all, by default: the field is left out of the request and the vendor's
#: own default applies. This is not a preference, it is what the current models accept.
#: Every one of the three frontier models in this project's set refuses a value. OpenAI's
#: answer "Unsupported value: 'temperature' does not support 0.0 with this model. Only the
#: default (1) value is supported."; Anthropic's claude-sonnet-5 answers "`temperature` is
#: deprecated for this model." A measurement would rather pin it: a rerun would then differ
#: by the vendor's nondeterminism alone and not by ours as well. That is no longer on offer
#: from the frontier, so the runs say so and the manifest records what was sent.
#:
#: A served open-weights model does accept it, and the quantisation deltas are small enough
#: that sampling noise would swamp them, so those runs pass --temperature 0 explicitly.
TEMPERATURE: Final[float | None] = None

PREDICTIONS: Final = "predictions.jsonl"
MANIFEST: Final = "run.json"

#: What the gateway writes on every row this project causes, and the key its spend cap is
#: filed under in the gateway's caps file. It has to match, or the cap guards nothing.
PROJECT: Final = "fraction-of-the-bill"


def open_gateway(config: Path, *, raw_store: Path) -> Gateway:
    """The gateway, from `boundary.yaml` and the gitignored `.env` beside it.

    `from_config` rather than the constructor, because it is the entry point that reads
    the `.env` file: keys are never in the configuration, and a real environment variable
    always wins over the file. `raw_store` is where pass-through keeps the request and
    response bytes; it belongs with the run, so that a published number and the bytes
    behind it are in one directory.
    """
    return Gateway.from_config(config, project=PROJECT, raw_store=raw_store)


class Caller(Protocol):
    """What the runner needs from the gateway.

    A protocol rather than `Gateway` itself so the runner can be tested against a stub
    that returns known responses without a network or a ledger. `Gateway` satisfies it.
    """

    def chat(
        self,
        request: ChatRequest,
        *,
        purpose: str,
        run_id: str | None = ...,
        mode: Mode = ...,
    ) -> ChatResponse: ...


class Prediction(BaseModel):
    """One filing, one call, and everything about it that any published number uses."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    #: The answer as the model wrote it. None when no response arrived at all.
    text: str | None
    finish_reason: str | None = None
    status: int | str
    ok: bool
    #: The gateway's row id, so any number here can be traced to what was billed.
    ledger_id: int
    cost_usd: float | None = None
    costed: bool = False
    price_list: str | None = None
    price_sha256: str | None = None
    latency_ms: float
    ttft_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    retries: int = 0
    model_returned: str | None = None


class RunManifest(BaseModel):
    """What was run, on what, and how. Written before the first call."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    model: str
    split: str
    style: PromptStyle
    thinking: bool
    prompt_fingerprint: str
    example_ids: tuple[str, ...] = ()
    mode: Mode
    max_tokens: int
    temperature: float | None
    build_dir: str
    requested: int
    started_at: dt.datetime
    finished_at: dt.datetime | None = None

    def incompatible_with(self, other: RunManifest) -> list[str]:
        """The fields that make two runs different runs. Empty means one may resume the other."""
        keys = (
            "model",
            "split",
            "style",
            "thinking",
            "prompt_fingerprint",
            "mode",
            "max_tokens",
            "temperature",
        )
        return [k for k in keys if getattr(self, k) != getattr(other, k)]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def read_predictions(out_dir: Path) -> list[Prediction]:
    """Every filing's latest prediction, in the order the filings were first attempted.

    Latest rather than every row, because a resumed run calls again for the filings whose
    last attempt failed, and the file is append-only so both attempts are in it. The
    earlier row stays on disk as the record that the attempt happened; what is graded is
    what the model last answered.
    """
    path = out_dir / PREDICTIONS
    if not path.exists():
        return []
    latest: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                prediction = Prediction.model_validate_json(line)
                latest[prediction.item_id] = prediction
    return list(latest.values())


def read_manifest(out_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((out_dir / MANIFEST).read_text(encoding="utf-8"))


def _write_manifest(out_dir: Path, manifest: RunManifest) -> None:
    (out_dir / MANIFEST).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )


def _prediction(item_id: str, response: ChatResponse) -> Prediction:
    return Prediction(
        item_id=item_id,
        text=response.text,
        finish_reason=response.finish_reason,
        status=response.status,
        ok=response.ok,
        ledger_id=response.ledger_id,
        cost_usd=response.cost_usd,
        costed=response.costed,
        price_list=response.price_list,
        price_sha256=response.price_sha256,
        latency_ms=response.latency_ms,
        ttft_ms=response.ttft_ms,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cached=response.cached,
        retries=response.retries,
        model_returned=response.model_returned,
    )


def run(
    caller: Caller,
    *,
    model: str,
    items: Sequence[SplitItem],
    prompt: Prompt,
    out_dir: Path,
    build_dir: Path,
    run_id: str,
    mode: Mode = Mode.PASSTHROUGH,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = TEMPERATURE,
    on_result: Callable[[int, int, Prediction], None] | None = None,
) -> RunManifest:
    """Run `model` over `items`, appending a prediction per filing. Resumable.

    `items` must be one split: mixing the post-cutoff test set with anything else would
    produce a directory whose headline number is an average over two different claims.
    """
    if not items:
        raise ValueError("no items to run")
    splits = {s.split for s in items}
    if len(splits) != 1:
        raise ValueError(
            f"one run covers one split; got {', '.join(sorted(s.value for s in splits))}"
        )
    split = next(iter(splits))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        run_id=run_id,
        model=model,
        split=split.value,
        style=prompt.style,
        thinking=prompt.thinking,
        prompt_fingerprint=prompt.fingerprint,
        example_ids=prompt.example_ids,
        mode=mode,
        max_tokens=max_tokens,
        temperature=temperature,
        build_dir=str(build_dir),
        requested=len(items),
        started_at=_now(),
    )
    if (out_dir / MANIFEST).exists():
        existing = read_manifest(out_dir)
        differences = manifest.incompatible_with(existing)
        if differences:
            raise ValueError(
                f"{out_dir} holds a different run; {', '.join(differences)} differ. "
                "Run it elsewhere rather than mixing two runs in one directory"
            )
        manifest = existing.model_copy(update={"requested": len(items), "finished_at": None})
    _write_manifest(out_dir, manifest)

    # Only an answered filing is done. A filing whose last attempt failed is called again,
    # because the usual reason a whole run fails is one the next run has fixed: a rejected
    # parameter, an expired key, a vendor having an afternoon. Errors are not billed, so
    # this cannot spend twice on the same filing in any case that matters.
    done = {p.item_id for p in read_predictions(out_dir) if p.ok}
    remaining = [s for s in items if s.item.item_id not in done]
    purpose = f"baseline:{prompt.style.value}"

    with (out_dir / PREDICTIONS).open("a", encoding="utf-8", newline="\n") as handle:
        for n, s in enumerate(remaining, start=1):
            request = ChatRequest(
                model=model,
                messages=prompt.messages(s.item.text),
                system=prompt.system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = caller.chat(request, purpose=purpose, run_id=run_id, mode=mode)
            prediction = _prediction(s.item.item_id, response)
            handle.write(prediction.model_dump_json() + "\n")
            handle.flush()
            if on_result is not None:
                on_result(n, len(remaining), prediction)

    manifest = manifest.model_copy(update={"finished_at": _now()})
    _write_manifest(out_dir, manifest)
    return manifest


def grade_run(
    predictions: Sequence[Prediction], items: Sequence[SplitItem]
) -> tuple[list[ItemGrade], list[str]]:
    """Grade the answers that arrived, and name the calls that did not arrive.

    A call that failed at the transport or the vendor is not an answer the model gave, so
    it is not graded as a wrong one; it is returned separately and reported beside the
    accuracy, because a model measured on the nine tenths of items it managed to answer is
    not measured on the task.
    """
    truth = {s.item.item_id: s.item for s in items}
    grades: list[ItemGrade] = []
    failed: list[str] = []
    for prediction in predictions:
        item = truth.get(prediction.item_id)
        if item is None:
            raise ValueError(f"prediction for {prediction.item_id}, which is not in these items")
        if not prediction.ok or prediction.text is None:
            failed.append(prediction.item_id)
            continue
        grades.append(grade_item(item.item_id, prediction.text, item.truth, item.context))
    return grades, failed


def within(
    predictions: Sequence[Prediction], items: Sequence[SplitItem]
) -> tuple[list[Prediction], int]:
    """The predictions for items this build contains, and how many were set aside.

    A rebuild drops an item a run has already answered when the filter learns its label
    was not on its page, as the hand audit taught it to. The answer stays in the run's
    files and is not graded, because there is no label left to grade it against; the count
    is reported beside the result. A run none of whose answers are in the build was pointed
    at the wrong build, and is refused rather than summarised as empty.
    """
    wanted = {s.item.item_id for s in items}
    kept = [p for p in predictions if p.item_id in wanted]
    if predictions and not kept:
        raise ValueError("none of these predictions are for items in this build")
    return kept, len(predictions) - len(kept)


class BaselineSummary(BaseModel):
    """One model on one split: what it got right, what it cost, how long it took."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    model: str
    split: str
    style: PromptStyle
    thinking: bool
    mode: Mode
    graded: int
    failed: int
    #: Answers to items a later build dropped, kept on disk and not graded.
    set_aside: int = 0
    cached_calls: int
    accuracy: Interval
    exact_match: Interval
    unparseable: Interval
    fields: tuple[FieldReport, ...]
    usd_per_1000: Measured | None
    uncosted_calls: int
    price_list: str | None
    #: Set when the cost is not the ledger's as written but recomputed from the run's raw
    #: bytes (smallprint/recost.py); says how. None means the recorded cost.
    cost_basis: str | None = None
    input_tokens: int
    output_tokens: int
    latency_p50_ms: Measured
    latency_p99_ms: Measured
    ttft_p50_ms: Measured | None

    def __str__(self) -> str:
        cost = "not costed" if self.usd_per_1000 is None else f"US${self.usd_per_1000}"
        lines = [
            f"{self.model} on {self.split}, {self.style.value}"
            + (", thinking" if self.thinking else "")
            + f", {self.mode.value} mode, run {self.run_id}",
            f"  fields correct   {self.accuracy}",
            f"  every field      {self.exact_match}",
            f"  unparseable      {self.unparseable}",
            f"  cost per 1,000   {cost}"
            + (f" ({self.uncosted_calls} calls uncosted)" if self.uncosted_calls else "")
            + (f", {self.cost_basis}" if self.cost_basis else ""),
            f"  latency p50      {self.latency_p50_ms}",
            f"  latency p99      {self.latency_p99_ms}",
        ]
        if self.ttft_p50_ms is not None:
            lines.append(f"  first token p50  {self.ttft_p50_ms}")
        if self.failed:
            lines.append(f"  calls that failed and are not graded: {self.failed}")
        if self.set_aside:
            lines.append(f"  answers to items the build has since dropped: {self.set_aside}")
        if self.cached_calls:
            lines.append(f"  answers served from the development cache: {self.cached_calls}")
        return "\n".join(lines)


def summarise(
    manifest: RunManifest,
    predictions: Sequence[Prediction],
    items: Sequence[SplitItem],
    *,
    seed: int = 0,
    recosted: Recost | None = None,
) -> BaselineSummary:
    """Everything one run says, with an interval on every proportion and every time.

    With `recosted`, a call's cost is the one recomputed from its raw bytes where there is
    one, and the summary says so; the predictions on disk are not changed.
    """
    if not predictions:
        raise ValueError("no predictions to summarise")
    predictions, set_aside = within(predictions, items)
    grades, failed = grade_run(predictions, items)
    if not grades:
        raise ValueError(f"every one of {len(predictions)} calls failed; nothing to grade")

    answered = [p for p in predictions if p.ok and p.text is not None]
    unparseable = [
        0.0 if p.text is not None and parse_extraction(p.text) is not None else 1.0
        for p in answered
    ]
    costed = [
        recosted.costs.get(p.ledger_id, p.cost_usd) if recosted is not None else p.cost_usd
        for p in predictions
        if p.costed and p.cost_usd is not None
    ]
    ttft = [p.ttft_ms for p in predictions if p.ttft_ms is not None]
    prices = {p.price_list for p in predictions if p.price_list is not None}

    return BaselineSummary(
        run_id=manifest.run_id,
        model=manifest.model,
        split=manifest.split,
        style=manifest.style,
        thinking=manifest.thinking,
        mode=manifest.mode,
        graded=len(grades),
        failed=len(failed),
        set_aside=set_aside,
        cached_calls=sum(1 for p in predictions if p.cached),
        accuracy=accuracy_ci(grades, seed=seed),
        exact_match=mean_ci([float(g.exact) for g in grades], seed=seed),
        unparseable=mean_ci(unparseable, seed=seed),
        fields=field_report(grades, seed=seed),
        usd_per_1000=(
            mean_measured([c * 1000 for c in costed], unit="per 1,000", seed=seed)
            if costed
            else None
        ),
        uncosted_calls=sum(1 for p in predictions if not p.costed),
        price_list=(
            recosted.price_list if recosted is not None else ", ".join(sorted(prices)) or None
        ),
        cost_basis=recosted.basis if recosted is not None else None,
        input_tokens=sum(p.input_tokens for p in predictions),
        output_tokens=sum(p.output_tokens for p in predictions),
        latency_p50_ms=percentile_measured([p.latency_ms for p in predictions], 50, seed=seed),
        latency_p99_ms=percentile_measured([p.latency_ms for p in predictions], 99, seed=seed),
        ttft_p50_ms=percentile_measured(ttft, 50, seed=seed) if ttft else None,
    )


def write_summary(out_dir: Path, summary: BaselineSummary) -> Path:
    path = out_dir / "summary.json"
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    return path


def read_split(build_dir: Path, split: str, *, limit: int | None = None) -> list[SplitItem]:
    """The items of one split, in the order the build wrote them."""
    chosen = [s for s in read_build(build_dir) if s.split.value == split]
    if not chosen:
        raise ValueError(f"no items in split {split!r} in {build_dir}")
    return chosen[:limit] if limit is not None else chosen
