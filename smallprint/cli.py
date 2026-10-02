"""The command line.

`PLAN.md` section 4 lays out the full surface: `data build`, `baseline`, `train`,
`quantise`, `serve`, `bench`, `breakeven`, `report`. Only the commands whose work exists
appear here. A command that parses its arguments and then raises `NotImplementedError` is
worse than an absent one: it reads as a feature in `--help`, and the first person to find
out otherwise is whoever trusted it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import subprocess
from pathlib import Path

import typer
import yaml
from boundary import Mode

from smallprint import __version__
from smallprint.baseline import (
    DEFAULT_MAX_TOKENS,
    TEMPERATURE,
    BaselineSummary,
    Prediction,
    grade_run,
    open_gateway,
    read_manifest,
    read_predictions,
    read_split,
    summarise,
    within,
    write_summary,
)
from smallprint.baseline import run as baseline_run_items
from smallprint.bench.breakeven import BreakEvenInputs, curve
from smallprint.bench.client import LevelResult, requests_for
from smallprint.bench.cost import GpuPrice, PriceKind
from smallprint.bench.run import mean_tokens
from smallprint.bench.run import read as read_load
from smallprint.bench.run import sweep as load_sweep
from smallprint.data import audit as hand_audit
from smallprint.data.build import SplitItem, build, write_build
from smallprint.data.datasheet import read_items, verify, write_datasheet
from smallprint.data.edgar import CONTACT_ENV, ContactNotDeclared, EdgarClient, declared_contact
from smallprint.data.split import Split
from smallprint.data.statements import Statement, locate
from smallprint.gate import side as gate_side
from smallprint.gate import spec as gate_spec
from smallprint.grade import ItemGrade
from smallprint.prompts import PromptStyle, build_prompt
from smallprint.quant import formats
from smallprint.quant.quality import Format, judge
from smallprint.recost import recost
from smallprint.report import SERVING_CONCURRENCY, baselines, finetuned, money
from smallprint.schema import REQUIRED_FIELDS, SCHEMA, json_schema_for_prompt
from smallprint.serve import launch
from smallprint.serve.overlay import prefill_share, rates
from smallprint.serve.overlay import write as write_overlay
from smallprint.train.checkpoint import CheckpointStore, LocalSyncer, Syncer, runpod_s3
from smallprint.train.dataset import TRAINABLE, VOLUME_SEED, Example
from smallprint.train.dataset import build as build_examples
from smallprint.train.dataset import read as read_examples
from smallprint.train.dataset import write as write_examples
from smallprint.train.merge import merge as merge_adapter
from smallprint.train.qlora import describe as describe_run
from smallprint.train.qlora import read_record as read_run_record
from smallprint.train.qlora import train as train_qlora
from smallprint.train.recipe import BASES, Base, TrainConfig, gpu_hours, sweep

app = typer.Typer(
    add_completion=False,
    help="Frontier quality at a fraction of the bill: structured extraction from SEC filings.",
)
data_app = typer.Typer(help="The corpus: fair access, the cache, and what has been fetched.")
app.add_typer(data_app, name="data")
baseline_app = typer.Typer(help="Running a model over held-out filings, through the gateway.")
train_app = typer.Typer(help="Fine-tuning: the dataset, the recipe, the run.")
quantise_app = typer.Typer(help="Merging a fine-tune, making its formats, and judging each.")
serve_app = typer.Typer(help="Serving a fine-tune and registering it with the gateway.")
gate_app = typer.Typer(help="The files the release gate reads.")
bench_app = typer.Typer(help="The load test, and the price overlay made from it.")
app.add_typer(baseline_app, name="baseline")
app.add_typer(train_app, name="train")
app.add_typer(quantise_app, name="quantise")
app.add_typer(serve_app, name="serve")
app.add_typer(gate_app, name="gate")
app.add_typer(bench_app, name="bench")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def schema(
    prompt: bool = typer.Option(False, help="Print the field list as the prompt shows it."),
) -> None:
    """The extraction task: fifteen fields, their kinds and their truth concepts."""
    if prompt:
        typer.echo(json_schema_for_prompt())
        return
    for spec in SCHEMA:
        required = "required" if not spec.optional else "optional"
        typer.echo(f"{spec.name:<24} {spec.kind.value:<14} {required:<9} {spec.concepts[0]}")
        for extra in spec.concepts[1:]:
            typer.echo(f"{'':<24} {'':<14} {'':<9} {extra}")
    typer.echo(f"\n{len(SCHEMA)} fields, {len(REQUIRED_FIELDS)} of them required of every filing.")


@data_app.command("check-access")
def check_access() -> None:
    """Report whether this machine may fetch from EDGAR, without fetching anything.

    The SEC's access policy requires a contact address in the User-Agent of every request,
    and this project refuses to supply a default for it. This command says plainly whether
    the gate is open, so that finding out does not require starting a corpus fetch.
    """
    try:
        contact = declared_contact()
    except ContactNotDeclared as refusal:
        typer.echo(f"Not configured: {refusal}")
        raise typer.Exit(code=1) from refusal
    typer.echo(f"{CONTACT_ENV} is set to {contact}; requests will be paced at 8 per second.")


@data_app.command("cache")
def cache(
    cache_dir: Path = typer.Option(Path("data/raw"), help="Where fetched documents are kept."),
) -> None:
    """What the local EDGAR cache holds, with the dates and digests a rebuild is checked against."""
    if not cache_dir.exists():
        typer.echo(f"No cache at {cache_dir}. Nothing has been fetched.")
        return
    # Offline: listing what was fetched must never itself become a reason to fetch.
    with EdgarClient(cache_dir, offline=True) as edgar:
        records = list(edgar.manifest())
    if not records:
        typer.echo(f"Cache at {cache_dir} is empty. Nothing has been fetched.")
        return
    total = sum(r.bytes for r in records)
    for record in records:
        typer.echo(
            f"{record.fetched_at:%Y-%m-%d} {record.sha256[:12]} {record.bytes:>10,} {record.url}"
        )
    typer.echo(f"\n{len(records):,} documents, {total / 1e6:,.1f} MB.")


@data_app.command("compact-cache")
def compact_cache(
    cache_dir: Path = typer.Option(Path("data/raw"), help="Where fetched documents are kept."),
) -> None:
    """Compress documents cached before compression, checking each against its digest."""
    if not cache_dir.exists():
        typer.echo(f"No cache at {cache_dir}.")
        return
    with EdgarClient(cache_dir, offline=True) as edgar:
        count, saved = edgar.compact()
    typer.echo(f"Compressed {count:,} documents, saving {saved / 1e9:,.2f} GB.")


@data_app.command("datasheet")
def datasheet_command(
    build_dir: Path = typer.Option(Path("data/build"), help="A build written by data build."),
) -> None:
    """Write DATASHEET.md and SHA256SUMS for a build, from the build's own outputs."""
    sheet, sums = write_datasheet(build_dir)
    typer.echo(f"Wrote {sheet} and {sums}.")


@data_app.command("verify")
def verify_command(
    build_dir: Path = typer.Option(Path("data/build"), help="A build with a SHA256SUMS file."),
) -> None:
    """Check a build's files against its SHA256SUMS."""
    changed = verify(build_dir)
    if changed:
        typer.echo(f"Changed since the checksums were written: {', '.join(changed)}")
        raise typer.Exit(code=1)
    typer.echo("Every file matches its checksum.")


@data_app.command("audit-sample")
def audit_sample(
    build_dir: Path = typer.Option(Path("data/build"), help="A build written by data build."),
    out: Path = typer.Option(Path("data/audit"), help="Where the pages and the sheet go."),
    n: int = typer.Option(200, help="How many items to audit."),
    seed: int = typer.Option(20260919, help="Seed for the sample."),
) -> None:
    """Draw the hand-audit sample: one page per item and a verdict sheet to fill in."""
    chosen = hand_audit.sample(read_items(build_dir), n, seed=seed)
    sheet = hand_audit.write_audit(out, chosen)
    typer.echo(f"{len(chosen)} items. Pages in {out / 'items'}; verdicts go in {sheet}.")


@data_app.command("audit-serve")
def audit_serve(
    build_dir: Path = typer.Option(Path("data/build"), help="The build the sample was drawn from."),
    sheet: Path = typer.Option(Path("data/audit/audit.csv"), help="The verdict sheet."),
    port: int = typer.Option(8765, help="Local port. Bound to 127.0.0.1 only."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open it in a browser."),
) -> None:
    """The hand audit in a browser: each label beside the line it was read from.

    Verdicts are written into the sheet as they are given, so stopping and starting again
    resumes where you stopped. Ctrl+C to finish.
    """
    import webbrowser

    from smallprint.data.audit_server import AuditApp, server

    ids = {row["item_id"] for row in hand_audit.read_sheet(sheet)}
    app_state = AuditApp(sheet, [s for s in read_items(build_dir) if s.item.item_id in ids])
    httpd = server(app_state, port=port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    typer.echo(f"Auditing {len(app_state.items)} items from {sheet} at {url}. Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        typer.echo("Stopped. Verdicts are in the sheet; smallprint data audit-report reads them.")
    finally:
        httpd.server_close()


@data_app.command("audit-report")
def audit_report(
    sheet: Path = typer.Option(Path("data/audit/audit.csv"), help="The filled verdict sheet."),
) -> None:
    """The label error rate from the hand audit so far, with its 95% interval."""
    result = hand_audit.report(sheet)
    typer.echo(f"Audited {result.audited}, pending {result.pending}.")
    typer.echo(f"Items with a wrong label: {result.items_wrong}")
    for field, count in result.wrong_by_field.items():
        typer.echo(f"  {field}: {count}")


def _describe(name: str, statement: Statement | None) -> str:
    if statement is None:
        return f"{name}: not found"
    scale = f"{statement.scale:,.0f}" if statement.scale_declared else "not printed, assumed 1"
    shares = "" if statement.share_scale is None else f", shares at {statement.share_scale:,.0f}"
    return f"{name}: {len(statement.rows)} rows, scale {scale}{shares}"


@data_app.command("locate")
def locate_command(
    document: Path = typer.Argument(..., exists=True, dir_okay=False, help="A filing document."),
) -> None:
    """Show what a model would be shown from one filing document, and the scales read from it.

    Reads a local file only. This is the view the hand audit of pairs works from: whether the
    right tables were taken and whether the scale under them was read correctly.
    """
    located = locate(document.read_bytes())
    typer.echo(located.render())
    typer.echo("")
    typer.echo(_describe("income statement", located.income))
    typer.echo(_describe("balance sheet", located.balance))
    typer.echo(f"auditor block: {'found' if located.auditor else 'not found'}")
    if located.income is None or located.balance is None:
        raise typer.Exit(code=1)


@data_app.command("build")
def build_command(
    first: str = typer.Option(..., help="First quarter of filing dates, like 2022Q1."),
    last: str = typer.Option(..., help="Last quarter of filing dates, like 2027Q1."),
    cutoff: str = typer.Option(
        ...,
        help=(
            "The latest training cutoff of the base models, YYYY-MM-DD, from docs/models.md. "
            "No default: the headline test set is defined by it."
        ),
    ),
    companies: int | None = typer.Option(
        None, help="Take this many companies, chosen by a keyed hash. All when omitted."
    ),
    seed: int = typer.Option(20270405, help="Seed for selection and pool assignment."),
    cache_dir: Path = typer.Option(Path("data/raw"), help="Where fetched documents are kept."),
    out: Path = typer.Option(Path("data/build"), help="Where the corpus is written."),
    offline: bool = typer.Option(
        False, help="Read the cache only. A rebuild needs no contact and fetches nothing."
    ),
    workers: int = typer.Option(
        1,
        help=(
            "Parallel processes for an offline rebuild. A live fetch is always sequential, "
            "to stay within the SEC's rate limit."
        ),
    ),
) -> None:
    """Select, fetch, pair and split the corpus, and write it with the reasons for every drop."""
    try:
        cutoff_date = dt.date.fromisoformat(cutoff)
    except ValueError as bad:
        raise typer.BadParameter(f"not a date: {cutoff!r}", param_hint="--cutoff") from bad
    try:
        edgar = EdgarClient(cache_dir, offline=offline)
    except ContactNotDeclared as refusal:
        typer.echo(f"Not configured: {refusal}")
        raise typer.Exit(code=1) from refusal
    with edgar:
        items, dropped, report = build(
            edgar,
            first=first,
            last=last,
            cutoff=cutoff_date,
            companies=companies,
            seed=seed,
            progress=lambda done, total: typer.echo(f"{done:,} of {total:,} companies", err=True),
            workers=workers,
        )
    write_build(out, items, dropped, report)
    typer.echo(f"{report.filings_selected:,} filings from {report.companies_selected:,} companies.")
    for reason, count in report.pairing.items():
        typer.echo(f"  {reason}: {count:,}")
    typer.echo(str(report.split))
    typer.echo(f"Written to {out}.")


def _gpu_price(
    gpu: str, provider: str, kind: PriceKind, usd_per_hour: float, checked: str, source: str
) -> GpuPrice:
    try:
        checked_on = dt.date.fromisoformat(checked)
    except ValueError as bad:
        raise typer.BadParameter(f"not a date: {checked!r}", param_hint="--checked") from bad
    return GpuPrice(
        gpu=gpu,
        provider=provider,
        kind=kind,
        usd_per_hour=usd_per_hour,
        checked=checked_on,
        source=source,
    )


@app.command()
def breakeven(
    gpu: str = typer.Option(..., help="The GPU, as the price table names it."),
    provider: str = typer.Option(..., help="Who rents it."),
    kind: PriceKind = typer.Option(..., help="Spot or on-demand."),
    usd_per_hour: float = typer.Option(..., help="The GPU-hour rate in US dollars."),
    checked: str = typer.Option(
        ..., help="The date the rate was read, YYYY-MM-DD. Every price carries its date."
    ),
    source: str = typer.Option(..., help="Where the rate was read."),
    requests_per_second: float = typer.Option(..., help="Measured throughput on that GPU."),
    api_usd_per_call: float = typer.Option(..., help="The API's cost per call, from the ledger."),
    fixed_usd_per_month: float = typer.Option(
        0.0, help="Any fixed monthly cost of self-hosting to count. Printed either way."
    ),
) -> None:
    """The monthly volume at which self-hosting stops costing more, over utilisation.

    Every input is an argument so that a reader can substitute their own rate, throughput,
    API cost and fixed costs; nothing here is a default except the fixed cost of zero.
    """
    inputs = BreakEvenInputs(
        price=_gpu_price(gpu, provider, kind, usd_per_hour, checked, source),
        requests_per_second=requests_per_second,
        api_usd_per_call=api_usd_per_call,
        fixed_usd_per_month=fixed_usd_per_month,
    )
    typer.echo(f"{inputs.price}")
    typer.echo(
        f"Throughput {requests_per_second:,.2f} req/s; API US${api_usd_per_call:.6f} per call; "
        f"fixed US${fixed_usd_per_month:,.2f} per month; {inputs.hours_per_month} h per month."
    )
    typer.echo("")
    typer.echo(
        f"{'utilisation':>11}  {'self $/1k':>10}  {'API $/1k':>10}  {'break-even per month':>22}  gpus"
    )
    for point in curve(inputs):
        volume = "never" if point.volume_per_month is None else f"{point.volume_per_month:,.0f}"
        gpus = "" if point.gpus_at_break_even is None else str(point.gpus_at_break_even)
        typer.echo(
            f"{point.utilisation:>11.0%}  {point.self_hosted_usd_per_call * 1000:>10.4f}  "
            f"{point.api_usd_per_call * 1000:>10.4f}  {volume:>22}  {gpus}"
        )


@baseline_app.command("prompt")
def baseline_prompt(
    style: PromptStyle = typer.Option(PromptStyle.ZERO_SHOT, help="Zero-shot or few-shot."),
    build_dir: Path = typer.Option(Path("data/build"), help="Where few-shot examples come from."),
    k: int = typer.Option(2, help="How many examples, few-shot only."),
    thinking: bool = typer.Option(False, help="Open the system prompt with the think token."),
    full: bool = typer.Option(False, help="Print the whole prompt, examples included."),
) -> None:
    """Print the prompt a run would send, and its fingerprint.

    Cheap to look at and worth looking at: every accuracy number in the project is a
    number about this text, and two runs are comparable only if their fingerprints match.
    """
    pool = _training_pool(build_dir) if style is PromptStyle.FEW_SHOT else []
    prompt = build_prompt(style, pool=pool, k=k, thinking=thinking)
    typer.echo(prompt.system if full else prompt.system.split("Rules:")[0].rstrip())
    if full:
        for text, answer in prompt.examples:
            typer.echo("")
            typer.echo("--- example user ---")
            typer.echo(text)
            typer.echo("--- example answer ---")
            typer.echo(answer)
    typer.echo("")
    typer.echo(
        f"Style {prompt.style.value}, {len(prompt.examples)} examples, "
        f"fingerprint {prompt.fingerprint[:16]}."
    )
    if prompt.example_ids:
        typer.echo(f"Examples: {', '.join(prompt.example_ids)}")


def _training_pool(build_dir: Path) -> list[SplitItem]:
    return [s for s in read_items(build_dir) if s.split is Split.TRAIN]


@baseline_app.command("run")
def baseline_run(
    model: str = typer.Option(..., help="An alias from the routes file, or provider/model-id."),
    out: Path = typer.Option(..., help="Where the predictions, the manifest and the raw bytes go."),
    config: Path = typer.Option(Path("boundary.yaml"), help="The gateway configuration."),
    build_dir: Path = typer.Option(Path("data/build"), help="A build written by data build."),
    split: str = typer.Option(
        Split.TEST_POST_CUTOFF.value, help="Which split to run. The headline is post-cutoff."
    ),
    style: PromptStyle = typer.Option(PromptStyle.ZERO_SHOT, help="Zero-shot or few-shot."),
    k: int = typer.Option(2, help="How many examples, few-shot only."),
    thinking: bool = typer.Option(False, help="Measure the model with reasoning turned on."),
    limit: int | None = typer.Option(None, help="Run only the first N items. For a smoke run."),
    mode: Mode = typer.Option(
        Mode.PASSTHROUGH.value,
        help="Pass-through for anything published: no retries, no cache, bytes kept.",
    ),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, help="Ceiling on the answer."),
    temperature: float | None = typer.Option(
        TEMPERATURE,
        help=(
            "Sent with every call. Omitted when not given, which is what all three "
            "frontier models require; a served open-weights model takes 0."
        ),
    ),
    run_id: str | None = typer.Option(None, help="Defaults to the model, style and the date."),
) -> None:
    """Run one model over one split and write a prediction per filing. Resumable.

    Every call goes through the gateway, so what this costs is read from the ledger rather
    than estimated. Re-running against the same directory sends only the filings that have
    no prediction yet.
    """
    items = read_split(build_dir, split, limit=limit)
    pool = _training_pool(build_dir) if style is PromptStyle.FEW_SHOT else []
    prompt = build_prompt(style, pool=pool, k=k, thinking=thinking)
    identifier = run_id or f"{style.value}-{model.replace('/', '-')}-{dt.date.today().isoformat()}"

    def progress(n: int, total: int, prediction: Prediction) -> None:
        if n == 1 or n % 25 == 0 or n == total:
            typer.echo(f"  {n:,}/{total:,} {prediction.item_id} {prediction.status}")

    gateway = open_gateway(config, raw_store=out / "raw")
    try:
        typer.echo(
            f"{model} on {len(items):,} {split} items, {style.value}, {mode.value} mode, "
            f"prompt {prompt.fingerprint[:16]}."
        )
        manifest = baseline_run_items(
            gateway,
            model=model,
            items=items,
            prompt=prompt,
            out_dir=out,
            build_dir=build_dir,
            run_id=identifier,
            mode=mode,
            max_tokens=max_tokens,
            temperature=temperature,
            on_result=progress,
        )
    finally:
        gateway.close()
    predictions = read_predictions(out)
    typer.echo(f"{len(predictions):,} predictions in {out}. Run {manifest.run_id}.")
    summary = summarise(manifest, predictions, items, recosted=recost(out, manifest.model))
    typer.echo(str(summary))
    write_summary(out, summary)


@baseline_app.command("report")
def baseline_report(
    run_dir: Path = typer.Option(..., help="A directory written by baseline run."),
    build_dir: Path | None = typer.Option(None, help="Defaults to the build the run names."),
    fields: bool = typer.Option(False, help="Per-field accuracy and the named failure modes."),
) -> None:
    """Grade a finished run and print what it measured, with an interval on every number."""
    manifest = read_manifest(run_dir)
    source = build_dir if build_dir is not None else Path(manifest.build_dir)
    items = read_split(source, manifest.split)
    predictions = read_predictions(run_dir)
    summary = summarise(manifest, predictions, items, recosted=recost(run_dir, manifest.model))
    typer.echo(str(summary))
    if fields:
        typer.echo("")
        for report in summary.fields:
            reasons = ", ".join(f"{k} {v}" for k, v in report.reasons.items()) or "none"
            typer.echo(f"  {report.field:<24} {report.accuracy}")
            typer.echo(f"  {'':<24} {reasons}")
    path = write_summary(run_dir, summary)
    typer.echo("")
    typer.echo(f"Written to {path}.")


def _base(size: str) -> Base:
    """The base for a size label, or a clear refusal naming the three that exist."""
    if size not in BASES:
        raise typer.BadParameter(f"{size!r} is not one of {', '.join(BASES)}", param_hint="--size")
    return BASES[size]


@train_app.command("dataset")
def train_dataset(
    build_dir: Path = typer.Option(Path("data/build"), help="A build written by data build."),
    out: Path = typer.Option(Path("data/train"), help="Where examples.jsonl is written."),
    volume: int | None = typer.Option(None, help="Training filings to use. All when omitted."),
    seed: int = typer.Option(VOLUME_SEED, help="Seed for the volume subset."),
    targets_from: Path | None = typer.Option(
        None, help="A run over the training filings whose answers replace the facts as targets."
    ),
) -> None:
    """Write the training file: the same prompt the baselines used, the graded answer as target.

    Only the training and validation pools are ever written. The volume subsets are nested,
    so the data-scaling curve varies how many filings and not which ones. With
    `--targets-from`, the targets are that run's answers instead: the distillation ablation.
    """
    from smallprint.grade import parse_extraction
    from smallprint.train.dataset import distil

    items = [s for s in read_items(build_dir) if s.split in TRAINABLE]
    examples, manifest = build_examples(items, volume=volume, seed=seed)
    if targets_from is not None:
        run = read_manifest(targets_from)
        if run.prompt_fingerprint != manifest.prompt_fingerprint or run.split != Split.TRAIN.value:
            raise typer.BadParameter(
                f"{targets_from} is {run.split} under prompt {run.prompt_fingerprint[:16]}; the "
                f"targets must be the training filings under {manifest.prompt_fingerprint[:16]}",
                param_hint="--targets-from",
            )
        answers = {}
        for p in read_predictions(targets_from):
            parsed = parse_extraction(p.text) if p.ok and p.text else None
            if parsed is not None:
                answers[p.item_id] = parsed
        examples, manifest = distil(examples, manifest, answers, source=run.model)
    path = write_examples(out, examples, manifest, build_dir=build_dir)
    typer.echo(
        f"{manifest.train:,} training and {manifest.validation:,} validation examples in {path}"
        + (
            f", targets from {manifest.targets}, {manifest.dropped:,} dropped."
            if manifest.targets != "xbrl"
            else "."
        )
    )
    typer.echo(
        f"Median {manifest.median_chars:,} characters, p95 {manifest.p95_chars:,}; "
        f"prompt {manifest.prompt_fingerprint[:16]}."
    )


@train_app.command("inherited")
def train_inherited(
    teacher: Path = typer.Option(..., help="The teacher's run over held-out filings."),
    student: list[Path] = typer.Option(
        ..., help="Runs over the same filings. The first is the reference for the deltas."
    ),
    build_dir: Path | None = typer.Option(None, help="Defaults to the build the teacher names."),
) -> None:
    """The distillation ablation's numbers: accuracy, the paired delta, the errors inherited.

    Each student is graded over the teacher's filings, paired with the first student, and
    asked what share of the teacher's wrong fields it got wrong the same way. Run it with the
    model trained on the facts as the first student and the distilled one second: the first
    is the control, the share any model trained on this task repeats by sharing a mistake.
    """
    from smallprint.grade import inherited_ci, paired_delta_ci

    manifest = read_manifest(teacher)
    items = read_split(build_dir or Path(manifest.build_dir), manifest.split)

    def grades(run_dir: Path) -> list[ItemGrade]:
        graded, failed = grade_run(within(read_predictions(run_dir), items)[0], items)
        if failed:
            raise typer.BadParameter(f"{run_dir}: {len(failed):,} filings failed")
        return graded

    t = grades(teacher)
    reference: list[ItemGrade] | None = None
    typer.echo(f"{len(t):,} {manifest.split} filings; teacher {manifest.model}")
    for run_dir in student:
        s = grades(run_dir)
        accuracy = sum(g.accuracy for g in s) / len(s)
        delta = "reference" if reference is None else str(paired_delta_ci(s, reference))
        typer.echo(f"  {run_dir.name}")
        typer.echo(f"    fields correct {accuracy:.1%}, paired delta {delta}")
        typer.echo(f"    vs teacher {paired_delta_ci(s, t)}")
        typer.echo(f"    teacher's errors repeated {inherited_ci(s, t)}")
        reference = reference if reference is not None else s


@train_app.command("plan")
def train_plan(
    dataset_dir: Path = typer.Option(Path("data/train"), help="Written by train dataset."),
    size: str = typer.Option(..., help="2b, 4b or 7b, as docs/models.md records them."),
    seconds_per_step: float | None = typer.Option(
        None, help="Measured on the card. Given, this prints GPU hours; never assumed."
    ),
) -> None:
    """The ablation schedule: every run, its steps and its checkpoints, before renting anything."""
    _, manifest = read_examples(dataset_dir)
    runs = sweep(TrainConfig(base=_base(size).repo, size=size))
    typer.echo(f"{len(runs)} runs over {manifest.train:,} training examples")
    for config in runs:
        typer.echo(
            f"  {config.run_id:<44} {config.steps(manifest.train):>6} steps, "
            f"{config.checkpoints(manifest.train):>4} checkpoints"
        )
    if seconds_per_step is not None:
        hours = gpu_hours(runs, manifest.train, seconds_per_step)
        typer.echo(f"{hours:,.1f} GPU hours at {seconds_per_step:g} s a step, seeds not included.")


@train_app.command("run")
def train_run(
    dataset_dir: Path = typer.Option(Path("data/train"), help="Written by train dataset."),
    size: str = typer.Option(..., help="2b, 4b or 7b. The base and its revision follow."),
    out: Path = typer.Option(..., help="Where the checkpoints and the adapter go."),
    rank: int = typer.Option(16, help="LoRA rank. Alpha follows at twice the rank."),
    learning_rate: float = typer.Option(1e-4, help="Peak learning rate."),
    epochs: int = typer.Option(2, help="Passes over the training pool."),
    seed: int = typer.Option(0, help="The run's seed."),
    volume: int | None = typer.Option(None, help="Training filings. All when omitted."),
    save_steps: int = typer.Option(50, help="Steps between checkpoints."),
    checkpoint_dir: Path | None = typer.Option(
        None, help="A second directory to copy every checkpoint to. Use a mounted bucket."
    ),
    checkpoint_uri: str | None = typer.Option(
        None,
        help=(
            "A Runpod network volume over its S3 API, s3://<volume id>/<path>. Needs "
            "--s3-datacenter and the S3 key in AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        ),
    ),
    s3_datacenter: str | None = typer.Option(None, help="The volume's data center, like EU-RO-1."),
) -> None:
    """Fine-tune one base. Needs the GPU extra; this does not run on the laptop, by rule."""
    examples, manifest = read_examples(dataset_dir)
    chosen = _base(size)
    config = TrainConfig(
        base=chosen.repo,
        size=size,
        rank=rank,
        alpha=2 * rank,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
        volume=volume,
        save_steps=save_steps,
    )
    if checkpoint_dir is not None and checkpoint_uri is not None:
        raise typer.BadParameter(
            "one place for checkpoints, not two", param_hint="--checkpoint-uri"
        )
    syncer: Syncer | None = None
    if checkpoint_uri is not None:
        if s3_datacenter is None:
            raise typer.BadParameter("the volume's data center", param_hint="--s3-datacenter")
        syncer = runpod_s3(checkpoint_uri, s3_datacenter)
    elif checkpoint_dir is not None:
        syncer = LocalSyncer(checkpoint_dir)
    store = CheckpointStore(out, syncer, keep=2)
    typer.echo(
        f"{config.run_id}: {config.steps(manifest.train):,} steps, resuming from {store.resume_step()}."
    )
    record = train_qlora(
        config,
        examples,
        manifest,
        out_dir=out,
        store=store,
        base_revision=chosen.revision,
        chat_template_from=(chosen.baseline_repo, chosen.baseline_revision),
    )
    typer.echo(describe_run(record))


@app.command()
def report(
    runs: Path = typer.Option(Path("data/baseline"), help="A directory of run directories."),
    also: list[Path] = typer.Option(
        [], help="More directories of runs for the baseline table, such as the untuned bases."
    ),
    build_dir: Path = typer.Option(Path("data/build"), help="The build the runs were made on."),
    split: str = typer.Option(Split.TEST_POST_CUTOFF.value, help="Which split the runs cover."),
    against: str | None = typer.Option(
        None, help="Pair every delta with this run, as 'model style'. Default: the cheapest."
    ),
    finetuned_runs: Path | None = typer.Option(
        None,
        "--finetuned",
        help="A directory of fine-tuned runs, post- and pre-cutoff. Adds their tables.",
    ),
    bench_runs: Path | None = typer.Option(
        None,
        "--bench",
        help="A directory of load tests. With --finetuned, adds serving, money and break-even.",
    ),
    chart: Path | None = typer.Option(
        None, help="Write the Pareto chart here as SVG. Needs --finetuned and --bench."
    ),
    out: Path | None = typer.Option(None, help="Write the markdown here instead of printing it."),
) -> None:
    """The results tables, written from the runs. Nobody edits these tables by hand."""
    if (bench_runs is not None or chart is not None) and finetuned_runs is None:
        raise typer.BadParameter("serving numbers need the fine-tuned runs", param_hint="--bench")
    if chart is not None and bench_runs is None:
        raise typer.BadParameter("the chart needs the load tests", param_hint="--chart")
    markdown = baselines(runs, build_dir, split, against=against, also=also)
    if finetuned_runs is not None:
        markdown += "\n\n" + finetuned(finetuned_runs, runs, build_dir)
    if finetuned_runs is not None and bench_runs is not None:
        markdown += "\n\n" + money(finetuned_runs, runs, bench_runs, build_dir, chart=chart)
    if out is None:
        typer.echo(markdown)
        return
    out.write_text(markdown + "\n", encoding="utf-8", newline="\n")
    typer.echo(f"Written to {out}.")


# -- merging and the formats ------------------------------------------------------------------


def _check_example(dataset_dir: Path) -> Example:
    """The validation filing a merge is checked on: the first by id, the same every time."""
    examples, _ = read_examples(dataset_dir)
    validation = sorted(
        (e for e in examples if e.split is Split.VALIDATION), key=lambda e: e.item_id
    )
    if not validation:
        raise typer.BadParameter("no validation filings", param_hint="--dataset-dir")
    return validation[0]


@quantise_app.command("merge")
def quantise_merge(
    run_dir: Path = typer.Option(
        ..., help="A finished run: its run.json and its adapter/, as sweep.sh keeps them."
    ),
    out: Path = typer.Option(..., help="Where the merged bf16 weights go."),
    dataset_dir: Path = typer.Option(Path("data/train"), help="For the merge check's filing."),
) -> None:
    """Merge a run's adapter into its base in bf16, checked against the adapter."""
    record = read_run_record(run_dir)
    size = record.config.size
    chosen = _base(size)
    if record.base_revision != chosen.revision:
        raise typer.BadParameter(
            f"the run trained on {record.base_revision} and the {size} base is pinned at "
            f"{chosen.revision}; merging into another revision is not this fine-tune",
            param_hint="--run-dir",
        )
    merged = merge_adapter(
        run_dir / "adapter",
        chosen,
        out,
        run_id=record.run_id,
        size=size,
        check=_check_example(dataset_dir),
    )
    typer.echo(
        f"{out}: next tokens agree on {merged.agreement:.2%} of the answer's "
        f"{merged.answer_tokens} and {merged.window_agreement or 0:.2%} of the last "
        f"{merged.check_tokens} positions, largest logit difference "
        f"{merged.max_abs_logit_diff:.4f}."
    )


@quantise_app.command("format")
def quantise_format(
    fmt: Format = typer.Option(..., "--format", help="awq or gptq."),
    model: Path = typer.Option(..., help="Merged bf16 weights, written by quantise merge."),
    out: Path = typer.Option(..., help="Where the quantised weights go."),
    dataset_dir: Path = typer.Option(Path("data/train"), help="Calibration comes from here."),
    max_seq_len: int = typer.Option(8192, help="Longest calibration sequence, in tokens."),
) -> None:
    """AWQ or GPTQ weights with llm-compressor, calibrated on training filings only."""
    examples, _ = read_examples(dataset_dir)
    record = formats.quantise(fmt, model, out, examples, max_seq_len=max_seq_len)
    typer.echo(
        f"{fmt.value} in {out}, {record.scheme}, calibrated on "
        f"{len(record.calibration_ids)} training filings."
    )


@quantise_app.command("gguf")
def quantise_gguf(
    model: Path = typer.Option(..., help="Merged bf16 weights, written by quantise merge."),
    out: Path = typer.Option(..., help="A directory for the GGUF files."),
    name: str = typer.Option(..., help="The run's name, which the files are named after."),
    llama_cpp: Path = typer.Option(..., help="A llama.cpp checkout, for its converter."),
    quantize_binary: Path = typer.Option(
        Path("llama-quantize"), help="The llama-quantize binary built from that checkout."
    ),
) -> None:
    """GGUF Q8_0 and Q4_K_M, through one bf16 GGUF, with llama.cpp's own tools."""
    out.mkdir(parents=True, exist_ok=True)
    intermediate = out / f"{name}-bf16.gguf"
    # The argv is built from typed paths by formats.convert_argv, not from a shell string.
    subprocess.run(formats.convert_argv(llama_cpp, model, intermediate), check=True)
    for fmt in formats.GGUF_TYPES:
        target = formats.gguf_file(out, name, fmt)
        subprocess.run(
            formats.quantize_argv(quantize_binary, intermediate, target, fmt), check=True
        )
        formats.gguf_record(fmt, model, target)
        typer.echo(f"{fmt.value}: {target}")


def _items_for(run_dir: Path, build_dir: Path | None) -> list[SplitItem]:
    manifest = read_manifest(run_dir)
    source = build_dir if build_dir is not None else Path(manifest.build_dir)
    return read_split(source, manifest.split)


def _grades(run_dir: Path, items: list[SplitItem]) -> list[ItemGrade]:
    """A run's grades over the items the build still holds."""
    graded, _ = grade_run(within(read_predictions(run_dir), items)[0], items)
    return graded


@app.command()
def card(
    run_dir: Path = typer.Option(..., help="The training run: its run.json, as sweep.sh keeps it."),
    evaluation: Path = typer.Option(..., help="Its bf16 run on the post-cutoff filings."),
    anchor: Path = typer.Option(..., help="The cost anchor's run on the same filings."),
    out: Path = typer.Option(..., help="Where the card goes, a README.md."),
    pre_cutoff: Path | None = typer.Option(None, help="Its bf16 run on the pre-cutoff filings."),
    fmt_runs: list[Path] = typer.Option(
        [], "--format-run", help="A quantised format's run on the same filings. Repeat."
    ),
    bench_dir: Path | None = typer.Option(None, "--bench", help="Its bf16 load test."),
    utilisation: float = typer.Option(0.5, help="The utilisation the served cost assumes."),
    gate_ledger: Path | None = typer.Option(
        None, help="The release gate's ledger; its decisions on this model go on the card."
    ),
    build_dir: Path | None = typer.Option(None, help="Defaults to the build the runs name."),
) -> None:
    """The model card for one fine-tune, generated from its records and never typed."""
    from smallprint import cards
    from smallprint.bench.cost import usd_per_call

    record = read_run_record(run_dir)
    manifest = read_manifest(evaluation)
    items = _items_for(evaluation, build_dir)
    reference = _grades(evaluation, items)
    verdicts = []
    for run in fmt_runs:
        _, run_name, fmt = launch.parse_served_name(read_manifest(run).model)
        if run_name != record.run_id and run_name not in manifest.model:
            raise typer.BadParameter(f"{run} is another model's", param_hint="--format-run")
        verdicts.append(judge(fmt, _grades(run, items), reference))
    served = None
    if bench_dir is not None:
        load = read_load(bench_dir)
        level = load.level(SERVING_CONCURRENCY)
        if level is None or level.summary is None:
            raise typer.BadParameter(f"no level {SERVING_CONCURRENCY}", param_hint="--bench")
        rps = level.summary.requests_per_second.point
        per_1000 = 1000 * usd_per_call(load.price.usd_per_hour, rps, utilisation)
        served = (
            f"US${per_1000:,.3f} served ({load.price}; {rps:.2f} requests a second at "
            f"{SERVING_CONCURRENCY} in flight, {utilisation:.0%} utilisation)"
        )

    def summary_of(run: Path) -> BaselineSummary:
        run_items = _items_for(run, build_dir)
        run_manifest = read_manifest(run)
        return summarise(
            run_manifest,
            read_predictions(run),
            run_items,
            recosted=recost(run, run_manifest.model),
        )

    text = cards.render(
        name=launch.parse_served_name(manifest.model)[1],
        base=BASES[record.config.size],
        record=record,
        evaluation=summary_of(evaluation),
        evaluation_prompt=manifest.prompt_fingerprint,
        anchor=summary_of(anchor),
        verdicts=verdicts,
        pre_cutoff=summary_of(pre_cutoff) if pre_cutoff is not None else None,
        served_cost=served,
        gate=[
            record
            for line in (
                gate_ledger.read_text(encoding="utf-8").splitlines() if gate_ledger else []
            )
            if line.strip()
            for record in [json.loads(line)]
            if record.get("candidate", {}).get("label") == manifest.model
        ],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(f"Written to {out}.")


@quantise_app.command("judge")
def quantise_judge(
    reference: Path = typer.Option(..., help="The bf16 run of the fine-tune."),
    candidate: Path = typer.Option(..., help="The quantised run of the same fine-tune."),
    build_dir: Path | None = typer.Option(None, help="Defaults to the build the runs name."),
) -> None:
    """Whether a format ships: its paired delta against bf16, per field, on the same filings."""
    ref_manifest, cand_manifest = read_manifest(reference), read_manifest(candidate)
    _, ref_run, ref_fmt = launch.parse_served_name(ref_manifest.model)
    _, cand_run, cand_fmt = launch.parse_served_name(cand_manifest.model)
    if ref_fmt is not Format.BF16 or ref_run != cand_run:
        raise typer.BadParameter(
            f"{cand_manifest.model} is judged against the bf16 build of {cand_run}, "
            f"not {ref_manifest.model}",
            param_hint="--reference",
        )
    if ref_manifest.split != cand_manifest.split:
        raise typer.BadParameter("the two runs cover different splits", param_hint="--candidate")
    items = _items_for(candidate, build_dir)
    verdict = judge(cand_fmt, _grades(candidate, items), _grades(reference, items))
    typer.echo(
        f"{cand_fmt.value}: {'ships' if verdict.ships else 'does not ship'}. {verdict.reason}"
    )
    path = candidate / "verdict.json"
    path.write_text(verdict.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    typer.echo(f"Written to {path}.")


# -- serving ------------------------------------------------------------------------------------


@serve_app.command("argv")
def serve_argv(
    model: Path = typer.Option(..., help="The weights: a directory, or a .gguf file."),
    run: str = typer.Option(..., help="The training run's name; the served name follows."),
    fmt: Format = typer.Option(..., "--format", help="Which format the weights are."),
    port: int = typer.Option(8000, help="The port the server listens on."),
    kv_tokens: int | None = typer.Option(
        None, help="llama.cpp only: the shared cache, in tokens. Sized to the card."
    ),
) -> None:
    """Print the command line that serves these weights, every flag that matters set."""
    name = launch.served_name(run, fmt)
    options: dict[str, int] = {"port": port}
    if kv_tokens is not None:
        if fmt.server != "llamacpp":
            raise typer.BadParameter("vLLM sizes its own cache", param_hint="--kv-tokens")
        options["kv_tokens"] = kv_tokens
    typer.echo(shlex.join(launch.argv_for(model.as_posix(), name, fmt, **options)))


@serve_app.command("config")
def serve_config(
    base_url: str = typer.Option(..., help="The server, like http://127.0.0.1:8000/v1."),
    base: Path = typer.Option(Path("boundary.yaml"), help="The project's gateway config."),
    out: Path = typer.Option(Path("boundary-served.yaml"), help="Where the new config goes."),
    self_hosted_prices: Path | None = typer.Option(
        None, help="The overlay directory, once the load test has written a price file to it."
    ),
) -> None:
    """The gateway configuration with the served model added as a self-hosted provider.

    Written beside the project's own, so its relative paths, caps and ledger resolve to the
    same files and a self-hosted call lands in the ledger every frontier call is in.
    """
    from boundary.config import BoundaryConfig

    if out.resolve().parent != base.resolve().parent:
        raise typer.BadParameter("write it beside the base config", param_hint="--out")
    raw = yaml.safe_load(base.read_text(encoding="utf-8"))
    prices = None
    if self_hosted_prices is not None:
        if not any(self_hosted_prices.glob("*.yaml")):
            raise typer.BadParameter(
                "no price file in it yet; leave it out and the calls are written uncosted",
                param_hint="--self-hosted-prices",
            )
        prices = self_hosted_prices.as_posix()
    config = launch.served_config(raw, base_url, self_hosted_prices=prices)
    BoundaryConfig.model_validate(config)
    out.write_text(
        "# Written by `smallprint serve config`; regenerate it rather than editing it.\n"
        + yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    typer.echo(f"Written to {out}. Call the model as {launch.PROVIDER}/<served name>.")


# -- the load test and the price overlay ------------------------------------------------------


@bench_app.command("run")
def bench_run(
    model: str = typer.Option(..., help="The served model, selfhosted/<run>-<format>."),
    out: Path = typer.Option(..., help="Where bench.json goes, one directory per model and card."),
    gpu: str = typer.Option(..., help="The card, as the price table names it."),
    provider: str = typer.Option(..., help="Who rents it."),
    kind: PriceKind = typer.Option(..., help="Spot or on-demand."),
    usd_per_hour: float = typer.Option(..., help="The GPU-hour rate paid for this card."),
    checked: str = typer.Option(..., help="The date the rate was read, YYYY-MM-DD."),
    source: str = typer.Option(..., help="Where the rate was read."),
    config: Path = typer.Option(
        Path("boundary-served.yaml"), help="Written by serve config, which turns retries off."
    ),
    build_dir: Path = typer.Option(Path("data/build"), help="The requests are its test filings."),
    concurrency: list[int] = typer.Option(
        list(launch.CONCURRENCY), help="Levels to run, lowest first. Repeat the option."
    ),
    warmup_seconds: float = typer.Option(30.0, help="Discarded from the start of each level."),
) -> None:
    """The closed-loop load test at each concurrency, through the gateway, streamed."""
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if raw.get("retry", {}).get("max_attempts") != 1:
        raise typer.BadParameter(
            "retries must be off, or the latencies measure the retry policy; "
            "use the config `smallprint serve config` writes",
            param_hint="--config",
        )
    launch.parse_served_name(model)
    price = _gpu_price(gpu, provider, kind, usd_per_hour, checked, source)
    items = read_split(build_dir, Split.TEST_POST_CUTOFF.value)
    requests = requests_for(items, build_prompt(PromptStyle.ZERO_SHOT), model)

    def progress(level: LevelResult) -> None:
        summary = level.summary
        typer.echo(
            f"  c={level.concurrency}: "
            + (
                f"{summary.requests_per_second}, TTFT p99 {summary.ttft_p99_ms}"
                if summary is not None
                else "nothing to summarise"
            )
            + (f", {level.retried} retried" if level.retried else "")
        )

    gateway = open_gateway(config, raw_store=out / "raw")
    try:
        typer.echo(f"{model} on {price}, {len(requests):,} distinct filings.")
        load_sweep(
            gateway,
            requests,
            model=model,
            price=price,
            levels=concurrency,
            out_dir=out,
            warmup_seconds=warmup_seconds,
            on_level=progress,
        )
    finally:
        gateway.close()
    typer.echo(f"Written to {out / 'bench.json'}.")


@bench_app.command("overlay")
def bench_overlay(
    bench: list[Path] = typer.Option(..., help="Load-test directories. Repeat the option."),
    out: Path = typer.Option(Path("prices/self-hosted"), help="The overlay directory."),
    utilisation: float = typer.Option(..., help="The utilisation the ledger's cost assumes."),
    date: str = typer.Option(..., help="The overlay file's date, YYYY-MM-DD."),
    concurrency: int = typer.Option(
        SERVING_CONCURRENCY, help="The level whose throughput is the rate."
    ),
) -> None:
    """The dated price file the gateway costs self-hosted calls with, from the load tests."""
    entries = []
    for directory in bench:
        run = read_load(directory)
        level = run.level(concurrency)
        if level is None or level.summary is None or level.retried:
            raise typer.BadParameter(
                f"no clean level at c={concurrency} in {directory}", param_hint="--bench"
            )
        mean_in, mean_out = mean_tokens(level, warmup_seconds=run.warmup_seconds)
        entries.append(
            rates(
                launch.PROVIDER,
                run.model.split("/", 1)[1],
                run.price,
                requests_per_second=level.summary.requests_per_second.point,
                utilisation=utilisation,
                mean_input_tokens=mean_in,
                mean_output_tokens=mean_out,
                prefill=prefill_share(level.summary),
            )
        )
    try:
        on = dt.date.fromisoformat(date)
    except ValueError as bad:
        raise typer.BadParameter(f"not a date: {date!r}", param_hint="--date") from bad
    path = write_overlay(out, entries, date=on)
    for entry in entries:
        typer.echo(f"{entry.model}: US${entry.usd_per_call * 1000:,.4f} per 1,000")
    typer.echo(f"Written to {path}.")


# -- the release gate ---------------------------------------------------------------------------


@gate_app.command("export")
def gate_export(
    candidate: Path = typer.Option(..., help="The fine-tune's run on the post-cutoff filings."),
    baseline: Path = typer.Option(..., help="The best frontier run on the same filings."),
    out: Path = typer.Option(..., help="Where the spec and the two sides go."),
    build_dir: Path | None = typer.Option(None, help="Defaults to the build the runs name."),
) -> None:
    """The eval spec and both sides' per-field outcomes, in the shapes the gate holds."""
    sides = {}
    for role, run_dir in (("baseline", baseline), ("candidate", candidate)):
        manifest = read_manifest(run_dir)
        if manifest.split != Split.TEST_POST_CUTOFF.value:
            raise typer.BadParameter(
                f"{run_dir} covers {manifest.split}; the headline is the post-cutoff set",
                param_hint=f"--{role}",
            )
        sides[role] = gate_side(
            manifest.model,
            read_predictions(run_dir),
            _items_for(run_dir, build_dir),
            source={"run_id": manifest.run_id, "directory": run_dir.as_posix()},
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.yaml").write_text(
        yaml.safe_dump(gate_spec(), sort_keys=False), encoding="utf-8", newline="\n"
    )
    for role, content in sides.items():
        (out / f"{role}.json").write_text(
            json.dumps(content, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
        )
    typer.echo(f"spec.yaml, baseline.json and candidate.json in {out}.")


def main() -> None:
    # Typer reads the terminal width from the environment; pinning it keeps the help text
    # reproducible in CI logs and in the documentation this command is quoted in.
    os.environ.setdefault("COLUMNS", "100")
    app()


if __name__ == "__main__":
    main()
