"""The command line.

`PLAN.md` section 4 lays out the full surface: `data build`, `baseline`, `train`,
`quantise`, `serve`, `bench`, `breakeven`, `report`. Only the commands whose work exists
appear here. A command that parses its arguments and then raises `NotImplementedError` is
worse than an absent one: it reads as a feature in `--help`, and the first person to find
out otherwise is whoever trusted it.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import typer
from boundary import Mode

from smallprint import __version__
from smallprint.baseline import (
    DEFAULT_MAX_TOKENS,
    Prediction,
    open_gateway,
    read_manifest,
    read_predictions,
    read_split,
    summarise,
    write_summary,
)
from smallprint.baseline import run as baseline_run_items
from smallprint.bench.breakeven import BreakEvenInputs, curve
from smallprint.bench.cost import GpuPrice, PriceKind
from smallprint.data import audit as hand_audit
from smallprint.data.build import SplitItem, build, write_build
from smallprint.data.datasheet import read_items, verify, write_datasheet
from smallprint.data.edgar import CONTACT_ENV, ContactNotDeclared, EdgarClient, declared_contact
from smallprint.data.split import Split
from smallprint.data.statements import Statement, locate
from smallprint.prompts import PromptStyle, build_prompt
from smallprint.schema import REQUIRED_FIELDS, SCHEMA, json_schema_for_prompt

app = typer.Typer(
    add_completion=False,
    help="Frontier quality at a fraction of the bill: structured extraction from SEC filings.",
)
data_app = typer.Typer(help="The corpus: fair access, the cache, and what has been fetched.")
app.add_typer(data_app, name="data")
baseline_app = typer.Typer(help="Running a model over held-out filings, through the gateway.")
app.add_typer(baseline_app, name="baseline")


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
    try:
        checked_on = dt.date.fromisoformat(checked)
    except ValueError as bad:
        raise typer.BadParameter(f"not a date: {checked!r}", param_hint="--checked") from bad
    inputs = BreakEvenInputs(
        price=GpuPrice(
            gpu=gpu,
            provider=provider,
            kind=kind,
            usd_per_hour=usd_per_hour,
            checked=checked_on,
            source=source,
        ),
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
            on_result=progress,
        )
    finally:
        gateway.close()
    predictions = read_predictions(out)
    typer.echo(f"{len(predictions):,} predictions in {out}. Run {manifest.run_id}.")
    typer.echo(str(summarise(manifest, predictions, items)))
    write_summary(out, summarise(manifest, predictions, items))


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
    summary = summarise(manifest, predictions, items)
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


def main() -> None:
    # Typer reads the terminal width from the environment; pinning it keeps the help text
    # reproducible in CI logs and in the documentation this command is quoted in.
    os.environ.setdefault("COLUMNS", "100")
    app()


if __name__ == "__main__":
    main()
