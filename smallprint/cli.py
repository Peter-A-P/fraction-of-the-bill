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

from smallprint import __version__
from smallprint.bench.breakeven import BreakEvenInputs, curve
from smallprint.bench.cost import GpuPrice, PriceKind
from smallprint.data.build import build, write_build
from smallprint.data.edgar import CONTACT_ENV, ContactNotDeclared, EdgarClient, declared_contact
from smallprint.data.statements import Statement, locate
from smallprint.schema import REQUIRED_FIELDS, SCHEMA, json_schema_for_prompt

app = typer.Typer(
    add_completion=False,
    help="Frontier quality at a fraction of the bill: structured extraction from SEC filings.",
)
data_app = typer.Typer(help="The corpus: fair access, the cache, and what has been fetched.")
app.add_typer(data_app, name="data")


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


def main() -> None:
    # Typer reads the terminal width from the environment; pinning it keeps the help text
    # reproducible in CI logs and in the documentation this command is quoted in.
    os.environ.setdefault("COLUMNS", "100")
    app()


if __name__ == "__main__":
    main()
