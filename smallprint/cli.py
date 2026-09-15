"""The command line.

`PLAN.md` section 4 lays out the full surface: `data build`, `baseline`, `train`,
`quantise`, `serve`, `bench`, `breakeven`, `report`. Only the commands whose work exists
appear here. A command that parses its arguments and then raises `NotImplementedError` is
worse than an absent one: it reads as a feature in `--help`, and the first person to find
out otherwise is whoever trusted it.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from smallprint import __version__
from smallprint.data.edgar import CONTACT_ENV, ContactNotDeclared, EdgarClient, declared_contact
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


def main() -> None:
    # Typer reads the terminal width from the environment; pinning it keeps the help text
    # reproducible in CI logs and in the documentation this command is quoted in.
    os.environ.setdefault("COLUMNS", "100")
    app()


if __name__ == "__main__":
    main()
