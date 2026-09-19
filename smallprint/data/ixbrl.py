"""Cover-page facts from the filing's own inline XBRL.

Four fields of the schema are `dei` facts: the period end, the fiscal period, the state of
incorporation and the auditor. The `companyfacts` API organises facts by unit and is not
where these live. The filing document itself is: since 2019 every 10-K and 10-Q is inline
XBRL, and its cover page tags exactly these facts with `ix:nonNumeric`, against contexts
declared in the hidden header. Those tags are what the company filed, which is this
project's definition of truth, so they are read from there rather than joined in from a
bulk data set.

Reading them from the document the model is shown is not a leak. The hidden header is
stripped before the model's input is rendered (`statements.py`), and the visible tags are
visible because they are printed on the page, which is the locatability rule working as
intended.

Values are normalised to the form the schema declares. A date is tagged as printed, with a
transformation format naming how to read it ("September 27, 2025" under
`ixt:date-monthname-day-year-en`); a state is sometimes tagged as its name rather than its
code. Rather than implement the transformation registry, the few shapes these four facts
actually take are parsed, and a value that fits none of them is left out, which drops the
filing with its reason named rather than guessing.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Final

import lxml.etree
import lxml.html
from lxml.html import HtmlElement

from smallprint.data.pair import STATE_NAMES
from smallprint.data.xbrl import Fact
from smallprint.schema import SCHEMA

#: The schema's cover-page concepts. Anything else tagged `dei:` is not read.
COVER_CONCEPTS: Final[frozenset[str]] = frozenset(
    c for spec in SCHEMA for c in spec.concepts if c.startswith("dei:")
)

_MONTHS: Final[dict[str, int]] = {
    name: number
    for number, full in enumerate(
        (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ),
        start=1,
    )
    for name in (full, full[:3], "sept" if number == 9 else full[:3])
}  # fmt: skip

_MONTH_DAY_YEAR = re.compile(r"^([a-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$")
_DAY_MONTH_YEAR = re.compile(r"^(\d{1,2})\s+([a-z]+)\.?,?\s+(\d{4})$")
_NUMERIC_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_SPACES = re.compile(r"\s+")
_SPACED_COMMA = re.compile(r"\s+,")

_CODE_OF_STATE: Final[dict[str, str]] = {
    name.casefold(): code for code, name in STATE_NAMES.items()
}


def parse_date(text: str) -> dt.date | None:
    """A cover-page date in any of the shapes filers tag it in, or None."""
    value = _SPACES.sub(" ", text.replace(chr(0xA0), " ")).strip().casefold()
    # "December 31 , 2024", with the comma set apart, is on real cover pages.
    value = _SPACED_COMMA.sub(",", value)
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        pass
    try:
        if match := _MONTH_DAY_YEAR.match(value):
            month = _MONTHS.get(match.group(1))
            if month is not None:
                return dt.date(int(match.group(3)), month, int(match.group(2)))
        if match := _DAY_MONTH_YEAR.match(value):
            month = _MONTHS.get(match.group(2))
            if month is not None:
                return dt.date(int(match.group(3)), month, int(match.group(1)))
        if match := _NUMERIC_DATE.match(value):
            return dt.date(int(match.group(3)), int(match.group(1)), int(match.group(2)))
    except ValueError:
        return None
    return None


def _normalise(concept: str, text: str) -> str | None:
    value = _SPACES.sub(" ", text.replace(chr(0xA0), " ")).strip()
    if not value:
        return None
    match concept:
        case "dei:DocumentPeriodEndDate":
            day = parse_date(value)
            return day.isoformat() if day is not None else None
        case "dei:DocumentFiscalPeriodFocus":
            return value.upper()
        case "dei:EntityIncorporationStateCountryCode":
            if len(value) == 2:
                return value.upper()
            return _CODE_OF_STATE.get(value.casefold())
        case _:
            return value


def _tag(element: HtmlElement) -> str:
    tag = element.tag
    return tag.lower() if isinstance(tag, str) else ""


def _contexts(root: HtmlElement) -> dict[str, tuple[dt.date | None, dt.date]]:
    """Context id to (start, end), for contexts without dimensions.

    A dimensional context qualifies a fact to a class of shares or a segment. The cover
    facts this task reads are entity-wide, so a dimensional one is not the one wanted.
    """
    out: dict[str, tuple[dt.date | None, dt.date]] = {}
    for context in root.iter():
        if _tag(context) != "xbrli:context":
            continue
        context_id = context.get("id")
        if context_id is None:
            continue
        found: dict[str, str] = {}
        dimensional = False
        for child in context.iter():
            tag = _tag(child)
            if tag in ("xbrli:segment", "xbrli:scenario"):
                dimensional = True
            elif tag in ("xbrli:startdate", "xbrli:enddate", "xbrli:instant"):
                found[tag] = (child.text or "").strip()
        if dimensional:
            continue
        try:
            if "xbrli:instant" in found:
                out[context_id] = (None, dt.date.fromisoformat(found["xbrli:instant"]))
            elif "xbrli:startdate" in found and "xbrli:enddate" in found:
                out[context_id] = (
                    dt.date.fromisoformat(found["xbrli:startdate"]),
                    dt.date.fromisoformat(found["xbrli:enddate"]),
                )
        except ValueError:
            continue
    return out


def cover_facts(document: bytes, *, accession: str, form: str, filed: dt.date) -> list[Fact]:
    """The schema's `dei` facts as this filing tagged them, in document order."""
    try:
        root = lxml.html.document_fromstring(document)
    except (lxml.etree.ParserError, ValueError):
        return []
    contexts = _contexts(root)
    facts: list[Fact] = []
    for element in root.iter():
        if _tag(element) != "ix:nonnumeric":
            continue
        concept = element.get("name")
        if concept not in COVER_CONCEPTS:
            continue
        period = contexts.get(element.get("contextref") or "")
        if period is None:
            continue
        text = " ".join(t for t in element.itertext() if isinstance(t, str))
        value = _normalise(concept, text)
        if value is None:
            continue
        facts.append(
            Fact(
                concept=concept,
                unit="",
                value=value,
                start=period[0],
                end=period[1],
                accession=accession,
                form=form,
                filed=filed,
            )
        )
    return facts
