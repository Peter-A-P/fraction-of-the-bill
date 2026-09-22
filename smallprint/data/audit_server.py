"""A faster hand audit: every label beside the line of the filing it was read from.

The Markdown pages make the auditor do the slow part, which is finding each of fifteen
numbers in six thousand characters. Most of the time goes there, not into judging. So this
does the finding: for every label it locates the row of the statement where that value is
printed, shows the row with the number marked, and says which column it sat in. Checking a
field becomes reading one line, and a filing with nothing wrong is one keystroke.

It also points at what is worth a second look, without deciding anything: a value found
only in a later column (a prior period?), a loss not printed in parentheses, a label that
could not be found in its own section at all, a null whose reason should be confirmed.
These are hints for a person. The verdict is still theirs, and the error rate this audit
publishes is only worth anything because a person gave it.

It is a local server rather than a published page because the verdicts belong in
`data/audit/audit.csv` beside the build, written as they are given, so an audit stopped
halfway resumes where it stopped. It listens on the loopback interface only.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import tempfile
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from smallprint.data.audit import AUDIT_COLUMNS, VERDICTS, read_sheet, wilson
from smallprint.data.build import SplitItem
from smallprint.data.pair import NULL_WHEN_LINE_ABSENT, STATE_NAMES, date_forms
from smallprint.data.statements import scale_heading
from smallprint.grade import normalise_categorical, tolerance
from smallprint.schema import FIELDS, SCHEMA, FieldKind, FieldSpec, Section

_HEADERS: Final[dict[str, Section]] = {
    "[COVER PAGE]": Section.COVER,
    "[INCOME STATEMENT]": Section.INCOME,
    "[BALANCE SHEET]": Section.BALANCE,
    "[AUDITOR]": Section.AUDITOR,
}
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
#: Fields whose sign carries meaning, so a label disagreeing with the parentheses around
#: it is worth a look. Cost of revenue is not one: the schema asks for it positive, and
#: filers print it in parentheses as an expense, so the disagreement there is by design.
_SIGNED: Final[frozenset[str]] = frozenset(
    {"operating_income", "net_income", "eps_basic", "eps_diluted", "stockholders_equity"}
)
#: A cell that prints zero as a dash, which is how statements print a zero. The dollar sign
#: is sometimes glued straight to an en or em dash with no space at all, a dollar sign
#: immediately followed by U+2013 or U+2014.
_DASHES: Final[frozenset[str]] = frozenset(
    {"-", "$-", "$ -", chr(0x2013), chr(0x2014), f"${chr(0x2013)}", f"${chr(0x2014)}"}
)

#: How many places a label is shown found, closest to it first. More is noise: a value that
#: recurs five times is a subtotal repeated down the statement.
_MAX_MATCHES: Final = 3


class Match(BaseModel):
    """One place in the input where a label's value is printed."""

    model_config = ConfigDict(frozen=True)

    line: str
    #: The matched characters, within `line`.
    start: int
    end: int
    #: The same, within the whole input text, so the page can mark it there too.
    text_start: int
    text_end: int
    #: 1 for the first value column of a table row, 2 for the next; None outside a table.
    column: int | None
    parenthesised: bool


class FieldEvidence(BaseModel):
    """What the auditor needs to judge one field, in one row of the page."""

    model_config = ConfigDict(frozen=True)

    field: str
    section: Section
    label: str | None
    #: The label as the page would print it after the reporting scale, for a number.
    printed_as: str | None
    #: The heading line that prints the scale, "(in thousands, except per share amounts)",
    #: since the row the value is found on does not say it.
    scale_line: Match | None = None
    matches: tuple[Match, ...]
    #: Things worth a second look. Never a verdict.
    flags: tuple[str, ...]


def _lines(text: str) -> list[tuple[int, Section | None, str]]:
    """Every line with its offset in the text and the section it sits in."""
    out, offset, section = [], 0, None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in _HEADERS:
            section = _HEADERS[stripped]
        else:
            out.append((offset, section, line))
        offset += len(line) + 1
    return out


def _column(line: str, position: int) -> int | None:
    if " | " not in line:
        return None
    return line[:position].count(" | ")


def _scale_for(spec: FieldSpec, item: SplitItem) -> float:
    context = item.item.context
    match spec.kind:
        case FieldKind.PER_SHARE:
            return 1.0
        case FieldKind.SHARE_COUNT:
            return context.share_scale if context.share_scale is not None else context.scale
        case _:
            return context.scale


_NAMES: Final[dict[Section, str]] = {
    Section.INCOME: "income statement",
    Section.BALANCE: "balance sheet",
}

#: A scale word in a parenthesised heading, "(thousands, except per share data)". Looser
#: than the build's reading, and only used to show the reader where a scale is printed.
_LOOSE_SCALE = re.compile(r"\((?:[^)]*?\b)?(thousands|millions|billions)\b", re.IGNORECASE)


def _scale_line(spec: FieldSpec, item: SplitItem) -> tuple[Match, Section] | None:
    """The line that prints the field's scale, and the section it is in.

    Its own section first. A statement that prints none was given the other statement's
    scale by the build, and the reader is shown that one.
    """
    kinds = [False]
    if spec.kind is FieldKind.SHARE_COUNT and item.item.context.share_scale is not None:
        kinds = [True, False]  # a scale of the shares' own, else the heading's
    others = [s for s in (Section.INCOME, Section.BALANCE) if s is not spec.section]
    lines = _lines(item.item.text)
    for section in [spec.section, *others]:
        for shares in kinds:
            for loose in (False, True):
                for offset, where, line in lines:
                    if where is not section:
                        continue
                    found = (
                        _LOOSE_SCALE.search(line) if loose else scale_heading(line, shares=shares)
                    )
                    if found is not None:
                        match = Match(
                            line=line,
                            start=found.start(),
                            end=found.end(),
                            text_start=offset + found.start(),
                            text_end=offset + found.end(),
                            column=None,
                            parenthesised=False,
                        )
                        return match, section
    return None


def _number(value: float) -> str:
    """Two decimal places, more if that would print a nonzero value as zero.

    A per-share loss of four tenths of a cent, $(0.004) as the filing prints it, must not
    round to the same "0.00" a genuine nil would show: the reader would see a value the
    label does not report at all.
    """
    if float(value).is_integer():
        return f"{value:,.0f}"
    places = 2
    while places < 6:
        text = f"{value:,.{places}f}"
        if value == 0.0 or float(text.replace(",", "")) != 0.0:
            return text
        places += 1
    return text


def _find_number(spec: FieldSpec, truth: float, item: SplitItem) -> list[Match]:
    scale = _scale_for(spec, item)
    allowed = tolerance(spec, truth, item.item.context)
    target = abs(truth)
    found: list[tuple[float, Match]] = []
    for offset, section, line in _lines(item.item.text):
        if section is not spec.section:
            continue
        if target == 0:
            found.extend((0.0, m) for m in _dash_cells(line, offset))
        for number in _NUMBER.finditer(line):
            value = float(number.group(0).replace(",", "")) * scale
            if abs(value - target) > allowed:
                continue
            before = line[: number.start()].rstrip().rstrip("$").rstrip()
            found.append(
                (
                    abs(value - target),
                    Match(
                        line=line,
                        start=number.start(),
                        end=number.end(),
                        text_start=offset + number.start(),
                        text_end=offset + number.end(),
                        column=_column(line, number.start()),
                        parenthesised=before.endswith("("),
                    ),
                )
            )
    # Closest first, in page order among equals: within the tolerance a neighbouring row can
    # match too, "Basic" a line above "Diluted", and the exact one must not be cut off.
    found.sort(key=lambda hit: hit[0])
    return [m for _, m in found]


def _dash_cells(line: str, offset: int) -> list[Match]:
    """Table cells printing zero as a dash, the way the locatability filter counted them."""
    out, at = [], 0
    for index, cell in enumerate(line.split(" | ")):
        stripped = cell.strip()
        if index > 0 and stripped in _DASHES:
            start = at + cell.index(stripped)
            out.append(
                Match(
                    line=line,
                    start=start,
                    end=start + len(stripped),
                    text_start=offset + start,
                    text_end=offset + start + len(stripped),
                    column=index,
                    parenthesised=False,
                )
            )
        at += len(cell) + 3
    return out


def _find_text(spec: FieldSpec, forms: Sequence[str], item: SplitItem) -> list[Match]:
    found = []
    for offset, section, line in _lines(item.item.text):
        if section is not spec.section:
            continue
        for form in forms:
            # Whole words only, or a state code is found inside "INC" and "incorporation".
            hit = _whole(form).search(line)
            if hit is not None:
                at = hit.start()
                found.append(
                    Match(
                        line=line,
                        start=at,
                        end=at + len(form),
                        text_start=offset + at,
                        text_end=offset + at + len(form),
                        column=_column(line, at),
                        parenthesised=False,
                    )
                )
                break
    return found


def _whole(form: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(form) + r"(?!\w)", re.IGNORECASE)


def _loose(form: str) -> re.Pattern[str]:
    """A printed form, allowing what real cover pages do to it: a comma set apart from the
    day ("March 31 , 2025"), and a line break or a run of spaces between the words."""
    words = [re.escape(w) for w in form.replace(",", " , ").split()]
    return re.compile(r"\s*".join(words), re.IGNORECASE)


def _find_date(spec: FieldSpec, day: dt.date, item: SplitItem) -> list[Match]:
    """A date anywhere in its section, across a line break if it was printed across one.

    The same leniency the locatability filter applied when it kept the filing, so a date
    the build found is a date the page shows.
    """
    text = item.item.text
    lines = _lines(text)
    spans = [(o, o + len(line)) for o, section, line in lines if section is spec.section]
    if not spans:
        return []
    start, end = spans[0][0], spans[-1][1]
    for form in date_forms(day):
        hit = _loose(form).search(text, start, end)
        if hit is None:
            continue
        for offset, _, line in lines:
            if offset <= hit.start() <= offset + len(line):
                within = hit.start() - offset
                return [
                    Match(
                        line=line,
                        start=within,
                        end=min(len(line), within + (hit.end() - hit.start())),
                        text_start=hit.start(),
                        text_end=hit.end(),
                        column=_column(line, within),
                        parenthesised=False,
                    )
                ]
    return []


def _find_name(spec: FieldSpec, name: str, item: SplitItem) -> list[Match]:
    """A name as printed, allowing the punctuation and suffixes the grader also allows."""
    direct = _find_text(spec, [name], item)
    if direct:
        return direct
    wanted = normalise_categorical(name)
    return [
        Match(
            line=line,
            start=0,
            end=len(line),
            text_start=offset,
            text_end=offset + len(line),
            column=None,
            parenthesised=False,
        )
        for offset, section, line in _lines(item.item.text)
        if section is spec.section and wanted and _whole(wanted).search(normalise_categorical(line))
    ]


def evidence_for(spec: FieldSpec, item: SplitItem) -> FieldEvidence:
    truth = getattr(item.item.truth, spec.name)
    flags: list[str] = []
    matches: list[Match] = []
    printed_as = None
    scale_line = None

    if truth is None:
        if spec.name in item.item.not_on_page:
            flags.append("Null: tagged in the XBRL but the statement prints no line for it.")
        elif spec.name in NULL_WHEN_LINE_ABSENT:
            flags.append("Null: the statement prints no line for it.")
        else:
            flags.append("Null: the filing reports no such fact. Check nothing on the page is it.")
        label = None
    elif spec.kind in (FieldKind.MONETARY, FieldKind.PER_SHARE, FieldKind.SHARE_COUNT):
        value = float(truth)
        scale = _scale_for(spec, item)
        label = _number(value)
        if scale != 1.0:
            printed_as = f"{_number(value / scale)} at a scale of {_number(scale)}"
            heading = _scale_line(spec, item)
            if heading is None:
                flags.append("No scale printed anywhere. Check the scale by hand.")
            else:
                scale_line, where = heading
                if where is not spec.section:
                    flags.append(f"Its statement prints no scale; the {_NAMES[where]}'s is used.")
        matches = _find_number(spec, value, item)
        if not matches:
            flags.append("Not found in its section. Look for it by hand.")
        elif all(m.column is not None and m.column > 1 for m in matches):
            flags.append(
                f"Found only in column {matches[0].column}: check that is the current period."
            )
        if spec.name in _SIGNED and matches:
            if value < 0 and not any(m.parenthesised for m in matches):
                flags.append("A negative label, but not printed in parentheses where found.")
            if value > 0 and all(m.parenthesised for m in matches):
                flags.append("A positive label, but printed in parentheses where found.")
    elif spec.kind is FieldKind.DATE:
        assert isinstance(truth, dt.date)
        label = truth.isoformat()
        matches = _find_date(spec, truth, item)
        if not matches:
            flags.append("Date not found on the cover page as printed.")
    elif spec.name == "fiscal_period":
        label = str(truth)
        annual = item.item.form.startswith("10-K")
        if annual != (truth == "FY"):
            flags.append(f"A {item.item.form} labelled {truth}.")
    else:
        label = str(truth)
        names = [label]
        if spec.name == "state_of_incorporation" and label.upper() in STATE_NAMES:
            names.insert(0, STATE_NAMES[label.upper()])  # cover pages print the name
        for name in names:
            matches = _find_name(spec, name, item)
            if matches:
                break
        if not matches:
            flags.append("Not found in its section as printed.")

    return FieldEvidence(
        field=spec.name,
        section=spec.section,
        label=label,
        printed_as=printed_as,
        scale_line=scale_line,
        matches=tuple(matches[:_MAX_MATCHES]),
        flags=tuple(flags),
    )


def evidence(item: SplitItem) -> list[FieldEvidence]:
    return [evidence_for(spec, item) for spec in SCHEMA]


# -- the verdict sheet ------------------------------------------------------------------


def save_verdict(sheet: Path, n: int, verdict: str, wrong_fields: Sequence[str], note: str) -> None:
    """Write one verdict into the sheet, atomically, refusing anything the report would.

    The whole file is rewritten through a temporary file and a rename, so a crash midway
    leaves the previous sheet intact rather than half of one. Verdicts already given are
    the one thing in this project that cannot be regenerated.
    """
    verdict = verdict.strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be ok or wrong, got {verdict!r}")
    unknown = [f for f in wrong_fields if f not in FIELDS]
    if unknown:
        raise ValueError(f"not schema fields: {unknown}")
    if verdict == "wrong" and not wrong_fields:
        raise ValueError("a wrong verdict has to name the wrong fields")
    if verdict == "ok" and wrong_fields:
        raise ValueError("an ok verdict cannot name wrong fields")
    rows = read_sheet(sheet)
    target = [r for r in rows if int(r["n"]) == n]
    if not target:
        raise KeyError(f"no row {n} in {sheet}")
    target[0].update(
        {"verdict": verdict, "wrong_fields": " ".join(wrong_fields), "note": note.strip()}
    )
    handle, temporary = tempfile.mkstemp(dir=sheet.parent, prefix=".audit-", suffix=".csv")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=AUDIT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, sheet)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def progress(sheet: Path) -> dict[str, Any]:
    rows = read_sheet(sheet)
    done = [r for r in rows if r["verdict"].strip()]
    wrong = sum(1 for r in done if r["verdict"].strip().lower() == "wrong")
    pending = [int(r["n"]) for r in rows if not r["verdict"].strip()]
    rate = wilson(wrong, len(done)) if done else None
    return {
        "total": len(rows),
        "done": len(done),
        "wrong": wrong,
        "first_pending": pending[0] if pending else None,
        "rate": None if rate is None else {"point": rate.point, "low": rate.low, "high": rate.high},
    }


# -- the server -------------------------------------------------------------------------


class AuditApp:
    """The state the handler reads: the sampled items and where verdicts go."""

    def __init__(self, sheet: Path, items: Sequence[SplitItem]) -> None:
        self.sheet = sheet
        rows = read_sheet(sheet)
        by_id = {s.item.item_id: s for s in items}
        missing = [r["item_id"] for r in rows if r["item_id"] not in by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} audited items are not in this build, first {missing[0]}; "
                "point --build-dir at the build the sample was drawn from"
            )
        self.items = {int(r["n"]): by_id[r["item_id"]] for r in rows}

    def item(self, n: int) -> dict[str, Any]:
        s = self.items[n]
        row = next(r for r in read_sheet(self.sheet) if int(r["n"]) == n)
        fields = evidence(s)
        return {
            "n": n,
            "total": len(self.items),
            "item_id": s.item.item_id,
            "form": s.item.form,
            "fiscal_period": s.item.fiscal_period,
            "period_end": s.item.period_end.isoformat(),
            "filed": s.item.filed.isoformat(),
            "split": s.split.value,
            "scale": s.item.context.scale,
            "share_scale": s.item.context.share_scale,
            "verdict": row["verdict"],
            "wrong_fields": row["wrong_fields"].split(),
            "note": row["note"],
            "fields": [f.model_dump(mode="json") for f in fields],
            "text": s.item.text,
        }


def handler(app: AuditApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return  # the terminal is for the person running it, not for every request

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _n(self) -> int | None:
            match = re.fullmatch(r"/api/item/(\d+)", self.path)
            return int(match.group(1)) if match else None

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(progress(app.sheet))
            elif (n := self._n()) is not None and n in app.items:
                self._json(app.item(n))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            n = self._n()
            if n is None or n not in app.items:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                save_verdict(
                    app.sheet,
                    n,
                    str(body.get("verdict", "")),
                    [str(f) for f in body.get("wrong_fields", [])],
                    str(body.get("note", "")),
                )
            except (ValueError, KeyError) as problem:
                self._json({"error": str(problem)}, HTTPStatus.BAD_REQUEST)
                return
            self._json(progress(app.sheet))

    return Handler


def server(app: AuditApp, *, port: int = 8765) -> ThreadingHTTPServer:
    """Bound to the loopback interface only. Nothing here is for anyone else on the network."""
    return ThreadingHTTPServer(("127.0.0.1", port), handler(app))


PAGE: Final = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label audit</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--line:#e7e5e4;--card:#fff;--ok:#15803d;
--bad:#b91c1c;--badbg:#fee2e2;--warn:#a16207;--warnbg:#fef3c7;--mark:#fde68a;--accent:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--line:#44403c;
--card:#292524;--ok:#4ade80;--bad:#f87171;--badbg:#450a0a;--warn:#fbbf24;--warnbg:#422006;
--mark:#854d0e;--accent:#93c5fd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--line);
padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.bar{flex:1;min-width:160px;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent)}
main{max-width:1200px;margin:0 auto;padding:16px}
.meta{color:var(--muted);margin-bottom:10px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}
td,th{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{font-weight:600;color:var(--muted);font-size:12px}
tr.flagged td{background:var(--warnbg)}tr.wrong td{background:var(--badbg)}
tr.cursor{outline:2px solid var(--accent);outline-offset:-2px}
td.field{font-family:ui-monospace,monospace;white-space:nowrap;cursor:pointer}
td.label{font-family:ui-monospace,monospace;white-space:nowrap}
.line{font-family:ui-monospace,monospace;font-size:12.5px;white-space:pre-wrap;word-break:break-word}
mark{background:var(--mark);color:inherit;padding:0 1px;border-radius:2px}
.scale{color:var(--muted)}
.col{font-size:11px;color:var(--muted);margin-left:6px}
.flag{color:var(--warn);font-size:12.5px}
.sub{color:var(--muted);font-size:12px}
.actions{display:flex;gap:8px;margin:12px 0;flex-wrap:wrap;align-items:center}
button{font:inherit;padding:7px 12px;border-radius:6px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer}
button.ok{border-color:var(--ok);color:var(--ok)}button.bad{border-color:var(--bad);color:var(--bad)}
input.note{flex:1;min-width:200px;font:inherit;padding:7px;border:1px solid var(--line);
border-radius:6px;background:var(--card);color:var(--fg)}
details{margin-top:14px}pre.text{background:var(--card);border:1px solid var(--line);padding:12px;
font-size:12.5px;white-space:pre-wrap;word-break:break-word;max-height:70vh;overflow:auto}
.keys{color:var(--muted);font-size:12px}.status{font-weight:600}
.given{font-size:12px;padding:2px 8px;border-radius:10px;border:1px solid var(--line)}
</style></head><body>
<header><span class="status" id="count">Loading</span><div class="bar"><i id="fill"></i></div>
<span id="rate" class="sub"></span></header>
<main>
<div class="meta" id="meta"></div>
<table><thead><tr><th>Field</th><th>Label</th><th>Where it is printed</th></tr></thead>
<tbody id="rows"></tbody></table>
<div class="actions">
<button class="ok" id="okb">All correct (Enter)</button>
<button class="bad" id="badb">Save as wrong (Enter, with fields marked)</button>
<button id="prev">Previous (Left)</button><button id="next">Next (Right)</button>
<input class="note" id="note" placeholder="Note (N to focus, optional)">
</div>
<div class="keys">Enter saves: all correct if nothing is marked, otherwise wrong with the
marked fields. Click a field, or move with J and K and press Space, to mark it wrong.
Amber rows are hints worth a second look, never verdicts.</div>
<details id="full"><summary>The whole input, as the model sees it</summary>
<pre class="text" id="text"></pre></details>
</main>
<script>
let n=1,total=0,item=null,wrong=new Set(),cursor=0;
const $=id=>document.getElementById(id);
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function state(){const s=await (await fetch('/api/state')).json();total=s.total;
$('count').textContent=s.done+' of '+s.total+' audited';
$('fill').style.width=(100*s.done/s.total)+'%';
$('rate').textContent=s.rate?('Wrong so far: '+s.wrong+', '+(100*s.rate.point).toFixed(1)+
'% ('+(100*s.rate.low).toFixed(1)+' to '+(100*s.rate.high).toFixed(1)+'%)'):'';return s}
function marked(line,a,b){return esc(line.slice(0,a))+'<mark>'+esc(line.slice(a,b))+'</mark>'+
esc(line.slice(b))}
function render(){const rows=$('rows');rows.innerHTML='';
item.fields.forEach((f,i)=>{const tr=document.createElement('tr');
if(f.flags.length)tr.classList.add('flagged');if(wrong.has(f.field))tr.classList.add('wrong');
if(i===cursor)tr.classList.add('cursor');
const scale=f.scale_line?'<div class="line scale">'+marked(f.scale_line.line,
f.scale_line.start,f.scale_line.end)+'<span class="col">the scale, from the heading</span></div>':'';
const where=scale+f.matches.map(m=>'<div class="line">'+marked(m.line,m.start,m.end)+
(m.column?'<span class="col">column '+m.column+'</span>':'')+'</div>').join('');
const flags=f.flags.map(x=>'<div class="flag">'+esc(x)+'</div>').join('');
tr.innerHTML='<td class="field">'+(wrong.has(f.field)?'&#10007; ':'')+esc(f.field)+
'<div class="sub">'+f.section+'</div></td><td class="label">'+(f.label===null?'null':esc(f.label))+
(f.printed_as?'<div class="sub">'+esc(f.printed_as)+'</div>':'')+'</td><td>'+where+flags+'</td>';
tr.querySelector('td.field').onclick=()=>{cursor=i;toggle(f.field)};
tr.onclick=e=>{if(!e.target.closest('td.field')&&f.matches.length)jump(f.matches[0])};
rows.appendChild(tr)});
const spans=item.fields.flatMap(f=>f.matches).sort((a,b)=>a.text_start-b.text_start);
let out='',at=0;for(const m of spans){if(m.text_start<at)continue;
out+=esc(item.text.slice(at,m.text_start))+'<mark id="t'+m.text_start+'">'+
esc(item.text.slice(m.text_start,m.text_end))+'</mark>';at=m.text_end}
$('text').innerHTML=out+esc(item.text.slice(at))}
function jump(m){$('full').open=true;const el=document.getElementById('t'+m.text_start);
if(el)el.scrollIntoView({block:'center'})}
function toggle(f){wrong.has(f)?wrong.delete(f):wrong.add(f);render()}
async function load(k){if(k<1||k>total)return;n=k;item=await (await fetch('/api/item/'+n)).json();
wrong=new Set(item.wrong_fields);cursor=0;$('note').value=item.note||'';
const given=item.verdict?' <span class="given">given: '+item.verdict+'</span>':'';
$('meta').innerHTML='<b>'+n+' of '+item.total+'</b> &middot; '+esc(item.item_id)+' &middot; '+
item.form+' '+item.fiscal_period+', period ended '+item.period_end+' &middot; filed '+item.filed+
' &middot; '+item.split+' &middot; money scale '+item.scale.toLocaleString()+
(item.share_scale?', shares '+item.share_scale.toLocaleString():'')+given;
render();window.scrollTo(0,0)}
async function save(){const verdict=wrong.size?'wrong':'ok';
const r=await fetch('/api/item/'+n,{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({verdict,wrong_fields:[...wrong],note:$('note').value})});
if(!r.ok){alert((await r.json()).error);return}const s=await state();
if(s.first_pending)await load(s.first_pending);else{await load(n);
$('count').textContent='All '+s.total+' audited. Run smallprint data audit-report.'}}
$('okb').onclick=()=>{wrong.clear();save()};$('badb').onclick=()=>{if(wrong.size)save()};
$('prev').onclick=()=>load(n-1);$('next').onclick=()=>load(n+1);
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'){
if(e.key==='Enter'){e.preventDefault();e.target.blur();save()}
if(e.key==='Escape')e.target.blur();return}
if(e.key==='Enter'){e.preventDefault();save()}
else if(e.key==='ArrowRight')load(n+1);else if(e.key==='ArrowLeft')load(n-1);
else if(e.key==='j'){cursor=Math.min(item.fields.length-1,cursor+1);render()}
else if(e.key==='k'){cursor=Math.max(0,cursor-1);render()}
else if(e.key===' '){e.preventDefault();toggle(item.fields[cursor].field)}
else if(e.key==='n'){e.preventDefault();$('note').focus()}});
(async()=>{const s=await state();await load(s.first_pending||1)})();
</script></body></html>
"""
