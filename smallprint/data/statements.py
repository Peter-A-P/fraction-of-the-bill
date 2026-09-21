"""Finding the primary statements inside a filing document, and the scale they are printed in.

A 10-K is a few hundred pages of HTML in which the income statement and the balance sheet are
two tables among several hundred. Most of the others carry the same line items: the summary
in management's discussion, the selected financial data, the segment note, the quarterly
table, the parent-only schedule. The model is shown the statements, not the document, so this
module decides which tables those are, and a wrong decision here becomes a wrong label that
no amount of training or grading can recover. The hand audit of 200 pairs before training
exists mainly to check this module.

A table is taken as a statement by what it contains, not only by its title. Titles vary more
than line items do ("statements of operations", "of income", "of earnings", "of financial
position"), and a statement whose title sits in a page header the parser could not attach to
it is still a statement. The title counts in its favour when it is present, and a table
titled as a *different* statement is never taken, which is what keeps the cash flow
statement, with its net income and its cash and cash equivalents, out of both.

Two things are removed before anything is read. The inline XBRL header, `ix:header`, is a
hidden block that carries the filing's own tagged facts, including the cover-page facts that
are this task's labels; leaving it in would put the answers in the question. And anything
styled `display:none`, which is hidden for the same kind of reason.

The reporting scale is read from the statement's own heading, "(in thousands, except per
share data)" and its many variants, because the grader needs it: a statement in thousands
cannot tell two values four hundred dollars apart, so neither can the grade. Shares have a
scale of their own, because "except share data" and "shares in thousands" say different
things about the same row.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

import lxml.html
from lxml.html import HtmlElement
from pydantic import BaseModel, ConfigDict

from smallprint.schema import Section

#: Elements whose content is never shown to a reader of the filing.
_INVISIBLE_TAGS: Final[frozenset[str]] = frozenset(
    {"head", "script", "style", "title", "ix:header", "noscript"}
)
_DISPLAY_NONE = re.compile(r"display\s*:\s*none", re.IGNORECASE)

#: Elements that start a new line of text when rendered.
_BLOCK_TAGS: Final[frozenset[str]] = frozenset(
    {
        "p", "div", "br", "hr", "li", "ul", "ol", "center", "section", "blockquote", "pre",
        "h1", "h2", "h3", "h4", "h5", "h6", "body",
    }
)  # fmt: skip

#: The whitespace a filing uses that `str.split` does not all treat as a separator the same
#: way a browser does: no-break space, zero-width space, and the narrow no-break space.
_SPACES = re.compile("[\\s\u00a0\u200b\u202f]+")

#: Cells that belong to their neighbour. Filings set the dollar sign and the closing
#: parenthesis of a negative number in cells of their own so they align in a column.
_JOIN_RIGHT: Final[frozenset[str]] = frozenset({"$", "(", "$(", "($"})
_JOIN_LEFT: Final[frozenset[str]] = frozenset({")", "%", ")%", "%)"})

_APOSTROPHE = "['\u2019]?"

#: Statement titles. A table whose heading names one of these is that statement or nothing.
_TITLES: Final[dict[str, re.Pattern[str]]] = {
    "income": re.compile(
        r"statements?\s+of\s+(?:consolidated\s+)?(?:operations|income|earnings|(?:net\s+)?loss)"
        r"|income\s+statements?",
        re.IGNORECASE,
    ),
    "comprehensive": re.compile(r"statements?\s+of\s+comprehensive", re.IGNORECASE),
    "balance": re.compile(
        r"balance\s+sheets?"
        r"|statements?\s+of\s+(?:consolidated\s+)?financial\s+(?:position|condition)"
        r"|statements?\s+of\s+condition",
        re.IGNORECASE,
    ),
    "cash": re.compile(r"statements?\s+of\s+(?:consolidated\s+)?cash\s+flows?", re.IGNORECASE),
    "equity": re.compile(
        rf"statements?\s+of\s+(?:consolidated\s+)?(?:changes\s+in\s+)?"
        rf"(?:stockholders|shareholders|members|partners){_APOSTROPHE}\s+(?:equity|deficit)",
        re.IGNORECASE,
    ),
}

#: The lines a statement carries. The first pattern of each is the anchor: a table without it
#: is not that statement however many of the others it has.
_SIGNATURES: Final[dict[Section, tuple[re.Pattern[str], ...]]] = {
    Section.INCOME: tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            # "Net (loss) income" is a common label, parentheses and all.
            r"net\s+\(?(?:income|loss|earnings)",
            r"per\s+(?:common\s+|basic\s+|diluted\s+)?share|per[\s-]share",
            r"\brevenues?\b|\bnet\s+sales\b|\bsales\b",
            r"operating\s+(?:income|loss|expenses)|(?:income|loss)\s+from\s+operations",
            r"income\s+tax|provision\s+for\s+(?:income\s+)?taxes",
            r"weighted[\s-]+average",
        )
    ),
    Section.BALANCE: tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"total\s+assets",
            r"total\s+liabilities",
            rf"(?:stockholders|shareholders){_APOSTROPHE}\s+(?:equity|deficit)",
            r"cash\s+and\s+cash\s+equivalents",
            r"total\s+current\s+(?:assets|liabilities)",
            r"retained\s+earnings|accumulated\s+deficit",
        )
    ),
}

#: A table has to match at least this many of its statement's signatures, anchor included.
_MIN_SIGNATURES: Final = 3
#: How much a matching title is worth against the line-item count.
_TITLE_WEIGHT: Final = 3
#: A statement has numbers. A table of contents that lists every statement by name does not.
_MIN_NUMERIC_CELLS: Final = 5

#: Short text blocks immediately above a table are its heading: company name, title, the
#: period, the scale. A long block is prose, and ends the heading.
_HEADING_BLOCKS: Final = 4
_HEADING_MAX_CHARS: Final = 300
#: How many of a table's first rows can carry its title and scale. ONEOK prints company,
#: title, period, years and then the scale, all as rows of the statement table.
_TITLE_ROWS: Final = 6

_NUMERIC_CELL = re.compile(r"^[($\-\u2212\s]*\d[\d,]*(?:\.\d+)?\s*\)?%?$")

_SCALE_WORDS: Final[dict[str, float]] = {
    "thousands": 1e3,
    "millions": 1e6,
    "billions": 1e9,
}
_MONEY_SCALE = re.compile(
    r"\bin\s+(thousands|millions|billions)\b"
    # "(Millions of dollars, except per share amounts)", ONEOK's form.
    r"|\b(thousands|millions|billions)\s+of\s+(?:u\.?s\.?\s+)?dollars\b",
    re.IGNORECASE,
)
_THOUSANDS_OMITTED = re.compile(r"\b000['\u2019]?s\s+omitted|\bin\s+\$\s*000['\u2019]?s\b", re.I)
_SHARE_SCALE = re.compile(
    r"\bshares?\b[^)]{0,40}?\bin\s+(thousands|millions|billions)\b", re.IGNORECASE
)
_SHARE_ROW = re.compile(r"\bshares\b", re.IGNORECASE)
_LABEL_SCALE = re.compile(
    r"\((?:in\s+)?(thousands|millions)\)|\bin\s+(thousands|millions)\b"
    rf"|\(000{_APOSTROPHE}s\)",
    re.IGNORECASE,
)
_INTRODUCING_SENTENCE = re.compile(
    r"\bthe\s+following\s+table\b|\bas\s+a\s+percentage\s+of\b|:\s*$", re.IGNORECASE
)
_EXCEPT_CLAUSE = re.compile(r"\bexcept\b([^)]*)", re.IGNORECASE)
_SHARE_WORD = re.compile(r"(per[\s-]+)?\bshares?\b", re.IGNORECASE)

#: Where the cover page ends. Ten-Ks put a table of contents after it; the budget catches the
#: rest.
_COVER_END = re.compile(r"^\s*(?:table\s+of\s+contents|index)\s*$|^\s*part\s+i\b", re.I)
_COVER_BUDGET_CHARS: Final = 5000
#: Enough cover text has been read that a heading like "Index" is the end and not the start.
_COVER_MIN_CHARS: Final = 200

#: The audit report's signature block sits around this sentence, which the PCAOB has
#: required in every audit report since 2017.
#: The wording varies: "the Company's auditor", "Aditxt's auditor", "the auditor of the
#: Company", "the Company's independent registered public accounting firm". Found on the
#: 300-company build, where the one fixed phrase missed a third of annual reports' auditors.
_AUDITOR_TENURE = re.compile(
    r"served\s+as\s+[^.]{0,80}?(?:auditors?|accounting\s+firm)\s+(?:of\s+[^.]{0,40}?\s+)?since",
    re.IGNORECASE,
)
#: How far around the tenure sentence the signature block reaches, and how long a line in
#: it may be. The firm's name, city, date and PCAOB ID are short lines; the opinion and the
#: critical audit matters above them are paragraphs, and are left out.
_AUDIT_WINDOW_BEFORE: Final = 8
_AUDIT_WINDOW_AFTER: Final = 4
_AUDIT_LINE_MAX_CHARS: Final = 160


class Block(BaseModel):
    """One piece of a filing in reading order: a run of text, or a table."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    rows: tuple[tuple[str, ...], ...] | None = None

    @property
    def is_table(self) -> bool:
        return self.rows is not None

    def render(self) -> str:
        if self.rows is None:
            return self.text
        return "\n".join(" | ".join(row) for row in self.rows)


class Statement(BaseModel):
    """A located statement, rendered as the model sees it."""

    model_config = ConfigDict(frozen=True)

    section: Section
    heading: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    #: The monetary scale printed in the heading, 1.0 when none is printed.
    scale: float
    #: Whether a scale was printed at all. An unprinted scale is assumed to be whole units,
    #: and a filing where that assumption is wrong fails the locatability filter rather than
    #: being labelled a thousand times too small.
    scale_declared: bool
    #: The scale share counts are printed in, when the heading gives them one of their own.
    #: None means shares follow `scale`.
    share_scale: float | None

    def render(self) -> str:
        lines = [*self.heading, *(" | ".join(row) for row in self.rows)]
        return "\n".join(lines)


class LocatedFiling(BaseModel):
    """The parts of a filing the model is shown, found and rendered."""

    model_config = ConfigDict(frozen=True)

    cover: str
    income: Statement | None
    balance: Statement | None
    #: The audit report's signature block. Empty in a quarterly report, which is not audited.
    auditor: str

    def section_text(self, section: Section) -> str:
        match section:
            case Section.COVER:
                return self.cover
            case Section.INCOME:
                return self.income.render() if self.income is not None else ""
            case Section.BALANCE:
                return self.balance.render() if self.balance is not None else ""
            case Section.AUDITOR:
                return self.auditor

    def render(self) -> str:
        """The model's input: every section, labelled, in a fixed order.

        The labels name the section and nothing else. The scale is left where the filing
        printed it, because reading it is part of the task.
        """
        parts = []
        for section, label in (
            (Section.COVER, "COVER PAGE"),
            (Section.INCOME, "INCOME STATEMENT"),
            (Section.BALANCE, "BALANCE SHEET"),
            (Section.AUDITOR, "AUDITOR"),
        ):
            text = self.section_text(section)
            if text:
                parts.append(f"[{label}]\n{text}")
        return "\n\n".join(parts)


def _clean(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def _tag(element: HtmlElement) -> str:
    tag = element.tag
    return tag.lower() if isinstance(tag, str) else ""


def _hidden(element: HtmlElement) -> bool:
    # Comments and processing instructions are nodes whose tag is not a string. Their
    # content is never shown; the text after them, their tail, belongs to the parent and is
    # still read. On the 300-company build one filing agent's comments ("Field: Set; Name:
    # xdx; ...") were read as text and pushed the cover page past its real start.
    if not isinstance(element.tag, str):
        return True
    if _tag(element) in _INVISIBLE_TAGS:
        return True
    style = element.get("style")
    return style is not None and _DISPLAY_NONE.search(style) is not None


def _visible_text(element: HtmlElement) -> str:
    """Text content without the parts a reader cannot see."""
    parts: list[str] = []

    def walk(node: HtmlElement) -> None:
        if _hidden(node):
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(element)
    return _clean(" ".join(parts))


def _render_rows(table: HtmlElement) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for tr in table.iter("tr"):
        if _hidden(tr):
            continue
        cells: list[str] = []
        pending = ""
        for cell in tr:
            if _tag(cell) not in ("td", "th") or _hidden(cell):
                continue
            text = _visible_text(cell)
            if not text:
                continue
            if text in _JOIN_RIGHT:
                pending += text
                continue
            if text in _JOIN_LEFT and cells:
                cells[-1] += text
                continue
            cells.append(pending + text)
            pending = ""
        if pending and cells:
            cells[-1] += pending
        if cells:
            rows.append(tuple(cells))
    return tuple(rows)


def blocks(document: bytes) -> list[Block]:
    """The filing in reading order, as text blocks and tables, with hidden content removed."""
    root = lxml.html.document_fromstring(document)
    out: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        text = _clean(" ".join(buffer))
        buffer.clear()
        if text:
            out.append(Block(text=text))

    def walk(node: HtmlElement) -> None:
        if _hidden(node):
            return
        tag = _tag(node)
        if tag == "table":
            flush()
            rows = _render_rows(node)
            if rows:
                out.append(Block(rows=rows))
            return
        is_block = tag in _BLOCK_TAGS
        if is_block:
            flush()
        if node.text:
            buffer.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                buffer.append(child.tail)
        if is_block:
            flush()

    walk(root)
    flush()
    return out


def _heading(items: Sequence[Block], index: int) -> tuple[str, ...]:
    """The short text blocks directly above a table, nearest last.

    A sentence introducing the table ends the heading: "The following table sets forth the
    components of our Consolidated Statements of Income as a percentage of net sales:" names
    a statement without being one, and on the 300-company build it made a discussion-section
    table outscore the statement it describes.
    """
    lines: list[str] = []
    for block in reversed(items[max(0, index - _HEADING_BLOCKS) : index]):
        if block.is_table or len(block.text) > _HEADING_MAX_CHARS:
            break
        if _INTRODUCING_SENTENCE.search(block.text):
            break
        lines.append(block.text)
    return tuple(reversed(lines))


def _share_label_scale(rows: Sequence[Sequence[str]]) -> float | None:
    """The scale printed in a share-count row's label, like "(in thousands)" or "(000's)"."""
    for row in rows:
        if not row or not _SHARE_ROW.search(row[0]):
            continue
        match = _LABEL_SCALE.search(row[0])
        if match is not None:
            # The third form, "(000's)", has no word and means thousands.
            word = match.group(1) or match.group(2) or "thousands"
            return _SCALE_WORDS[word.lower()]
    return None


def _titles(text: str) -> set[str]:
    return {kind for kind, pattern in _TITLES.items() if pattern.search(text)}


def _numeric_cells(rows: Sequence[Sequence[str]]) -> int:
    return sum(1 for row in rows for cell in row if _NUMERIC_CELL.match(cell))


def _statement_titles(section: Section, title_text: str, labels: str) -> set[str]:
    """The statements a table's heading names, with combined statements resolved.

    Many filers print one "statement of comprehensive income" that is the income statement
    with other comprehensive income appended. It carries per-share lines; a standalone
    comprehensive income statement, which starts from net income, does not. Deckers titles
    its income statement this way, and on the first live build the locator took the
    discussion section's results table instead.
    """
    titles = _titles(title_text)
    per_share = _SIGNATURES[Section.INCOME][1]
    if section is Section.INCOME and "comprehensive" in titles and per_share.search(labels):
        titles = (titles - {"comprehensive"}) | {"income"}
    return titles


def _score(section: Section, block: Block, heading: Sequence[str]) -> int | None:
    """How much a table looks like `section`, or None if it cannot be that statement."""
    assert block.rows is not None
    if _numeric_cells(block.rows) < _MIN_NUMERIC_CELLS:
        return None
    title_text = " ".join([*heading, *(" ".join(r) for r in block.rows[:3])])
    labels = " ".join(row[0] for row in block.rows if row)
    titles = _statement_titles(section, title_text, labels)
    own = "income" if section is Section.INCOME else "balance"
    if titles and own not in titles:
        # Titled as another statement. The cash flow statement has net income and cash and
        # cash equivalents; it is still the cash flow statement.
        return None

    signatures = _SIGNATURES[section]
    if not signatures[0].search(labels):
        return None
    matched = sum(1 for pattern in signatures if pattern.search(labels))
    if matched < _MIN_SIGNATURES:
        return None
    return matched + (_TITLE_WEIGHT if own in titles else 0)


def scale_heading(line: str, *, shares: bool = False) -> re.Match[str] | None:
    """Where a line prints the statement's scale, for showing a reader where it came from.

    With `shares`, only a scale the heading gives share counts of their own.
    """
    if shares:
        return _SHARE_SCALE.search(line)
    return _MONEY_SCALE.search(line) or _THOUSANDS_OMITTED.search(line)


def _scales(text: str) -> tuple[float, bool, float | None]:
    """The monetary scale, whether one was printed, and the share scale if it differs."""
    money = _MONEY_SCALE.search(text)
    if money is not None:
        word = money.group(1) or money.group(2)
        scale, declared = _SCALE_WORDS[word.lower()], True
    elif _THOUSANDS_OMITTED.search(text):
        scale, declared = 1e3, True
    else:
        scale, declared = 1.0, False

    share_scale: float | None = None
    explicit = _SHARE_SCALE.search(text)
    if explicit is not None:
        share_scale = _SCALE_WORDS[explicit.group(1).lower()]
    else:
        clause = _EXCEPT_CLAUSE.search(text)
        # "except share and per share data": shares are excepted from the scale and printed
        # whole. "except per share data" excepts only the per-share amounts, which are never
        # scaled anyway, and leaves shares at the statement's scale.
        if clause is not None and any(
            m.group(1) is None for m in _SHARE_WORD.finditer(clause.group(1))
        ):
            share_scale = 1.0
    return scale, declared, share_scale


def _continuation(
    section: Section, items: Sequence[Block], index: int
) -> tuple[tuple[str, ...], ...]:
    """Rows of the next table, when the statement runs on to it across a page break.

    Balance sheets especially: assets on one page, liabilities and equity on the next, with
    a page number and a repeated company name in between.
    """
    for j in range(index + 1, min(len(items), index + 2 + _HEADING_BLOCKS)):
        block = items[j]
        if not block.is_table:
            if len(block.text) > _HEADING_MAX_CHARS:
                return ()
            continue
        assert block.rows is not None
        heading = _heading(items, j)
        labels = " ".join(row[0] for row in block.rows if row)
        title_text = " ".join([*heading, *(" ".join(r) for r in block.rows[:3])])
        titles = _statement_titles(section, title_text, labels)
        own = "income" if section is Section.INCOME else "balance"
        if titles and own not in titles:
            return ()
        matched = sum(1 for pattern in _SIGNATURES[section] if pattern.search(labels))
        if matched >= 2 and _numeric_cells(block.rows) >= _MIN_NUMERIC_CELLS:
            return block.rows
        return ()
    return ()


def locate_statement(section: Section, items: Sequence[Block]) -> Statement | None:
    """The table that is this statement, or None.

    Ties go to the earliest table. A titled statement outscores every untitled table with
    the same lines, and among untitled ones the earliest is the least bad guess; the hand
    audit is where a wrong guess is caught.
    """
    if section not in _SIGNATURES:
        raise ValueError(f"{section.value} is not a statement")
    best: tuple[int, int] | None = None
    for index, block in enumerate(items):
        if not block.is_table:
            continue
        score = _score(section, block, _heading(items, index))
        if score is not None and (best is None or score > best[0]):
            best = (score, index)
    if best is None:
        return None

    index = best[1]
    block = items[index]
    assert block.rows is not None
    heading = _heading(items, index)
    # In an annual report the audit opinion runs straight into the statements, and its
    # closing date and city would otherwise read as part of the heading. When the title is
    # there, the heading starts at it.
    own_titles = (
        (_TITLES["income"], _TITLES["comprehensive"])
        if section is Section.INCOME
        else (_TITLES["balance"],)
    )

    def is_title(line: str) -> bool:
        return any(t.search(line) for t in own_titles)

    titled_at = next((i for i, line in enumerate(heading) if is_title(line)), None)
    if titled_at is not None:
        heading = heading[titled_at:]
    elif any(is_title(" ".join(row)) for row in block.rows[:_TITLE_ROWS]):
        # The title is inside the table, so what sits above it belongs to something else:
        # on ONEOK's 10-K, the last lines of the audit report and a page footer.
        heading = ()
    rows = block.rows + _continuation(section, items, index)
    scale, declared, share_scale = _scales(
        " ".join([*heading, *(" ".join(r) for r in block.rows[:_TITLE_ROWS])])
    )
    # A scale printed in the share rows' own label is the most specific statement of it,
    # and overrides the heading: Insulet's "(in millions, except share and per share data)"
    # sits over "Weighted-average number of common shares outstanding (in thousands):".
    share_scale = _share_label_scale(rows) or share_scale
    return Statement(
        section=section,
        heading=heading,
        rows=rows,
        scale=scale,
        scale_declared=declared,
        share_scale=share_scale,
    )


def _cover(items: Sequence[Block]) -> str:
    lines: list[str] = []
    used = 0
    for block in items:
        text = block.render()
        if used >= _COVER_MIN_CHARS and not block.is_table and _COVER_END.search(text):
            break
        lines.append(text)
        used += len(text)
        if used >= _COVER_BUDGET_CHARS:
            break
    return "\n".join(lines)


def _auditor(items: Sequence[Block]) -> str:
    """The signature blocks of every audit report in the document.

    Every report, because a 10-K filed after a change of auditor carries two, one per
    auditor, and both are tagged. Short lines only, because the signature is short lines
    and the paragraphs around it would add a page of text that holds no answer.
    """
    lines: list[str] = []
    for index, block in enumerate(items):
        if block.is_table or not _AUDITOR_TENURE.search(block.text):
            continue
        window = items[max(0, index - _AUDIT_WINDOW_BEFORE) : index + _AUDIT_WINDOW_AFTER]
        for near in window:
            text = near.text
            if near.is_table or not text or text in lines:
                continue
            if len(text) <= _AUDIT_LINE_MAX_CHARS or near is block:
                lines.append(text)
    return "\n".join(lines)


def locate(document: bytes) -> LocatedFiling:
    """Find the cover page, the two primary statements and the audit signature."""
    items = blocks(document)
    return LocatedFiling(
        cover=_cover(items),
        income=locate_statement(Section.INCOME, items),
        balance=locate_statement(Section.BALANCE, items),
        auditor=_auditor(items),
    )
