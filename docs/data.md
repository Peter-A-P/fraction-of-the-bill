# The dataset

The result of this project is a curve, and the curve is only as good as the labels under
it. This document says where truth comes from, which of several true numbers is the label,
how the corpus is divided, and what is still unknown. It is written before the fetch rather
than after it, so that the decisions can be argued with before thirty thousand documents
are pulled from a public regulator on the strength of them.

**Status: built and tested against fixtures; nothing has been fetched.** The fetcher is
gated on a declared contact address (see Fair access below). Written 2026-09-14; the
statement locator and the locatability filter added 2026-09-18.

## The task

Structured extraction from the primary financial statements of a 10-K or 10-Q into fifteen
fields. The schema, the fields and their tolerances are in
[`smallprint/schema.py`](../smallprint/schema.py), which is the single definition the
prompt, the training targets and the grader all read from. Three consumers agreeing about
the task by convention rather than by construction is how a measurement quietly becomes
wrong, so `Extraction` fails to build at all if it and `SCHEMA` have drifted apart.

## Truth

Truth for every field is the XBRL fact the company itself filed. Nothing is judged by a
model, here or anywhere else in this project.

### Which fact

A filing does not report one revenue. A third-quarter 10-Q reports revenue for the three
months just ended, for the nine months to date, and for both of the prior year's matching
periods: four numbers on one line, all correctly tagged, all true. The label is the one
whose period ends when the report ends and whose duration is the length the report covers.
Balance sheet lines are instants and take the position at the period end.

`select_fact` in [`smallprint/data/xbrl.py`](../smallprint/data/xbrl.py) does this, with
the bands it allows written down:

| | Target | Accepted |
|---|---|---|
| Fiscal year | 365 days | 300 to 400, because a fiscal year is 52 or 53 weeks |
| Quarter | 91 days | 60 to 120, wide enough for a retail calendar and narrow enough that nine months can never pass as three |
| Period end | the cover page date | within 7 days, because filers tag the Saturday a 52/53-week year ended while the cover page carries the month end |

The three readings that are not the label are kept as **distractors** and handed to the
grader. They are never correct. They exist so that a model which reads the prior-year
column is reported as having read the wrong period, by name, instead of disappearing into
a single undifferentiated wrong-answer count. The failure modes in the model cards are
written from that table.

### Only the filing being labelled

Facts are taken only from the filing whose accession number is being labelled. A later
filing restating the same period is a different document saying a different thing, and
training a model to produce a number that was not on the page it was shown is training it
to guess.

### Where the facts come from, and one open question

Numeric facts come from the `companyfacts` API per company, cross-checked against the
quarterly Financial Statement Data Sets in bulk. `Fact` records are deliberately
source-agnostic: the same selection rule has to run over both, or the two are not a
cross-check of each other at all.

**Open:** `companyfacts` organises facts by unit (USD, shares, USD/shares, pure), and three
fields of the schema are not numeric: `auditor_name`, `state_of_incorporation` and
`fiscal_period`. The expectation is that they are not served by that API and have to come
from the Financial Statement Data Sets submission table and the Notes Data Sets text table,
or from the filing's own inline XBRL. This is checked against the live API on the first
fetch, and the answer is recorded here before any filing is paired. `Fact.value` already
accepts a string for that reason. Nothing downstream changes either way; if the three
fields turn out to be unavailable at corpus scale they are dropped from the schema and the
task becomes twelve fields, which is a change to `SCHEMA` and to this document in the same
commit.

## Fair access

The SEC's access policy is a condition of use: a declared User-Agent carrying a contact
address, and under ten requests a second. The client in
[`smallprint/data/edgar.py`](../smallprint/data/edgar.py) enforces both rather than
documenting them.

- It **refuses to be constructed** without a contact address in `SMALLPRINT_EDGAR_CONTACT`.
  There is deliberately no default, because a default would be someone else's address or a
  fiction. This is why nothing has been fetched yet.
- It paces itself at eight requests a second, in the client rather than at the call sites,
  and refuses to be configured above ten.
- It backs off on 429 and 403 rather than retrying immediately, and does not retry a status
  that will not improve.
- Everything it fetches is cached to disk with the date and a SHA-256 digest. A rebuild
  reads the cache, so the corpus is fetched once. An offline client reads the cache and
  needs no address at all, which is what makes a rebuild reproducible without asking the
  SEC for the corpus again.

The tests for all of this run against a mock transport and never touch sec.gov. A suite
that hammers a public regulator to prove it is polite to that regulator has missed the
point, and CI would run it on every push.

## The splits

Two constraints, satisfied in this order.

**No filer is used for two purposes.** Companies file the same statements quarter after
quarter with the same layout, the same auditor and numbers that barely move. A split that
divides one company's filings between train and test measures memorisation. So companies,
not filings, are partitioned into three pools: train, validation and test. A company is in
exactly one pool for ever, chosen by a blake2b hash of its CIK with a fixed seed, ranked
within its stratum so the pool sizes come out at the fractions asked for rather than
wandering by several percent.

**The headline is measured after the base models' training cutoff.** The cutoff then
divides the test pool's own filings in time: before it is the pre-cutoff test set, after it
is the post-cutoff test set that carries the headline.

A test filer therefore appears in *both* test sets, deliberately. `PLAN.md` says only that
no filer appears in both train and test; this is the exact form of that rule, and it
matters. If the pre-cutoff and post-cutoff sets held different companies, the gap between
their scores would mix contamination with whichever set happened to hold easier filers.
Sharing the filers makes the contamination gap a within-filer comparison, which is the only
version of it worth publishing.

Training-pool filings dated after the cutoff are **dropped**, and the count is published in
the split report. Nothing about them is unusable, but a reader should not have to take on
trust that filings from the held-out period were not the reason the score held up.

`check_no_leakage` runs on every build, not only in the test suite. A split file is the
kind of artefact that gets hand-edited once, late at night, to put back a few filings that
were dropped, and that edit is what it refuses to let pass.

## What the model is shown

Not the filing. A 10-K is a few hundred pages; the model is shown four sections of it,
found by [`smallprint/data/statements.py`](../smallprint/data/statements.py) and labelled
in a fixed order: the cover page, the income statement, the balance sheet, and in an annual
report the audit report's signature block. Each field of the schema names the section it is
read from (`FieldSpec.section`).

**Two things are removed before anything is read.** The inline XBRL header (`ix:header`) is
a hidden block carrying the filing's own tagged facts, which include this task's cover-page
labels; left in, it would put the answers in the question. Anything styled `display:none`
goes for the same reason. A test plants a sentinel number in the hidden header and fails
if it ever reaches the rendered text.

**A table is taken as a statement by its lines, not only by its title.** Most tables in a
10-K carry the same line items as the statements: the discussion section's highlights, the
segment note, the quarterly table. The locator scores each table against the lines its
statement carries (net income, per-share amounts, revenue, operating income, tax, weighted
shares for the income statement; total assets, total liabilities, equity, cash, current
totals, retained earnings for the balance sheet), with an anchor line it must have. A title
in the heading adds weight. A table titled as a *different* statement is never taken, which
is what keeps the cash flow statement out of both. A balance sheet broken across a page is
read on to its second table. Ties go to the earliest table, and that is the choice the hand
audit is most likely to catch wrong.

**The scale is read from the statement's heading**, "(in thousands, except per share data)"
and its variants, because the grader needs it as its precision floor. Shares have their own
scale: "except share and per share data" means share counts are printed whole, and "number
of shares, which are reflected in thousands" means they are not. A statement with no printed
scale is assumed to be in whole units, and one where that is wrong fails the filter below
rather than being labelled a thousand times too small.

`smallprint data locate FILE` prints exactly what a model would be shown from one local
filing document, with the scales read. The hand audit works from that view.

## The locatability filter

A filing becomes an item only when every fact it is labelled with can be found in the text
it is shown, so the task is extraction rather than inference
([`smallprint/data/pair.py`](../smallprint/data/pair.py)). "Found" is exact:

| Field kind | Found when |
|---|---|
| Money, shares, per share | A number printed in the field's own section, times that section's scale, is within the grader's tolerance of the fact. The grader and the filter share one function, so a filing is kept only if a reading the grader would accept is on the page |
| Period end | The date is printed on the cover page, in any of its usual spellings |
| Auditor | The name appears in the signature block after the grader's normalisation (case, punctuation, "and" for "&", legal suffix) |
| State of incorporation | The code or, for US states and territories, the name is printed on the cover page |
| Fiscal period | Not searched. It is never printed as "Q3"; it is read from which columns the income statement carries. That is still reading the page, but no string search can confirm it |

The field's own section matters. A revenue figure found in the balance sheet, or a cash
figure found only in the discussion section's highlights table, is a coincidence and not a
location, and tests assert that both are dropped. Sign is not required to match: reading a
loss out of its parentheses is the model's job.

Every drop has a named reason, and the tally goes in the datasheet: no income statement, no
balance sheet, mixed scales between the two statements, no cover-period facts, a required
fact missing, or a named field not locatable. A filter whose losses are not published could
be removing exactly the hard cases without anyone knowing.

**Known weaknesses, to be measured on the first fetch rather than guessed at.** Any printed
number counts as a location for a field in its section, including a year in a column
heading, so a coincidental match is possible; the tolerance makes it rare and the hand
audit is the check. Filers incorporated outside the US whose cover page spells out a
country rather than printing the EDGAR code are dropped for their state field; the tally
says how many. And the rule that the two statements must declare the same scale drops a
small class of filings that could in principle be kept.

## Still to build

- The corpus driver: selecting filings from the quarterly indexes, fetching them through the
  fair-access client, and running pairing and the splits over the lot.
- The answer to the open question above, from the first live fetch.
- The datasheet with the drop tally, the checksums and the Hugging Face publication.
- A hand audit of 200 pairs before any training starts.
