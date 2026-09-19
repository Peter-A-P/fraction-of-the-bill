# The dataset

The result of this project is a curve, and the curve is only as good as the labels under
it. This document says where truth comes from, which of several true numbers is the label,
how the corpus is divided, and what is still unknown. It is written before the fetch rather
than after it, so that the decisions can be argued with before thirty thousand documents
are pulled from a public regulator on the strength of them.

**Status: built, tested against fixtures, and run live over 300 companies** (below). The
full corpus waits on the base models' training cutoff. Written 2026-09-14; the statement
locator, the locatability filter, the cover-facts decision and the build added 2026-09-18;
the live builds and the bank decision 2026-09-19.

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

### Where the facts come from

Numeric facts come from the `companyfacts` API per company, cross-checked against the
quarterly Financial Statement Data Sets in bulk. `Fact` records are deliberately
source-agnostic: the same selection rule has to run over both, or the two are not a
cross-check of each other at all.

**Decided 2026-09-18: the cover-page facts come from the filing itself.** Four fields are
`dei` cover-page facts, and three of them are not numeric: `auditor_name`,
`state_of_incorporation` and `fiscal_period`, with `period_end` alongside. Rather than join
them in from the Financial Statement Data Sets, as the plan first expected, they are read
from the filing's own inline XBRL ([`smallprint/data/ixbrl.py`](../smallprint/data/ixbrl.py)).
Every 10-K and 10-Q since 2019 tags exactly these facts on its cover page with
`ix:nonNumeric`, against contexts declared in the hidden header, and those tags are what the
company filed, which is this project's definition of truth. The document is fetched anyway
to build the model's input, so this costs no request, and it removes a second source whose
rows would have to be matched to filings by accession.

It is not a leak. The hidden header is stripped before the input is rendered, and the tags
that remain visible are printed on the page, which is the locatability rule working as
intended. Tags against dimensional contexts (a subsidiary, a class of shares) are ignored.
A tagged date is parsed from the few shapes cover pages print it in, and a state tagged by
name becomes its code; a value that fits no known shape is left out, which drops the filing
with its reason named rather than guessing.

Whether `companyfacts` also serves these concepts no longer matters to the build. If it
does, the document's own tags come first.

## Fair access

The SEC's access policy is a condition of use: a declared User-Agent carrying a contact
address, and under ten requests a second. The client in
[`smallprint/data/edgar.py`](../smallprint/data/edgar.py) enforces both rather than
documenting them.

- It **refuses to be constructed** without a contact address in `SMALLPRINT_EDGAR_CONTACT`.
  There is deliberately no default, because a default would be someone else's address or a
  fiction.
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

## The build

`smallprint data build` ([`smallprint/data/build.py`](../smallprint/data/build.py)) runs the
whole construction: select, fetch, pair, split, write.

- **Selection is by company.** Every 10-K and 10-Q filed in the quarter range, from
  companies ranked by a keyed hash; `--companies N` takes the first N. Amendments are left
  out, because a 10-K/A often restates only the part that changed. The selection hash is
  keyed differently from the pool assignment, so being selected and landing in the test
  pool are independent draws.
- **Requests per company:** the filing history (older pages only when they reach back into
  the range), the facts, and one document per filing, all through the fair-access client.
- **The stratum** the pools are balanced across is the filer's size band, from the median
  of its own balance sheets, so that it is constant for a company as pool assignment needs.
- **Nothing that cannot be fetched stops the build.** A withdrawn document or a company with
  no XBRL facts is a drop with a named reason, counted with the rest.
- **`--cutoff` has no default.** It is the latest base-model training cutoff from
  `docs/models.md`, it defines the headline test set, and a default would be a guess.
- **Output:** `items.jsonl` (each item with its split), `dropped.jsonl`, and `report.json`
  with the selection, the pairing tally and the split report. All sorted, so two builds
  diff cleanly. `data/build/` is not committed; the corpus is published to Hugging Face.
- **A rebuild runs offline** from the cache with `--offline`, needs no contact address, and
  reproduces the corpus without asking the SEC for it again. A test builds once through a
  mock EDGAR and again offline, and asserts the two are identical.

## What the first live build found

2026-09-18, a smoke build of 50 companies, filings dated 2025Q1 to 2026Q2, 236 filings,
provisional cutoff 2025-12-31 (not the real one; the bases are not chosen). The first run
kept **41 of 236**. Every large loss was a bug in this code rather than a property of the
filings, and each fix has a regression test named for the filer that exposed it:

| Found | Cost | Fix |
|---|---:|---|
| `form.idx` rows sit five characters right of the header's column labels; offsets read from the header cut the date in half | the whole build | Rows are read from the right, where fields have shapes |
| A Q2 or Q3 10-Q tags its cover facts against the year to date, not the quarter | 75 | Cover (`dei`) facts are matched on period end alone |
| Cover pages print "December 31 , 2024", with the comma set apart, and "03/31/2025" | 62 + 40 | Both shapes parsed and searched for |
| The Andersons tag $371m as contract revenue against $2,659m of sales: the schema preferred the ASC 606 concept, which is only part of the top line | a wrong label, caught by the filter | `us-gaap:Revenues` first in `SCHEMA` |
| ONEOK prints title and "(Millions of dollars ...)" as rows of the statement table | 10 | Scale read from six rows, "millions of dollars" recognised, heading dropped when the title is inside the table |
| St. Joe prints shares whole under "(Dollars in thousands ...)" | 3 | Shares found whole when the heading scales only money |
| Lifetime Brands labels the line "Net (loss) income" | 1 | Anchor allows the parentheses |

After the fixes, rebuilt offline from the cache in 70 seconds: **141 of 236 kept (60%)**.
What remains is mostly the corpus rather than the code:

| Reason | Filings | What they are |
|---|---:|---|
| `truth_missing:revenue` | 41 | SPACs and pre-revenue biotechs and developers. They report no revenue fact; the task requires one |
| `no_income_statement` | 13 | A BDC and commodity and currency trusts, whose statements are investment-company formats |
| `unlocatable:shares_diluted` | 10 | Not yet examined |
| `not_fetched` | 9 | Documents or facts EDGAR did not serve |
| `truth_missing:cash_and_equivalents` | 6 | Not yet examined; banks tag cash differently |
| other | 16 | Scattered, five reasons or fewer each |

Spot checks of kept items against their text found the labels right, including the
attributable-to-parent net income where a filer prints consolidated net income on the line
above. The model's input runs to about 6,000 characters, well under the 3.5k-token budget
in the plan's cost table. Parsing costs about a second of CPU per document; the live build
is bound by the paced fetch, not by parsing.

Two more fixes followed from the last unexamined reasons, taking the smoke build to **152 of
236 (64%)**: a combined "statement of comprehensive income" that carries per-share lines is
the income statement (Deckers' is titled so, and the locator had taken the discussion
section's results table instead), and `us-gaap:Cash` is the last fallback for cash and
equivalents (a filer with no equivalents tags its one line so). Lifetime Brands prints no
weighted shares on its income statement at all, and is dropped as the plan's rule says.

## The 300-company build

2026-09-19, 300 companies, same range and provisional cutoff: 1,456 filings, fetched in
about an hour at the paced rate, 3.9 GB of cache in all, rebuilt offline in 14 minutes.
**715 kept (49%).** Lower than the smoke build, because a wider draw reaches more kinds of
filer:

| Reason | Filings | |
|---|---:|---|
| `truth_missing:revenue` | 280 | Broken down below |
| `no_income_statement` | 105 | Not yet examined at this scale; in the smoke build, funds and trusts |
| `unlocatable:shares_diluted` | 66 | Not yet examined at this scale |
| `unlocatable:revenue` | 64 | Not yet examined at this scale |
| `not_fetched` | 59 | |
| `unlocatable:period_end` | 40 | |
| `unlocatable:auditor_name` | 34 | New at this scale |
| `no_balance_sheet` | 26 | New at this scale |
| other | 72 | Twelve reasons, 17 or fewer each |

The 280 filings (71 companies) with no revenue label, by what they do tag:

| | Filings | |
|---|---:|---|
| No revenue concept at all | 95 | Pre-revenue companies and shells. Correctly dropped |
| SPACs | 41 | Correctly dropped |
| Banks and other lenders | about 54 | Interest income and gains on loans, no single revenue line. **Excluded by decision, below** |
| A revenue concept the selection did not take | 40 | Probably a bug in period selection. To examine first |
| Other revenue concepts | about 50 | Broker-dealer revenue net of interest, and others |

### Banks are out of scope (decided 2026-09-19)

Banks and savings institutions report interest income, fee income and gains on loans, and
no single line that is their revenue. The task requires a revenue label and grades it
against one tagged fact, so for a bank there is nothing to grade against; defining bank
revenue (interest plus non-interest income, say) would be a second task with its own
truth rule. They are excluded by industry code before anything but their filing history is
fetched (`BANK_INDUSTRY_CODES` in `build.py`: 6021, 6022, 6029, 6035, 6036), and counted
under their own drop reason, `bank`. **The corpus therefore underweights financial
companies, and the datasheet says so.** Insurers, REITs and broker-dealers are not
excluded; they have revenue lines, and whether their filings survive the filter is left to
the filter.

### What the 300-company build fixed

Examining the larger build's drops found five more faults in this code, each with a test
named for the filer that exposed it:

| Found | Fix |
|---|---|
| Weis Markets tags `Revenues` for prior years only and the current year as contract revenue; Amcor tags `Revenues` by quarter only. Selection stopped at the first concept reported at all | A concept with no fact for the period falls through to the next; its other periods stay as distractors |
| Hasbro tags `Revenues` at $4,745.9m while its statement prints "Net revenues" of $4,135.5m, its contract revenue. The Andersons are the opposite case, so no concept order is right for both | Among the synonyms reported for the period, the label is the one the statement prints; the schema's order decides when both are. The label is always a filed fact |
| Audit reports say "served as Aditxt's auditor since", "the auditor of the Company since", and so on; the signature can sit several lines above; a change of auditor gives two reports | Wider wording, a wider window of short lines, every report collected |
| Insulet and Linde print the share scale in the share row's label ("(in thousands):", "(000's)"), not in the heading | The label's scale overrides the heading's |
| A discussion-section table introduced by "The following table sets forth ... our Consolidated Statements of Income as a percentage of net sales:" counted that sentence as its title | A sentence introducing a table ends its heading |

A sixth fault came from the 40 cover-page misses: **HTML comments were read as text.** One
filing agent's software writes comments ("Field: Set; Name: xdx; ...") at the top of every
document, and they pushed the cover page past its real start. Anything a filer left in a
comment could have reached the model's input the same way. Comments and processing
instructions are now never read; the text after them still is.

Rebuilt offline with every fix and the bank exclusion: **775 kept of 1,456, or 775 of the
1,373 filings not excluded as banks (56%).**

| Reason | Filings | What they are |
|---|---:|---|
| `truth_missing:revenue` | 225 | Pre-revenue companies, shells, SPACs and other revenue concepts |
| `no_income_statement` | 103 | Investment vehicles: 56 with no industry code (BDCs and closed-end funds), 32 commodity and crypto trusts, 3 SPACs, 3 asset-backed trusts, 8 others |
| `bank` | 83 | Excluded by decision |
| `unlocatable:shares_diluted` | 62 | 46 print weighted shares only in a note, not on the statement. Dropped by the plan's rule; see below |
| `not_fetched` | 59 | |
| `unlocatable:revenue` | 36 | About 20 print no scale at all over statements in thousands (Thor). Dropped: the model would have to guess the scale |
| `unlocatable:auditor_name` | 30 | Mostly signatures printed as images, with no text to find |
| `no_balance_sheet` | 24 | Statements laid out without HTML tables, holding companies with one asset line |
| other | 64 | Eleven reasons, 17 or fewer each |

**Changed by decision, 2026-09-19 (PLAN.md rev. 5).** About 3% of filings print weighted
diluted shares only in a note. When the income statement has no share-count line at all,
that field's label is null and the filing is kept (`NULL_WHEN_LINE_ABSENT` in `pair.py`),
and the item records it in `not_on_page`. When a share line is printed but does not match,
the filing is still dropped. The tally counts these as `kept_with_null:shares_diluted`.
No other field is treated this way.

**The cache is stored compressed** (from 2026-09-19). Filing HTML compresses about
fourteen times: the 300-company cache went from 3.9 GB to 299 MB, and the full corpus is
about 5 GB rather than 70. Each body's digest is still of the bytes EDGAR served.
`smallprint data compact-cache` converts a cache written before this, checking every body
against its digest first.

## Still to build

- The full corpus, split at **2025-01-31** (Gemma 4 E4B's documented cutoff).
- The datasheet, checksums and Hugging Face publication, and the hand audit of 200 pairs.
- The base-model cutoff, from `docs/models.md`, which the full build needs.
- The datasheet with the drop tally, the checksums and the Hugging Face publication.
- A hand audit of 200 pairs before any training starts.
