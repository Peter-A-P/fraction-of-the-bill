"""A fair-access client for SEC EDGAR.

The SEC's access policy is short and it is enforced: declare who you are in the User-Agent
with a contact address, and stay under ten requests a second. Both are conditions of use
rather than suggestions, and a project whose whole subject is measurement should not be
sloppy about the one part of it that touches someone else's server.

So three things are true of this client and are tested rather than asserted:

* It refuses to make a request at all until a contact string has been configured. There is
  no default, because a default would be someone else's address or a fiction.
* It paces itself, and the pacing is in the client rather than at the call sites, so no
  caller can forget.
* Everything it fetches is written to a local cache with the date and a digest. A rebuild
  of the dataset reads the cache, so the corpus is fetched once and the measurement is
  reproducible without asking the SEC for thirty thousand documents again.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final, Self

import httpx
from pydantic import BaseModel, ConfigDict

#: The environment variable that carries the declared contact. Named rather than taken from
#: a config file so that it is obvious it is per-machine and never committed.
CONTACT_ENV: Final = "SMALLPRINT_EDGAR_CONTACT"

#: The SEC publishes a limit of ten requests a second. Eight leaves headroom for the fact
#: that the limit is enforced on their clock and not on ours.
DEFAULT_RATE_PER_SECOND: Final = 8.0

#: Statuses worth retrying. 429 is the rate limiter; 403 is what EDGAR returns when it has
#: decided a client is misbehaving, and backing off is the correct response to it.
_RETRY_STATUSES: Final[frozenset[int]] = frozenset({403, 429, 500, 502, 503, 504})

DATA_HOST: Final = "https://data.sec.gov"
ARCHIVE_HOST: Final = "https://www.sec.gov"

_CONTACT_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ContactNotDeclared(RuntimeError):
    """Raised when a fetch is attempted without a declared contact address."""


class FairAccessViolation(RuntimeError):
    """Raised when the client is asked to do something the access policy forbids."""


def declared_contact(explicit: str | None = None) -> str:
    """The contact address that goes in the User-Agent, or an error explaining its absence."""
    contact = explicit if explicit is not None else os.environ.get(CONTACT_ENV, "").strip()
    if not contact:
        raise ContactNotDeclared(
            "SEC fair access requires a contact address in the User-Agent of every request. "
            f"Set {CONTACT_ENV} to an address that reaches whoever is running this fetch. "
            "There is deliberately no default."
        )
    if not _CONTACT_PATTERN.match(contact):
        raise ContactNotDeclared(
            f"{CONTACT_ENV} should be an email address that reaches a person; got {contact!r}"
        )
    return contact


class _Pacer:
    """A minimum interval between requests, held on the client rather than at call sites."""

    def __init__(self, rate_per_second: float) -> None:
        if rate_per_second <= 0 or rate_per_second > 10:
            raise FairAccessViolation(
                f"rate must be above zero and at or under the SEC's ten per second; got {rate_per_second}"
            )
        self._interval = 1.0 / rate_per_second
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
                now = time.monotonic()
            self._next_at = now + self._interval


class CachedResponse(BaseModel):
    """What the cache records about one fetch, so a rebuild can be audited."""

    model_config = ConfigDict(frozen=True)

    url: str
    fetched_at: dt.datetime
    status: int
    sha256: str
    bytes: int


class EdgarClient:
    """Fetches from EDGAR, once, politely, and remembers what it fetched.

    Not a general HTTP client: every method here maps to one EDGAR endpoint, because a
    general `get` would be the thing a caller reaches for when a new endpoint is needed and
    would sooner or later be pointed somewhere the cache and the pacer do not cover.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        contact: str | None = None,
        rate_per_second: float = DEFAULT_RATE_PER_SECOND,
        transport: httpx.BaseTransport | None = None,
        offline: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.offline = offline
        self._pacer = _Pacer(rate_per_second)
        # The contact is resolved now rather than at the first request, so a misconfigured
        # run fails before it has fetched anything rather than part way through a corpus.
        # An offline client reads the cache only and needs none.
        self._contact = None if offline else declared_contact(contact)
        self._client = (
            None
            if offline
            else httpx.Client(
                headers={
                    # The form the SEC asks for: who, and how to reach them.
                    "User-Agent": f"smallprint/0.1 (research; {self._contact})",
                    "Accept-Encoding": "gzip, deflate",
                },
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
                transport=transport,
            )
        )
        (self.cache_dir / "meta").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "body").mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    # -- cache ---------------------------------------------------------------------------

    def _key(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def _cached(self, url: str) -> bytes | None:
        key = self._key(url)
        compressed = self.cache_dir / "body" / f"{key}.gz"
        if compressed.exists():
            return gzip.decompress(compressed.read_bytes())
        # Caches written before bodies were compressed still read.
        plain = self.cache_dir / "body" / key
        return plain.read_bytes() if plain.exists() else None

    def _store(self, url: str, status: int, body: bytes) -> None:
        """Keep the body compressed and its digest over the original bytes.

        Filing HTML compresses about fourteen times. The full corpus is some seventy gigabytes
        as fetched and about five compressed, which is the difference between fitting on a
        laptop and not. The digest is of what EDGAR served, so compression changes nothing a
        rebuild is audited against.
        """
        key = self._key(url)
        (self.cache_dir / "body" / f"{key}.gz").write_bytes(gzip.compress(body, compresslevel=6))
        record = CachedResponse(
            url=url,
            fetched_at=dt.datetime.now(dt.UTC),
            status=status,
            sha256=hashlib.sha256(body).hexdigest(),
            bytes=len(body),
        )
        (self.cache_dir / "meta" / f"{key}.json").write_text(
            record.model_dump_json(indent=2), encoding="utf-8"
        )

    def compact(self) -> tuple[int, int]:
        """Compress bodies cached before compression, checking each against its digest.

        Returns how many were compressed and how many bytes that saved. A body whose digest
        does not match its record is left alone and reported by raising, because a cache
        that no longer holds what was fetched is not one to quietly rewrite.
        """
        records = {self._key(r.url): r for r in self.manifest()}
        count = saved = 0
        for plain in sorted((self.cache_dir / "body").iterdir()):
            if plain.suffix == ".gz" or plain.stem not in records:
                continue
            body = plain.read_bytes()
            if hashlib.sha256(body).hexdigest() != records[plain.stem].sha256:
                raise ValueError(f"cached body {plain.name} does not match its recorded digest")
            packed = gzip.compress(body, compresslevel=6)
            plain.with_name(f"{plain.name}.gz").write_bytes(packed)
            plain.unlink()
            count += 1
            saved += len(body) - len(packed)
        return count, saved

    def manifest(self) -> Iterator[CachedResponse]:
        """Every document this cache holds, with the date it was fetched and its digest."""
        for path in sorted((self.cache_dir / "meta").glob("*.json")):
            yield CachedResponse.model_validate_json(path.read_text(encoding="utf-8"))

    # -- fetching ------------------------------------------------------------------------

    def get(self, url: str, *, attempts: int = 4) -> bytes:
        """Fetch a URL, from the cache if it is there, from EDGAR at the paced rate if not."""
        cached = self._cached(url)
        if cached is not None:
            return cached
        if self._client is None:
            raise FairAccessViolation(
                f"offline client has no cached copy of {url}; run the fetch online first"
            )

        backoff = 1.0
        last_status = 0
        for attempt in range(attempts):
            self._pacer.wait()
            response = self._client.get(url)
            if response.status_code == 200:
                self._store(url, response.status_code, response.content)
                return response.content
            last_status = response.status_code
            if response.status_code not in _RETRY_STATUSES:
                break
            if attempt < attempts - 1:
                # Backing off rather than retrying immediately is the whole point: a 429
                # answered with another request is what gets a client blocked.
                time.sleep(backoff)
                backoff *= 2
        raise httpx.HTTPStatusError(
            f"EDGAR returned {last_status} for {url} after {attempts} attempts",
            request=httpx.Request("GET", url),
            response=httpx.Response(last_status),
        )

    def get_json(self, url: str) -> object:
        return json.loads(self.get(url))

    # -- endpoints -----------------------------------------------------------------------

    def submissions(self, cik: int) -> object:
        """A company's filing history, including the accession numbers and primary documents."""
        return self.get_json(f"{DATA_HOST}/submissions/CIK{cik:010d}.json")

    def submissions_page(self, name: str) -> object:
        """An older page of a company's filing history, named in the first page's `files`.

        The first page holds only the most recent thousand or so filings. A large filer's
        insider and current reports push its 2022 10-Qs off it.
        """
        if not re.fullmatch(r"CIK\d{10}-submissions-\d{3}\.json", name):
            raise ValueError(f"not a submissions page name: {name!r}")
        return self.get_json(f"{DATA_HOST}/submissions/{name}")

    def company_facts(self, cik: int) -> object:
        """Every XBRL fact a company has ever filed. This is where truth comes from."""
        return self.get_json(f"{DATA_HOST}/api/xbrl/companyfacts/CIK{cik:010d}.json")

    def quarterly_index(self, year: int, quarter: int) -> bytes:
        """The form index for one quarter: every filing of every type, one line each.

        The cheap way to select a corpus. One request per quarter covers what would
        otherwise be a request per company.
        """
        if quarter not in (1, 2, 3, 4):
            raise ValueError(f"quarter must be 1 to 4; got {quarter}")
        return self.get(f"{ARCHIVE_HOST}/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx")

    def document(self, cik: int, accession: str, filename: str) -> bytes:
        """One document from one filing."""
        plain = accession.replace("-", "")
        return self.get(f"{ARCHIVE_HOST}/Archives/edgar/data/{cik}/{plain}/{filename}")


class IndexEntry(BaseModel):
    """One line of a quarterly form index."""

    model_config = ConfigDict(frozen=True)

    form: str
    cik: int
    company: str
    filed: dt.date
    path: str

    @property
    def accession(self) -> str:
        """The accession number, which the index carries only inside the file path."""
        return Path(self.path).stem


#: One row of a form index, read from the right. The CIK, the date and the path have fixed
#: shapes; the company name is whatever lies between them and the form type, and the form
#: type ends at the first run of two or more spaces (form types contain single spaces, as in
#: "SC 13G/A", never double ones).
_INDEX_ROW = re.compile(
    r"^(?P<form>\S(?:\S| (?! ))*)\s{2,}(?P<company>.*?)\s+(?P<cik>\d+)\s+"
    r"(?P<filed>\d{4}-\d{2}-\d{2})\s+(?P<path>edgar/\S+)\s*$"
)


def parse_form_index(body: bytes, forms: frozenset[str] | None = None) -> Iterator[IndexEntry]:
    """Parse a quarterly `form.idx` into entries, optionally keeping only some forms.

    The file looks fixed-width, but its header does not line up with its rows: in the files
    EDGAR actually serves, every data column sits five characters to the right of the label
    above it. Reading offsets from the header, which this function first did against a
    fixture that happened to be aligned, cuts the date in half on the first real file. So
    each row is read from the right, where the fields have shapes, and the company name
    (spaces, commas, slashes and all) is what is left in the middle.
    """
    text = body.decode("latin-1")
    lines = text.splitlines()

    header_at = next((i for i, line in enumerate(lines) if set(line.strip()) == {"-"}), None)
    if header_at is None or header_at == 0:
        raise ValueError("form.idx has no dashed separator line; the format has changed")

    rows = [line for line in lines[header_at + 1 :] if line.strip()]
    matched = 0
    for line in rows:
        row = _INDEX_ROW.match(line)
        if row is None:
            continue
        matched += 1
        form = row.group("form")
        if forms is not None and form not in forms:
            continue
        yield IndexEntry(
            form=form,
            cik=int(row.group("cik")),
            company=row.group("company").strip(),
            filed=dt.date.fromisoformat(row.group("filed")),
            path=row.group("path"),
        )
    if rows and not matched:
        raise ValueError("no form.idx row has the expected shape; the format has changed")
