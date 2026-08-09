"""Rebuild the document-type detection corpus from public sources.

The section-schema detector classifies a document into one cell of the
document-type partition. Verifying it needs a real sample per cell -- vocabulary
authored without one is vocabulary nobody can check.

Full source documents are deliberately *not* committed: they are large and
variously licensed. What the detector actually consumes is a document's ordered
heading sequence plus a sample of its body paragraphs, so only those extracts
land in ``test/data/schema_corpus.json``, each carrying the URL and licence it
came from. That keeps the test suite offline, deterministic and a few tens of kB.

Usage:
    uv run python run/fetch_schema_samples.py            # rebuild every entry
    uv run python run/fetch_schema_samples.py --only standard fiction

Entries for ``academic`` and ``financial`` are extracted from documents already
in ``data/json/`` and need no network access.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import date

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "test" / "data" / "schema_corpus.json"

USER_AGENT = "ontocast-schema-corpus/1.0 (+https://github.com/growgraph/ontocast)"
MAX_HEADINGS = 60
MAX_PARAGRAPHS = 24
PARAGRAPH_MIN_CHARS = 300
PARAGRAPH_MAX_CHARS = 600


@dataclass
class Source:
    """One corpus entry and how to reproduce it."""

    schema_id: str
    name: str
    license: str
    url: str | None = None
    local: str | None = None
    style: str = "markdown"
    notes: str = ""
    # Body text outside this character window is skipped (front/back matter).
    window: tuple[int, int | None] = (0, None)
    headings: list[str] = field(default_factory=list)


SOURCES: list[Source] = [
    Source(
        schema_id="academic",
        name="chem.204703 (J. Chem. Phys. research article)",
        license="publisher copyright; extract only, already vendored in data/",
        local="data/json/chem.204703_1_5.0167542.json",
    ),
    Source(
        schema_id="financial",
        name="Apple Inc. Form 10-Q",
        license="US SEC filing, public record",
        local="data/json/fin.10Q.apple.json",
    ),
    Source(
        schema_id="standard",
        name="RFC 7231 (HTTP/1.1 Semantics and Content)",
        license="IETF Trust, freely redistributable",
        url="https://www.rfc-editor.org/rfc/rfc7231.txt",
        style="rfc",
    ),
    Source(
        schema_id="fiction",
        name="Pride and Prejudice (Jane Austen)",
        license="public domain (Project Gutenberg)",
        url="https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        style="gutenberg",
    ),
    Source(
        schema_id="patent",
        name="US patent full text",
        license="US patent document, public record",
        url="https://patents.google.com/patent/US20190131529A1/en",
        style="html",
    ),
    Source(
        schema_id="manual",
        name="nginx beginner's guide",
        license="nginx docs, BSD-style licence",
        url="https://nginx.org/en/docs/beginners_guide.html",
        style="html",
    ),
    Source(
        schema_id="legal",
        name="Creative Commons Attribution 4.0 legal code",
        license="CC licence text, published for reuse",
        url="https://creativecommons.org/licenses/by/4.0/legalcode.en",
        style="html",
    ),
    Source(
        schema_id="clinical",
        name="USPIO-enhanced MRI in CNS tumours (UMIC) study protocol",
        license="CC BY (Europe PMC open access)",
        url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13285558/fullTextXML",
        style="pmc",
        notes=(
            "A published trial protocol, which sits on the academic/clinical "
            "partition edge by construction and is therefore the sharpest test "
            "of that boundary. Europe PMC full-text XML is used rather than a "
            "publisher page because <sec><title> gives real document headings "
            "without site furniture. Note data/json/clinical.trials.*.json are "
            "raw registry API records ({'protocolSection': ...}), not document "
            "text -- they extract to ~0 characters and cannot be used here."
        ),
    ),
    Source(
        schema_id="news",
        name="Wikinews article",
        license="CC BY 2.5",
        url="https://en.wikinews.org/wiki/Special:Random/main",
        style="html",
        notes=(
            "News is genuinely heading-poor -- a typical article yields a "
            "handful of headings, most of them site furniture. This entry "
            "therefore leans on paragraphs, and news is expected to abstain "
            "under heading-only detection."
        ),
    ),
]

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(\S.*?)\s*$")
# RFC section headings: flush-left, numbered, e.g. "4.3.1.  GET".
_RFC_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s{1,3}(\S.*?)\s*$")
_GUTENBERG_HEADING = re.compile(
    r"^\s*((?:chapter|part|book|volume)\s+[\dIVXLC]+|prologue|epilogue|"
    r"afterword|dedication|epigraph)\s*\.?\s*$",
    re.I,
)
_HTML_HEADING = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.I | re.S)
# Europe PMC full-text XML: <sec><title> is the document's own heading, free of
# the navigation furniture a publisher's HTML page carries.
_PMC_HEADING = re.compile(r"<title>(.*?)</title>", re.S)
_PMC_PARAGRAPH = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def _unescape(text: str) -> str:
    import html

    return html.unescape(_TAG.sub(" ", text)).strip()


def extract_headings(text: str, style: str) -> list[str]:
    """Pull a document's ordered heading sequence out of its raw text."""
    headings: list[str] = []
    if style == "pmc":
        headings = [_unescape(m.group(1)) for m in _PMC_HEADING.finditer(text)]
    elif style == "html":
        headings = [_unescape(m.group(1)) for m in _HTML_HEADING.finditer(text)]
    elif style == "rfc":
        for line in text.splitlines():
            match = _RFC_HEADING.match(line)
            if match and not line.startswith(" "):
                headings.append(f"{match.group(1)}. {match.group(2)}")
    elif style == "gutenberg":
        for line in text.splitlines():
            match = _GUTENBERG_HEADING.match(line)
            if match:
                headings.append(match.group(1).strip())
    else:
        for line in text.splitlines():
            match = _MD_HEADING.match(line)
            if match:
                headings.append(match.group(1))
    # Collapse consecutive duplicates (page furniture repeats) but keep order.
    deduped: list[str] = []
    for heading in headings:
        cleaned = " ".join(heading.split())
        if cleaned and (not deduped or deduped[-1] != cleaned) and len(cleaned) < 120:
            deduped.append(cleaned)
    return deduped[:MAX_HEADINGS]


def extract_paragraphs(
    text: str, style: str, window: tuple[int, int | None]
) -> list[str]:
    """Sample body paragraphs, evenly spread, excluding headings and tables."""
    start, end = window
    body = text[start:end]
    if style == "pmc":
        blocks = [_unescape(m.group(1)) for m in _PMC_PARAGRAPH.finditer(body)]
        blocks = [" ".join(b.split()) for b in blocks if len(b) >= PARAGRAPH_MIN_CHARS]
        if not blocks:
            return []
        step = max(1, len(blocks) // MAX_PARAGRAPHS)
        return [b[:PARAGRAPH_MAX_CHARS] for b in blocks[::step]][:MAX_PARAGRAPHS]
    if style == "html":
        body = _unescape(body)
    blocks = [block.strip() for block in re.split(r"\n\s*\n", body)]
    blocks = [
        " ".join(block.split())
        for block in blocks
        if len(block) >= PARAGRAPH_MIN_CHARS
        and not block.lstrip().startswith(("#", "|", "<"))
    ]
    if not blocks:
        return []
    step = max(1, len(blocks) // MAX_PARAGRAPHS)
    return [block[:PARAGRAPH_MAX_CHARS] for block in blocks[::step]][:MAX_PARAGRAPHS]


def build_entry(source: Source) -> dict[str, object] | None:
    """Fetch or read one source and reduce it to a corpus entry."""
    if source.local is not None:
        payload = json.loads((REPO_ROOT / source.local).read_text(encoding="utf-8"))
        raw = payload.get("text", "") if isinstance(payload, dict) else ""
        origin = source.local
    elif source.url is not None:
        raw = _fetch(source.url)
        origin = source.url
    else:
        return None

    headings = extract_headings(raw, source.style)
    paragraphs = extract_paragraphs(raw, source.style, source.window)
    if not headings and not paragraphs:
        print(f"  !! {source.schema_id}: nothing extracted from {origin}")
        return None
    return {
        "schema_id": source.schema_id,
        "name": source.name,
        "source": origin,
        "license": source.license,
        "retrieved": date.today().isoformat(),
        "notes": source.notes,
        "headings": headings,
        "paragraphs": paragraphs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Limit to these schema ids.")
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="Report cells with no source configured and exit.",
    )
    args = parser.parse_args()

    if args.list_missing:
        for source in SOURCES:
            if source.url is None and source.local is None:
                print(f"{source.schema_id}: {source.notes or 'no source configured'}")
        return 0

    selected = [
        source for source in SOURCES if not args.only or source.schema_id in args.only
    ]
    entries: list[dict[str, object]] = []
    if CORPUS_PATH.exists() and args.only:
        existing = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        keep = {source.schema_id for source in selected}
        entries = [e for e in existing["entries"] if e["schema_id"] not in keep]

    for source in selected:
        print(f"-- {source.schema_id}: {source.name}")
        try:
            entry = build_entry(source)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  !! failed: {exc}")
            continue
        if entry is None:
            print("  .. skipped (no source configured)")
            continue
        print(
            f"  ok {len(entry['headings'])} headings, "
            f"{len(entry['paragraphs'])} paragraphs"
        )
        entries.append(entry)

    entries.sort(key=lambda entry: (entry["schema_id"], entry["name"]))
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(
        json.dumps({"version": 1, "entries": entries}, indent=1, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {CORPUS_PATH.relative_to(REPO_ROOT)} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
