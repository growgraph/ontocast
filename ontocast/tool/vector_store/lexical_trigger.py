"""Case-sensitive exact-match retrieval for notation-bearing ontology terms.

Vocabularies keyed by a literal token in source text — unit symbols, chemical
formulae, gene symbols, CAS numbers — are a poor fit for dense semantic search.
This module indexes those tokens at atomization time and matches them directly
against raw chunk text at query time, outside the semantic atom budget.
"""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import Field, PrivateAttr

from ontocast.tool.onto import Tool
from ontocast.tool.vector_store.core import GraphAtom

# Symbol characters that begin common unit/quantity tokens (µm, °C, Å, %, Δν).
# Without them a token like "µm" tokenized to "m" and "%" to nothing at all.
_SYMBOL_START = "µμÅΩ°%‰Δ"
_SYMBOL_CONTINUE = "°·/⁻²³µμÅΩ%‰Δ"

# Word-like tokens tolerant of unit/formula punctuation (UCUM is case-sensitive).
_TOKEN_PATTERN = re.compile(
    rf"[A-Za-z{_SYMBOL_START}][A-Za-z0-9{_SYMBOL_CONTINUE}]*(?:[A-Za-z0-9]+)?|"
    rf"[0-9]+(?:\.[0-9]+)?[A-Za-z{_SYMBOL_START}][A-Za-z0-9{_SYMBOL_CONTINUE}]*"
)

_HEURISTIC_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "so",
        "the",
        "to",
        "up",
        "we",
        "nc",
        "sl",
        "pl",
    }
)

DEFAULT_LEXICAL_TRIGGER_PREDICATES: list[str] = [
    "http://www.w3.org/2004/02/skos/core#notation",
    "http://qudt.org/schema/qudt/symbol",
    "http://qudt.org/schema/qudt/ucumCode",
]


def tokenize_for_lexical_match(text: str) -> list[str]:
    """Extract case-preserved candidate tokens from raw source text."""
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text)


def looks_like_lexical_code(
    value: str,
    *,
    min_len: int,
    max_len: int,
) -> bool:
    """True when a bare label/altLabel plausibly denotes a formal code."""
    stripped = value.strip()
    if not stripped or any(ch.isspace() for ch in stripped):
        return False
    if len(stripped) < min_len or len(stripped) > max_len:
        return False
    if stripped.lower() in _HEURISTIC_STOPWORDS:
        return False
    has_digit = any(ch.isdigit() for ch in stripped)
    has_mixed_case = any(ch.isupper() for ch in stripped) and any(
        ch.islower() for ch in stripped
    )
    has_symbol_punct = any(not ch.isalnum() for ch in stripped)
    if stripped.isalpha() and stripped.islower() and len(stripped) > 4:
        return False
    return (
        has_digit
        or has_mixed_case
        or has_symbol_punct
        or (stripped.isalpha() and len(stripped) <= 4 and stripped[0].isupper())
    )


def dedupe_preserve_case(values: Iterable[str]) -> list[str]:
    """Deduplicate trigger strings without normalizing case."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = raw.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


class LexicalTriggerIndex(Tool):
    """In-memory case-sensitive token → atom_id index."""

    max_match_atoms: int = Field(
        default=16,
        ge=0,
        description="Maximum atom IDs returned per match call.",
    )

    _token_to_atom_ids: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    _atom_to_tokens: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    _ontology_to_atoms: dict[str, set[str]] = PrivateAttr(default_factory=dict)
    _substring_triggers: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    def clear(self) -> None:
        self._token_to_atom_ids = {}
        self._atom_to_tokens = {}
        self._ontology_to_atoms = {}
        self._substring_triggers = {}

    def register_atoms(self, atoms: Iterable[GraphAtom]) -> None:
        for atom in atoms:
            self.register_atom(atom)

    def register_atom(self, atom: GraphAtom) -> None:
        triggers = dedupe_preserve_case(atom.lexical_triggers)
        if not triggers:
            return
        atom_id = atom.atom_id
        self._atom_to_tokens[atom_id] = triggers
        if atom.ontology_iri:
            self._ontology_to_atoms.setdefault(atom.ontology_iri, set()).add(atom_id)
        for trigger in triggers:
            self._token_to_atom_ids.setdefault(trigger, []).append(atom_id)
            if _needs_substring_scan(trigger):
                self._substring_triggers.setdefault(trigger, []).append(atom_id)

    def unregister_ontology(self, ontology_iri: str) -> None:
        atom_ids = self._ontology_to_atoms.pop(ontology_iri, set())
        for atom_id in atom_ids:
            self._unregister_atom(atom_id)

    def _unregister_atom(self, atom_id: str) -> None:
        triggers = self._atom_to_tokens.pop(atom_id, [])
        for trigger in triggers:
            bucket = self._token_to_atom_ids.get(trigger)
            if bucket is not None:
                self._token_to_atom_ids[trigger] = [
                    aid for aid in bucket if aid != atom_id
                ]
                if not self._token_to_atom_ids[trigger]:
                    del self._token_to_atom_ids[trigger]
            sub_bucket = self._substring_triggers.get(trigger)
            if sub_bucket is not None:
                self._substring_triggers[trigger] = [
                    aid for aid in sub_bucket if aid != atom_id
                ]
                if not self._substring_triggers[trigger]:
                    del self._substring_triggers[trigger]

    def match(self, text: str, *, max_atoms: int | None = None) -> list[str]:
        """Return atom IDs whose lexical triggers appear in ``text``."""
        limit = self.max_match_atoms if max_atoms is None else max_atoms
        if limit <= 0 or not text:
            return []

        hits: list[str] = []
        seen: set[str] = set()

        def add(atom_id: str) -> None:
            if atom_id in seen:
                return
            seen.add(atom_id)
            hits.append(atom_id)

        for token in tokenize_for_lexical_match(text):
            for atom_id in self._token_to_atom_ids.get(token, ()):
                add(atom_id)
                if len(hits) >= limit:
                    return hits[:limit]

        for trigger, atom_ids in sorted(
            self._substring_triggers.items(), key=lambda item: -len(item[0])
        ):
            if _substring_match_with_boundaries(trigger, text):
                for atom_id in atom_ids:
                    add(atom_id)
                    if len(hits) >= limit:
                        return hits[:limit]

        return hits[:limit]


def _substring_match_with_boundaries(trigger: str, text: str) -> bool:
    """True when ``trigger`` occurs in ``text`` at token boundaries.

    A bare ``in`` check let a shorter unit symbol fire inside a longer one:
    ``mA/cm²`` in the text also fired the ``A/cm²`` trigger, and ``/cm``
    fired inside ``/cm²``. An occurrence counts only when the adjacent
    characters are not alphanumeric (superscripts count as numeric).
    """
    start = text.find(trigger)
    while start != -1:
        end = start + len(trigger)
        before_ok = start == 0 or not text[start - 1].isalnum()
        after_ok = end == len(text) or not text[end].isalnum()
        if before_ok and after_ok:
            return True
        start = text.find(trigger, start + 1)
    return False


def _needs_substring_scan(trigger: str) -> bool:
    """Triggers with internal punctuation may not survive tokenization."""
    has_non_ascii = any(ord(ch) > 127 for ch in trigger)
    if len(trigger) < 3 and not has_non_ascii:
        return False
    return any(ch in trigger for ch in "/·⁻²³°")
