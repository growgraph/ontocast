import re
from urllib.parse import urlparse

from ontocast.util import CONVENTIONAL_MAPPINGS


def derive_ontology_id(iri: str) -> str | None:
    if not isinstance(iri, str) or not iri.strip():
        return None

    normalized_iri = iri.strip().rstrip("/#")

    if normalized_iri in CONVENTIONAL_MAPPINGS:
        return CONVENTIONAL_MAPPINGS[normalized_iri]

    parsed = urlparse(normalized_iri)

    candidate = (
        parsed.path.rsplit("/", 1)[-1]
        if parsed.path and "/" in parsed.path
        else parsed.netloc.split(".")[0]
        if parsed.netloc
        else normalized_iri
    )

    return _clean_derived_id(candidate)


def _clean_derived_id(value: str) -> str | None:
    value = re.sub(r"\.(owl|ttl|rdf|xml)$", "", value, flags=re.IGNORECASE)
    match = re.match(r"^(.*?)\.(org|com|net|io|edu|gov|int|mil)$", value, re.IGNORECASE)
    if match:
        value = match.group(1)
    result = re.sub(r"[^a-zA-Z0-9_-]", "", value).lower()
    return result if result else None
