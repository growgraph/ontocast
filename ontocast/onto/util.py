import re
from urllib.parse import urlparse

from ontocast.onto.constants import ONTOLOGY_NULL_ID
from ontocast.util import CONVENTIONAL_MAPPINGS


def derive_ontology_id(iri: str) -> str:
    if not isinstance(iri, str) or not iri.strip():
        return ONTOLOGY_NULL_ID

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


def _clean_derived_id(value: str) -> str:
    value = re.sub(r"\.(owl|ttl|rdf|xml)$", "", value, flags=re.IGNORECASE)
    match = re.match(r"^(.*?)\.(org|com|net|io|edu|gov|int|mil)$", value, re.IGNORECASE)
    if match:
        value = match.group(1)
    return re.sub(r"[^a-zA-Z0-9_-]", "", value).lower() or ONTOLOGY_NULL_ID
