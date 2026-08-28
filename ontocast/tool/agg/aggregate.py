"""Embedding-based RDF graph aggregator.

This module provides the main aggregator class that orchestrates entity
disambiguation using embedding-based clustering.

Pipeline:
1. Collect entities from all content units
2. Normalize entities: e -> r(e) (string representation with semantic context)
3. Generate embedding-based identity candidates
4. Validate candidate merges with symbolic identity checks
5. Select canonical identity per validated cluster
6. Assign final URIs from canonical identity + document namespace policy
7. Rewrite graphs: apply mapping e -> e' to all triples
"""

import logging
import re
from difflib import SequenceMatcher
from enum import StrEnum
from itertools import combinations
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import DCTERMS, FOAF, OWL, RDF, RDFS, XSD

from ontocast.config import AggregationConfig
from ontocast.onto.constants import (
    DEFAULT_IRI,
    PROV,
    SCHEMA,
    prefix_lookup_for_ingest,
)
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.iri_policy import (
    is_in_namespace,
    join_namespace_local,
    normalize_namespace_iri,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.representation_text import (
    normalize_identifier,
    normalize_text,
    normalize_uri_local_name,
)

from .clustering import ClusterRepresentativeSelector
from .normalizer import EntityNormalizer, EntityRepresentation
from .rewriter import GraphRewriter
from .signatures import (
    MergeGuardContext,
    build_sibling_pairs,
    empirically_functional_predicates,
    harvest_max_one_predicates,
    label_tokens,
    labels_alias_with_initials,
    labels_differ_only_by_initials,
    string_values_compatible,
    tokens_alias_compatible,
)
from .uri_builder import EntityRole, URIBuilder, to_lower_camel_case

logger = logging.getLogger(__name__)
_INSTANCE_LOCAL_NAME_RE = re.compile(r"^(?P<stem>.+?)(?P<index>\d+)$")

# Natural-key evidence bounds: values longer than this are prose payloads
# (notes, descriptions), not identifiers; values shared by more entities than
# this are generic attributes ("bulgaria"), not identifying marks.
_NATURAL_KEY_MAX_VALUE_LENGTH = 64
_NATURAL_KEY_MAX_VALUE_ENTITIES = 8

_DOC_METADATA_FIRST_CLASS: dict[str, URIRef] = {
    "title": DCTERMS.title,
    "published": DCTERMS.issued,
    "issued": DCTERMS.issued,
    "source_system": PROV.wasAttributedTo,
}
_DOC_METADATA_IDENTIFIER_KEYS = frozenset({"doi", "isbn", "pmid", "arxiv_id", "handle"})
_DOC_METADATA_SOURCE_KEYS = frozenset({"source_uri", "source_url"})
_DOC_METADATA_ENTITY_LINKS: dict[str, tuple[URIRef, URIRef]] = {
    "author": (DCTERMS.creator, SCHEMA.Person),
    "authors": (DCTERMS.creator, SCHEMA.Person),
    "creator": (DCTERMS.creator, SCHEMA.Person),
    "project": (DCTERMS.relation, PROV.Entity),
}
# Canonicals that get optional ``id`` prefix/suffix expansion in the alias map.
_DOC_METADATA_ID_AFFIX_CANONICALS = (
    _DOC_METADATA_IDENTIFIER_KEYS
    | _DOC_METADATA_SOURCE_KEYS
    | frozenset({"stable_source_iri"})
)
_DEFAULT_ENTITY_LINK_PREDICATE = DCTERMS.relation
_DEFAULT_ENTITY_TYPE = PROV.Entity

_DCTERMS_IDENTIFIER_CLASS = URIRef(str(DCTERMS) + "Identifier")

# Closed set of identifier-shaped affixes for the fallback structured-id path.
# ``key`` is gated here (never expanded into the registry alias map).
_IDENTIFIER_AFFIXES = frozenset(
    {
        "id",
        "uid",
        "uuid",
        "guid",
        "ref",
        "reference",
        "no",
        "num",
        "number",
        "code",
        "slug",
        "handle",
        "accession",
        "key",
    }
)


def _aliases_for(canonical: str, *, with_id_affix: bool = False) -> set[str]:
    """Expand a canonical metadata key into separator / optional ``id`` aliases."""
    toks = normalize_identifier(canonical).split()
    if not toks:
        return set()
    forms = {"_".join(toks), "".join(toks)}
    if not with_id_affix:
        return forms
    if toks[-1] == "id":
        stem = toks[:-1]
        if stem:
            forms.add("_".join(stem))
            forms.add("".join(stem))
    else:
        stem = toks
        forms |= {
            "_".join([*stem, "id"]),
            "id_" + "_".join(stem),
            "".join(stem) + "id",
            "id" + "".join(stem),
        }
    return forms


def _build_doc_metadata_key_aliases() -> dict[str, str]:
    """Map normalized alias forms to canonical registry keys.

    ``id`` affix expansion is scoped to bibliographic identifiers, source keys,
    and ``stable_source_iri`` so ``project_id`` / ``title_id`` are not silently
    folded into entity-link or first-class keys (those go through the fallback
    structured-identifier path instead).
    """
    aliases: dict[str, str] = {}
    for canonical in sorted(_DOC_METADATA_ID_AFFIX_CANONICALS):
        for alias in _aliases_for(canonical, with_id_affix=True):
            aliases[alias] = canonical

    plain_canonicals = (
        set(_DOC_METADATA_FIRST_CLASS)
        | set(_DOC_METADATA_ENTITY_LINKS)
        | frozenset({"identifiers"})
    )
    for canonical in sorted(plain_canonicals):
        for alias in _aliases_for(canonical, with_id_affix=False):
            aliases.setdefault(alias, canonical)
    return aliases


_DOC_METADATA_KEY_ALIASES = _build_doc_metadata_key_aliases()


def _resolve_metadata_key(key: str) -> str:
    """Resolve a caller metadata key to a canonical registry name.

    Matching is case-insensitive and tolerant of camelCase / snake_case /
    kebab-case. Optional leading/trailing ``id`` affixes apply only to
    bibliographic identifier and source keys. Unknown keys are returned
    unchanged.
    """
    tokens = normalize_identifier(key).split()
    if not tokens:
        return key
    for form in ("_".join(tokens), "".join(tokens)):
        canonical = _DOC_METADATA_KEY_ALIASES.get(form)
        if canonical is not None:
            return canonical
    return key


def _split_identifier_affix(key: str) -> tuple[str, str] | None:
    """Split a key into ``(stem, affix)`` when it carries an identifier affix.

    Recognizes a leading or trailing token from :data:`_IDENTIFIER_AFFIXES`
    after camel/snake/kebab normalization. Returns ``None`` when there is no
    affix or the stem would be empty.
    """
    tokens = normalize_identifier(key).split()
    if len(tokens) < 2:
        return None
    if tokens[-1] in _IDENTIFIER_AFFIXES:
        stem = "_".join(tokens[:-1])
        return (stem, tokens[-1]) if stem else None
    if tokens[0] in _IDENTIFIER_AFFIXES:
        stem = "_".join(tokens[1:])
        return (stem, tokens[0]) if stem else None
    return None


def _as_iri_or_literal(value: object) -> URIRef | Literal:
    text = str(value).strip()
    if text.startswith(("http://", "https://", "urn:")):
        return URIRef(text)
    return Literal(text)


def _add_structured_identifier(
    graph: RDFGraph,
    doc_iri: URIRef,
    *,
    scheme: str,
    value: object,
) -> None:
    bnode = BNode()
    graph.add((doc_iri, DCTERMS.identifier, bnode))
    graph.add((bnode, RDF.type, _DCTERMS_IDENTIFIER_CLASS))
    graph.add((bnode, DCTERMS.type, Literal(scheme)))
    graph.add((bnode, RDF.value, Literal(str(value))))


def _resolve_entity_type(type_hint: str | None, default: URIRef) -> URIRef:
    if not type_hint:
        return default
    text = str(type_hint).strip()
    if not text:
        return default
    if text.startswith(("http://", "https://", "urn:")):
        return URIRef(text)
    prefix, sep, local = text.partition(":")
    if not sep or not local:
        return default
    ns = prefix_lookup_for_ingest().get(prefix)
    if not ns:
        return default
    return URIRef(f"{ns}{local}")


def _mint_metadata_entity(
    graph: RDFGraph,
    entity_namespace: str,
    name: str,
    *,
    rdf_type: URIRef,
    identifier: str | None,
) -> URIRef:
    local = to_lower_camel_case(normalize_text(name)) or "entity"
    local = re.sub(r"[^\w]", "", local) or "entity"
    entity_iri = URIRef(join_namespace_local(entity_namespace, local, context="facts"))
    graph.add((entity_iri, RDF.type, rdf_type))
    graph.add((entity_iri, RDFS.label, Literal(name)))
    if identifier:
        graph.add((entity_iri, DCTERMS.identifier, Literal(str(identifier))))
    return entity_iri


def _emit_metadata_entities(
    graph: RDFGraph,
    doc_iri: URIRef,
    entity_namespace: str,
    *,
    link: URIRef,
    default_type: URIRef,
    value: object,
) -> list[URIRef]:
    """Mint linked entities for a metadata value; return minted IRIs."""
    items = value if isinstance(value, list) else [value]
    minted: list[URIRef] = []
    for item in items:
        if item is None or item == "":
            continue
        if isinstance(item, dict):
            name = item.get("name")
            if not name:
                continue
            raw_type = item.get("type")
            type_hint = raw_type if isinstance(raw_type, str) else None
            rdf_type = _resolve_entity_type(type_hint, default_type)
            raw_id = item.get("identifier")
            identifier = str(raw_id) if raw_id is not None else None
        else:
            name = str(item)
            rdf_type = default_type
            identifier = None
        entity_iri = _mint_metadata_entity(
            graph,
            entity_namespace,
            str(name),
            rdf_type=rdf_type,
            identifier=identifier,
        )
        graph.add((doc_iri, link, entity_iri))
        minted.append(entity_iri)
    return minted


def apply_document_metadata_provenance(
    doc_iri: URIRef,
    metadata: dict[str, Any],
    graph: RDFGraph,
    *,
    entity_namespace: str | None = None,
) -> None:
    """Emit caller-asserted document identity triples on ``doc_iri``.

    Document-level identity is provenance-adjacent but intentionally kept on the
    facts graph (survives chunk-level ``strip_provenance``) so query/RAG clients
    can filter by DOI, business id, filename, etc.

    Business-oriented keys (``author``, ``project``, and any non-reserved key)
    mint typed RDF entities under ``entity_namespace`` (defaults to the document
    facts namespace) so they are SPARQL-discoverable via ``rdf:type``.

    Registry keys are matched via :func:`_resolve_metadata_key` (case /
    separator / optional ``id`` affix for identifier and source keys). Keys with
    an identifier-shaped affix (``id``, ``ref``, ``no``, ``key``, …) that do not
    resolve to a registry entry become structured ``dcterms:identifier`` blank
    nodes, or attach to a companion entity-link stem when one was minted in the
    same payload (e.g. ``project`` + ``project_id``).
    """
    if not metadata:
        return

    graph.bind("prov", str(PROV))
    graph.bind("foaf", str(FOAF))
    graph.bind("dcterms", str(DCTERMS))
    graph.bind("owl", str(OWL))
    graph.bind("rdfs", str(RDFS))
    graph.bind("schema", str(SCHEMA))

    graph.add((doc_iri, RDF.type, PROV.Entity))
    graph.add((doc_iri, RDF.type, FOAF.Document))

    ns = entity_namespace or normalize_namespace_iri(str(doc_iri), context="facts")
    entity_iri_by_stem: dict[str, URIRef] = {}
    deferred_affix: list[tuple[str, object, tuple[str, str]]] = []

    for raw_key, value in metadata.items():
        if value is None or value == "":
            continue
        key = _resolve_metadata_key(raw_key)
        if key == "stable_source_iri":
            graph.add((doc_iri, OWL.sameAs, _as_iri_or_literal(value)))
            continue
        if key in _DOC_METADATA_SOURCE_KEYS:
            graph.add((doc_iri, DCTERMS.source, _as_iri_or_literal(value)))
            continue
        if key in _DOC_METADATA_IDENTIFIER_KEYS:
            graph.add((doc_iri, DCTERMS.identifier, Literal(str(value))))
            continue
        if key == "identifiers":
            items = value if isinstance(value, list) else [value]
            for item in items:
                if not isinstance(item, dict):
                    continue
                scheme = item.get("scheme") or item.get("type")
                val = item.get("value")
                if scheme and val is not None and val != "":
                    _add_structured_identifier(
                        graph, doc_iri, scheme=str(scheme), value=val
                    )
            continue
        if key in _DOC_METADATA_FIRST_CLASS:
            predicate = _DOC_METADATA_FIRST_CLASS[key]
            if isinstance(value, list):
                for item in value:
                    if item is not None and item != "":
                        graph.add((doc_iri, predicate, Literal(str(item))))
            else:
                graph.add((doc_iri, predicate, Literal(str(value))))
            continue

        if key in _DOC_METADATA_ENTITY_LINKS:
            link, default_type = _DOC_METADATA_ENTITY_LINKS[key]
            minted = _emit_metadata_entities(
                graph,
                doc_iri,
                ns,
                link=link,
                default_type=default_type,
                value=value,
            )
            # Companion ``*_id`` attachment only for a singular non-list entity.
            if not isinstance(value, list) and len(minted) == 1:
                entity_iri_by_stem[key] = minted[0]
            continue

        split = _split_identifier_affix(key)
        if split is not None:
            deferred_affix.append((key, value, split))
            continue

        _emit_metadata_entities(
            graph,
            doc_iri,
            ns,
            link=_DEFAULT_ENTITY_LINK_PREDICATE,
            default_type=_DEFAULT_ENTITY_TYPE,
            value=value,
        )

    for _key, value, (stem, _affix) in deferred_affix:
        entity_iri = entity_iri_by_stem.get(stem)
        if entity_iri is not None:
            if isinstance(value, list):
                for item in value:
                    if item is not None and item != "":
                        graph.add((entity_iri, DCTERMS.identifier, Literal(str(item))))
            else:
                graph.add((entity_iri, DCTERMS.identifier, Literal(str(value))))
            continue
        if isinstance(value, list):
            for item in value:
                if item is not None and item != "":
                    _add_structured_identifier(graph, doc_iri, scheme=stem, value=item)
        else:
            _add_structured_identifier(graph, doc_iri, scheme=stem, value=value)


class EntityClassification(StrEnum):
    """Classification of entities during aggregation."""

    FACT = "fact"
    KNOWN_ONTOLOGY = "known_ontology"
    TENTATIVE_ONTOLOGY = "tentative_ontology"


class EntityDecision(BaseModel):
    """Decision record for one entity across aggregation stages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    classification: EntityClassification
    identity_target: URIRef
    final_uri: URIRef | None = None
    suppress_fact_subject_assertions: bool = False
    suppress_sameas: bool = False


def build_merged_clusters(
    final_mapping: dict[URIRef, URIRef],
    identity_mapping: dict[URIRef, URIRef],
) -> dict[str, list[str]]:
    """Group merge clusters by canonical identity, keyed by every final URI.

    One canonical spanning several source documents mints one final URI per
    doc base; keying by final URI alone would split the same merge decision
    into per-document clusters, and a validation veto on one flagged URI would
    leave the sibling document's half of the cluster merged. Every final URI
    rendering a canonical therefore keys the *full* cross-document member set.
    """
    members_by_canonical: dict[str, set[str]] = {}
    final_uris_by_canonical: dict[str, set[str]] = {}
    for entity, final_uri in final_mapping.items():
        canonical = str(identity_mapping.get(entity, entity))
        members_by_canonical.setdefault(canonical, set()).add(str(entity))
        final_uris_by_canonical.setdefault(canonical, set()).add(str(final_uri))
    merged_clusters: dict[str, list[str]] = {}
    for canonical, final_uris in final_uris_by_canonical.items():
        members = members_by_canonical[canonical]
        if len(members) < 2:
            continue
        for final_uri in final_uris:
            merged_clusters[final_uri] = sorted(members)
    return merged_clusters


class AggregationResult(BaseModel):
    """Outcome of one aggregation pass, including merge bookkeeping.

    Attributes:
        graph: Merged facts graph with provenance annotations.
        decisions: Per-entity decision records (classification, identity
            target, final URI).
        merged_clusters: Final URI -> all source entities sharing the same
            canonical identity, restricted to clusters where >= 2 distinct
            entities merged. A canonical spanning several documents mints one
            final URI per doc base; every such final URI keys the *full*
            cross-document cluster, so a validation veto dissolves the whole
            merge decision rather than one document's half. Keys/values are
            strings so the mapping can live on
            :class:`~ontocast.onto.state.AgentState` between graph nodes.
        rejected_merge_count: Candidate merges rejected by symbolic
            validation (guards, roles, types, lexical bar).
        key_supported_clusters: Final URIs of merged clusters containing at
            least one natural-key pair (a shared identifier value). The
            validation gate treats label disagreement inside these clusters
            as name variance rather than a merge signature.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: RDFGraph
    decisions: dict[URIRef, EntityDecision] = Field(default_factory=dict)
    merged_clusters: dict[str, list[str]] = Field(default_factory=dict)
    rejected_merge_count: int = 0
    key_supported_clusters: list[str] = Field(default_factory=list)


class _EntityCollectionState(BaseModel):
    """Mutable state for entity collection across content units."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    known_entities: set[URIRef]
    entities: set[URIRef] = Field(default_factory=set)
    source_entities: set[URIRef] = Field(default_factory=set)
    entity_graphs: dict[URIRef, RDFGraph] = Field(default_factory=dict)
    entity_doc_iris: dict[URIRef, URIRef] = Field(default_factory=dict)
    entity_classification: dict[URIRef, EntityClassification] = Field(
        default_factory=dict
    )
    direct_relation_pairs: set[frozenset[URIRef]] = Field(default_factory=set)
    object_groups: dict[tuple[URIRef, URIRef], set[URIRef]] = Field(
        default_factory=dict
    )


_STANDARD_NAMESPACES = (
    str(RDF),
    str(RDFS),
    str(OWL),
    str(XSD),
    str(SCHEMA),
    str(PROV),
)


class EmbeddingBasedAggregator:
    """Main aggregator using embedding-based entity disambiguation.

    Pipeline stages:
    1. Entity normalisation (with semantic context)
    2. Parallel embedding
    3. Similarity-based clustering
    4. Representative selection (prefer ontology, then simplicity)
    5. URI normalisation (PascalCase/camelCase under DEFAULT_IRI)
    6. Graph rewriting

    ContentUnit types are handled as follows:
    - ``facts``: entities under ``base_iri`` are normalised.
    - ``ontology``: all other entities are considered ontology entities and preserved.
    """

    def __init__(
        self,
        config: AggregationConfig | None = None,
        *,
        add_sameas_links: bool = True,
        base_iri: str = DEFAULT_IRI,
        candidate_similarity_threshold: float | None = None,
    ):
        """Initialise the embedding-based aggregator.

        Every tunable lives on :class:`AggregationConfig`, so ``settings.py``
        stays the single source of their defaults rather than restating them in
        this signature and again at the call site.

        Args:
            config: Aggregation tunables. Defaults to :class:`AggregationConfig`,
                i.e. the environment-resolved settings.
            add_sameas_links: Whether to add ``owl:sameAs`` for merged entities.
                Not config-driven: callers choose it per use, and the entity
                aligner wants different behaviour from the pipeline.
            base_iri: Base IRI for fact entity URIs. Entities under this
                namespace are facts; everything else is treated as an ontology
                entity and left unchanged.
            candidate_similarity_threshold: Overrides the configured permissive
                candidate threshold. The entity aligner pins it to its own
                similarity threshold rather than the pipeline's.
        """
        cfg = config or AggregationConfig()

        self.base_iri = base_iri
        self.candidate_similarity_threshold = (
            cfg.candidate_similarity_threshold
            if candidate_similarity_threshold is None
            else candidate_similarity_threshold
        )
        self.lexical_label_jaccard = cfg.lexical_label_jaccard
        self.lexical_sequence_ratio = cfg.lexical_sequence_ratio
        self.lexical_token_jaccard = cfg.lexical_token_jaccard
        self.functional_min_empirical_support = cfg.functional_min_empirical_support
        self.sibling_guard_scope = str(cfg.sibling_guard_scope)
        self.literal_conflict_guard = cfg.literal_conflict_guard
        self.initials_distinct_guard = cfg.initials_distinct_guard
        self.natural_key_merge = cfg.natural_key_merge
        self.type_guard_untyped = str(cfg.type_guard_untyped)

        # Pipeline components (EntityClusterer imports sklearn/ST lazily).
        # The clusterer runs at the permissive candidate threshold: candidates
        # are validated symbolically afterwards, so there is exactly one
        # clustering threshold on this path.
        from .clustering import EntityClusterer

        self.normalizer = EntityNormalizer(facts_iri=self.base_iri)
        self.clusterer = EntityClusterer(
            embedding_model=cfg.embedding_model,
            similarity_threshold=self.candidate_similarity_threshold,
        )
        self.selector = ClusterRepresentativeSelector()
        self.uri_builder = URIBuilder(base_iri=self.base_iri)
        self.rewriter = GraphRewriter(
            add_sameas_links=add_sameas_links,
            blocked_sameas_namespaces=(self.base_iri,),
        )

    @staticmethod
    def _entity_in_namespace(entity: URIRef, namespace: URIRef | str | None) -> bool:
        """Return True when *entity* is under the provided namespace."""
        if namespace is None:
            return False
        return is_in_namespace(str(entity), str(namespace), context="auto")

    def _is_fact_entity_in_unit(self, entity: URIRef, unit: ContentUnit) -> bool:
        """Classify whether an entity should be treated as a fact in this unit.

        Facts are entities in either:
        - the configured base facts namespace (``base_iri``), or
        - the unit document namespace (``unit.doc_iri``).
        """
        return self._entity_in_namespace(
            entity, self.base_iri
        ) or self._entity_in_namespace(entity, unit.doc_iri)

    @staticmethod
    def _is_standard_ontology_entity(entity: URIRef) -> bool:
        """Return True for entities from built-in standard RDF vocabularies."""
        entity_str = str(entity)
        return any(entity_str.startswith(prefix) for prefix in _STANDARD_NAMESPACES)

    def _build_known_ontology_entities(
        self, ontology_graph: RDFGraph | None
    ) -> set[URIRef]:
        """Build a set of known ontology entities from ontology and std vocabularies."""
        known_entities: set[URIRef] = set()

        if ontology_graph is not None:
            for s, p, o in ontology_graph:
                if isinstance(s, URIRef):
                    known_entities.add(s)
                if isinstance(p, URIRef):
                    known_entities.add(p)
                if isinstance(o, URIRef):
                    known_entities.add(o)

        return known_entities

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        # Short tokens stay: initials and single-letter identifiers
        # ("company S." vs "company T.") are often the only distinguishing
        # mark, and dropping them made such labels compare identical.
        return set(label_tokens(text))

    @staticmethod
    def _role_key(representation: EntityRepresentation) -> str:
        role = (
            representation.role
            if representation.role is not None
            else EntityRole.INSTANCE
        )
        return str(role)

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        return len(left & right) / len(union)

    @staticmethod
    def _instance_like_local_name(entity: URIRef) -> str | None:
        """Return normalized local name when URI ends with numeric suffix."""
        local_name = normalize_uri_local_name(entity).replace(" ", "")
        if not local_name:
            return None
        match = _INSTANCE_LOCAL_NAME_RE.match(local_name)
        if match is None:
            return None
        if len(match.group("stem")) < 3:
            return None
        return local_name

    def _are_roles_compatible(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> bool:
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        return self._role_key(left_rep) == self._role_key(right_rep)

    def _are_types_compatible(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> bool:
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        left_types = set(left_rep.types)
        right_types = set(right_rep.types)
        if not left_types or not right_types:
            if self.type_guard_untyped == "strict":
                # Strict mode fails a typed-vs-untyped pair closed; two
                # untyped entities stay comparable — there is no type
                # evidence in either direction.
                return not left_types and not right_types
            return True
        return bool(left_types & right_types)

    def _entity_label_values(self, rep: EntityRepresentation) -> set[str]:
        """Normalized name strings an entity is identified by.

        ``alt_labels`` (string literals from arbitrary domain predicates)
        stand in only when the entity carries no ``rdfs:label``: for a
        labeled entity they are payload, not names — an honorific or role
        literal shared by several people must not read as label agreement.
        """
        source = rep.labels if rep.labels else rep.alt_labels
        return {
            self.normalizer.normalize_string(label) for label in source if label.strip()
        }

    def _are_lexical_aliases(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> bool:
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        if left_rep.normal_form == right_rep.normal_form:
            return True

        left_instance_name = self._instance_like_local_name(left)
        right_instance_name = self._instance_like_local_name(right)
        if (
            left_instance_name is not None
            and right_instance_name is not None
            and left_instance_name == right_instance_name
        ):
            return True

        left_label_tokens = self._entity_label_values(left_rep)
        right_label_tokens = self._entity_label_values(right_rep)
        if left_label_tokens & right_label_tokens:
            return True

        # Abbreviation-aware tier: "baranov d" vs "dmitry baranov" alias when
        # every token of one label matches a token of the other exactly or as
        # a single-character initial, with at least one shared full token.
        if self._labels_alias_with_initials(left_label_tokens, right_label_tokens):
            return True

        # Guard-literal-bearing entities (measurements, dated events) are
        # individuated by their payload, not their phrasing: "PL red shift of
        # SL1" vs "PL red shift of SL2" share most tokens yet denote distinct
        # values. Only the exact tiers above may merge them. String literals
        # (names, descriptions) do not raise this bar — disjoint identifier
        # strings are handled by _have_conflicting_literals instead.
        if left_rep.has_guard_literal and right_rep.has_guard_literal:
            return False

        if left_label_tokens and right_label_tokens:
            max_label_overlap = 0.0
            for left_label in left_label_tokens:
                left_tokens = self._tokenize(left_label)
                for right_label in right_label_tokens:
                    right_tokens = self._tokenize(right_label)
                    overlap = self._jaccard(left_tokens, right_tokens)
                    max_label_overlap = max(max_label_overlap, overlap)
            if max_label_overlap >= self.lexical_label_jaccard:
                return True

        left_normalized = left_rep.normal_form.strip()
        right_normalized = right_rep.normal_form.strip()
        if left_normalized and right_normalized:
            if left_normalized != right_normalized and (
                left_normalized.startswith(f"{right_normalized} ")
                or right_normalized.startswith(f"{left_normalized} ")
            ):
                return False

        ratio = SequenceMatcher(
            None, left_rep.normal_form, right_rep.normal_form
        ).ratio()
        if ratio >= self.lexical_sequence_ratio:
            return True

        left_tokens = self._tokenize(left_rep.normal_form)
        right_tokens = self._tokenize(right_rep.normal_form)
        if len(left_tokens) >= 2 and len(right_tokens) >= 2:
            if self._jaccard(left_tokens, right_tokens) >= self.lexical_token_jaccard:
                return True

        return False

    # Thin delegates: the shared implementations live in ``signatures`` so the
    # validation gate can consult the same string-compatibility notion without
    # importing the aggregator.
    _tokens_alias_compatible = staticmethod(tokens_alias_compatible)
    _labels_alias_with_initials = staticmethod(labels_alias_with_initials)
    _string_values_compatible = staticmethod(string_values_compatible)

    @classmethod
    def _have_conflicting_literals(
        cls,
        left_rep: EntityRepresentation,
        right_rep: EntityRepresentation,
    ) -> bool:
        """Return True when the entities assert disjoint values per predicate.

        A shared predicate with two non-empty, disjoint canonical value sets
        (numeric/temporal) marks the entities as distinct individuals; overlap
        or one-sided values read as re-mention/enrichment and stay mergeable.
        String payloads (identifiers, codes) conflict only when NO cross-pair
        is compatible (equality, prefix, or initial-abbreviation) — "d" vs
        "dmitry" is a re-mention, "S-2024-001" vs "S-2024-002" is a conflict.
        """
        for predicate, left_values in left_rep.predicate_literals.items():
            right_values = right_rep.predicate_literals.get(predicate)
            if not right_values or not left_values:
                continue
            if left_values.isdisjoint(right_values):
                return True
        for predicate, left_strings in left_rep.predicate_string_literals.items():
            right_strings = right_rep.predicate_string_literals.get(predicate)
            if not right_strings or not left_strings:
                continue
            if not any(
                cls._string_values_compatible(left_value, right_value)
                for left_value in left_strings
                for right_value in right_strings
            ):
                return True
        return False

    @staticmethod
    def _have_conflicting_functional_objects(
        left_rep: EntityRepresentation,
        right_rep: EntityRepresentation,
        functional_predicates: set[URIRef],
    ) -> bool:
        """Return True when a max-1 object predicate points at disjoint IRIs.

        Catches conflicts invisible to value comparison — e.g. two "10"
        quantities whose ``qudt:unit`` objects are ``DEG_C`` vs ``KiloHZ``.
        """
        if not functional_predicates:
            return False
        for predicate, left_objects in left_rep.predicate_iri_objects.items():
            if predicate not in functional_predicates:
                continue
            right_objects = right_rep.predicate_iri_objects.get(predicate)
            if not right_objects or not left_objects:
                continue
            if left_objects.isdisjoint(right_objects):
                return True
        return False

    def _labels_confirm_identity(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> bool:
        """Exact or initials-aware label agreement strong enough to skip cosine."""
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return False
        left_labels = self._entity_label_values(left_rep)
        right_labels = self._entity_label_values(right_rep)
        if not left_labels or not right_labels:
            return False
        if left_labels & right_labels:
            return True
        return self._labels_alias_with_initials(left_labels, right_labels)

    def _labels_mark_distinct_entities(
        self,
        left_rep: EntityRepresentation,
        right_rep: EntityRepresentation,
    ) -> bool:
        """Label pairs identical except for conflicting initials mark distinctness."""
        if not self.initials_distinct_guard:
            return False
        return labels_differ_only_by_initials(
            self._entity_label_values(left_rep),
            self._entity_label_values(right_rep),
        )

    def _pair_distinctness_veto(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
        direct_relation_pairs: set[frozenset[URIRef]] | None = None,
        guard_context: MergeGuardContext | None = None,
    ) -> bool:
        """Positive evidence that *left* and *right* denote distinct entities.

        Unlike the absence of a lexical alias — which merely fails to support
        a merge — a veto is grounds to keep the pair apart in *any* identity
        cluster, including transitively: two entities that a guard separates
        must not end up merged through a chain of intermediate aliases.
        """
        pair = frozenset((left, right))
        if direct_relation_pairs is not None and pair in direct_relation_pairs:
            return True
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if guard_context is not None:
            if pair in guard_context.sibling_pairs:
                return True
            if left_rep is not None and right_rep is not None:
                if self.literal_conflict_guard and self._have_conflicting_literals(
                    left_rep, right_rep
                ):
                    return True
                if self._have_conflicting_functional_objects(
                    left_rep, right_rep, guard_context.functional_predicates
                ):
                    return True
        if left_rep is not None and right_rep is not None:
            if self._labels_mark_distinct_entities(left_rep, right_rep):
                return True
        if not self._are_roles_compatible(left, right, representations):
            return True
        if not self._are_types_compatible(left, right, representations):
            return True
        return False

    def _can_merge_as_identity(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
        direct_relation_pairs: set[frozenset[URIRef]] | None = None,
        guard_context: MergeGuardContext | None = None,
        key_pairs: set[frozenset[URIRef]] | None = None,
    ) -> bool:
        if self._pair_distinctness_veto(
            left,
            right,
            representations,
            direct_relation_pairs=direct_relation_pairs,
            guard_context=guard_context,
        ):
            return False
        if key_pairs is not None and frozenset((left, right)) in key_pairs:
            # A shared value on a single-valued identifier-like predicate is
            # positive identity evidence in its own right; the guards above
            # still had their say.
            return True
        return self._are_lexical_aliases(left, right, representations)

    def _collect_natural_key_pairs(
        self,
        representations: dict[URIRef, EntityRepresentation],
        schema_functional_predicates: set[URIRef],
    ) -> set[frozenset[URIRef]]:
        """Instance pairs sharing a value on a single-valued identifier predicate.

        Every guard in this module is a veto; this is the one source of
        *positive* symbolic identity evidence: two instances asserting the
        same value for a predicate that behaves like an identifier (declared
        max-1 by the schema, or observed single-valued on every subject) are
        candidate re-mentions of one entity — "Application no. 36760/06" is
        the same case wherever its number appears. Pairs found here are still
        subject to all distinctness vetoes; string values only (dates and
        numbers are coordinates, not identifiers), short values only (prose
        payloads such as notes and descriptions are not keys), and values
        shared too widely are treated as generic rather than identifying.
        """
        instance_role = str(EntityRole.INSTANCE)
        by_predicate: dict[URIRef, dict[URIRef, set[str]]] = {}
        for entity, rep in representations.items():
            if self._role_key(rep) != instance_role:
                continue
            for predicate, values in rep.predicate_string_literals.items():
                filtered = {
                    value
                    for value in values
                    if 0 < len(value) <= _NATURAL_KEY_MAX_VALUE_LENGTH
                }
                if filtered:
                    by_predicate.setdefault(predicate, {})[entity] = filtered

        pairs: set[frozenset[URIRef]] = set()
        for predicate, entity_values in by_predicate.items():
            if predicate not in schema_functional_predicates:
                if len(entity_values) < self.functional_min_empirical_support:
                    continue
                if any(len(values) != 1 for values in entity_values.values()):
                    continue
            value_index: dict[str, list[URIRef]] = {}
            for entity, values in entity_values.items():
                for value in values:
                    value_index.setdefault(value, []).append(entity)
            for value, entities in value_index.items():
                if not 2 <= len(entities) <= _NATURAL_KEY_MAX_VALUE_ENTITIES:
                    continue
                for left, right in combinations(sorted(entities, key=str), 2):
                    pairs.add(frozenset((left, right)))
        return pairs

    @staticmethod
    def _merge_candidate_clusters_by_key_pairs(
        candidate_clusters: list[list[URIRef]],
        key_pairs: set[frozenset[URIRef]],
    ) -> list[list[URIRef]]:
        """Join candidate clusters bridged by a natural-key pair.

        Embedding clustering only proposes pairs that read alike; two mentions
        of one entity under different surface forms ("Application no. X" vs
        "Case A v. B") never co-cluster, so a key pair spanning two candidate
        clusters must pull them into one before symbolic validation — which
        still adjudicates every pair inside the joined cluster.
        """
        if not key_pairs:
            return candidate_clusters
        cluster_of: dict[URIRef, int] = {}
        for index, cluster in enumerate(candidate_clusters):
            for entity in cluster:
                cluster_of[entity] = index

        parent = list(range(len(candidate_clusters)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for pair in key_pairs:
            left, right = tuple(pair)
            left_index = cluster_of.get(left)
            right_index = cluster_of.get(right)
            if left_index is None or right_index is None:
                continue
            left_root, right_root = find(left_index), find(right_index)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        grouped: dict[int, list[URIRef]] = {}
        for index, cluster in enumerate(candidate_clusters):
            grouped.setdefault(find(index), []).extend(cluster)
        return list(grouped.values())

    def _cluster_entities_by_role(
        self, representations: dict[URIRef, EntityRepresentation]
    ) -> tuple[list[list[URIRef]], dict[URIRef, np.ndarray]]:
        grouped_entities: dict[str, dict[URIRef, EntityRepresentation]] = {}
        for entity, representation in representations.items():
            grouped_entities.setdefault(self._role_key(representation), {})[entity] = (
                representation
            )

        all_clusters: list[list[URIRef]] = []
        all_embeddings: dict[URIRef, np.ndarray] = {}
        for role_representations in grouped_entities.values():
            role_clusters, role_embeddings = self.clusterer.cluster_entities(
                role_representations
            )
            all_clusters.extend(role_clusters)
            all_embeddings.update(role_embeddings)
        return all_clusters, all_embeddings

    @staticmethod
    def _candidate_similarity(
        left: URIRef,
        right: URIRef,
        embeddings: dict[URIRef, np.ndarray],
    ) -> float | None:
        left_embedding = embeddings.get(left)
        right_embedding = embeddings.get(right)
        if left_embedding is None or right_embedding is None:
            return None

        denominator = float(
            np.linalg.norm(left_embedding) * np.linalg.norm(right_embedding)
        )
        if denominator == 0:
            return None
        return float(np.dot(left_embedding, right_embedding) / denominator)

    def _merge_validation_failures(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
        guard_context: MergeGuardContext | None = None,
    ) -> list[str]:
        failures: list[str] = []
        if guard_context is not None:
            if frozenset((left, right)) in guard_context.sibling_pairs:
                failures.append("sibling")
            left_rep = representations.get(left)
            right_rep = representations.get(right)
            if left_rep is not None and right_rep is not None:
                if self.literal_conflict_guard and self._have_conflicting_literals(
                    left_rep, right_rep
                ):
                    failures.append("literal_conflict")
                if self._have_conflicting_functional_objects(
                    left_rep, right_rep, guard_context.functional_predicates
                ):
                    failures.append("functional_iri_conflict")
                if self._labels_mark_distinct_entities(left_rep, right_rep):
                    failures.append("initials_conflict")
        if not self._are_roles_compatible(left, right, representations):
            failures.append("role")
        if not self._are_types_compatible(left, right, representations):
            failures.append("type")
        if not self._are_lexical_aliases(left, right, representations):
            failures.append("lexical")
        return failures

    def _build_identity_clusters(
        self,
        candidate_clusters: list[list[URIRef]],
        representations: dict[URIRef, EntityRepresentation],
        embeddings: dict[URIRef, np.ndarray],
        direct_relation_pairs: set[frozenset[URIRef]] | None = None,
        guard_context: MergeGuardContext | None = None,
        key_pairs: set[frozenset[URIRef]] | None = None,
    ) -> tuple[
        list[list[URIRef]], list[tuple[URIRef, URIRef, float | None, tuple[str, ...]]]
    ]:
        validated_clusters: list[list[URIRef]] = []
        rejected_merges: list[tuple[URIRef, URIRef, float | None, tuple[str, ...]]] = []

        for candidate_cluster in candidate_clusters:
            if len(candidate_cluster) <= 1:
                validated_clusters.append(candidate_cluster)
                continue

            ordered_cluster = sorted(candidate_cluster, key=str)
            parents: dict[URIRef, URIRef] = {
                entity: entity for entity in ordered_cluster
            }
            members: dict[URIRef, set[URIRef]] = {
                entity: {entity} for entity in ordered_cluster
            }

            # Distinctness vetoes hold cluster-wide: an accepted A–B edge and
            # an accepted B–C edge must not merge a vetoed A–C pair through
            # transitive closure. Computed for every pair up front (the guards
            # are cheap symbolic checks) so unions can be checked against all
            # current members of both sides.
            vetoed_pairs: set[frozenset[URIRef]] = {
                frozenset((left, right))
                for left, right in combinations(ordered_cluster, 2)
                if self._pair_distinctness_veto(
                    left,
                    right,
                    representations,
                    direct_relation_pairs=direct_relation_pairs,
                    guard_context=guard_context,
                )
            }

            def find(entity: URIRef) -> URIRef:
                root = parents[entity]
                if root != entity:
                    parents[entity] = find(root)
                return parents[entity]

            def union_blocked(left_root: URIRef, right_root: URIRef) -> bool:
                left_members = members[left_root]
                right_members = members[right_root]
                return any(
                    frozenset((left_member, right_member)) in vetoed_pairs
                    for left_member in left_members
                    for right_member in right_members
                )

            def union(left: URIRef, right: URIRef) -> None:
                left_root = find(left)
                right_root = find(right)
                if left_root == right_root:
                    return
                if str(left_root) <= str(right_root):
                    parents[right_root] = left_root
                    members[left_root] |= members.pop(right_root)
                else:
                    parents[left_root] = right_root
                    members[right_root] |= members.pop(left_root)

            for left, right in combinations(ordered_cluster, 2):
                pair = frozenset((left, right))
                score = self._candidate_similarity(left, right, embeddings)
                if score is not None and score < self.candidate_similarity_threshold:
                    # Label-confirmed and key-confirmed pairs bypass the cosine
                    # gate (mirrors EntityAligner): short-string embeddings of
                    # aliases like "Baranov, D." vs "Dmitry Baranov" hover
                    # around the threshold, which made identity linking
                    # nondeterministic — and a shared identifier value needs no
                    # embedding agreement at all.
                    if not (
                        (key_pairs is not None and pair in key_pairs)
                        or self._labels_confirm_identity(left, right, representations)
                    ):
                        continue
                if self._can_merge_as_identity(
                    left,
                    right,
                    representations,
                    direct_relation_pairs=direct_relation_pairs,
                    guard_context=guard_context,
                    key_pairs=key_pairs,
                ):
                    left_root = find(left)
                    right_root = find(right)
                    if left_root == right_root:
                        continue
                    if union_blocked(left_root, right_root):
                        # The pair itself is mergeable, but somewhere in the
                        # two groups sits a vetoed pair — accepting the edge
                        # would chain around that guard.
                        rejected_merges.append((left, right, score, ("cluster_veto",)))
                        continue
                    union(left, right)
                    continue
                rejected_merges.append(
                    (
                        left,
                        right,
                        score,
                        tuple(
                            self._merge_validation_failures(
                                left,
                                right,
                                representations,
                                guard_context=guard_context,
                            )
                        ),
                    )
                )

            grouped: dict[URIRef, list[URIRef]] = {}
            for entity in ordered_cluster:
                grouped.setdefault(find(entity), []).append(entity)

            for group in grouped.values():
                sorted_group = sorted(group, key=str)
                validated_clusters.append(sorted_group)

        return validated_clusters, rejected_merges

    def _select_ontology_anchor_candidates(
        self,
        tentative_entities: list[URIRef],
        tentative_representations: dict[URIRef, EntityRepresentation],
        tentative_doc_iris: dict[URIRef, URIRef],
        ontology_graph: RDFGraph | None,
        known_ontology_entities: set[URIRef],
    ) -> dict[URIRef, URIRef]:
        """Pick ontology anchors and preserve the triggering document IRI."""
        if (
            ontology_graph is None
            or not tentative_entities
            or not known_ontology_entities
        ):
            return {}

        ontology_entities = [
            entity
            for entity in known_ontology_entities
            if not self._is_standard_ontology_entity(entity)
        ]
        if not ontology_entities:
            return {}

        ontology_graphs = {entity: ontology_graph for entity in ontology_entities}
        ontology_representations = self.normalizer.create_representations_batch(
            ontology_entities, ontology_graphs
        )

        token_index: dict[str, set[URIRef]] = {}
        for entity, representation in ontology_representations.items():
            for token in self._tokenize(representation.representation):
                token_index.setdefault(token, set()).add(entity)

        selected: dict[URIRef, URIRef] = {}
        for tentative_entity in tentative_entities:
            tentative_representation = tentative_representations.get(tentative_entity)
            if tentative_representation is None:
                continue
            tentative_doc_iri = tentative_doc_iris.get(tentative_entity)
            if tentative_doc_iri is None:
                continue
            tentative_tokens = self._tokenize(tentative_representation.representation)
            if not tentative_tokens:
                continue

            candidate_pool: set[URIRef] = set()
            for token in tentative_tokens:
                candidate_pool.update(token_index.get(token, set()))

            if not candidate_pool:
                continue

            scored: list[tuple[int, URIRef]] = []
            for candidate in candidate_pool:
                candidate_representation = ontology_representations.get(candidate)
                if candidate_representation is None:
                    continue
                candidate_tokens = self._tokenize(
                    candidate_representation.representation
                )
                overlap = len(tentative_tokens & candidate_tokens)
                if overlap >= 2:
                    scored.append((overlap, candidate))

            scored.sort(key=lambda item: (-item[0], str(item[1])))
            for _, candidate in scored[:3]:
                selected.setdefault(candidate, tentative_doc_iri)

        return selected

    def _classify_entity_for_unit(
        self,
        entity: URIRef,
        unit: ContentUnit,
        known_ontology_entities: set[URIRef],
    ) -> EntityClassification:
        """Classify an entity as fact, known ontology, or tentative ontology."""
        if unit.type == OutputType.ONTOLOGIES:
            return EntityClassification.KNOWN_ONTOLOGY

        if self._is_fact_entity_in_unit(entity, unit):
            return EntityClassification.FACT

        if entity in known_ontology_entities or self._is_standard_ontology_entity(
            entity
        ):
            return EntityClassification.KNOWN_ONTOLOGY

        return EntityClassification.TENTATIVE_ONTOLOGY

    @staticmethod
    def _classification_priority(classification: EntityClassification) -> int:
        """Return priority for multi-unit classification merging."""
        if classification == EntityClassification.KNOWN_ONTOLOGY:
            return 3
        if classification == EntityClassification.TENTATIVE_ONTOLOGY:
            return 2
        return 1

    @staticmethod
    def _merge_into_context_graph(target: RDFGraph, source: RDFGraph) -> None:
        """Merge source triples/namespaces into a per-entity context graph."""
        target += source

    def _register_entity(
        self,
        *,
        entity: URIRef,
        unit: ContentUnit,
        state: _EntityCollectionState,
    ) -> None:
        """Register one URI entity with merged context and stable classification."""
        state.entities.add(entity)
        state.source_entities.add(entity)
        if entity not in state.entity_graphs:
            state.entity_graphs[entity] = unit.graph.copy()
        else:
            self._merge_into_context_graph(state.entity_graphs[entity], unit.graph)
        state.entity_doc_iris.setdefault(entity, unit.doc_iri)
        current = state.entity_classification.get(entity, EntityClassification.FACT)
        candidate = self._classify_entity_for_unit(entity, unit, state.known_entities)
        state.entity_classification[entity] = (
            candidate
            if self._classification_priority(candidate)
            >= self._classification_priority(current)
            else current
        )

    @staticmethod
    def _register_direct_relation(
        state: _EntityCollectionState,
        subject: URIRef,
        obj: URIRef,
    ) -> None:
        """Record direct subject-object URI relation pair in collection state."""
        if subject == obj:
            return
        state.direct_relation_pairs.add(frozenset((subject, obj)))

    def _collect_all_entities(
        self,
        units: list[ContentUnit],
        known_ontology_entities: set[URIRef] | None = None,
    ) -> tuple[
        list[URIRef],
        set[URIRef],
        dict[URIRef, RDFGraph],
        dict[URIRef, URIRef],
        dict[URIRef, EntityClassification],
        set[frozenset[URIRef]],
        dict[tuple[URIRef, URIRef], set[URIRef]],
    ]:
        """Collect all entities from all content unit graphs.

        Each entity is associated with the graph it was found in and the
        ``doc_iri`` of the :class:`ContentUnit` that produced it.  When an
        entity appears in several units the *last-seen* ``doc_iri`` wins (in
        practice most pipelines aggregate chunks of the same document, so all
        ``doc_iri`` values are identical).

        Args:
            units: List of content units to aggregate.

        Returns:
            Tuple of (
                entities,
                entity_to_graph,
                entity_to_doc_iri,
                entity_to_is_ontology,
            ).
        """
        state = _EntityCollectionState(known_entities=known_ontology_entities or set())

        for unit in units:
            if unit.graph is None:
                continue
            unit.graph.sanitize_prefixes_namespaces()
            # Keep collection in the same URI space that rewrite/merge consumes
            # (unit.graph). Using graph_absolute here causes mapping keys to miss
            # during rewrite, because unit.graph still contains the original terms.
            for s, p, o in unit.graph:
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    self._register_direct_relation(state=state, subject=s, obj=o)
                    if isinstance(p, URIRef) and p != RDF.type:
                        state.object_groups.setdefault((s, p), set()).add(o)
                for term in (s, p, o):
                    if isinstance(term, URIRef):
                        self._register_entity(entity=term, unit=unit, state=state)

        return (
            list(state.entities),
            state.source_entities,
            state.entity_graphs,
            state.entity_doc_iris,
            state.entity_classification,
            state.direct_relation_pairs,
            state.object_groups,
        )

    def aggregate_graphs(
        self,
        units: list[ContentUnit],
        ontology_graph: RDFGraph,
        merge_vetoes: set[frozenset[URIRef]] | None = None,
    ) -> AggregationResult:
        """Aggregate multiple content unit graphs with embedding-based disambiguation.

        Args:
            units: List of ContentUnits to aggregate.
            ontology_graph: Selected ontology graph used to distinguish
                known ontology entities from tentative ontology-like aliases.
            merge_vetoes: Extra entity pairs that must never identity-merge —
                the targeted un-merge lever used by the post-aggregation
                validation gate. Unioned into the direct-relation veto set.

        Returns:
            :class:`AggregationResult` with the merged graph and merge
            bookkeeping (decisions, merged clusters, rejection count).
        """
        logger.info(f"Starting aggregation with metadata for {len(units)} units")
        if ontology_graph is None:
            raise ValueError("ontology_graph must not be None for facts aggregation")

        if not units:
            return AggregationResult(graph=RDFGraph())

        # Steps 1-3: Collect, normalise, candidate clustering
        known_ontology_entities = self._build_known_ontology_entities(ontology_graph)
        (
            entities,
            source_entities,
            entity_graphs,
            entity_doc_iris,
            entity_classification,
            direct_relation_pairs,
            object_groups,
        ) = self._collect_all_entities(units, known_ontology_entities)
        if merge_vetoes:
            direct_relation_pairs = direct_relation_pairs | merge_vetoes
        schema_functional_predicates = harvest_max_one_predicates(ontology_graph)
        guard_context = MergeGuardContext(
            sibling_pairs=build_sibling_pairs(
                object_groups, scope=self.sibling_guard_scope
            ),
            functional_predicates=schema_functional_predicates
            | empirically_functional_predicates(
                object_groups,
                min_support=self.functional_min_empirical_support,
            ),
        )
        representations = self.normalizer.create_representations_batch(
            entities, entity_graphs
        )
        decisions: dict[URIRef, EntityDecision] = {
            entity: EntityDecision(
                classification=classification,
                identity_target=entity,
            )
            for entity, classification in entity_classification.items()
        }
        tentative_entities = [
            entity
            for entity, decision in decisions.items()
            if decision.classification == EntityClassification.TENTATIVE_ONTOLOGY
        ]
        anchor_candidates = self._select_ontology_anchor_candidates(
            tentative_entities=tentative_entities,
            tentative_representations=representations,
            tentative_doc_iris=entity_doc_iris,
            ontology_graph=ontology_graph,
            known_ontology_entities=known_ontology_entities,
        )
        if anchor_candidates:
            for ontology_entity, anchor_doc_iri in anchor_candidates.items():
                if ontology_entity in entity_graphs:
                    continue
                entities.append(ontology_entity)
                entity_graphs[ontology_entity] = ontology_graph
                entity_doc_iris[ontology_entity] = anchor_doc_iri
                entity_classification[ontology_entity] = (
                    EntityClassification.KNOWN_ONTOLOGY
                )
                decisions[ontology_entity] = EntityDecision(
                    classification=EntityClassification.KNOWN_ONTOLOGY,
                    identity_target=ontology_entity,
                )
                representations[ontology_entity] = (
                    self.normalizer.create_representation(
                        ontology_entity, ontology_graph
                    )
                )
        entity_is_known_ontology = {
            entity: decision.classification == EntityClassification.KNOWN_ONTOLOGY
            for entity, decision in decisions.items()
        }
        if logger.isEnabledFor(logging.INFO):
            known_count = sum(
                1 for is_known in entity_is_known_ontology.values() if is_known
            )
            fact_count = sum(
                1
                for decision in decisions.values()
                if decision.classification == EntityClassification.FACT
            )
            logger.info(
                "Aggregation entity classification stats: fact=%d known_ontology=%d "
                "tentative_ontology=%d",
                fact_count,
                known_count,
                len(tentative_entities),
            )

        candidate_clusters, embeddings = self._cluster_entities_by_role(representations)
        key_pairs: set[frozenset[URIRef]] = set()
        if self.natural_key_merge:
            key_pairs = self._collect_natural_key_pairs(
                representations, schema_functional_predicates
            )
            if key_pairs:
                logger.info(
                    "Natural-key evidence proposed %d candidate pair(s)",
                    len(key_pairs),
                )
                candidate_clusters = self._merge_candidate_clusters_by_key_pairs(
                    candidate_clusters, key_pairs
                )
        clusters, rejected_merges = self._build_identity_clusters(
            candidate_clusters=candidate_clusters,
            representations=representations,
            embeddings=embeddings,
            direct_relation_pairs=direct_relation_pairs,
            guard_context=guard_context,
            key_pairs=key_pairs or None,
        )
        if rejected_merges:
            logger.info(
                "Rejected %d candidate merges after symbolic validation",
                len(rejected_merges),
            )
            for left, right, score, failed_checks in rejected_merges:
                logger.debug(
                    "Rejected candidate merge: %s <-> %s (score=%s, failed=%s)",
                    left,
                    right,
                    f"{score:.3f}" if score is not None else "n/a",
                    ",".join(failed_checks) if failed_checks else "unknown",
                )

        # Step 4: Canonical identity mapping (no URI policy yet)
        identity_mapping = self.selector.create_mapping(
            clusters,
            representations,
            entity_is_known_ontology=entity_is_known_ontology,
        )

        # Keep known ontology entities stable. Tentative ontology-like entities are:
        # - mapped to known ontology representatives when present in a mixed cluster
        # - preserved as-is when only tentative entities are present
        suppress_sameas_origins: set[URIRef] = set()
        suppress_fact_subject_sources: set[URIRef] = set()
        for cluster in clusters:
            known_ontology_entities_in_cluster = [
                entity
                for entity in cluster
                if decisions.get(entity) is not None
                and decisions[entity].classification
                == EntityClassification.KNOWN_ONTOLOGY
            ]
            tentative_entities_in_cluster = [
                entity
                for entity in cluster
                if decisions.get(entity) is not None
                and decisions[entity].classification
                == EntityClassification.TENTATIVE_ONTOLOGY
            ]
            fact_entities_in_cluster = [
                entity
                for entity in cluster
                if decisions.get(entity) is not None
                and decisions[entity].classification == EntityClassification.FACT
            ]

            for entity in known_ontology_entities_in_cluster:
                identity_mapping[entity] = entity

            if known_ontology_entities_in_cluster:
                canonical_known_ontology = self.selector.select_representative(
                    known_ontology_entities_in_cluster,
                    representations,
                    entity_is_known_ontology=entity_is_known_ontology,
                )
                for tentative_entity in tentative_entities_in_cluster:
                    if self._can_merge_as_identity(
                        tentative_entity,
                        canonical_known_ontology,
                        representations,
                        direct_relation_pairs=direct_relation_pairs,
                        guard_context=guard_context,
                    ):
                        identity_mapping[tentative_entity] = canonical_known_ontology
                        decisions[tentative_entity].suppress_sameas = True
                    else:
                        identity_mapping[tentative_entity] = tentative_entity
                for fact_entity in fact_entities_in_cluster:
                    if self._can_merge_as_identity(
                        fact_entity,
                        canonical_known_ontology,
                        representations,
                        direct_relation_pairs=direct_relation_pairs,
                        guard_context=guard_context,
                    ):
                        identity_mapping[fact_entity] = canonical_known_ontology
                        decisions[fact_entity].suppress_sameas = True
                        decisions[fact_entity].suppress_fact_subject_assertions = True
                    else:
                        identity_mapping[fact_entity] = fact_entity

            elif tentative_entities_in_cluster:
                # In mixed FACT + TENTATIVE clusters with no known ontology
                # entity, prefer the FACT side when symbolic identity checks
                # agree (e.g. hallucinated ontology prefix on an instance).
                if fact_entities_in_cluster:
                    canonical_fact = self.selector.select_representative(
                        fact_entities_in_cluster,
                        representations,
                        entity_is_known_ontology=entity_is_known_ontology,
                    )
                    for fact_entity in fact_entities_in_cluster:
                        identity_mapping[fact_entity] = canonical_fact
                    for tentative_entity in tentative_entities_in_cluster:
                        if self._can_merge_as_identity(
                            tentative_entity,
                            canonical_fact,
                            representations,
                            direct_relation_pairs=direct_relation_pairs,
                            guard_context=guard_context,
                        ):
                            identity_mapping[tentative_entity] = canonical_fact
                            decisions[tentative_entity].suppress_sameas = True
                        else:
                            identity_mapping[tentative_entity] = tentative_entity
                else:
                    for tentative_entity in tentative_entities_in_cluster:
                        identity_mapping[tentative_entity] = tentative_entity

        for entity, target in identity_mapping.items():
            if entity in decisions:
                decisions[entity].identity_target = target

        suppress_sameas_origins = {
            entity for entity, decision in decisions.items() if decision.suppress_sameas
        }
        suppress_fact_subject_sources = {
            entity
            for entity, decision in decisions.items()
            if decision.suppress_fact_subject_assertions
        }

        # Step 5: URI assignment from canonical identity + namespace policy
        final_mapping = self.uri_builder.create_entity_uri_mapping(
            identity_mapping=identity_mapping,
            representations=representations,
            entity_doc_iris=entity_doc_iris,
            entity_is_ontology={
                entity: (
                    decisions.get(entity) is not None
                    and decisions[entity].classification != EntityClassification.FACT
                )
                for entity in representations
            },
        )
        for entity, final_uri in final_mapping.items():
            if entity in decisions:
                decisions[entity].final_uri = final_uri
        known_ontology_entities_all = {
            entity
            for entity, decision in decisions.items()
            if decision.classification == EntityClassification.KNOWN_ONTOLOGY
        }
        assert all(
            identity_mapping.get(entity, entity) == entity
            for entity in known_ontology_entities_all
        ), "Known ontology entities must remain identity-mapped"
        assert not (known_ontology_entities_all & suppress_sameas_origins), (
            "Known ontology entities cannot be suppress_sameas origins"
        )
        assert not (known_ontology_entities_all & suppress_fact_subject_sources), (
            "Known ontology entities cannot be suppress_fact_subject origins"
        )
        assert all(entity in decisions for entity in source_entities), (
            "Every source entity must have a decision record"
        )
        final_mapping = {
            entity: mapped
            for entity, mapped in final_mapping.items()
            if entity in source_entities
        }

        # Step 7: Rewrite and merge with provenance
        active_units = [u for u in units if u.graph is not None and len(u.graph) > 0]
        merged_graph = self.rewriter.merge_graphs_with_provenance(
            active_units,
            final_mapping,
            suppress_sameas_origins=suppress_sameas_origins,
            suppress_fact_subject_sources=suppress_fact_subject_sources,
        )

        merged_clusters = build_merged_clusters(final_mapping, identity_mapping)
        key_supported_clusters = sorted(
            {
                str(final_mapping[left])
                for pair in key_pairs
                for left, right in [tuple(pair)]
                if left in final_mapping
                and final_mapping.get(right) == final_mapping[left]
            }
        )

        logger.info("Aggregation with metadata complete")
        return AggregationResult(
            graph=merged_graph,
            decisions=decisions,
            merged_clusters=merged_clusters,
            rejected_merge_count=len(rejected_merges),
            key_supported_clusters=key_supported_clusters,
        )

    def postprocess_facts_units(
        self,
        units: list[ContentUnit],
        ontology_graph: RDFGraph,
        *,
        doc_iri: URIRef | None = None,
        document_metadata: dict[str, Any] | None = None,
        doc_namespace: str | None = None,
        merge_vetoes: set[frozenset[URIRef]] | None = None,
    ) -> AggregationResult:
        """Sanitize facts units, then run aggregation/normalization.

        This method is intentionally safe for both single-unit and multi-unit
        inputs so unit-pipeline and graph-pipeline paths share the same
        post-processing behavior.

        When ``doc_iri`` and non-empty ``document_metadata`` are provided,
        caller-asserted document identity triples are attached to the merged
        facts graph. Business-oriented keys mint typed entities under
        ``doc_namespace`` (defaults to the document facts namespace).

        Args:
            units: Facts content units to aggregate.
            ontology_graph: Merged ontology context for classification/guards.
            doc_iri: Document IRI for metadata provenance attachment.
            document_metadata: Caller-asserted document identity metadata.
            doc_namespace: Namespace for metadata-minted entities.
            merge_vetoes: Entity pairs that must never identity-merge
                (validation-gate un-merge lever).

        Returns:
            :class:`AggregationResult`; its ``graph`` carries the merged facts
            plus any document-metadata provenance.
        """
        for unit in units:
            unit.sanitize()
        result = self.aggregate_graphs(
            units=units, ontology_graph=ontology_graph, merge_vetoes=merge_vetoes
        )
        if doc_iri is not None and document_metadata:
            apply_document_metadata_provenance(
                doc_iri,
                document_metadata,
                result.graph,
                entity_namespace=doc_namespace,
            )
        # Cross-unit prefix conflicts surface only on the merged graph (e.g.
        # aliases of one namespace arriving from different units), so sanitize
        # once more after aggregation.
        result.graph.sanitize_prefixes_namespaces()
        return result
