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
from collections import Counter
from collections.abc import Set as AbstractSet
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
    DOCUMENT_METADATA_TERMS,
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
from ontocast.onto.model import MergeProvenance
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import well_known_value_classes
from ontocast.tool.facts_validation.terms import CatalogRoles, superclass_closure
from ontocast.tool.representation_text import (
    normalize_identifier,
    normalize_text,
    normalize_uri_local_name,
)

from .clustering import ClusterRepresentativeSelector
from .merge_provenance import build_merge_provenance
from .normalizer import EntityNormalizer, EntityRepresentation
from .rewriter import GraphRewriter
from .signatures import (
    MergeGuardContext,
    empirically_functional_predicates,
    harvest_disjoint_classes,
    harvest_max_one_predicates,
    label_tokens,
    labels_alias_with_initials,
    labels_differ_only_by_initials,
    numbers_agree,
    string_values_compatible,
    tokens_alias_compatible,
)
from .statement_conflict import (
    Same,
    StatementIndex,
    dimension_conflict,
    statement_conflict,
)
from .unit_scope import scope_fact_iris, unit_scope_index, unscoped_iri
from .uri_builder import EntityRole, URIBuilder, to_lower_camel_case
from .value_names import merge_equal_values

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
# Predicates above plus ``FOAF.Document`` / ``DCTERMS.source`` /
# ``DCTERMS.identifier`` are exactly :data:`DOCUMENT_METADATA_TERMS` — the
# gate's exemption set is derived from that constant, not from this map.
assert set(_DOC_METADATA_FIRST_CLASS.values()) <= DOCUMENT_METADATA_TERMS
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


#: Lexical tiers that confirm identity by name. Any other tier (a name inside
#: another, overlapping words, similar strings, an IRI-derived name) only makes
#: identity a question: such a pair merges on a coreference "same" verdict.
CONFIRMING_ALIAS_TIERS: frozenset[str] = frozenset({"label", "initials"})


def _identical(left: object, right: object) -> bool:
    return left == right


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
        rejected_by_reason: How many of those each reason refused
            (``same_unit``, ``relation``, ``literal_conflict``,
            ``functional_iri_conflict``, ``statement_conflict``,
            ``dimension_conflict``, ``initials_conflict``, ``role``,
            ``type``, ``overlap``,
            ``lexical``, ``cluster_veto``); a pair refused for several
            reasons counts under each.
        confirmed_clusters: Final URIs of merged clusters containing a pair
            confirmed as one entity: a shared identifier value or a
            coreference "same" verdict. The validation gate treats text-value
            disagreement inside these clusters as name variance rather than a
            merge signature.
        merge_provenance: Which multi-valued facts of the graph only a merge
            brought together. See :func:`build_merge_provenance`.
        lexical_near_misses: Candidate source pairs refused only for want of
            identity evidence -- no guard said they are distinct -- as
            ``(left, right, similarity, reason)``: ``overlap`` when named alike
            but not equally, ``lexical`` when not named alike. What
            coreference resolution may merge.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: RDFGraph
    decisions: dict[URIRef, EntityDecision] = Field(default_factory=dict)
    merged_clusters: dict[str, list[str]] = Field(default_factory=dict)
    rejected_merge_count: int = 0
    rejected_by_reason: dict[str, int] = Field(default_factory=dict)
    confirmed_clusters: list[str] = Field(default_factory=list)
    merge_provenance: MergeProvenance = Field(default_factory=MergeProvenance)
    lexical_near_misses: list[tuple[str, str, float, str]] = Field(default_factory=list)


class _EntityCollectionState(BaseModel):
    """Mutable state for entity collection across content units."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    known_entities: set[URIRef]
    entities: set[URIRef] = Field(default_factory=set)
    source_entities: set[URIRef] = Field(default_factory=set)
    # Every unit triple, once: the shared context of all collected entities.
    context: RDFGraph = Field(default_factory=RDFGraph)
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
        self.natural_key_merge = cfg.natural_key_merge
        self.llm_coreference = cfg.llm_coreference
        self.llm_coreference_max_pairs = cfg.llm_coreference_max_pairs
        self.type_guard_untyped = str(cfg.type_guard_untyped)
        self.unit_scoped_fact_iris = cfg.unit_scoped_fact_iris
        self.enabled = cfg.enabled
        if candidate_similarity_threshold is None:
            self._warn_inert_similarity_threshold(cfg)

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
    def _warn_inert_similarity_threshold(cfg: AggregationConfig) -> None:
        """Warn when the aligner threshold is tuned but the pipeline one is not.

        ``similarity_threshold`` drives only the cross-graph
        :class:`~ontocast.tool.agg.entity_aligner.EntityAligner`; this
        aggregator clusters and gates at ``candidate_similarity_threshold``.
        Setting the former away from its default while the latter stays at its
        default is the signature of someone tuning the wrong knob.
        """
        fields = AggregationConfig.model_fields
        if (
            cfg.similarity_threshold != fields["similarity_threshold"].default
            and cfg.candidate_similarity_threshold
            == fields["candidate_similarity_threshold"].default
        ):
            logger.warning(
                "AGG_SIMILARITY_THRESHOLD=%s is set, but the in-pipeline "
                "aggregator does not read it: it clusters and gates at "
                "AGG_CANDIDATE_SIMILARITY_THRESHOLD (still at its default %s). "
                "AGG_SIMILARITY_THRESHOLD is only the cross-graph aligner's "
                "fallback threshold (POST /match/entities, match-graphs, the "
                "ontocast_align_entities tool).",
                cfg.similarity_threshold,
                cfg.candidate_similarity_threshold,
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
        local_name = normalize_uri_local_name(unscoped_iri(entity)).replace(" ", "")
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
        guard_context: MergeGuardContext | None = None,
    ) -> bool:
        """Whether the two entities' types allow one identity at all.

        Between two instances the document minted, asserted types are
        evidence, not identity: units type one thing at different levels or
        from different vocabularies (a place as a location in one, a feature
        of interest in another), so only a declared ``owl:disjointWith``
        between their superclass closures says they are two things; types
        that merely differ ask for a name that confirms identity instead
        (:meth:`_types_overlap`). A catalog entity's types define it, so a
        merge with one needs a shared type.
        """
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
        if guard_context is None or not {left, right} <= guard_context.fact_entities:
            return bool(left_types & right_types)
        disjoint = guard_context.disjoint_classes
        if not disjoint:
            return True
        right_closure = self._type_closure(right_types, guard_context)
        return not any(
            frozenset((left_class, right_class)) in disjoint
            for left_class in self._type_closure(left_types, guard_context)
            for right_class in right_closure
        )

    @staticmethod
    def _type_closure(
        types: set[URIRef], guard_context: MergeGuardContext | None
    ) -> set[URIRef]:
        ancestors = guard_context.class_ancestors if guard_context else {}
        return {
            ancestor
            for class_iri in types
            for ancestor in ancestors.get(class_iri, frozenset((class_iri,)))
        }

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
        return self._lexical_alias_tier(left, right, representations) is not None

    def _lexical_alias_tier(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
    ) -> str | None:
        """The lexical tier that makes *left* and *right* aliases, or None.

        ``label`` and ``initials`` confirm identity by name and are tried
        first; the other tiers (``sub_label``, ``label_overlap``,
        ``sequence``, ``token_overlap`` and the IRI-derived ``normal_form`` /
        ``local_name``) only make it a question: such a pair is a coreference
        candidate and merges on a "same" verdict alone.
        """
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if left_rep is None or right_rep is None:
            return None
        left_label_tokens = self._entity_label_values(left_rep)
        right_label_tokens = self._entity_label_values(right_rep)
        # Identity is in the names an entity carries, not its IRI. Two named
        # fact entities are decided by their names alone: units mint local
        # names independently and can mint one alike for unrelated things. A
        # local name still counts for an unnamed side, and against a catalog
        # term, whose name the renderer saw and reused.
        names_decide = (
            bool(left_label_tokens and right_label_tokens)
            and not left_rep.is_ontology_entity
            and not right_rep.is_ontology_entity
        )

        if left_label_tokens & right_label_tokens:
            return "label"

        # Abbreviation-aware tier: "baranov d" vs "dmitry baranov" alias when
        # every token of one label matches a token of the other exactly or as
        # a single-character initial, with at least one shared full token.
        if self._labels_alias_with_initials(left_label_tokens, right_label_tokens):
            # Every word of one name inside the other, no initial involved: a
            # name variant ("Mr Beer" / "Mr Karlheinz Beer") or a narrower
            # thing ("temperature" / "daily maximum temperature").
            if any(
                set(label_tokens(a)) <= set(label_tokens(b))
                or set(label_tokens(b)) <= set(label_tokens(a))
                for a in left_label_tokens
                for b in right_label_tokens
            ):
                return "sub_label"
            return "initials"

        if not names_decide:
            if left_rep.normal_form == right_rep.normal_form:
                return "normal_form"
            left_instance_name = self._instance_like_local_name(left)
            right_instance_name = self._instance_like_local_name(right)
            if (
                left_instance_name is not None
                and right_instance_name is not None
                and left_instance_name == right_instance_name
            ):
                return "local_name"

        # Guard-literal-bearing entities (measurements, dated events) are
        # individuated by their payload, not their phrasing: "PL red shift of
        # SL1" vs "PL red shift of SL2" share most tokens yet denote distinct
        # values. Only the exact tiers above may merge them. String literals
        # (names, descriptions) do not raise this bar — disjoint identifier
        # strings are handled by _have_conflicting_literals instead.
        if left_rep.has_guard_literal and right_rep.has_guard_literal:
            return None

        if left_label_tokens and right_label_tokens:
            max_label_overlap = 0.0
            for left_label in left_label_tokens:
                left_tokens = self._tokenize(left_label)
                for right_label in right_label_tokens:
                    if not numbers_agree(left_label, right_label):
                        continue
                    right_tokens = self._tokenize(right_label)
                    overlap = self._jaccard(left_tokens, right_tokens)
                    max_label_overlap = max(max_label_overlap, overlap)
            if max_label_overlap >= self.lexical_label_jaccard:
                return "label_overlap"
        if names_decide:
            return None

        left_normalized = left_rep.normal_form.strip()
        right_normalized = right_rep.normal_form.strip()
        if left_normalized and right_normalized:
            if left_normalized != right_normalized and (
                left_normalized.startswith(f"{right_normalized} ")
                or right_normalized.startswith(f"{left_normalized} ")
            ):
                return None

        ratio = SequenceMatcher(
            None, left_rep.normal_form, right_rep.normal_form
        ).ratio()
        if ratio >= self.lexical_sequence_ratio:
            return "sequence"

        left_tokens = self._tokenize(left_rep.normal_form)
        right_tokens = self._tokenize(right_rep.normal_form)
        if len(left_tokens) >= 2 and len(right_tokens) >= 2:
            if self._jaccard(left_tokens, right_tokens) >= self.lexical_token_jaccard:
                return "token_overlap"

        return None

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
        name_lists: AbstractSet[URIRef] = frozenset(),
    ) -> bool:
        """Return True when the entities assert disjoint values per predicate.

        A shared predicate with two non-empty, disjoint canonical value sets
        (numeric/temporal) marks the entities as distinct individuals; overlap
        or one-sided values read as re-mention/enrichment and stay mergeable.
        String payloads (identifiers, codes) conflict only when NO cross-pair
        is compatible (equality, prefix, or initial-abbreviation) — "d" vs
        "dmitry" is a re-mention, "S-2024-001" vs "S-2024-002" is a conflict.
        A predicate in ``name_lists`` holds several names per entity (alternate
        names), so two entities listing different ones are not told apart.
        """
        for predicate, left_values in left_rep.predicate_literals.items():
            right_values = right_rep.predicate_literals.get(predicate)
            if not right_values or not left_values:
                continue
            if left_values.isdisjoint(right_values):
                return True
        for predicate, left_strings in left_rep.predicate_string_literals.items():
            right_strings = right_rep.predicate_string_literals.get(predicate)
            if not right_strings or not left_strings or predicate in name_lists:
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
        same: Same = _identical,
    ) -> bool:
        """Return True when a max-1 object predicate points at distinct objects.

        Objects compare by ``same``: the aggregator passes the identity it has
        decided so far, so two objects merged in the same pass are one object.
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
            if not any(
                same(left_object, right_object)
                for left_object in left_objects
                for right_object in right_objects
            ):
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
        return labels_differ_only_by_initials(
            self._entity_label_values(left_rep),
            self._entity_label_values(right_rep),
        )

    @staticmethod
    def _minted_by_one_unit(left: URIRef, right: URIRef) -> bool:
        """True when one unit minted both: a render names each thing once.

        A unit's renderer sees its whole text, so two IRIs it mints are two
        things it chose to tell apart ("current" and "future" results, the
        deaths in Oregon and in Washington). Units mint independently of each
        other, so the evidence holds only within one.
        """
        left_unit = unit_scope_index(str(left))
        return left_unit is not None and left_unit == unit_scope_index(str(right))

    def _distinctness_reasons(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
        direct_relation_pairs: set[frozenset[URIRef]] | None = None,
        guard_context: MergeGuardContext | None = None,
        same: Same = _identical,
        *,
        first_only: bool = False,
    ) -> list[str]:
        """The evidence that *left* and *right* denote distinct entities.

        Unlike the absence of a lexical alias -- which merely fails to support
        a merge -- each reason is grounds to keep the pair apart in *any*
        identity cluster, including transitively: two entities a guard
        separates must not end up merged through a chain of intermediate
        aliases. Objects and statements compare by ``same``, the identity
        decided so far. ``first_only`` stops at the first reason.
        """
        reasons: list[str] = []

        def found(reason: str) -> bool:
            reasons.append(reason)
            return first_only

        if self._minted_by_one_unit(left, right) and found("same_unit"):
            return reasons
        pair = frozenset((left, right))
        # Direct relations are recorded by name; the gate's merge vetoes name
        # scoped source IRIs. Both spellings are checked.
        name_pair = frozenset((unscoped_iri(left), unscoped_iri(right)))
        if (
            direct_relation_pairs is not None
            and (pair in direct_relation_pairs or name_pair in direct_relation_pairs)
            and found("relation")
        ):
            return reasons
        left_rep = representations.get(left)
        right_rep = representations.get(right)
        if guard_context is not None:
            if left_rep is not None and right_rep is not None:
                if self._have_conflicting_literals(
                    left_rep, right_rep, guard_context.name_list_predicates
                ) and found("literal_conflict"):
                    return reasons
                if self._have_conflicting_functional_objects(
                    left_rep, right_rep, guard_context.functional_predicates, same
                ) and found("functional_iri_conflict"):
                    return reasons
            if (
                guard_context.statements is not None
                and statement_conflict(guard_context.statements, left, right, same)
                is not None
                and found("statement_conflict")
            ):
                return reasons
            if (
                guard_context.statements is not None
                and dimension_conflict(guard_context.statements, left, right)
                is not None
                and found("dimension_conflict")
            ):
                return reasons
        if (
            left_rep is not None
            and right_rep is not None
            and self._labels_mark_distinct_entities(left_rep, right_rep)
            and found("initials_conflict")
        ):
            return reasons
        if not self._are_roles_compatible(left, right, representations) and found(
            "role"
        ):
            return reasons
        if not self._are_types_compatible(
            left, right, representations, guard_context
        ) and found("type"):
            return reasons
        return reasons

    def _pair_distinctness_veto(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
        direct_relation_pairs: set[frozenset[URIRef]] | None = None,
        guard_context: MergeGuardContext | None = None,
        same: Same = _identical,
    ) -> bool:
        """Positive evidence that *left* and *right* denote distinct entities
        (:meth:`_distinctness_reasons`)."""
        return bool(
            self._distinctness_reasons(
                left,
                right,
                representations,
                direct_relation_pairs,
                guard_context,
                same,
                first_only=True,
            )
        )

    def _can_merge_as_identity(
        self,
        left: URIRef,
        right: URIRef,
        representations: dict[URIRef, EntityRepresentation],
        direct_relation_pairs: set[frozenset[URIRef]] | None = None,
        guard_context: MergeGuardContext | None = None,
        key_pairs: set[frozenset[URIRef]] | None = None,
        same: Same = _identical,
    ) -> bool:
        """Whether the pair alone may merge: no distinctness, and identity evidence.

        Identity evidence is a shared key or a "same" verdict (``key_pairs``),
        or a name that confirms identity (:data:`CONFIRMING_ALIAS_TIERS`).
        """
        if self._pair_distinctness_veto(
            left,
            right,
            representations,
            direct_relation_pairs=direct_relation_pairs,
            guard_context=guard_context,
            same=same,
        ):
            return False
        if key_pairs is not None and frozenset((left, right)) in key_pairs:
            return True
        return (
            self._lexical_alias_tier(left, right, representations)
            in CONFIRMING_ALIAS_TIERS
        )

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

    def _collect_label_pairs(
        self, representations: dict[URIRef, EntityRepresentation]
    ) -> set[frozenset[URIRef]]:
        """Instance pairs that share a normalised name.

        A name shared by more than ``_NATURAL_KEY_MAX_VALUE_ENTITIES``
        instances is generic, not identifying. The pairs only bring entities
        into one candidate cluster; every veto still decides.
        """
        instance_role = str(EntityRole.INSTANCE)
        by_label: dict[str, list[URIRef]] = {}
        for entity, rep in representations.items():
            if self._role_key(rep) != instance_role:
                continue
            for label in self._entity_label_values(rep):
                by_label.setdefault(label, []).append(entity)
        pairs: set[frozenset[URIRef]] = set()
        for entities in by_label.values():
            if 2 <= len(entities) <= _NATURAL_KEY_MAX_VALUE_ENTITIES:
                pairs.update(
                    frozenset(pair)
                    for pair in combinations(sorted(entities, key=str), 2)
                )
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
        direct_relation_pairs: set[frozenset[URIRef]] | None = None,
        same: Same = _identical,
        *,
        confirmed: bool = False,
    ) -> list[str]:
        """Every reason the pair alone may not merge.

        The distinctness reasons, then the missing identity evidence:
        ``overlap`` for a name that only makes identity a question (a
        coreference candidate), ``lexical`` for no name evidence at all. A
        ``confirmed`` pair (a shared key or a "same" verdict) lacks no evidence.
        """
        failures = self._distinctness_reasons(
            left,
            right,
            representations,
            direct_relation_pairs,
            guard_context,
            same,
        )
        if confirmed:
            return failures
        tier = self._lexical_alias_tier(left, right, representations)
        if tier is None:
            failures.append("lexical")
        elif tier not in CONFIRMING_ALIAS_TIERS:
            failures.append("overlap")
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
        list[list[URIRef]],
        list[tuple[URIRef, URIRef, float | None, tuple[str, ...]]],
    ]:
        """Merge the candidate pairs that identity evidence supports.

        Evidence order: unions for shared keys and "same" verdicts
        (``key_pairs``) first -- those a confirming name also supports before
        the rest -- then for names that confirm identity (an equal label,
        initials); within each, the most similar pair first. A pair
        named only alike is never merged here; it is returned as refused for
        ``overlap``, a coreference candidate.

        Distinctness is read on merged identity: objects and statements
        compare by the clusters built so far, and a union is refused when any
        two members of the joined cluster are distinct. A union can make two
        members of a cluster distinct (two subjects become distinct once the
        properties they report are one), so after each batch of unions, and after each
        round, the pairs inside the joined clusters are read again on live
        identity; a union that brought two now-distinct members together is
        undone, and its pairs are not joined again. Pairs refused in one round
        are tried again once the clusters have changed, until a round changes
        nothing. Pairs of exactly equal rank are decided together: when
        joining them would bring two distinct members together, none of them
        merges, and undoing one undoes them all, so no union depends on IRI
        order.

        Returns:
            The clusters, and every candidate pair left in two clusters with
            its similarity and reasons (``cluster_veto`` when the pair alone
            could merge).
        """
        keyed = key_pairs or set()
        parents: dict[URIRef, URIRef] = {
            entity: entity for cluster in candidate_clusters for entity in cluster
        }
        members: dict[URIRef, set[URIRef]] = {entity: {entity} for entity in parents}

        def find(entity: URIRef) -> URIRef:
            root = entity
            while parents.get(root, root) != root:
                root = parents[root]
            while entity != root:
                parents[entity], entity = root, parents[entity]
            return root

        def same(left: object, right: object) -> bool:
            if left == right:
                return True
            if not isinstance(left, URIRef) or not isinstance(right, URIRef):
                return False
            return left in parents and right in parents and find(left) == find(right)

        def union(left_root: URIRef, right_root: URIRef) -> None:
            keep, drop = sorted((left_root, right_root), key=str)
            parents[drop] = keep
            members[keep] |= members.pop(drop)

        # The unions kept, in the order applied: each the tied pairs of one
        # joined group. Undoing one rebuilds the clusters from the rest.
        applied: list[list[tuple[URIRef, URIRef]]] = []
        # Pairs whose union was undone: never joined again.
        blocked: set[frozenset[URIRef]] = set()

        def rebuild() -> None:
            for entity in parents:
                parents[entity] = entity
            members.clear()
            members.update({entity: {entity} for entity in parents})
            for edges in applied:
                for left, right in edges:
                    a, b = find(left), find(right)
                    if a != b:
                        union(a, b)

        def distinct(left: URIRef, right: URIRef) -> bool:
            return self._pair_distinctness_veto(
                left,
                right,
                representations,
                direct_relation_pairs=direct_relation_pairs,
                guard_context=guard_context,
                same=same,
            )

        def undo_vetoed(roots: set[URIRef]) -> bool:
            """Undo every union that joined two members now distinct.

            Distinctness is read on the clusters as they now are, inside the
            clusters of ``roots``. The union undone is the one that first
            joined the two, found by replaying the unions in order; its pairs
            are blocked. Returns whether anything was undone.
            """
            vetoed = [
                (left, right)
                for root in sorted(roots, key=str)
                for left, right in combinations(sorted(members[root], key=str), 2)
                if distinct(left, right)
            ]
            if not vetoed:
                return False
            scratch: dict[URIRef, URIRef] = {}

            def scratch_find(entity: URIRef) -> URIRef:
                while scratch.get(entity, entity) != entity:
                    entity = scratch[entity]
                return entity

            undone: set[int] = set()
            pending = list(vetoed)
            for index, edges in enumerate(applied):
                for left, right in edges:
                    a, b = scratch_find(left), scratch_find(right)
                    if a != b:
                        scratch[max(a, b, key=str)] = min(a, b, key=str)
                joined = [
                    pair
                    for pair in pending
                    if scratch_find(pair[0]) == scratch_find(pair[1])
                ]
                if joined:
                    undone.add(index)
                    pending = [pair for pair in pending if pair not in joined]
                if not pending:
                    break
            for index in undone:
                blocked.update(frozenset(pair) for pair in applied[index])
            applied[:] = [
                edges for index, edges in enumerate(applied) if index not in undone
            ]
            rebuild()
            return True

        # Candidate pairs, each with its rank of identity evidence: 0 for a key
        # or a verdict with a confirming name, 1 for a key or a verdict alone,
        # 2 for a confirming name alone, None for none.
        cluster_pairs: list[tuple[URIRef, URIRef]] = []
        candidates: list[tuple[URIRef, URIRef, float | None, int | None]] = []
        for candidate_cluster in candidate_clusters:
            ordered = sorted(candidate_cluster, key=str)
            for left, right in combinations(ordered, 2):
                cluster_pairs.append((left, right))
                pair = frozenset((left, right))
                score = self._candidate_similarity(left, right, embeddings)
                confirmed = pair in keyed
                if (
                    score is not None
                    and score < self.candidate_similarity_threshold
                    and not confirmed
                    and not self._labels_confirm_identity(left, right, representations)
                ):
                    # Label- and key-confirmed pairs bypass the cosine gate:
                    # short-string embeddings of aliases ("Baranov, D." vs
                    # "Dmitry Baranov") hover around the threshold, and a
                    # shared identifier needs no embedding agreement at all.
                    continue
                named = (
                    self._lexical_alias_tier(left, right, representations)
                    in CONFIRMING_ALIAS_TIERS
                )
                rank: int | None = (
                    (0 if named else 1) if confirmed else 2 if named else None
                )
                candidates.append((left, right, score, rank))

        def order(item: tuple[URIRef, URIRef, float | None, int | None]):
            _, _, score, rank = item
            return (rank, -score if score is not None else float("inf"))

        while True:
            # Every pair a candidate cluster holds, not only those above the
            # cosine gate: a union must not join two distinct members.
            vetoed = {
                frozenset((left, right))
                for left, right in cluster_pairs
                if find(left) != find(right) and distinct(left, right)
            }

            def holds_veto(group: set[URIRef]) -> bool:
                return any(
                    frozenset((a, b)) in vetoed for a, b in combinations(group, 2)
                )

            mergeable = sorted(
                (
                    item
                    for item in candidates
                    if item[3] is not None
                    and find(item[0]) != find(item[1])
                    and frozenset((item[0], item[1])) not in vetoed
                    and frozenset((item[0], item[1])) not in blocked
                ),
                key=lambda item: (*order(item), str(item[0]), str(item[1])),
            )
            changed = False
            index = 0
            while index < len(mergeable):
                key = order(mergeable[index])
                tied: list[tuple[URIRef, URIRef]] = []
                while index < len(mergeable) and order(mergeable[index]) == key:
                    left, right = mergeable[index][0], mergeable[index][1]
                    if find(left) != find(right):
                        tied.append((left, right))
                    index += 1
                # Join the tied pairs on a scratch copy of the clusters, then
                # keep a joined group only if it holds no distinct members.
                scratch: dict[URIRef, URIRef] = {}

                def scratch_find(root: URIRef) -> URIRef:
                    while scratch.get(root, root) != root:
                        root = scratch[root]
                    return root

                for left, right in tied:
                    a, b = scratch_find(find(left)), scratch_find(find(right))
                    if a != b:
                        scratch[max(a, b, key=str)] = min(a, b, key=str)
                groups: dict[URIRef, list[tuple[URIRef, URIRef]]] = {}
                roots_of: dict[URIRef, set[URIRef]] = {}
                for left, right in tied:
                    head = scratch_find(find(left))
                    groups.setdefault(head, []).append((left, right))
                    roots_of.setdefault(head, set()).update((find(left), find(right)))
                joined_roots: set[URIRef] = set()
                for head, edges in groups.items():
                    roots = roots_of[head]
                    if len(roots) < 2:
                        continue
                    joined = set().union(*(members[root] for root in roots))
                    if holds_veto(joined):
                        continue
                    applied.append(edges)
                    first, *rest = sorted(roots, key=str)
                    for root in rest:
                        union(find(first), find(root))
                    joined_roots.add(find(first))
                    changed = True
                # The batch moved identity: a pair it joined, or one it made
                # distinct, is read again on the clusters as they now are.
                if joined_roots:
                    undo_vetoed({find(root) for root in joined_roots})
            # A union may make two members of an older cluster distinct.
            if undo_vetoed({root for root, group in members.items() if len(group) > 1}):
                changed = True
            if not changed:
                break

        rejected: list[tuple[URIRef, URIRef, float | None, tuple[str, ...]]] = []
        for left, right, score, rank in candidates:
            if find(left) == find(right):
                continue
            failures = self._merge_validation_failures(
                left,
                right,
                representations,
                guard_context=guard_context,
                direct_relation_pairs=direct_relation_pairs,
                same=same,
                confirmed=rank in (0, 1),
            )
            rejected.append((left, right, score, tuple(failures or ["cluster_veto"])))

        grouped: dict[URIRef, list[URIRef]] = {}
        for candidate_cluster in candidate_clusters:
            for entity in candidate_cluster:
                grouped.setdefault(find(entity), []).append(entity)
        clusters = [sorted(group, key=str) for group in grouped.values()]
        return clusters, rejected

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

    def _register_entity(
        self,
        *,
        entity: URIRef,
        unit: ContentUnit,
        state: _EntityCollectionState,
    ) -> None:
        """Register one URI entity with its document and stable classification."""
        state.entities.add(entity)
        state.source_entities.add(entity)
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

        Each entity is associated with a graph holding its triples and the
        ``doc_iri`` of the first :class:`ContentUnit` that mentions it (in
        practice most pipelines aggregate chunks of the same document, so all
        ``doc_iri`` values are identical).

        Args:
            units: List of content units to aggregate.
            known_ontology_entities: Entities of the selected ontology, used
                for classification.

        Returns:
            Tuple of (
                entities,
                source_entities,
                entity_to_graph,
                entity_to_doc_iri,
                entity_to_classification,
                direct_relation_pairs,
                object_groups,
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
            for triple in unit.graph:
                state.context.add(triple)
                s, p, o = triple
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    # Direct relations and the single-valued inference key on
                    # *names*: a subject mentioned in two units points at two
                    # scoped objects, and one subject named in two units must
                    # not make a predicate look single-valued on twice the
                    # subjects.
                    name_s, name_o = unscoped_iri(s), unscoped_iri(o)
                    self._register_direct_relation(
                        state=state, subject=name_s, obj=name_o
                    )
                    if isinstance(p, URIRef) and p != RDF.type:
                        state.object_groups.setdefault((name_s, p), set()).add(name_o)
                for term in (s, p, o):
                    if isinstance(term, URIRef):
                        self._register_entity(entity=term, unit=unit, state=state)

        # One graph serves every entity: a representation reads only the
        # triples that mention its entity, through the graph's indexes.
        return (
            list(state.entities),
            state.source_entities,
            dict.fromkeys(state.entities, state.context),
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
        confirmed_pairs: set[frozenset[URIRef]] | None = None,
    ) -> AggregationResult:
        """Aggregate multiple content unit graphs with embedding-based disambiguation.

        Args:
            units: List of ContentUnits to aggregate.
            ontology_graph: Selected ontology graph used to distinguish
                known ontology entities from tentative ontology-like aliases.
            merge_vetoes: Extra entity pairs that must never identity-merge —
                the targeted un-merge lever used by the post-aggregation
                validation gate. Unioned into the direct-relation veto set.
            confirmed_pairs: Entity pairs confirmed as one entity (coreference
                resolution): candidates whatever their embeddings, merged like a
                natural-key pair, and still subject to every distinctness veto.

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
        contract = CatalogRoles.of([ontology_graph]).statement_contract(
            frozenset(well_known_value_classes())
        )
        guard_context = MergeGuardContext(
            functional_predicates=schema_functional_predicates
            | empirically_functional_predicates(
                object_groups,
                min_support=self.functional_min_empirical_support,
            ),
            disjoint_classes=harvest_disjoint_classes(ontology_graph),
            fact_entities={
                entity
                for entity, classification in entity_classification.items()
                if classification == EntityClassification.FACT
            },
            statements=StatementIndex.of(
                (unit.graph for unit in units if unit.graph is not None), contract
            ),
        )
        representations = self.normalizer.create_representations_batch(
            entities, entity_graphs
        )
        guard_context.name_list_predicates = {
            predicate
            for representation in representations.values()
            for predicate, values in representation.predicate_string_literals.items()
            if len(values) >= 2
        } - guard_context.functional_predicates
        for representation in representations.values():
            for class_iri in representation.types:
                if class_iri not in guard_context.class_ancestors:
                    guard_context.class_ancestors[class_iri] = frozenset(
                        superclass_closure(class_iri, ontology_graph)
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
        anchor_candidates = (
            self._select_ontology_anchor_candidates(
                tentative_entities=tentative_entities,
                tentative_representations=representations,
                tentative_doc_iris=entity_doc_iris,
                ontology_graph=ontology_graph,
                known_ontology_entities=known_ontology_entities,
            )
            if self.enabled
            else {}
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

        key_pairs: set[frozenset[URIRef]] = set()
        clusters: list[list[URIRef]]
        rejected_merges: list[tuple[URIRef, URIRef, float | None, tuple[str, ...]]]
        if self.enabled:
            candidate_clusters, embeddings = self._cluster_entities_by_role(
                representations
            )
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
            # Embedding clusters can split entities that share a name (their
            # types and neighbourhoods differ by unit); a shared name must at
            # least bring them together for symbolic validation.
            candidate_clusters = self._merge_candidate_clusters_by_key_pairs(
                candidate_clusters, self._collect_label_pairs(representations)
            )
            confirmed = {
                pair
                for pair in confirmed_pairs or set()
                if pair.issubset(representations)
            }
            if confirmed:
                candidate_clusters = self._merge_candidate_clusters_by_key_pairs(
                    candidate_clusters, confirmed
                )
            clusters, rejected_merges = self._build_identity_clusters(
                candidate_clusters=candidate_clusters,
                representations=representations,
                embeddings=embeddings,
                direct_relation_pairs=direct_relation_pairs,
                guard_context=guard_context,
                key_pairs=(key_pairs | confirmed) or None,
            )
        else:
            # Consolidation off: every entity is its own cluster, so nothing is
            # embedded or merged; naming and provenance below still run.
            clusters = [[entity] for entity in representations]
            rejected_merges = []
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

        cluster_of = {entity: cluster[0] for cluster in clusters for entity in cluster}

        def same(left: object, right: object) -> bool:
            return left == right or (
                isinstance(left, URIRef)
                and isinstance(right, URIRef)
                and cluster_of.get(left, left) == cluster_of.get(right, right)
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
                        same=same,
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
                        same=same,
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
                            same=same,
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
        merged_values = merge_equal_values(merged_graph, contract)
        if merged_values:
            logger.info("Merged %d value(s) equal by content", merged_values)

        merged_clusters = build_merged_clusters(final_mapping, identity_mapping)
        confirmed_clusters = sorted(
            {
                str(final_mapping[left])
                for pair in key_pairs | (confirmed_pairs or set())
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
            rejected_by_reason=dict(
                sorted(
                    Counter(
                        reason for *_, failed in rejected_merges for reason in failed
                    ).items()
                )
            ),
            confirmed_clusters=confirmed_clusters,
            merge_provenance=build_merge_provenance(active_units, final_mapping),
            lexical_near_misses=[
                (str(left), str(right), score or 0.0, failed[0])
                for left, right, score, failed in rejected_merges
                if failed in (("lexical",), ("overlap",))
            ],
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
        confirmed_pairs: set[frozenset[URIRef]] | None = None,
    ) -> AggregationResult:
        """Sanitize facts units, then run aggregation/normalization.

        This method is intentionally safe for both single-unit and multi-unit
        inputs so unit-pipeline and graph-pipeline paths share the same
        post-processing behavior.

        When ``doc_iri`` and non-empty ``document_metadata`` are provided,
        caller-asserted document identity triples are attached to the merged
        facts graph. Business-oriented keys mint typed entities under
        ``doc_namespace`` (defaults to the document facts namespace).

        Every facts unit first spells well-known terms in their canonical
        namespace (:meth:`RDFGraph.canonicalize_well_known_iris`), so a
        scheme or separator variant never reaches identity resolution.

        With ``unit_scoped_fact_iris`` on, every minted fact IRI is suffixed
        with its unit index *in place* on the unit graph before aggregation
        (see :mod:`ontocast.tool.agg.unit_scope`), so a local name minted by
        two units is a merge decision rather than an accidental fusion. The
        rewrite is idempotent, which keeps the validation gate's
        re-aggregation of the same units on identical input.

        Args:
            units: Facts content units to aggregate.
            ontology_graph: Merged ontology context for classification/guards.
            doc_iri: Document IRI for metadata provenance attachment.
            document_metadata: Caller-asserted document identity metadata.
            doc_namespace: Namespace for metadata-minted entities.
            merge_vetoes: Entity pairs that must never identity-merge
                (validation-gate un-merge lever).
            confirmed_pairs: Entity pairs confirmed as one entity (coreference
                resolution); see :meth:`aggregate_graphs`.

        Returns:
            :class:`AggregationResult`; its ``graph`` carries the merged facts
            plus any document-metadata provenance.
        """
        for unit in units:
            unit.sanitize()
            if unit.type == OutputType.ONTOLOGIES:
                continue
            # A well-known term spelled in a variant of its namespace is the
            # same term: canonical before any identity is weighed.
            unit.graph.canonicalize_well_known_iris()
            if self.unit_scoped_fact_iris:
                scope_fact_iris(
                    unit.graph, unit.index, (self.base_iri, str(unit.doc_iri))
                )
        result = self.aggregate_graphs(
            units=units,
            ontology_graph=ontology_graph,
            merge_vetoes=merge_vetoes,
            confirmed_pairs=confirmed_pairs,
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
