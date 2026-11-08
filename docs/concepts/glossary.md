# Glossary

The terms these pages use, each with the page that explains it.

**Candidate term**
:   A class or property a unit minted that the document's ontology plan did
    not list. Consolidation decides it; without a decision it is dropped. See
    [How OntoCast works](index.md#the-ontology-block).

**Catalog**
:   The set of ontologies OntoCast knows: those loaded at startup, uploaded to
    `/ontologies`, or created by earlier runs. Every ontology context is taken
    from it. See [Ontologies and facts](ontologies_and_facts.md#the-catalog).

**Completion pass**
:   An optional extra call, after the critic, that asks the model for
    measurements the render missed. It only adds facts. See [How OntoCast
    works](index.md#the-per-unit-loop).

**Consolidation**
:   The step that decides every term a document minted before it is written:
    keep, merge into another term, generalise, demote to a property value, or
    drop (`ONTOLOGY_CONSOLIDATION`). See [How OntoCast
    works](index.md#the-ontology-block).

**Content unit**
:   One part of a document, cut along its sections, that is extracted on its
    own and in parallel with the others. See [How OntoCast
    works](index.md#chunk).

**Coreference resolution**
:   At most one LLM call per document, after the merge, that decides the entity pairs
    the merge could not settle (`AGG_LLM_COREFERENCE`). See [How OntoCast
    works](index.md#the-facts-block).

**Critic**
:   The LLM call that reviews a render and names the statements to fix.
    OntoCast applies the fixes as a patch without another call. See [Validation
    layers](../internals/validation_layers.md).

**Curation**
:   A pass over a catalog ontology, across every document processed so far,
    that keeps, merges, generalises, demotes or drops the terms documents
    added (`ontocast curate`). See [Curating the
    catalog](../guides/curation.md).

**Entity disambiguation**
:   Deciding which entities from different content units are the same thing,
    and merging them. See [Entity disambiguation](entity_disambiguation.md).

**Facts**
:   The graph of what a document states: the entities it mentions and their
    relations and values, in the terms of the ontology. See [Ontologies and
    facts](ontologies_and_facts.md#facts-and-the-two-namespace-contract).

**Facts namespace (`cd:`)**
:   `https://growgraph.dev/facts/`, where every entity extracted from text is
    named. The ontology's own namespaces are never extended by facts. See
    [Ontologies and facts](ontologies_and_facts.md#facts-and-the-two-namespace-contract).

**GraphUpdate (patch)**
:   What the model returns instead of a whole graph: the triples to insert and
    the triples to delete. See [Ontologies and
    facts](ontologies_and_facts.md#changes-are-patches).

**Head**
:   The version of an ontology no other version names as prior. A run reads
    one head per ontology; an ontology with several heads is diverged, and runs
    cannot extend it until an upload (`PUT`) merges it. See [Ontologies and
    facts](ontologies_and_facts.md#versions).

**Extension ontology**
:   The ontology `CURRENT_DOMAIN/<id>-ext` that holds the terms OntoCast adds
    to the external ontology `<id>`, which it imports. See [Ontologies and
    facts](ontologies_and_facts.md#owned-and-external-ontologies).

**External ontology**
:   A catalog ontology whose IRI is not under `CURRENT_DOMAIN`, such as a
    published vocabulary. OntoCast reads it and never writes to it.

**Home ontology**
:   Where a document's new vocabulary goes: an existing catalog ontology it
    extends, or a new one minted under `CURRENT_DOMAIN`, resolved once per
    document. See [Ontologies and
    facts](ontologies_and_facts.md#the-home-ontology).

**IRI**
:   The web-style name of anything in a graph, such as
    `https://example.org/onto#Device`. Two things with the same IRI are the same
    thing, wherever they appear.

**Label**
:   The readable name of a term or entity (`rdfs:label`). The IRI identifies it;
    the label is what a person reads.

**Ontology**
:   A graph of classes, properties and declared individuals, identified by its
    IRI and stored in versions. See [Ontologies and
    facts](ontologies_and_facts.md).

**Ontology context**
:   The slice of the catalog a content unit is shown: one chosen ontology, the
    terms vector retrieval finds, or one fixed ontology
    (`ONTOLOGY_CONTEXT_MODE`). See [Choosing ontology
    context](../guides/ontology_context.md).

**Ontology plan**
:   The classes and properties a document adds to its domain, listed by one
    LLM call over the whole document before its units render and checked
    against the catalog without a model (`ONTOLOGY_PLAN_ENABLED`). Units mint
    from it. See [How OntoCast works](index.md#the-ontology-block).

**Prefix**
:   A short alias for the start of an IRI, so `ex:Device` can stand for
    `https://example.org/onto#Device` in Turtle. It changes how a file reads, not
    what it means.

**Provenance**
:   The link from each fact to the content unit it came from, written with RDF
    1.2 reifiers. See [Ontologies and facts](ontologies_and_facts.md#provenance).

**RDF**
:   The W3C standard data model for graphs: a graph is a set of triples. OWL and
    SHACL are written in it, and Turtle is one way to write it down.

**Render**
:   The LLM call that extracts an ontology patch or a unit's facts. See [How
    OntoCast works](index.md#the-per-unit-loop).

**Render mode**
:   Which blocks a run executes: `ontology_and_facts`, `ontology` or `facts`
    (`RENDER_MODE`). See [How OntoCast works](index.md#which-blocks-run).

**Run manifest**
:   The JSON file `ontocast process` writes beside each facts file
    (`<name>.run.json`): the model, the settings and what the run cost. See
    [Reading run telemetry](../guides/telemetry.md).

**Route**
:   What happens to a content unit before extraction: extracted, extracted as
    bibliographic records, kept but flagged as non-content (front or back
    matter, a table of contents), or dropped, with the reason recorded. See [Structured
    documents](structured_documents.md#how-a-unit-is-routed).

**Route class**
:   What a section label's sections hold, set in its section schema:
    `content`, `boilerplate`, `non_content` (front and back matter) or
    `citations` (a reference list). Evidence for the route, never a decision
    on its own.

**Section label**
:   The section a content unit came from, such as `methods` or `results`, used
    to select what to extract. A label is *strong* when it comes from an exact
    heading, a heading placed inside a strong section by its number or level,
    or a reference list or acknowledgements recognised from the content, and
    *weak* otherwise (a keyword in a longer heading, a heading's nearest label
    by embedding, inheritance by wording or from a weak parent, front matter
    recognised by position, other guesses from the content, the model); only a
    strong label can take part in dropping a unit.
    Each step's vote is kept as the label's evidence. See [Structured
    documents](structured_documents.md#section-labels).

**Section schema**
:   The set of section labels for one kind of document, such as `academic` or
    `financial`. See [Structured
    documents](structured_documents.md#section-schemas).

**SHACL shape**
:   A constraint the extracted facts are validated against, loaded from
    `FACTS_SHAPES_DIR` or uploaded to `/shapes`. See [Validation and
    SHACL](../guides/validation.md).

**Tenant and project**
:   Request parameters that partition the triple store and the vector store, so
    that several users or projects share one server without seeing each other's
    graphs. See [Tenancy](../guides/tenancy.md).

**Term sheet**
:   A compact form of the ontology context in facts prompts: one line per term
    instead of a serialized graph (`ONTOLOGY_CHAPTER_FORMAT`). See [Performance
    tuning](../guides/performance.md).

**Triple**
:   One statement in three parts: subject, predicate, object. In `ex:Device_C
    ex:madeOf ex:Perovskite .` the subject and object are entities and the
    predicate is the link between them. The object can also be a value, such as a
    number or text.

**Turtle**
:   A compact text format for RDF: triples ending in `.`, with prefixes keeping
    IRIs short. OntoCast returns graphs in it, and its models write it by
    default (`LLM_GRAPH_FORMAT`).

**Unit pipeline**
:   The per-unit loops run on a single piece of text, with no conversion,
    chunking or merging: `POST /process_unit` or `run_unit_pipeline`. See
    [Embedding OntoCast in your agent](../guides/embedding.md).

**User instruction**
:   Your own guidance, added to the prompts of one request
    (`facts_user_instruction`, `ontology_user_instruction`,
    `ontology_selection_user_instruction`). See [Writing user
    instructions](../guides/user_instructions.md).

**Vector retrieval**
:   Choosing a unit's ontology context by searching an index of the whole
    catalog (LanceDB or Qdrant) for the terms the unit needs. See
    [Retrieval](../internals/retrieval.md).

**Version IRI**
:   The named graph one immutable version of an ontology is stored as,
    `<iri>#<key>`; its header names the prior version(s) it was made from.
    See [Ontologies and facts](ontologies_and_facts.md#versions).
