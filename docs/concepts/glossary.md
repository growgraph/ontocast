# Glossary

The terms these pages use, each with the page that explains it.

**Catalog**
:   The set of ontologies OntoCast knows: those loaded at startup, uploaded to
    `/ontologies`, or created by earlier runs. Every ontology context is taken
    from it. See [Ontologies and facts](ontologies_and_facts.md#the-catalog).

**Completion pass**
:   An optional extra call, after the critic, that asks the model for
    measurements the render missed. It only adds facts. See [How OntoCast
    works](index.md#the-per-unit-loop).

**Consolidation**
:   An optional LLM call that refines a document's ontology against an excerpt
    of the whole document, after the units' changes are merged
    (`ENABLE_ONTOLOGY_CONSOLIDATION`). See [How OntoCast
    works](index.md#the-ontology-block).

**Content unit**
:   One part of a document, cut along its sections, that is extracted on its
    own and in parallel with the others. See [How OntoCast
    works](index.md#chunk).

**Critic**
:   The LLM call that reviews a render and names the statements to fix.
    OntoCast applies the fixes as a patch without another call. See [Validation
    layers](../internals/validation_layers.md).

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

**Ontology**
:   A graph of classes, properties and declared individuals, identified by its
    IRI and stored in versions. See [Ontologies and
    facts](ontologies_and_facts.md).

**Ontology context**
:   The slice of the catalog a content unit is shown: one chosen ontology, the
    terms vector retrieval finds, or one fixed ontology
    (`ONTOLOGY_CONTEXT_MODE`). See [Choosing ontology
    context](../guides/ontology_context.md).

**Provenance**
:   The link from each fact to the content unit it came from, written with RDF
    1.2 reifiers. See [Ontologies and facts](ontologies_and_facts.md#provenance).

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

**Section label**
:   The section a content unit came from, such as `methods` or `results`, used
    to select what to extract. See [Structured
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

**Unit pipeline**
:   The per-unit loops run on a single piece of text, with no conversion,
    chunking or merging: `POST /process_unit` or `run_unit_pipeline`. See
    [Embedding OntoCast in your agent](../guides/embedding.md).

**User instruction**
:   Your own guidance, added to the prompts of one request
    (`facts_user_instruction`, `ontology_user_instruction`). See [Writing user
    instructions](../guides/user_instructions.md).

**Vector retrieval**
:   Choosing a unit's ontology context by searching an index of the whole
    catalog (LanceDB or Qdrant) for the terms the unit needs. See
    [Retrieval](../internals/retrieval.md).
