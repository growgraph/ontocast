---
hide:
- navigation
- toc
---

<div class="oc-hero" markdown>
<div class="oc-hero__text" markdown>

# OntoCast

OntoCast reads documents and writes an RDF knowledge graph: an ontology that
describes the domain, and the facts the documents state in its terms.
{ .oc-lead }

Give it your ontologies and it extracts facts against them; give it none and it
builds one as it reads. Each part of a document is handled by a language model
in a render-and-critique loop, and every change to a graph is a small,
checked patch rather than a rewritten file.

It is for engineers who need a graph they can query, validate and trace back to
the text it came from.

```bash
pip install "ontocast[server,openai]"
```

[Quick start](getting_started/quickstart.md){ .md-button .md-button--primary }
[How it works](concepts/index.md){ .md-button }

</div>
<figure class="oc-layers">
<svg viewBox="0 4 480 267" role="img" aria-labelledby="oc-layers-title" xmlns="http://www.w3.org/2000/svg">
<title id="oc-layers-title">Documents pass through a render-and-critique loop and become two layers of one graph: an ontology of classes above, and facts below, each fact tied to its class.</title>
<defs>
<marker id="oc-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path class="oc-layers__arrowhead" d="M0,1 L9,5 L0,9 z"/></marker>
<marker id="oc-arrow-loop" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto"><path class="oc-layers__arrowhead oc-layers__arrowhead--loop" d="M0,1 L9,5 L0,9 z"/></marker>
</defs>
<g class="oc-layers__doc"><rect x="32" y="101" width="48" height="64" rx="4"/><rect x="24" y="109" width="48" height="64" rx="4"/><rect x="16" y="117" width="48" height="64" rx="4"/></g>
<g class="oc-layers__text-line"><path d="M25,132 H55"/><path d="M25,141 H50"/><path d="M25,150 H55"/><path d="M25,159 H46"/><path d="M25,168 H53"/></g>
<text class="oc-layers__caption" x="48" y="201" text-anchor="middle">Documents</text>
<path class="oc-layers__flow" d="M88,140 H104" marker-end="url(#oc-arrow)"/>
<path class="oc-layers__loop" d="M114.3,136.2 A22,22 0 0 1 157.7,136.2" marker-end="url(#oc-arrow-loop)"/>
<path class="oc-layers__loop oc-layers__loop--back" d="M157.7,143.8 A22,22 0 0 1 114.3,143.8" marker-end="url(#oc-arrow-loop)"/>
<text class="oc-layers__small" x="136" y="109" text-anchor="middle">render</text>
<text class="oc-layers__small" x="136" y="179" text-anchor="middle">critique</text>
<path class="oc-layers__flow" d="M166,131 Q180,76.4 211.1,64.4" marker-end="url(#oc-arrow)"/>
<path class="oc-layers__flow" d="M166,149 Q180,171.6 232.2,183.6" marker-end="url(#oc-arrow)"/>
<path class="oc-layers__slab oc-layers__slab--facts" d="M251.0,170.8 L463.6,206.2 L431.0,267.0 L210.5,228.6 Z"/>
<path class="oc-layers__sheet oc-layers__sheet--facts" d="M251.0,166.8 L463.6,202.2 L431.0,263.0 L210.5,224.6 Z"/>
<g class="oc-layers__fact-edge"><path d="M253.8,183.1 L248.4,213.9"/><path d="M253.8,183.1 L299.2,181.1"/><path d="M248.4,213.9 L353.0,214.9"/><path d="M299.2,181.1 L353.0,214.9"/><path d="M353.0,214.9 L417.6,253.8"/><path d="M353.0,214.9 L445.9,208.1"/><path d="M248.4,213.9 L299.2,181.1"/><path d="M445.9,208.1 L417.6,253.8"/></g>
<g class="oc-layers__typed"><path d="M253.8,183.1 L248.7,75.1"/><path d="M248.4,213.9 L332.9,47.6"/><path d="M299.2,181.1 L248.7,75.1"/><path d="M353.0,214.9 L332.9,47.6"/><path d="M417.6,253.8 L414.5,101.0"/><path d="M445.9,208.1 L414.5,101.0"/></g>
<g class="oc-layers__fact"><path d="M258.7,184.0 L258.0,184.7 L256.8,185.3 L255.3,185.7 L253.7,185.8 L252.0,185.7 L250.6,185.3 L249.5,184.7 L248.8,184.0 L248.6,183.1 L248.9,182.3 L249.7,181.6 L250.9,181.0 L252.3,180.6 L254.0,180.4 L255.6,180.6 L257.0,181.0 L258.1,181.6 L258.8,182.3 L259.0,183.1 Z"/><path d="M255.9,215.2 L254.7,216.4 L252.9,217.3 L250.7,217.9 L248.2,218.1 L245.7,217.9 L243.5,217.3 L241.8,216.4 L240.8,215.2 L240.5,213.9 L241.0,212.7 L242.2,211.5 L244.0,210.6 L246.2,210.0 L248.7,209.8 L251.1,210.0 L253.3,210.6 L255.0,211.5 L256.1,212.7 L256.4,213.9 Z"/><path d="M305.4,182.1 L304.5,183.0 L303.0,183.8 L301.1,184.3 L299.1,184.4 L297.1,184.3 L295.3,183.8 L293.8,183.0 L292.9,182.1 L292.6,181.1 L293.0,180.0 L293.9,179.1 L295.4,178.3 L297.3,177.9 L299.3,177.7 L301.3,177.9 L303.1,178.3 L304.6,179.1 L305.5,180.0 L305.8,181.1 Z"/><path d="M360.1,216.2 L359.0,217.2 L357.4,218.1 L355.3,218.7 L353.0,218.9 L350.7,218.7 L348.7,218.1 L347.0,217.2 L346.0,216.2 L345.6,214.9 L345.9,213.7 L347.0,212.7 L348.6,211.8 L350.7,211.2 L353.0,211.1 L355.3,211.2 L357.3,211.8 L359.0,212.7 L360.0,213.7 L360.4,214.9 Z"/><path d="M423.8,254.9 L422.9,255.8 L421.5,256.6 L419.8,257.1 L417.8,257.3 L415.8,257.1 L414.0,256.6 L412.5,255.8 L411.6,254.9 L411.2,253.8 L411.5,252.7 L412.3,251.8 L413.7,251.0 L415.5,250.5 L417.4,250.4 L419.4,250.5 L421.2,251.0 L422.7,251.8 L423.7,252.7 L424.0,253.8 Z"/><path d="M454.0,209.5 L452.9,210.7 L451.1,211.7 L448.8,212.3 L446.3,212.5 L443.7,212.3 L441.3,211.7 L439.4,210.7 L438.1,209.5 L437.6,208.1 L437.9,206.8 L439.0,205.6 L440.8,204.6 L443.1,204.0 L445.6,203.8 L448.2,204.0 L450.6,204.6 L452.5,205.6 L453.8,206.8 L454.3,208.1 Z"/></g>
<path class="oc-layers__slab oc-layers__slab--onto" d="M248.1,30.0 L468.0,63.0 L434.3,119.8 L206.0,83.9 Z"/>
<path class="oc-layers__sheet oc-layers__sheet--onto" d="M248.1,26.0 L468.0,59.0 L434.3,115.8 L206.0,79.9 Z"/>
<g class="oc-layers__schema-edge"><path d="M332.9,47.6 L359.9,93.5"/><path d="M414.5,101.0 L359.9,93.5"/><path d="M248.7,75.1 L332.9,47.6"/><path d="M332.9,47.6 L414.5,101.0"/></g>
<g class="oc-layers__class"><path d="M237.5,67.1 L268.3,71.9 L260.0,83.1 L229.0,78.3 Z"/><path d="M321.4,39.9 L351.9,44.5 L344.5,55.3 L313.8,50.6 Z"/><path d="M402.0,92.7 L434.1,97.7 L427.1,109.3 L394.8,104.3 Z"/><path d="M347.8,85.3 L379.5,90.2 L372.1,101.8 L340.1,96.7 Z"/></g>
<text class="oc-layers__label" x="234.1" y="32.0" text-anchor="end">Ontology</text>
<text class="oc-layers__label" x="196.5" y="230.6" text-anchor="end">Facts</text>
</svg>
</figure>
</div>

## What you can do with it

<div class="oc-columns" markdown>
<div markdown>

### Build or extend an ontology

From the text, OntoCast proposes classes and properties, checks them for
structure and consistency, and merges them into the ontology one version at a
time. Start from your own ontologies or from none.

</div>
<div markdown>

### Extract facts in its terms

Facts are written against the ontology, so every new entity is typed by it.
Mentions of the same entity across a document are merged into one, and the
result can be validated with SHACL shapes before it is stored.

</div>
<div markdown>

### Run it the way you work

Run OntoCast as an HTTP service, as a batch command over a folder, or inside
your own LangChain or LangGraph agent. Graphs stay in memory by default, or go
to Apache Jena Fuseki.

</div>
</div>

## What to read next

<div class="grid cards oc-next" markdown>

-   **[Installation](getting_started/installation.md)**

    Install the package with the extras you need.

-   **[Quick start](getting_started/quickstart.md)**

    Start the server and turn one document into a graph.

-   **[How it works](concepts/index.md)**

    The pipeline from document to graph, stage by stage.

-   **[Recipes](guides/recipes.md)**

    Settings for evaluating, building an ontology, or serving.

</div>
