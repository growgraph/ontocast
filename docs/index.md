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
<svg viewBox="0 0 440 300" role="img" aria-labelledby="oc-layers-title" xmlns="http://www.w3.org/2000/svg">
<title id="oc-layers-title">Documents pass through a render-and-critique loop and become two layers of one graph: an ontology of classes above, and facts below, each fact typed by a class.</title>
<defs>
<marker id="oc-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path class="oc-layers__arrowhead" d="M0,1 L9,5 L0,9 z"/>
</marker>
<marker id="oc-arrow-loop" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
<path class="oc-layers__arrowhead oc-layers__arrowhead--loop" d="M0,1 L9,5 L0,9 z"/>
</marker>
</defs>
<rect class="oc-layers__band oc-layers__band--onto" x="186" y="36" width="246" height="86" rx="8"/>
<rect class="oc-layers__band oc-layers__band--facts" x="186" y="172" width="246" height="100" rx="8"/>
<g class="oc-layers__doc">
<rect x="44" y="82" width="62" height="82" rx="4"/>
<rect x="36" y="90" width="62" height="82" rx="4"/>
<rect x="28" y="98" width="62" height="82" rx="4"/>
</g>
<g class="oc-layers__text-line">
<path d="M38,114 H80"/>
<path d="M38,124 H74"/>
<path d="M38,134 H80"/>
<path d="M38,144 H66"/>
<path d="M38,154 H78"/>
<path d="M38,164 H58"/>
</g>
<text class="oc-layers__caption" x="62" y="202" text-anchor="middle">Documents</text>
<path class="oc-layers__flow" d="M108,140 H121" marker-end="url(#oc-arrow)"/>
<path class="oc-layers__loop" d="M157,122.7 A20,20 0 1 1 137,122.7" marker-end="url(#oc-arrow-loop)"/>
<text class="oc-layers__small" x="147" y="178" text-anchor="middle">render</text>
<text class="oc-layers__small" x="147" y="192" text-anchor="middle">critique</text>
<path class="oc-layers__flow" d="M171,140 H182" marker-end="url(#oc-arrow)"/>
<g class="oc-layers__typed">
<path d="M212,208 L236,96"/>
<path d="M254,236 L236,96"/>
<path d="M290,204 L310,76"/>
<path d="M326,244 L310,76"/>
<path d="M362,212 L384,100"/>
<path d="M404,238 L384,100"/>
</g>
<g class="oc-layers__schema-edge">
<path d="M236,96 L310,76"/>
<path d="M310,76 L384,100"/>
</g>
<g class="oc-layers__class">
<rect x="219" y="85" width="34" height="22" rx="5"/>
<rect x="293" y="65" width="34" height="22" rx="5"/>
<rect x="367" y="89" width="34" height="22" rx="5"/>
</g>
<g class="oc-layers__fact-edge">
<path d="M212,208 L254,236 L290,204 L326,244 L362,212 L404,238"/>
<path d="M254,236 L326,244"/>
</g>
<g class="oc-layers__fact">
<circle cx="212" cy="208" r="5.5"/>
<circle cx="254" cy="236" r="5.5"/>
<circle cx="290" cy="204" r="5.5"/>
<circle cx="326" cy="244" r="5.5"/>
<circle cx="362" cy="212" r="5.5"/>
<circle cx="404" cy="238" r="5.5"/>
</g>
<text class="oc-layers__label" x="198" y="56">Ontology</text>
<text class="oc-layers__label" x="198" y="264">Facts</text>
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
