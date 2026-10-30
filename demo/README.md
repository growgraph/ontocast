# OntoCast Demo

This demo shows how OntoCast can extract a knowledge graph from a document and visualize it as a graph.

---

## What You'll See
- Turn a PDF or text document into a knowledge graph (RDF triples)
- Visualize the extracted ontology and facts using free online tools

---

## Demo Steps

Run every command below from the repository root, so the relative paths resolve.

### 1. Start the OntoCast Server

Make sure you have OntoCast installed and a triple store (optional) running. Then start the server:

```bash
ONTOCAST_ONTOLOGY_DIRECTORY=test/data/ontologies uv run ontocast serve
```

### 2. Process a Sample Document

Two example PDFs ship in this directory. Either works:

**For PDF:**
```bash
curl -X POST http://localhost:8999/process \
  -F "file=@demo/A_Brief_Introduction_To_AI.pdf"
```

**For plain text:**
```bash
curl -X POST http://localhost:8999/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here"}'
```

The response will include extracted facts and ontology in Turtle format.
`ttl/response.json` is a response you can inspect without running the server. It
was recorded with an older version, so its shape and namespaces differ from what
a current server returns.

---

### 3. Visualize the Output

Copy the Turtle from the `facts` or `ontology` field of the response into any RDF
viewer or triple store.

**Example Screenshot:**

![Demo Graph Screenshot](figs/thames-water.png)

---

## Files in This Directory
- `A_Brief_Introduction_To_AI.pdf` – Example document for the demo
- `water-resources-management-plan-overview.pdf` – Second example document
- `ttl/response.json` – `/process` response recorded with an older version
- `figs/thames-water.png` – Screenshot used above

---

## Tips
- You can use your own documents by changing the file path in the `curl` command.
- For more details, see the [main README](../README.md) or the [Triple Store Setup Guide](https://growgraph.github.io/ontocast/guides/triple_stores/).

---

## Need Help?
Open an issue or check the [OntoCast documentation](https://growgraph.github.io/ontocast).
