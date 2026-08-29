# OntoCast Demo

This demo shows how OntoCast can extract a knowledge graph from a document and visualize it as a graph.

---

## What You'll See
- Turn a PDF or text document into a knowledge graph (RDF triples)
- Visualize the extracted ontology and facts using free online tools

---

## Demo Steps

### 1. Start the OntoCast Server

Make sure you have OntoCast installed and a triple store (optional) running. Then start the server:

```bash
ONTOCAST_ONTOLOGY_DIRECTORY=../test/data/ontologies uv run ontocast serve
```

### 2. Process a Sample Document

You can use the provided `sample.pdf` or `sample.txt` in this directory.

**For PDF:**
```bash
curl -X POST http://localhost:8999/process -F "file=@demo/sample.pdf"
```

**For plain text:**
```bash
curl -X POST http://localhost:8999/process -H "Content-Type: application/json" -d '{"text": "Your document text here"}'
```

The response will include extracted facts and ontology in Turtle format.

---

### 3. Visualize the Output

1. Use neo4j built-in graph navigator
2. Copy the Turtle output from the `facts` or `ontology` field in the response.

**Example Screenshot:**

![Demo Graph Screenshot](figs/thames-water.png)

---

## Files in This Directory
- `sample.pdf` – Example document for demo
- `sample.txt` – Example plain text for demo
- (Optional) `sample.ttl` – Example Turtle output

---

## Tips
- You can use your own documents by changing the file path in the `curl` command.
- For more details, see the [main README](../README.md) or the [Triple Store Setup Guide](../docs/user_guide/triple_stores.md).

---

## Need Help?
Open an issue or check the [OntoCast documentation](https://growgraph.github.io/ontocast). 