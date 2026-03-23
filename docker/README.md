# Triple Store Configuration

OntoCast supports multiple triple store backends for storing and managing RDF data. This guide covers the setup and configuration of supported triple stores.

---

## Overview

OntoCast supports the following triple store backends:

1. **Apache Fuseki** (Recommended) - Native RDF triple store with SPARQL support
2. **Neo4j with n10s plugin** - Graph database with RDF capabilities
3. **Filesystem** - Local file-based storage (fallback)

When multiple triple stores are configured, OntoCast uses the following priority order:
1. Fuseki (if `FUSEKI_URI` and `FUSEKI_AUTH` are set)
2. Neo4j (if `NEO4J_URI` and `NEO4J_AUTH` are set)
3. Filesystem (default fallback)

---

## Environment Variables

Configure your triple store connection using environment variables in your main `.env` file:

```bash
# Fuseki Configuration (Preferred)
FUSEKI_URI=http://localhost:3032
FUSEKI_AUTH=admin/abc123-qwe

# Neo4j Configuration (Alternative)
NEO4J_URI=bolt://localhost:7689
NEO4J_AUTH=neo4j/test!passfortesting
```

---

## Apache Fuseki Setup

Sample configurations are provided here: [ontocast/docker](https://github.com/growgraph/ontocast/tree/main/docker).

**1. Prepare the environment file:**
```bash
cd docker/fuseki
cp .env.example .env
# Edit with your values
```

**Example `docker/fuseki/.env.example`:**
```bash
IMAGE_VERSION=secoresearch/fuseki:5.1.0
SPEC=test
CONTAINER_NAME="${SPEC}.fuseki"
STORE_FOLDER="$HOME/tmp/${CONTAINER_NAME}"
TS_PORT=3032
TS_PASSWORD="abc123-qwe"
TS_USERNAME="admin"
UID=1000
GID=1000
```

**2. Start/Stop Fuseki:**
```bash
# Start
cd docker/fuseki
docker compose --env-file .env up fuseki -d

# Stop
# (use the container name from your .env, e.g. test.fuseki)
docker compose stop test.fuseki
```

**3. Access Fuseki:**
- Web interface: http://localhost:3032
- Default dataset: `/test`
- SPARQL endpoint: http://localhost:3032/test/sparql

---

## Neo4j with n10s Plugin Setup

**1. Prepare the environment file:**
```bash
cd docker/neo4j
cp .env.example .env
# Edit with your values
```

**Example `docker/neo4j/.env.example`:**
```bash
IMAGE_VERSION=neo4j:5.20
SPEC=test
CONTAINER_NAME="${SPEC}.sem.neo4j"
NEO4J_PORT=7476
NEO4J_BOLT_PORT=7689
STORE_FOLDER="$HOME/tmp/${CONTAINER_NAME}"
NEO4J_PLUGINS='["apoc", "graph-data-science", "n10s"]'
NEO4J_AUTH="neo4j/test!passfortesting"
```

**2. Start/Stop Neo4j:**
```bash
# Start
cd docker/neo4j
docker compose --env-file .env up neo4j -d

# Stop
docker compose stop neo4j
```

**3. Access Neo4j:**
- Browser: http://localhost:7476
- Username: `neo4j`
- Password: `test!passfortesting`
- Bolt: bolt://localhost:7689

---

## Filesystem Storage (Fallback)

If neither Fuseki nor Neo4j is configured, OntoCast will store ontologies and facts as Turtle files in the working directory.

**No setup required.**

---

## Triple Store Comparison

| Feature | Fuseki | Neo4j + n10s | Filesystem |
|---------|--------|--------------|------------|
| **RDF Native** | ✅ Yes | ⚠️ Via plugin | ✅ Yes |
| **SPARQL** | ✅ Full 1.1 | ❌ Limited | ❌ No |
| **Setup Complexity** | ✅ Simple | ⚠️ Moderate | ✅ Very Simple |
| **Visualization** | ⚠️ Basic | ✅ Excellent | ❌ None |
| **Production Ready** | ✅ Yes | ✅ Yes | ❌ No |

---

## Best Practices

- Use **Filesystem** for quick setup and testing.
- Use **Fuseki** for RDF-focused or production deployments.
- Use **Neo4j** if you need advanced graph analytics or visualization.
- Monitor triple store performance and logs.
- Backup your data regularly.

---

## Troubleshooting

### Fuseki
```bash
# Check if Fuseki is running
curl http://localhost:3032/$/ping

# Restart Fuseki
docker compose restart fuseki
```

### Neo4j
```bash
# Check if Neo4j is running
curl http://localhost:7476

# Check n10s plugin
cypher-shell -u neo4j -p test!passfortesting "CALL n10s.graphconfig.show()"
```

### Common Problems
- **Connection Refused**: Triple store not running
- **Authentication Failed**: Incorrect credentials
- **Dataset Not Found**: Dataset not created in Fuseki
- **Plugin Not Loaded**: n10s plugin not installed in Neo4j
