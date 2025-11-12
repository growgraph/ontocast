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

## Configuration

### Environment Variables

Configure your triple store connection using environment variables in your `.env` file:

```bash
# Fuseki Configuration (Preferred)
FUSEKI_URI=http://localhost:3032/test
FUSEKI_AUTH=admin:password
FUSEKI_DATASET=dataset_name

# Neo4j Configuration (Alternative)
NEO4J_URI=bolt://localhost:7689
NEO4J_AUTH=neo4j:password

```

### Configuration Hierarchy

The new configuration system provides better organization:

```python
from ontocast.config import Config

config = Config()

# Access triple store configuration
tool_config = config.get_tool_config()

# Check which triple store is configured
if tool_config.fuseki.uri and tool_config.fuseki.auth:
    print("Using Fuseki triple store")
elif tool_config.neo4j.uri and tool_config.neo4j.auth:
    print("Using Neo4j triple store")
else:
    print("Using filesystem storage")
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
ENVIRONMENT_ACTUAL=test
CONTAINER_NAME="${ENVIRONMENT_ACTUAL}.fuseki"
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
docker compose --env-file .env fuseki up -d

# Stop
# (use the container name from your .env, e.g. test.fuseki)
docker compose stop test.fuseki
```

**3. Access Fuseki:**

- Web interface: http://localhost:3032
- Default dataset: `/test`
- SPARQL endpoint: http://localhost:3032/test/sparql

**4. Configure OntoCast for Fuseki:**

```bash
# In your .env file
FUSEKI_URI=http://localhost:3032/test
FUSEKI_AUTH=admin:abc123-qwe
FUSEKI_DATASET=dataset_name
```

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
docker compose --env-file .env neo4j up -d

# Stop
docker compose stop neo4j
```

**3. Access Neo4j:**

- Browser: http://localhost:7476
- Username: `neo4j`
- Password: `test!passfortesting`
- Bolt: bolt://localhost:7689

**4. Configure OntoCast for Neo4j:**

```bash
# In your .env file
NEO4J_URI=bolt://localhost:7689
NEO4J_AUTH=neo4j:test!passfortesting
```

---

## Filesystem Storage (Fallback)

If neither Fuseki nor Neo4j is configured, OntoCast will store ontologies and facts as Turtle files in the working directory.

**No setup required - works out of the box.**

---

## Triple Store Comparison

| Feature | Fuseki | Neo4j + n10s | Filesystem |
|---------|--------|--------------|------------|
| **RDF Native** | ✅ Yes | ⚠️ Via plugin | ✅ Yes |
| **SPARQL** | ✅ Full 1.1 | ❌ Limited | ❌ No |
| **Setup Complexity** | ✅ Simple | ⚠️ Moderate | ✅ Very Simple |
| **Visualization** | ⚠️ Basic | ✅ Excellent | ❌ None |
| **Production Ready** | ✅ Yes | ✅ Yes | ❌ No |
| **Configuration** | ✅ Environment vars | ✅ Environment vars | ✅ Automatic |

---

## Best Practices

- Use **Filesystem** for quick setup and testing
- Use **Fuseki** for RDF-focused or production deployments
- Use **Neo4j** if you need advanced graph analytics or visualization
- Monitor triple store performance and logs
- Backup your data regularly
- Use the `/flush` API endpoint to clean triple stores when needed (see below)

---

## Troubleshooting

### Fuseki
```bash
# Check if Fuseki is running
curl http://localhost:3032/$/ping

# Restart Fuseki
docker compose restart fuseki

# Check dataset exists
curl http://localhost:3032/$/datasets
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
- **Authentication Failed**: Incorrect credentials in environment variables
- **Dataset Not Found**: Dataset not created in Fuseki
- **Plugin Not Loaded**: n10s plugin not installed in Neo4j
- **Configuration Not Loaded**: Check `.env` file and environment variable names

---

## Flushing Triple Store Data

You can clean/flush data from the triple store using the `/flush` API endpoint. This endpoint allows you to explicitly delete data when needed.

### Using the Flush Endpoint

```bash
# Clean all datasets (Fuseki) or entire database (Neo4j)
curl -X POST http://localhost:8999/flush

# Clean specific Fuseki dataset
curl -X POST "http://localhost:8999/flush?dataset=my_dataset"
```

**For Fuseki:**
- If no `dataset` parameter is provided, both the main dataset and ontologies dataset are cleaned
- If a `dataset` parameter is provided, only that specific dataset is cleaned

**For Neo4j:**
- The `dataset` parameter is ignored (Neo4j doesn't support datasets)
- All nodes and relationships are deleted

**Warning:** This operation is irreversible and will delete all data. Use with caution in production environments!

---

## Migration from Previous Versions

If you're upgrading from a previous version of OntoCast:

1. **Update Environment Variables**: The configuration system has been refactored
2. **Check Triple Store Settings**: Ensure your triple store configuration is properly set
3. **Test Configuration**: Use the new configuration system to verify your setup

```python
# Test your configuration
from ontocast.config import Config

config = Config()
print("Configuration loaded successfully!")
print(f"LLM Provider: {config.tool_config.llm_config.provider}")
print(f"Working Directory: {config.tool_config.path_config.working_directory}")
```
