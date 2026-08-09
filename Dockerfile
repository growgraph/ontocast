# ─── builder stage ───────────────────────────────────────────────────────────
FROM python:3.12-slim-bullseye AS builder

RUN apt update -y \
 && apt install -y curl git \
 && curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="${PATH}:/root/.local/bin"
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN touch README.md

# Everything except `plot` (pygraphviz builds against system graphviz headers),
# `dev`, and `docs`. `server` + the LLM providers are what make the image a
# runnable API server; a base install has no CLI and no provider.
RUN uv sync --all-extras --no-extra plot --no-extra dev --no-extra docs

COPY ontocast ./ontocast
COPY README.md ./

# ─── runtime stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim-bullseye AS runtime

LABEL org.opencontainers.image.title="ontocast" \
      org.opencontainers.image.description="Ontology-assisted knowledge-graph extraction (API server)" \
      org.opencontainers.image.source="https://github.com/growgraph/ontocast"

# Install curl for healthcheck
RUN apt update -y \
 && apt install -y curl \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r ontocast && useradd -r -g ontocast ontocast
USER ontocast

WORKDIR /app
COPY --from=builder /app /app
# The console scripts live in the synced venv; uv itself is not shipped.
ENV PATH="/app/.venv/bin:${PATH}"

# ─── Volume Mounting Notes ──────────────────────────────────────────────────────
# Paths are configured via .env file (ONTOCAST_WORKING_DIRECTORY,
# ONTOCAST_ONTOLOGY_DIRECTORY, ONTOCAST_CACHE_DIR) and should be mounted
# as volumes in docker-compose to persist data and allow host access.
#
# Example docker-compose volumes:
#   volumes:
#     - ./data/working:/path/to/working
#     - ./data/ontologies:/path/to/ontologies
#     - ./data/cache:/path/to/cache
#
# Ensure the paths in .env match the container-side mount paths.
# Do NOT hardcode paths in Dockerfile - they must come from .env configuration.

# Expose & healthcheck
EXPOSE 8999
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8999/health || exit 1

# `serve` starts the API; override CMD (e.g. `process ...`) for batch runs.
ENTRYPOINT ["ontocast"]
CMD ["serve"]
