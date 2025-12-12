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

# uv creates a venv and installs deps
RUN --mount=type=ssh uv sync --no-group dev --no-group docs -v

COPY ontocast ./ontocast
COPY README.md ./

# ─── runtime stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim-bullseye AS runtime

LABEL org.opencontainers.image.title="ontocast-mcp-server" \
      org.opencontainers.image.version="0.1.1" \
      org.opencontainers.image.description="Ontology‑assisted semantic triple extractor (MCP‑compatible)"

# Install curl for healthcheck
RUN apt update -y \
 && apt install -y curl \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r ontocast && useradd -r -g ontocast ontocast
USER ontocast

WORKDIR /app
COPY --from=builder /app /app

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

ENTRYPOINT ["uv", "run", "ontocast"]