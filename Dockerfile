FROM python:3.12-slim

WORKDIR /app

# Java runtime for the Pellet reasoner (ontology_check).
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

# Embedding models are downloaded here on first start — mount a volume on it to keep them
# across container restarts (see docker-compose.yml).
ENV FASTEMBED_CACHE_PATH=/models

ENV MCP_TRANSPORT=sse
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

EXPOSE 8000

CMD ["olaf"]
