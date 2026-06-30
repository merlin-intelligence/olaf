# olaf_building_agent

Autonomous agent that reads text chunks from Qdrant and builds an OWL/RDFS ontology using OLAF MCP tools and LiteLLM.

## Architecture

```
agent.py  ──SSE──▶  OLAF MCP server  ──SPARQL──▶  Oxigraph (RDF)
                                      ──gRPC───▶  Qdrant (chunks)
LiteLLM   ──API──▶  Claude / OpenAI / Ollama
```

The agent loop runs up to `max_iterations` times (default 50). Each iteration sends the full conversation history to the LLM, which decides which OLAF tools to call next. The loop ends when the LLM returns a response with no tool calls.

## Prerequisites

- OLAF MCP server running in SSE mode (see root `docker-compose.yml`)
- Qdrant with a populated chunk collection
- An LLM API key (Anthropic, OpenAI, etc.)

## Setup

```bash
cd demos/olaf_building_agent
pip install -r requirements.txt
cp config.example.toml config.toml   # then edit config.toml
export ANTHROPIC_API_KEY=sk-ant-...
```

## Configuration (`config.toml`)

```toml
[olaf]
url = "http://localhost:8000/sse"   # OLAF SSE endpoint

[litellm]
model       = "anthropic/claude-haiku-4-5-20251001"  # or claude-sonnet-4-6, gpt-4o, ollama/llama3…
max_tokens  = 4096
temperature = 0

[agent]
max_iterations = 50          # hard stop for the LLM loop
export_path    = "out.ttl"   # Turtle file written when ontology_export is called
log_level      = "INFO"      # DEBUG | INFO | WARNING
```

**Model strings for LiteLLM** — always prefix with the provider:

| Provider  | Model string                              | API key env var    |
|-----------|-------------------------------------------|--------------------|
| Anthropic | `anthropic/claude-haiku-4-5-20251001`     | `ANTHROPIC_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-6`             | `ANTHROPIC_API_KEY` |
| OpenAI    | `openai/gpt-4o`                           | `OPENAI_API_KEY`   |
| Ollama    | `ollama/llama3`                           | *(none)*           |


## Running

```bash
python agent.py                       # uses config.toml in current directory
python agent.py --config my.toml
python agent.py --log-level DEBUG
```

The agent logs each tool call to stderr:
```
11:32:01 INFO     Loaded 20 OLAF tools.
11:32:01 INFO     --- Iteration 1 ---
11:32:03 INFO     Tool call: chunk_list(status='pending', limit='20')
11:32:05 INFO     Tool call: concept_create(label='Climate Risk', ...)
11:32:10 INFO     Agent finished.
11:32:10 INFO     Ontology exported to out.ttl
```

The agent is done when you see `Agent finished.` or `Reached max_iterations`.

## Export

When the agent calls `ontology_export`, the Turtle content is automatically saved to the path defined in `config.toml → [agent] export_path`. The file is overwritten on each export call.

To export manually from Oxigraph (replace `ontology_id` with your ontology_id):

```bash
curl -s -X POST http://oxigraph:7878/query \
  -H 'Content-Type: application/sparql-query' \
  -H 'Accept: text/turtle' \
  -d 'CONSTRUCT { ?s ?p ?o } WHERE { GRAPH <urn:olaf:ontology_id> { ?s ?p ?o } }' \
  > ontology.ttl
```

To delete an ontology from Oxigraph:

```bash
curl -s -X POST http://oxigraph:7878/update \
  -H 'Content-Type: application/sparql-update' \
  -d 'DROP GRAPH <urn:olaf:ontology_id>'
```

## Infrastructure (Docker)

The root `docker-compose.yml` runs OLAF and Oxigraph. If Qdrant runs as a standalone container on the host network, add `extra_hosts` to the `olaf` service and point Qdrant's URL to `host.docker.internal`:

```toml
# config.toml
[qdrant]
url = "http://host.docker.internal:6333"

[oxigraph]
url = "http://oxigraph:7878"   # Docker service name
```

```yaml
# docker-compose.yml
services:
  olaf:
    extra_hosts:
      - "host.docker.internal:host-gateway"   # required on Linux/WSL2
```

## Known issues

**LiteLLM async bug** — Some versions of LiteLLM have a bug where `acompletion` returns a coroutine object instead of a response for the Anthropic provider. The agent works around this by using `asyncio.to_thread(litellm.completion, ...)` with a 120-second timeout. If you see `RAW RESPONSE: <coroutine object ...>` in the logs, upgrading LiteLLM may fix it; the workaround is already in place.

**URI generation** — Concept labels must be space-separated title-case words (`"Climate Risk"`, not `"ClimateRisk"` or `"climate_risk"`). The server slugifies the label into a PascalCase URI automatically: `"Climate Risk"` → `ClimateRisk`. Passing a single compound word produces incorrect results (`"ClimateRisk"` → `Climaterisk`).
