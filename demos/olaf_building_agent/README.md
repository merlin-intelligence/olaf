# olaf_building_agent

Autonomous agent that reads text chunks from Qdrant and builds an OWL/RDFS ontology using OLAF MCP tools and LiteLLM.

## Architecture

```
agent.py  ──SSE──▶  OLAF MCP server  ──SPARQL──▶  Oxigraph (RDF)
                                      ──gRPC───▶  Qdrant (chunks)
LiteLLM   ──API──▶  Claude / OpenAI / Ollama
```

The code drives the pipeline; the LLM does the ontology work through OLAF tool calls. Each step below is a **fresh LLM conversation**, so the prompt size depends on `batch_size`, not on how far the build has gone — which keeps token usage per call bounded (and within provider rate limits).

1. **Extraction** — pending chunks are processed by batches of `batch_size`. Each batch gets the chunk texts plus a digest of the current ontology (existing classes, properties and seed classes with their URIs), so the LLM reuses them instead of re-creating them. Once the LLM is done, the pipeline marks the chunks processed: an interrupted run resumes where it stopped.
2. **Consolidation** (`consolidate = true`)
   - *Deduplication* — classes whose embeddings are closer than `dedup_threshold` are reviewed by the LLM, `dedup_pairs_per_task` pairs at a time: merge (`concept_merge`) or keep.
   - *Consistency* — `ontology_check` (reasoner + integrity checks) and `ontology_orphans` run; if they report anything, the LLM fixes it and re-checks.
   No LLM call is made when there is nothing to review or fix.
3. **Export** — the ontology is written to `export_path`.

Within a step, the LLM loops on tool calls until it replies with a plain-text summary (at most `max_iterations` rounds); older tool results are pruned from its context as it goes.

## Prerequisites

- OLAF MCP server running in SSE mode (see root `docker-compose.yml`)
- Qdrant with a populated chunk collection
- An LLM API key (Anthropic, OpenAI, etc.)
- For the deduplication step: embeddings enabled on the server (`[embedding] enabled = true`); otherwise it is skipped with a warning

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
url           = "http://localhost:8000/sse"   # OLAF SSE endpoint
ontology_id   = "demo"                        # ontology to build (created if missing)
ontology_name = "Demo Ontology"
base_uri      = "http://olaf.local/ontology#" # namespace, used when the ontology is created
collection    = "chunks"                      # Qdrant collection to process (default: server's)

[litellm]
model       = "anthropic/claude-haiku-4-5-20251001"  # or claude-sonnet-4-6, gpt-4o, ollama/llama3…
max_tokens  = 4096
temperature = 0

[agent]
batch_size     = 5                       # chunks per extraction batch
max_batches    = 0                       # 0 = all pending chunks
consolidate    = true                    # deduplication + consistency fixes after extraction
max_iterations = 30                      # max LLM rounds per task
export_path    = "ontology_output.ttl"   # Turtle file written at the end of the run
log_level      = "INFO"                  # DEBUG | INFO | WARNING
```

See [`config.example.toml`](config.example.toml) for all settings (digest size, dedup threshold, context pruning, rate-limit retries).

**Model strings for LiteLLM** — always prefix with the provider:

| Provider  | Model string                              | API key env var    |
|-----------|-------------------------------------------|--------------------|
| Anthropic | `anthropic/claude-haiku-4-5-20251001`     | `ANTHROPIC_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-6`             | `ANTHROPIC_API_KEY` |
| OpenAI    | `openai/gpt-4o`                           | `OPENAI_API_KEY`   |
| Ollama    | `ollama/llama3`                           | *(none)*           |


### OpenAI-compatible providers (Scaleway…)

Prefix the model with `openai/`, set `api_base`, and set `api_key_env` to the **name** of the environment variable holding the key (the key itself stays out of the config):

```toml
[litellm]
model       = "openai/llama-3.3-70b-instruct"   # a model of the catalogue that supports tool calling
api_base    = "https://api.scaleway.ai/v1"
api_key_env = "SCW_SECRET_KEY"
```

The agent does not read `.env` files: export the variable in the shell that runs it, e.g. `set -a; source ../../.env; set +a`.

### Rate limits

When the provider answers HTTP 429 (e.g. a tokens-per-minute quota), the call is retried after 15 s, 30 s, then 60 s, up to `[litellm] rate_limit_retries` times (default 6). If it keeps happening, lower `batch_size` (each LLM call carries one batch) and `max_tokens` (some providers count the requested maximum against the quota).

## Running

```bash
python agent.py                       # uses config.toml in current directory
python agent.py --config my.toml
python agent.py --log-level DEBUG
```

The agent logs each step and tool call to stderr:
```
11:32:01 INFO     Active ontology: demo (created)
11:32:01 INFO     [batch 1] iteration 1
11:32:05 INFO     Tool call: concept_create(label='Climate Risk', ...)
11:32:40 INFO     [batch 1] done: Added 6 classes, 3 properties…
11:32:40 INFO     Batch 1 done — chunks 5/120 processed, 6 classes, 3 object properties, 0 individuals.
…
11:58:02 INFO     Deduplication: 4 candidate pairs above 0.90.
11:59:30 INFO     Consistency check: 3 issues, 1 orphans — asking the LLM to fix them.
12:01:12 INFO     Consistency check after fixes: 0 issues left.
12:01:13 INFO     Ontology exported to ontology_output.ttl
12:01:13 INFO     Finished — 84 classes, 41 object properties, 0 datatype properties, 23 individuals; chunks 120/120 processed.
```

To try a configuration on a small sample, set `max_batches` (e.g. `2`); the remaining chunks stay pending for a later run. To rebuild from scratch, reset the chunk statuses with `olaf reset-chunks [--collection NAME]` and use a new `ontology_id`.

## Export

At the end of the run, the ontology is exported as Turtle to `config.toml → [agent] export_path` (overwritten on each run).

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
