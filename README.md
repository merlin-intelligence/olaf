# OLAF

**Ontology Learning Agentic Framework** — MCP server for building OWL/RDFS ontologies from text, incrementally and collaboratively with an LLM.

---

## Overview

OLAF exposes a set of MCP tools that let an LLM agent construct a formal OWL/RDFS ontology from raw text chunks. The agent reads chunks, extracts concepts and relations, checks for duplicates, and builds up the ontology piece by piece — resuming at any point without losing work.

```
Text chunks (Qdrant)  ──►  LLM agent  ──►  OWL ontology (Oxigraph)
                              │  ▲
                     seed TTLs│  │ concept_search / concept_semantic_search
                              ▼  │
                          22 MCP tools
```

**Storage split:**
- **Oxigraph** — the ontologies (OWL/RDFS triples, named graphs, SPARQL), runs as a separate HTTP service
- **Qdrant** — your existing chunk collection (read + status tracking) + one `olaf_concepts_{id}` collection per ontology for semantic search

---

## Requirements

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- A running [Qdrant](https://qdrant.tech/) instance with your chunk collection
- An MCP-compatible client (Claude Desktop, Claude Code, or any MCP host)

---

## Deployment

### Docker Compose (recommended)

```bash
git clone <repo>
cd olaf
cp config.example.toml config.toml
$EDITOR config.toml
docker compose up -d
```

This starts two services:
- **oxigraph** — RDF triplestore on port 7878, data persisted in a Docker volume
- **olaf** — MCP server on port 8000 (SSE transport)

### Connecting from Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "olaf": {
      "url": "http://your-server:8000/sse"
    }
  }
}
```

For a local Docker deployment, use `http://localhost:8000/sse`.

---

## Configuration

`config.toml` (or `olaf.toml`) is loaded automatically from the working directory.

```toml
[qdrant]
url                 = "https://your-qdrant-instance.example.com"
collection          = "chunks"           # your existing chunk collection
concepts_collection = "olaf_concepts"    # prefix — actual collections are olaf_concepts_{ontology_id}

[qdrant.field_mapping]
# names of the payload fields in your existing collection
text        = "text"
doc_id      = "doc_id"
chunk_index = "chunk_index"

[oxigraph]
url = "http://oxigraph:7878"   # Docker service name; use http://localhost:7878 if running locally

[ontology]
base_uri     = "http://olaf.local/ontology#"   # namespace for generated URIs
name         = "My Ontology"
ontology_id  = "main"                          # active ontology (urn:olaf:main)

[embedding]
model   = "intfloat/multilingual-e5-small"   # downloaded automatically via fastembed
enabled = true
```

**The `olaf_status` and `olaf_processed_at` payload fields are added to your existing Qdrant points via `set_payload` — your other fields are never modified.**

---

## Tool reference

### Ontology management

| Tool | Description |
|------|-------------|
| `ontology_list` | List all ontologies in the store with id, name, base_uri, class count, and active flag. |
| `ontology_create` | Create a new empty ontology. Parameters: `ontology_id`, `name`, optional `base_uri`. |
| `ontology_switch` | Switch the active ontology for this session. All subsequent tools operate on the selected ontology. |
| `ontology_summary` | Counts of classes, properties, individuals, restrictions, root classes, and chunk processing progress. Includes `active_ontology`. |
| `ontology_export` | Export the active ontology as Turtle. Pass `include_seeds=true` to append all seed graphs. |

### Chunks

| Tool | Description |
|------|-------------|
| `chunk_list` | List chunks from Qdrant. Filter by `doc_id` and/or `status` (`pending`/`processed`/`all`). Returns `id`, `doc_id`, `chunk_index`, `text_preview` (200 chars), `status`. |
| `chunk_read` | Read the full text of a chunk by its Qdrant point ID. |
| `chunk_mark_processed` | Mark a chunk as processed after extracting ontology elements from it. |

### Concepts (`owl:Class`)

| Tool | Description |
|------|-------------|
| `concept_search` | Substring search on `rdfs:label` and `rdfs:altLabel` via SPARQL. Returns `source_chunk_id` when available. Call before `concept_create`. |
| `concept_semantic_search` | Vector similarity search in the active ontology's Qdrant collection. Finds near-duplicates even with different wording. Returns `source_chunk_id` when available. Call alongside `concept_search`. |
| `concept_create` | Create an `owl:Class`. Generates a CamelCase URI from the label. Optional `source_chunk_id` to record which chunk the concept was extracted from. Returns `{uri, created}` — `created=false` if the URI already exists. |
| `concept_get` | Get all triples for a concept: type, label, definition, aliases, `subClassOf`, restrictions, `source_chunk_ids`. |
| `concept_update` | Update label, definition, or aliases (`aliases_add` / `aliases_remove`). |
| `concept_merge` | Merge two concepts: all triples from `merge_uri` move to `keep_uri`, all references re-pointed, `merge_uri` deleted. |

### Properties (`owl:ObjectProperty` / `owl:DatatypeProperty`)

| Tool | Description |
|------|-------------|
| `property_create` | Create a property. `type` = `"object"` (links two classes) or `"datatype"` (class → literal). Generates a lowerCamelCase URI. |
| `property_get` | Get all triples for a property. |
| `property_search` | Substring search on property labels. |

### Relations & Restrictions

| Tool | Description |
|------|-------------|
| `relation_add` | Insert any triple `(subject, property, object)`. Use full URIs. Set `is_literal=true` for literal objects; optionally pass `datatype` (XSD URI). |
| `relation_search` | Find triples by pattern. All three parameters are optional. |
| `restriction_add` | Add an `owl:Restriction` blank node to a class. Supports `some`, `all`, `has_value`, `exactly`, `min`, `max`. |

`restriction_add` produces:
```turtle
:Car rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty :hasEngine ;
    owl:someValuesFrom :Engine
] .
```

### Seeds

Seeds are **global** — shared across all ontologies in the store.

| Tool | Description |
|------|-------------|
| `seed_list` | List all seed ontologies loaded in the store (id, graph URI, triple count). |
| `seed_load` | Load a seed ontology (Turtle format) via `url` (HTTP URI) or `content` (raw Turtle string). Stored in `urn:olaf:seed:{id}`, accessible to all ontologies. |

---

## CLI commands

Administrative operations not exposed to the MCP agent:

```bash
# Delete an ontology (does not affect seeds)
olaf drop <ontology_id>

# Delete a global seed
olaf drop-seed <seed_id>
```

---

## Typical agent workflow

```
1. ontology_list()                  → discover existing ontologies
2. ontology_switch("my-project")    → or ontology_create("my-project", "My Project")
3. ontology_summary()               → understand current state
4. seed_list()                      → check available seeds
5. seed_load("./domain.ttl")        → optionally load a reference ontology

For each chunk:
6. chunk_list(doc_id=..., status="pending")
7. chunk_read(chunk_id)
8. concept_search(query) +          → run in parallel, deduplicate by URI
   concept_semantic_search(query)
9. concept_create / property_create / relation_add / restriction_add
10. chunk_mark_processed(chunk_id)

11. ontology_export()
```

---

## Named graphs

Oxigraph uses named graphs to separate ontologies and seeds:

| Graph | Contents |
|-------|----------|
| `urn:olaf:{id}` | An ontology (e.g. `urn:olaf:main`, `urn:olaf:legal`) |
| `urn:olaf:seed:{id}` | A seed ontology — global, shared across all ontologies |

The active ontology is set via `ontology_id` in config (default: `main`) and can be changed at runtime with `ontology_switch`.

---

## Architecture

```
                  ┌─────────────────────┐
Claude Desktop ──►│  olaf (port 8000)   │
Cloud agent    ──►│  MCP / SSE          │
                  │                     │
                  │  config.py          │
                  │  ontology.py  ──────┼──► Oxigraph HTTP (port 7878)
                  │  chunks.py    ──────┼──► Qdrant
                  │  embeddings.py──────┼──► Qdrant (olaf_concepts)
                  │  server.py          │
                  └─────────────────────┘
```

```
src/olaf/
├── config.py       Config dataclasses, TOML loading (stdlib tomllib)
├── ontology.py     OntologyStore — HTTP SPARQL backend, all reads/writes
├── chunks.py       ChunkStore — Qdrant wrapper for existing chunk collection
├── embeddings.py   EmbeddingService — fastembed + olaf_concepts Qdrant collection
└── server.py       MCP server, tool registration, call dispatch, CLI commands
```

**URI generation** — concept URIs are derived from labels:
- Classes: `CamelCase` → `http://olaf.local/ontology#MotorVehicle`
- Properties: `lowerCamelCase` → `http://olaf.local/ontology#hasEngine`

Change `base_uri` in config to use your own namespace.

---

## Dependencies

| Package | Role |
|---------|------|
| `mcp` | MCP server SDK (Anthropic) |
| `httpx` | HTTP client — calls Oxigraph SPARQL endpoint and loads remote seeds |
| `qdrant-client` | Qdrant vector database client |
| `fastembed` | Lightweight ONNX-based embeddings (no PyTorch) |
| `starlette` | ASGI framework for SSE transport |
| `uvicorn` | ASGI server |
