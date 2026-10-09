# OLAF

**Ontology Learning Agentic Framework** — MCP server for building OWL/RDFS ontologies from text, incrementally and collaboratively with an LLM, and for querying them.

---

## Overview

OLAF exposes a set of MCP tools that let an LLM agent construct a formal OWL/RDFS ontology from raw text chunks. The agent reads chunks, extracts concepts and relations, checks for duplicates, and builds up the ontology piece by piece — resuming at any point without losing work.

Once built, the ontology can be queried the same way: read-only tools and `sparql_query` let an agent answer natural-language questions and trace each fact back to its source chunks. See [`demos/`](demos/) for a building agent, a reasoning agent that repairs the built ontology, and a searching agent.

```
Text chunks (Qdrant)  ──►  LLM agent  ──►  OWL ontology (Oxigraph)
                              │  ▲
                     seed TTLs│  │ concept_search / concept_semantic_search
                              ▼  │
                          36 MCP tools
```

**Storage split:**
- **Oxigraph** — the ontologies (OWL/RDFS triples, named graphs, SPARQL), runs as a separate HTTP service
- **Qdrant** — your existing chunk collections (read + status tracking; each agent picks its collection with `chunk_collection_switch`) + one `olaf_concepts_{id}` collection per ontology for semantic search

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
- **olaf** — MCP server on port 8000 (SSE transport). The image includes the Java runtime used by the reasoner (`ontology_check`).

On first start, the olaf service downloads the embedding model (a few hundred MB, ~1.1 GB for `multilingual-e5-base`) before accepting connections — wait for `Application startup complete` in `docker compose logs olaf`. The model is cached in the `olaf_models` volume, so later starts are fast.

### Without Docker Compose (Oxigraph / Qdrant already running)

If an Oxigraph or Qdrant instance is already running on this machine, `docker compose up` fails on the busy ports. Run only the OLAF image instead, on the host network so that `localhost` in `config.toml` reaches them:

```bash
docker build -t olaf .
docker run --rm --network host \
  -v "$PWD/config.toml:/app/config.toml:ro" \
  -v olaf_models:/models \
  olaf
```

CLI commands run the same way, e.g. `docker run --rm --network host -v "$PWD/config.toml:/app/config.toml:ro" olaf olaf reset-chunks`.

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
collection          = "chunks"           # default chunk collection — each agent can pick its own
concepts_collection = "olaf_concepts"    # prefix — actual collections are olaf_concepts_{ontology_id}

[qdrant.field_mapping]
# default names of the payload fields in your chunk collections
text        = "text"
doc_id      = "doc_id"
chunk_index = "chunk_index"

[oxigraph]
url = "http://oxigraph:7878"   # Docker service name; use http://localhost:7878 if running locally

[ontology]
base_uri     = "http://olaf.local/ontology#"   # namespace for generated URIs
name         = "My Ontology"
ontology_id  = "main"                          # active ontology (urn:olaf:main)

[reasoner]
enabled         = true     # ontology_check: Pellet OWL reasoner (needs Java 11+, included in the image)
java            = "java"   # path to the java executable
memory_mb       = 2048     # JVM max heap
timeout_seconds = 120      # per reasoner call

[embedding]
model   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Any model fastembed supports (TextEmbedding.list_supported_models()), plus
# "intfloat/multilingual-e5-small" (384 d) and "intfloat/multilingual-e5-base" (768 d),
# loaded from their ONNX export on Hugging Face (~0.5 GB / ~1.1 GB, downloaded on first start).
# Concepts are indexed when they are created, and an existing index is not recomputed when
# the model changes (vector sizes differ): after changing it, build into a new ontology_id.
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
| `ontology_summary` | Counts of classes, properties, individuals, restrictions, root classes, and chunk processing progress. Includes `active_ontology`, the number of materialized inferences (`inferred_triples`) and the outcome of the last `ontology_check` since the server started (`last_check`: time, consistent, issues — later changes are not reflected until the next check). |
| `ontology_export` | Export the active ontology as Turtle. Pass `include_seeds=true` to append all seed graphs. |
| `ontology_orphans` | List classes/individuals with no relation to the rest of the graph beyond their own type/label/definition triples. |
| `ontology_check` | Check the ontology for logical errors: Pellet OWL 2 DL reasoning (consistency + unsatisfiable classes, each with its explanation) plus closed-world SPARQL checks (subclass cycles, untyped individuals, domain/range violations, object/datatype property misuse, unknown vocabulary terms such as `rdfs:subClassof`). See [Consistency checking](#consistency-checking). |
| `ontology_infer` | What the ontology entails but does not assert (Pellet): subclass relations, class membership of individuals, equivalent classes, sub-properties, property assertions, same-individual links. `action=preview` (default) lists them with why each holds; `materialize` writes them into the ontology, marked as inferred; `clear` removes them. See [Inference](#inference). |

### Chunks

| Tool | Description |
|------|-------------|
| `chunk_collection_switch` | Switch the Qdrant collection the chunk tools read from, for this session only (default: `[qdrant].collection`). Optional `field_mapping` overrides the payload field names. Returns chunk counts. |
| `chunk_list` | List chunks from Qdrant. Filter by `doc_id` and/or `status` (`pending`/`processed`/`all`). Returns `id`, `doc_id`, `chunk_index`, `text_preview` (200 chars), `status`. |
| `chunk_read` | Read the full text of a chunk by its Qdrant point ID. |
| `chunk_read_batch` | Read the full text of several chunks in one call. Returns `{id, doc_id, chunk_index, text, status}` for each. |
| `chunk_mark_processed` | Mark a chunk as processed after extracting ontology elements from it. |

### Concepts (`owl:Class`) and individuals

| Tool | Description |
|------|-------------|
| `concept_list` | List the classes of the active ontology with `uri`, `label`, `definition` and `parent_uri`. `root_only=true` returns only top-level classes. |
| `concept_search` | Substring search on `rdfs:label` and `rdfs:altLabel` via SPARQL. Returns `source_chunk_id` when available. Call before `concept_create`. |
| `concept_semantic_search` | Vector similarity search in the active ontology's Qdrant collection. Finds near-duplicates even with different wording. Returns `source_chunk_id` when available. Call alongside `concept_search`. |
| `concept_create` | Create an `owl:Class`. Generates a CamelCase URI from the label. Optional `source_chunk_id` to record which chunk the concept was extracted from. Returns `{uri, created}` — `created=false` if the URI already exists. |
| `concept_get` | Get all triples for a concept: type, label, definition, aliases, `subClassOf`, `source_chunk_ids`, and its OWL restrictions as `restrictions` (in `restriction_add` / `restriction_delete` terms). |
| `concept_update` | Update label, definition, or aliases (`aliases_add` / `aliases_remove`). |
| `individual_create` | Create an `owl:NamedIndividual` (a specific named entity such as "GDPR") as an instance of `class_uri`. Returns `{uri, created}`. If it already exists, `source_chunk_id` is added to its sources. |
| `concept_merge` | Merge two concepts: all triples from `merge_uri` move to `keep_uri`, all references re-pointed, `merge_uri` deleted. The kept concept keeps its label and definition: the merged one's labels become `rdfs:altLabel`, and its definition is used only if the kept one has none. A relation between the two (e.g. `merge_uri` subClassOf `keep_uri`) is dropped rather than turned into a self-loop. The provenance of the re-pointed relations is kept (`relation_sources` on the new triples). |

### Properties (`owl:ObjectProperty` / `owl:DatatypeProperty`)

| Tool | Description |
|------|-------------|
| `property_create` | Create a property. `type` = `"object"` (links two classes) or `"datatype"` (class → literal). Generates a lowerCamelCase URI. |
| `property_update` | Update `domain_uri`, `range_uri`, or `parent_uri` after creation. Pass `""` to clear a field, omit to leave unchanged. |
| `property_get` | Get all triples for a property. |
| `property_search` | Substring search on property labels. |

### Relations & Restrictions

| Tool | Description |
|------|-------------|
| `relation_add` | Insert any triple `(subject, property, object)`. Use full URIs. Set `is_literal=true` for literal objects; optionally pass `datatype` (XSD URI). Rejects subject/property/object URIs that look like they belong to this ontology but don't exist yet, and RDF/RDFS/OWL/SKOS terms that do not exist (e.g. `rdfs:subClassof`, with the closest real term as a hint). |
| `relation_delete` | Remove a triple `(subject, property, object)`. Same parameters as `relation_add`. |
| `relation_search` | Find triples by pattern. All three parameters are optional. |
| `relation_sources` | Get the chunk IDs a `(subject, property, object)` triple was extracted from. |
| `restriction_add` | Add an `owl:Restriction` blank node to a class. Supports `some`, `all`, `has_value`, `exactly`, `min`, `max`. |
| `restriction_delete` | Delete the restrictions of a class on a property — all, or those of a given `restriction_type` and/or `value`. Reaches the blank nodes `relation_delete` cannot. |
| `entity_delete` | Delete a class, individual or property of the ontology entirely: every triple it appears in, their provenance, its restrictions and those pointing to it, its semantic-index entry. Only entities of the ontology's namespace. Irreversible. |
| `disjoint_add` | Declare 2+ classes pairwise disjoint (`owl:disjointWith`). Rejects non-classes and classes in a subclass relation. Undo a pair with `relation_delete`. |

`restriction_add` produces:
```turtle
:Car rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty :hasEngine ;
    owl:someValuesFrom :Engine
] .
```

### Consistency checking

`ontology_check` combines two kinds of checks, because OWL semantics alone catch few of the mistakes an extraction agent makes:

- **Reasoner (open world).** The [Pellet](https://github.com/stardog-union/pellet) reasoner bundled with `owlready2` runs on the ontology's logical content (provenance triples and `owl:imports` are stripped). It reports whether the ontology is **consistent** and lists **unsatisfiable classes** — classes that can have no instance. Each problem comes with an *explanation*: the minimal set of axioms causing it, e.g. `Organisation disjointWith Person`, `Employee subClassOf Organisation`, `Employee subClassOf Person`. `entities` maps the local names used in explanations to full URIs. If the ontology is inconsistent, unsatisfiable classes are not computed: fix the inconsistency first and check again.
- **Integrity checks (closed world, SPARQL).** In OWL, `rdfs:domain`/`rdfs:range` do not *constrain* — they *infer* types — and a relation between two classes (punning) is not constrained at all; a subclass cycle silently means equivalence. These checks report what the reasoner considers fine but is almost always an extraction error — and **unknown vocabulary terms**: a typo such as `rdfs:subClassof` makes a triple no tool or reasoner understands, so it silently means nothing; each one comes with the closest real term as `suggestion`. `rdfs:altLabel`, which OLAF has always used for aliases, is accepted although the standard term is `skos:altLabel`.

The reasoner can only find contradictions the ontology states. **Without `owl:disjointWith` axioms, an ontology is almost never inconsistent** — declare sibling classes that cannot overlap disjoint with `disjoint_add`.

[`olaf_reasoning_agent`](demos/olaf_reasoning_agent/) runs these checks on a built ontology and has an LLM repair what they report.

Requires Java 11+ (included in the Docker image). Settings live under `[reasoner]` in `config.toml` (`enabled`, `java`, `memory_mb`, `timeout_seconds`). When the reasoner is disabled or Java is missing, `ontology_check` still runs the integrity checks and reports the reasoner error.

### Inference

`ontology_infer` runs Pellet on a **consistent** ontology and returns what it entails but does not assert: indirect superclasses, the inherited types of individuals, and what follows from domains, ranges, equivalences, restrictions and property axioms. Trivial entailments (`X ⊑ owl:Thing`, `X ≡ X`…) are left out, and so is the type a class gets when it is used as an individual in a relation between classes (punning).

- **`preview`** lists the inferences, each with why it holds. *Taxonomic* ones follow from the asserted class hierarchy alone and come with their chain of `rdf:type` / `rdfs:subClassOf` axioms; the reasoner explains the others (`max_explained`, a few seconds each). An absurd inference — e.g. an agent inferred to be a language model — reveals a wrong axiom in its explanation: fix that axiom, not the inference.
- **`materialize`** writes them into the ontology graph, replacing those written before. Each inferred triple gets a reification node marked `<urn:olaf:inferredBy> "pellet"` (and no source chunk), so it can be told apart from what was extracted. Asserting an inferred triple later (`relation_add`, …) removes its mark.
- **`clear`** removes them. `concept_merge` does too, as a merge changes what can be inferred.

`ontology_check` and `ontology_orphans` ignore materialized inferences: an inferred `rdf:type` would otherwise hide the very domain/range violation that caused it. With `include_seeds=true`, the seeds take part in the reasoning, but only inferences about the ontology's own entities are kept.

[`olaf_reasoning_agent`](demos/olaf_reasoning_agent/) has an LLM review the new inferences, then materializes them.

### Querying

| Tool | Description |
|------|-------------|
| `sparql_query` | Run a read-only SPARQL query (`SELECT` / `ASK` / `CONSTRUCT` / `DESCRIBE`; updates are rejected). No implicit default graph: target `GRAPH <urn:olaf:{id}>` or `urn:olaf:seed:*`. Optional `limit` (default 100, max 1000) for SELECT rows. |

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

# Mark processed chunks as pending again (all, or one document's), to rebuild an ontology.
# Removes only the olaf_status / olaf_processed_at payload fields.
olaf reset-chunks [doc_id] [--collection NAME]
```

---

## Typical agent workflow

```
1. ontology_list()                  → discover existing ontologies
2. ontology_switch("my-project")    → or ontology_create("my-project", "My Project")
   chunk_collection_switch("docs")   → optional: chunks from another collection than the server default
3. ontology_summary()               → understand current state
4. seed_list()                      → check available seeds
5. seed_load("./domain.ttl")        → optionally load a reference ontology

For each chunk:
6. chunk_list(doc_id=..., status="pending")
7. chunk_read(chunk_id)
8. concept_search(query) +          → run in parallel, deduplicate by URI
   concept_semantic_search(query)
9. concept_create / property_create / relation_add / restriction_add / disjoint_add
10. chunk_mark_processed(chunk_id)

11. ontology_orphans()               → connect or justify any isolated entity
12. ontology_check()                 → fix each reported problem, check again until clean
13. ontology_infer()                 → review the inferences, fix the axioms behind absurd ones,
    ontology_infer(action="materialize")  then store them
14. ontology_export()
```

### Querying workflow

Used by [`olaf_searching_agent`](demos/olaf_searching_agent/), with read-only tools only:

```
1. ontology_switch("my-project")
2. concept_search(term) +            → locate the entities named in the question
   concept_semantic_search(term)
3. concept_get / relation_search     → explore their neighbourhood
4. sparql_query(query)               → lists, counts, joins, hierarchies, paths
5. chunk_read_batch(source_chunk_ids) → ground the answer in the source text
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
                  │  reasoner.py  ──────┼──► Pellet (Java subprocess)
                  │  server.py          │
                  └─────────────────────┘
```

```
src/olaf/
├── config.py       Config dataclasses, TOML loading (stdlib tomllib)
├── ontology.py     OntologyStore — HTTP SPARQL backend, all reads/writes
├── chunks.py       ChunkStore — Qdrant wrapper for existing chunk collection
├── embeddings.py   EmbeddingService — fastembed + olaf_concepts Qdrant collection
├── reasoner.py     Reasoner — runs Pellet (bundled with owlready2) for ontology_check
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
| `owlready2` | Provides the Pellet OWL reasoner jars used by `ontology_check` (run with Java) |
