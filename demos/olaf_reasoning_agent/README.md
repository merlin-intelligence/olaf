# olaf_reasoning_agent

Agent that repairs the logical problems of an existing OWL/RDFS ontology — typically one built by [`olaf_building_agent`](../olaf_building_agent/) — using the OLAF reasoner (`ontology_check`) and LiteLLM.

## Architecture

```
agent.py  ──SSE──▶  OLAF MCP server  ──SPARQL──▶  Oxigraph (RDF)
                     ontology_check  ──JVM─────▶  Pellet reasoner
                                     ──gRPC────▶  Qdrant (source chunks)
LiteLLM   ──API──▶  Claude / OpenAI / Ollama
```

The code drives the repair; the LLM only decides and applies the fixes:

1. **Check** — `ontology_check` and `ontology_orphans` run. Problems reported:
   - by the Pellet reasoner: **inconsistency**, **unsatisfiable classes**, each with its explanation (the minimal set of axioms causing it);
   - by the SPARQL integrity checks: **subclass cycles**, **domain/range violations**, **untyped individuals**, **property kind mismatches**;
   - and **orphans** (`fix_orphans = true`).
2. **Fix** — the problems are split into groups of `problems_per_task`. For each group, the code gathers the context itself: the axioms at stake, the entities involved (label, definition, types, parents, all ancestors, domain/range) and the text of the chunks they were extracted from. For domain/range violations it also lists the classes the property could be widened to (common ancestors) and the property's other uses. A fresh LLM conversation gets it all in its task message and fixes the problems with a few write tools (`relation_delete`, `relation_add`, `property_update`, `concept_update`, `concept_merge`); a few read tools (`concept_get`, `property_get`, `concept_search`, `relation_search`) cover what the context misses, such as where to attach an orphan.
3. **Check again** — for up to `max_rounds` rounds. The run stops early when nothing is left, or when a round changed nothing. Several rounds are often needed: the reasoner only lists unsatisfiable classes once the ontology is consistent.

Because the context is given upfront, the LLM has little to explore: each task should take a few LLM calls, and the prompt size depends on `problems_per_task`, not on the size of the ontology.

Before the first change, the whole ontology graph is saved as Turtle in `backup_dir` (see [Undoing a run](#undoing-a-run)).

## Prerequisites

- OLAF MCP server running in SSE mode, with the reasoner enabled (`[reasoner] enabled = true`, Java 11+ — included in the Docker image). Without it, only the SPARQL checks run.
- The ontology to repair, and the Qdrant chunk collection it was built from (for the source text).
- An LLM API key (Anthropic, OpenAI, etc.)

## Setup

```bash
cd demos/olaf_reasoning_agent
pip install -r requirements.txt
cp config.example.toml config.toml   # then edit config.toml
export ANTHROPIC_API_KEY=sk-ant-...
```

## Configuration (`config.toml`)

```toml
[olaf]
url         = "http://localhost:8000/sse"   # OLAF SSE endpoint
ontology_id = "demo"                        # ontology to repair (must exist)
collection  = "chunks"                      # chunk collection it was built from (default: server's)

[litellm]
model       = "claude-sonnet-4-6"
max_tokens  = 4096
temperature = 0

[agent]
max_rounds        = 3       # check → fix → check again
problems_per_task = 5       # problems per LLM conversation
fix_orphans       = true    # also connect orphan classes/individuals
max_iterations    = 12      # max LLM rounds per task
log_level         = "INFO"
```

See [`config.example.toml`](config.example.toml) for all settings (context size, seeds, retries, backup, export). The `[litellm]` section works as in `olaf_building_agent`, including OpenAI-compatible providers (`api_base` + `api_key_env`).

## Running

```bash
python agent.py                       # uses config.toml in current directory
python agent.py --config my.toml
```

The agent logs each round, task, the model's reasoning text and every fix to stderr:
```
17:20:01 INFO     Active ontology: demo
17:20:04 INFO     Round 1 — consistent: True, unsatisfiable: 0, classes in cycles: 2, domain: 11, range: 8, untyped: 0, kind: 0, orphans: 8
17:20:04 INFO     Backup of the ontology before any change: backups/demo-20261007-172004.ttl
17:20:05 INFO     [round 1 · task 1/6] iteration 1
17:20:31 INFO     [round 1 · task 1/6] The text lists constraints among ontology concepts, so…
17:20:31 INFO     Tool call: relation_delete(subject_uri='http://olaf.local/ontology#OntologyConcept', property_uri='http://www.w3.org/2000/01/rdf-schema#subClassOf', object_value='http://olaf.local/ontology#OntologyConstraint')
17:20:31 INFO     Tool call: property_update(uri='http://olaf.local/ontology#produces', domain_uri='http://olaf.local/ontology#OntologytotoolCompilation')
17:20:58 INFO     [round 1 · task 1/6] done: Removed OntologyConcept subClassOf OntologyConstraint (the text…
…
17:31:40 INFO     Round 2 — consistent: True, unsatisfiable: 0, classes in cycles: 0, domain: 1, range: 0, untyped: 0, kind: 0, orphans: 3
…
```

With `--log-level DEBUG`, the full task message of each conversation (problems, entities, source text) is logged too; library traces (LiteLLM, HTTP) stay at INFO so the log remains readable.

## Undoing a run

Each run that has something to fix writes `backups/<ontology_id>-<date>-<time>.ttl` before changing anything — a dump of the whole graph, provenance included (`[agent] backup_dir`; set it to `""` to disable). To put the ontology back as it was, replace the graph with it in Oxigraph:

```bash
curl -X PUT 'http://localhost:7878/store?graph=urn:olaf:<ontology_id>' \
  -H 'Content-Type: text/turtle' --data-binary @backups/<ontology_id>-<date>-<time>.ttl
```

Every change is also logged with its full arguments (`Tool call: relation_delete(subject_uri='…', …)`).
