# olaf_merging_agent

Agent that merges two ontologies into a new one — typically two built by [`olaf_building_agent`](../olaf_building_agent/) from different corpora. It **aligns** them first, finding which entities of one correspond to which of the other (*ontology matching*), then **merges** them along that alignment: equivalent entities become one, and a subclass link is added between an entity and a broader one of the other ontology. It uses OLAF (`ontology_import`, `concept_semantic_search`, `concept_merge`) and LiteLLM.

The two source ontologies are left unchanged. The run produces the merged ontology and the alignment itself, as an [SSSOM](https://mapping-commons.github.io/sssom/) file.

## Architecture

```
agent.py  ──SSE──▶  OLAF MCP server  ──SPARQL──▶  Oxigraph (RDF)
                                     ──gRPC────▶  Qdrant (embeddings, source chunks)
LiteLLM   ──API──▶  Claude / OpenAI / Ollama
```

The code drives the work. The LLM only judges the candidate pairs:

1. **Import** — the target ontology (`[olaf] ontology_id`) is created, then the reference ontology and the other one are copied into it with `ontology_import`, provenance included, their entities moved to the target's namespace. Ontologies built with the same `base_uri` use the same URI for the same name, e.g. `#Fund`. A naive copy would merge them silently, yet two corpora often use one word for different notions. So an entity of the other ontology whose name the reference already uses is renamed (`#Fund_ipcc`), and the pair is handed to the LLM like any other. Seed terms that both reuse are the same entity already.
2. **Match** — for each class, then each property, then each individual of the other ontology, the agent looks for candidates among the reference entities of the same kind: the one with the same name, if any, then the semantically closest above `min_score`, up to `candidates_per_entity`. An entity with no candidate is kept as it is.
3. **Decide** — the LLM gets `entities_per_task` entities at a time, with their candidates. For each entity it sees the label, definition, parents, domain and range for a property, and the text of the source chunks. For each pair, it answers in JSON with a relation and a confidence:
   - `equivalent`: same extension;
   - `narrower`: the other entity is a kind of the reference one;
   - `broader`: the reverse;
   - `none`: siblings, part and whole, related topics… (the default).

   A same name is not taken as evidence. Individuals can only be `equivalent` or `none`. Before a `narrower` or `broader`, the LLM must answer the question "is *every* O an R?" at the start of its reason. Without this test, LLMs tend to answer `narrower` for anything merely related: a group of countries "under" the fund that supports them, an intervention "under" the livelihoods it improves. A specific case of a concept ("Readiness Programme" vs "Programme") is `narrower`, never `equivalent`, because merging would erase it.
4. **Merge** — decisions with a confidence of at least `min_confidence` are applied:
   - `equivalent`: `concept_merge` into the reference's entity, which keeps its URI, label and definition. The other one's labels become alternative labels, and its relations and source chunks move over.
   - `narrower` / `broader`: an `rdfs:subClassOf` link (`rdfs:subPropertyOf` for properties) between the two.

   Classes are merged before properties are matched: two properties merged afterwards then have the same domain and range, rather than two.

   Property correspondences are also checked by the code against the current domains and ranges. A subproperty inherits the domain and range of its parent, and two merged properties keep both, which OWL reads as their intersection. So `narrower` requires the other property's domain and range to be equal to, or subclasses of, the reference's; `broader` requires the reverse; and `equivalent` requires both. Otherwise the correspondence is only written to the alignment file. Example: `advances` (Readiness Programme → Strategic Policy Planning) under `funds` (Green Climate Fund → Programme) would type every Readiness Programme as a Green Climate Fund.
5. **Output** — the alignment is written to `alignment_path`, applied or not, and the merged ontology is checked (`ontology_check`).

## Prerequisites

- OLAF MCP server with `ontology_import` (after a code update, rebuild the container: `docker compose up -d --build olaf`), running in SSE mode, with embeddings enabled (`[embedding] enabled = true`). Matching needs semantic search. The reasoner is optional; without it, the final check only runs the SPARQL checks.
- The two ontologies to merge, and ideally the Qdrant chunk collections they were built from, for the source text.
- An LLM API key (Anthropic, OpenAI, etc.)

## Setup

```bash
cd demos/olaf_merging_agent
pip install -r requirements.txt
cp config.example.toml config.toml   # then edit config.toml
export ANTHROPIC_API_KEY=sk-ant-...
```

## Configuration (`config.toml`)

```toml
[olaf]
url           = "http://localhost:8000/sse"
ontology_id   = "merged"          # created by the run — must not exist yet
ontology_name = "Merged ontology"

[merge]
reference            = "gcf"          # keeps its URIs
other                = "ipcc"         # merged into it
reference_collection = "gcf_chunks"   # chunk collections, for the source text (optional)
other_collection     = "ipcc_chunks"

[litellm]
model = "claude-sonnet-4-6"

[agent]
min_score             = 0.7   # min similarity for a candidate
candidates_per_entity = 4
entities_per_task     = 8
min_confidence        = 0.7   # below: written to the alignment, not applied
alignment_path        = "alignment.sssom.tsv"
```

See [`config.example.toml`](config.example.toml) for all settings. The `[litellm]` section works as in `olaf_building_agent`, including OpenAI-compatible providers (`api_base` + `api_key_env`).

**Which one is the reference?** On equivalence, the reference's URI, label and definition are kept. Choose the ontology that is better curated, or the one other systems already point to.

## Running

```bash
python agent.py                       # uses config.toml in current directory
python agent.py --config my.toml
```

```
10:02:11 INFO     Active ontology: merged (created)
10:02:14 INFO     Imported 'gcf' (reference): 2841 triples, 212 entities, 0 renamed (name already used).
10:02:17 INFO     Imported 'ipcc' (other): 3120 triples, 240 entities, 37 renamed (name already used).
10:02:31 INFO     Matching class: 121 entities with candidates, 64 without (kept as they are).
10:02:58 INFO     [class · task 1/16] 5 correspondence(s) found for 8 entities.
10:02:58 INFO     Applied: Adaptation Fund equivalent Adaptation Fund (0.95) — both define the UNFCCC fund financing adaptation projects
10:02:58 INFO     Applied: Mangrove Restoration narrower Ecosystem Restoration (0.90) — restoring mangroves is a kind of ecosystem restoration
…
10:09:40 INFO     Merge done: 74 equivalences merged, 31 subsumptions added, 9 correspondences not applied.
10:09:40 INFO     Alignment (114 correspondences) written to alignment.sssom.tsv
10:09:52 INFO     Check of the merged ontology — consistent: True, unsatisfiable: 0, classes in cycles: 1, domain: 3, range: 2, issues: 6 — run olaf_reasoning_agent on it to fix them
```

Then run [`olaf_reasoning_agent`](../olaf_reasoning_agent/) on the merged ontology (`ontology_id = "merged"`). Its deduplication catches the remaining near-duplicates, and its repair fixes the problems the merge brought to light. A typical one is a subclass cycle when the two ontologies ordered two notions in opposite ways.

## The alignment file

`alignment.sssom.tsv` holds one row per correspondence. It uses the URIs the entities have in their **source** ontologies, so it describes the alignment between them, independently of the merge:

| subject_id (reference) | predicate_id | object_id (other) | confidence | applied | comment |
|---|---|---|---|---|---|
| `…#AdaptationFund` | `skos:exactMatch` | `…#AdaptationFund` | 0.95 | true | the LLM's reason |
| `…#EcosystemRestoration` | `skos:narrowMatch` | `…#MangroveRestoration` | 0.90 | true | … |

`skos:narrowMatch` means the object is narrower than the subject. Rows with `applied = false` say why at the end of their comment: below `min_confidence`, incompatible domain or range, or failed to apply (logged). Review them, and apply them by hand if they are right (`concept_merge`, `relation_add`).

## Limitations

- **Two ontologies per run.** To merge a third, run again with the merged ontology as the reference.
- **Starting over**: the target must not exist, since importing again would duplicate its content. Drop it first with `olaf drop <ontology_id>`.
- **Source text**: each source chunk keeps its id. Reading it in the merged ontology (searching agent, reasoning agent) needs the right chunk collection, and the two sources may have been built from different ones.
- **Chunk processing status** is per ontology: the merged ontology has none, so it is not meant to be extended with `olaf_building_agent` from the same chunks.
