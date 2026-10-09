# OLAF — Demo agents

Standalone demo agents that connect to a running OLAF MCP server and drive ontology construction, curation or querying autonomously. Typical order: build, then curate (reasoning agent), then query.

| Agent | Description |
|-------|-------------|
| [`olaf_building_agent`](olaf_building_agent/) | Reads Qdrant chunks and builds an OWL/RDFS ontology via OLAF + LiteLLM |
| [`olaf_reasoning_agent`](olaf_reasoning_agent/) | Curates a built ontology: merges duplicates, has the LLM fix each problem the reasoner and integrity checks report (from its axioms and source text), declares disjoint siblings, then reviews and materializes the reasoner's inferences |
| [`olaf_searching_agent`](olaf_searching_agent/) | Answers natural-language questions over an ontology: searches with read-only OLAF tools, generates and runs SPARQL, cites source chunks |

Each demo has its own `requirements.txt` and `config.example.toml`. They connect to OLAF over SSE and do not modify the OLAF package itself. Code they all need — the LLM call with retries, the tool-calling loop, context pruning, session setup, logging — lives in [`agent_common.py`](agent_common.py), next to them: run each agent from its own directory (`python agent.py`), it finds the module there.

See each agent's `README.md` for setup instructions, configuration reference, and known issues.
