# OLAF — Demo agents

Standalone demo agents that connect to a running OLAF MCP server and drive ontology construction, repair or querying autonomously. Typical order: build, then repair, then query.

| Agent | Description |
|-------|-------------|
| [`olaf_building_agent`](olaf_building_agent/) | Reads Qdrant chunks and builds an OWL/RDFS ontology via OLAF + LiteLLM |
| [`olaf_reasoning_agent`](olaf_reasoning_agent/) | Repairs a built ontology: runs the reasoner and integrity checks, then has the LLM fix each problem from its axioms and source text |
| [`olaf_searching_agent`](olaf_searching_agent/) | Answers natural-language questions over an ontology: searches with read-only OLAF tools, generates and runs SPARQL, cites source chunks |

Each demo has its own `requirements.txt` and `config.example.toml`. They connect to OLAF over SSE and do not modify the OLAF package itself.

See each agent's `README.md` for setup instructions, configuration reference, and known issues.
