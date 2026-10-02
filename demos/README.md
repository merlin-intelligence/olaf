# OLAF — Demo agents

Standalone demo agents that connect to a running OLAF MCP server and drive ontology construction or querying autonomously.

| Agent | Description |
|-------|-------------|
| [`olaf_building_agent`](olaf_building_agent/) | Reads Qdrant chunks and builds an OWL/RDFS ontology via OLAF + LiteLLM |
| [`olaf_searching_agent`](olaf_searching_agent/) | Answers natural-language questions over an ontology: searches with read-only OLAF tools, generates and runs SPARQL, cites source chunks |

Each demo has its own `requirements.txt` and `config.example.toml`. They connect to OLAF over SSE and do not modify the OLAF package itself.

See each agent's `README.md` for setup instructions, configuration reference, and known issues.
