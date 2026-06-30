# OLAF — Demo agents

Standalone demo agents that connect to a running OLAF MCP server and drive ontology construction autonomously.

| Agent | Description |
|-------|-------------|
| [`olaf_building_agent`](olaf_building_agent/) | Reads Qdrant chunks and builds an OWL/RDFS ontology via OLAF + LiteLLM |

Each demo has its own `requirements.txt` and `config.example.toml`. They connect to OLAF over SSE and do not modify the OLAF package itself.

See each agent's `README.md` for setup instructions, configuration reference, and known issues.
