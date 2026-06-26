from __future__ import annotations
import asyncio
import json
import os
import sys
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from qdrant_client import QdrantClient

from .chunks import ChunkStore
from .config import Config
from .embeddings import EmbeddingService
from .ontology import OntologyStore


def _text(content: Any) -> list[types.TextContent]:
    body = content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False)
    return [types.TextContent(type="text", text=body)]


def _err(msg: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=f"Error: {msg}")]


def create_server(config: Config) -> Server:
    server = Server("olaf")
    onto = OntologyStore(
        url=config.oxigraph.url,
        base_uri=config.ontology.base_uri,
        name=config.ontology.name,
        ontology_id=config.ontology.ontology_id,
    )
    qdrant = QdrantClient(url=config.qdrant.url)
    chunks = ChunkStore(config.qdrant.url, config.qdrant.collection, config.qdrant.field_mapping)

    embed: EmbeddingService | None = None
    if config.embedding.enabled:
        try:
            concepts_collection = f"{config.qdrant.concepts_collection}_{config.ontology.ontology_id}"
            embed = EmbeddingService(config.embedding.model, qdrant, concepts_collection)
        except Exception as e:
            print(f"Warning: embeddings disabled ({e})", file=sys.stderr)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="chunk_list",
                description=(
                    "List chunks from the Qdrant collection. "
                    "Filter by doc_id and/or status. Returns id, doc_id, chunk_index, "
                    "text_preview (200 chars), and processing status."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "Filter to one document"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "processed", "all"],
                            "description": "Filter by olaf processing status (default: all)",
                        },
                        "limit": {"type": "integer", "description": "Max results (default: 20)"},
                        "offset": {"type": "integer", "description": "Pagination offset"},
                    },
                },
            ),
            types.Tool(
                name="chunk_read",
                description="Read the full text of a chunk by its Qdrant point ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Qdrant point ID (integer or UUID string)"},
                    },
                    "required": ["chunk_id"],
                },
            ),
            types.Tool(
                name="chunk_mark_processed",
                description="Mark a chunk as processed after extracting ontology elements from it.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string"},
                    },
                    "required": ["chunk_id"],
                },
            ),
            types.Tool(
                name="concept_create",
                description=(
                    "Create an OWL class in the ontology. "
                    "Always call concept_search and concept_semantic_search first to check for duplicates. "
                    "Returns the URI and whether a new class was created (created=false if URI already exists)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Human-readable class name, e.g. 'Motor Vehicle'"},
                        "definition": {"type": "string", "description": "Natural language definition"},
                        "parent_uri": {"type": "string", "description": "Parent class URI for rdfs:subClassOf"},
                        "aliases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Alternative labels (rdfs:altLabel)",
                        },
                        "language": {"type": "string", "description": "BCP-47 language tag (default: en)"},
                        "source_chunk_id": {"type": "string", "description": "Qdrant point ID of the chunk this concept was extracted from"},
                    },
                    "required": ["label", "definition"],
                },
            ),
            types.Tool(
                name="concept_get",
                description="Get all triples for a concept URI: type, label, definition, aliases, subClassOf, restrictions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string"},
                    },
                    "required": ["uri"],
                },
            ),
            types.Tool(
                name="concept_search",
                description=(
                    "Search for existing concepts by label substring (SPARQL). "
                    "Call this before concept_create to avoid duplicates. "
                    "For broader similarity matching, also call concept_semantic_search."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "description": "Max results (default: 10)"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="concept_semantic_search",
                description=(
                    "Search for existing concepts by semantic similarity (vector search in Qdrant). "
                    "Finds concepts that are similar in meaning even with different wording. "
                    "Call alongside concept_search before concept_create to catch near-duplicates."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "description": "Max results (default: 10)"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="concept_update",
                description="Update a concept's label, definition, or aliases.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string"},
                        "label": {"type": "string"},
                        "definition": {"type": "string"},
                        "aliases_add": {"type": "array", "items": {"type": "string"}},
                        "aliases_remove": {"type": "array", "items": {"type": "string"}},
                        "language": {"type": "string"},
                    },
                    "required": ["uri"],
                },
            ),
            types.Tool(
                name="concept_merge",
                description=(
                    "Merge two concepts: all triples from merge_uri are copied to keep_uri, "
                    "all references to merge_uri are re-pointed to keep_uri, then merge_uri is deleted."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keep_uri": {"type": "string", "description": "URI to keep"},
                        "merge_uri": {"type": "string", "description": "URI to absorb (will be deleted)"},
                    },
                    "required": ["keep_uri", "merge_uri"],
                },
            ),
            types.Tool(
                name="property_create",
                description="Create an OWL property (ObjectProperty links two classes; DatatypeProperty links a class to a literal).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "e.g. 'has engine'"},
                        "type": {
                            "type": "string",
                            "enum": ["object", "datatype"],
                        },
                        "domain_uri": {"type": "string", "description": "rdfs:domain class URI"},
                        "range_uri": {
                            "type": "string",
                            "description": "rdfs:range class URI (object) or XSD datatype URI (datatype)",
                        },
                        "parent_uri": {"type": "string", "description": "rdfs:subPropertyOf URI"},
                        "language": {"type": "string"},
                    },
                    "required": ["label", "type"],
                },
            ),
            types.Tool(
                name="property_get",
                description="Get all details of a property by URI.",
                inputSchema={
                    "type": "object",
                    "properties": {"uri": {"type": "string"}},
                    "required": ["uri"],
                },
            ),
            types.Tool(
                name="property_search",
                description="Search for properties by label substring.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="relation_add",
                description=(
                    "Add a triple (subject, property, object) to the ontology. "
                    "Use full URIs for all arguments. "
                    "Common property URIs: rdfs:subClassOf = http://www.w3.org/2000/01/rdf-schema#subClassOf, "
                    "rdf:type = http://www.w3.org/1999/02/22-rdf-syntax-ns#type."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subject_uri": {"type": "string"},
                        "property_uri": {"type": "string"},
                        "object_value": {"type": "string", "description": "Target URI or literal value"},
                        "is_literal": {"type": "boolean", "description": "True if object_value is a literal (default: false)"},
                        "datatype": {
                            "type": "string",
                            "description": "XSD datatype URI when is_literal=true, e.g. http://www.w3.org/2001/XMLSchema#integer",
                        },
                    },
                    "required": ["subject_uri", "property_uri", "object_value"],
                },
            ),
            types.Tool(
                name="relation_search",
                description="Find triples matching a pattern. All parameters are optional — omit to get all triples.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subject_uri": {"type": "string"},
                        "property_uri": {"type": "string"},
                        "object_uri": {"type": "string"},
                        "limit": {"type": "integer", "description": "Max results (default: 20)"},
                    },
                },
            ),
            types.Tool(
                name="restriction_add",
                description=(
                    "Add an OWL restriction to a class via a blank node. "
                    "Examples: 'Car must have some Engine' → some, "
                    "'Person must have exactly 1 birthDate' → exactly+cardinality=1. "
                    "Creates: class rdfs:subClassOf [ a owl:Restriction; owl:onProperty prop; owl:someValuesFrom value ]."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "class_uri": {"type": "string"},
                        "property_uri": {"type": "string"},
                        "restriction_type": {
                            "type": "string",
                            "enum": ["some", "all", "has_value", "exactly", "min", "max"],
                            "description": "some=someValuesFrom, all=allValuesFrom, has_value=hasValue, exactly/min/max=cardinality",
                        },
                        "value": {
                            "type": "string",
                            "description": "Target class URI, or literal string for has_value with is_literal_value=true. Leave empty for unqualified cardinality.",
                        },
                        "cardinality": {"type": "integer", "description": "Required for exactly/min/max"},
                        "is_literal_value": {"type": "boolean", "description": "True if value is a literal (for has_value)"},
                    },
                    "required": ["class_uri", "property_uri", "restriction_type", "value"],
                },
            ),
            types.Tool(
                name="seed_list",
                description="List all seed ontologies loaded in the store (shared across all ontologies).",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="seed_load",
                description=(
                    "Load a seed ontology (Turtle format) into the global seed store. "
                    "Pass either 'url' (HTTP URI) or 'content' (raw Turtle string). "
                    "Seeds are shared across all ontologies."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "HTTP URI of a Turtle ontology to fetch"},
                        "content": {"type": "string", "description": "Raw Turtle content to load directly"},
                        "graph_id": {
                            "type": "string",
                            "description": "Named graph URI (auto-generated as urn:olaf:seed:{id} if omitted)",
                        },
                    },
                },
            ),
            types.Tool(
                name="ontology_list",
                description=(
                    "List all ontologies stored in Oxigraph. "
                    "Returns id, name, base_uri, class count, and whether it is the active ontology."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="ontology_create",
                description=(
                    "Create a new empty ontology. "
                    "Use ontology_switch afterwards to start working with it."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ontology_id": {"type": "string", "description": "Short identifier, e.g. 'legal', 'medical-v2'"},
                        "name": {"type": "string", "description": "Human-readable name"},
                        "base_uri": {"type": "string", "description": "Namespace URI (defaults to configured base_uri)"},
                    },
                    "required": ["ontology_id", "name"],
                },
            ),
            types.Tool(
                name="ontology_switch",
                description=(
                    "Switch the active ontology for this session. "
                    "All subsequent tool calls will operate on the selected ontology."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ontology_id": {"type": "string", "description": "Identifier of the ontology to activate"},
                    },
                    "required": ["ontology_id"],
                },
            ),
            types.Tool(
                name="ontology_summary",
                description=(
                    "Get a summary of the current ontology: "
                    "class/property/individual/restriction counts, root classes, chunk processing progress."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="ontology_export",
                description="Export the ontology as a Turtle string.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_seeds": {
                            "type": "boolean",
                            "description": "Also include seed graphs (default: false)",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            match name:
                case "chunk_list":
                    return _text(chunks.list_chunks(
                        doc_id=arguments.get("doc_id"),
                        status=arguments.get("status", "all"),
                        limit=arguments.get("limit", 20),
                        offset=arguments.get("offset"),
                    ))

                case "chunk_read":
                    chunk = chunks.get_chunk(arguments["chunk_id"])
                    return _text(chunk) if chunk else _err(f"Chunk not found: {arguments['chunk_id']}")

                case "chunk_mark_processed":
                    ok = chunks.mark_processed(arguments["chunk_id"])
                    return _text("ok" if ok else "failed")

                case "concept_create":
                    source_chunk_id = arguments.get("source_chunk_id")
                    result = onto.concept_create(
                        label=arguments["label"],
                        definition=arguments["definition"],
                        parent_uri=arguments.get("parent_uri"),
                        aliases=arguments.get("aliases"),
                        lang=arguments.get("language", "en"),
                        source_chunk_id=source_chunk_id,
                    )
                    if result["created"] and embed:
                        embed.upsert_concept(result["uri"], arguments["label"], arguments["definition"], source_chunk_id)
                    return _text(result)

                case "concept_get":
                    result = onto.concept_get(arguments["uri"])
                    return _text(result) if result else _err(f"Concept not found: {arguments['uri']}")

                case "concept_search":
                    return _text(onto.concept_search_sparql(arguments["query"], arguments.get("top_k", 10)))

                case "concept_semantic_search":
                    if not embed:
                        return _err("Semantic search is disabled (embedding model not loaded)")
                    return _text(embed.search_concepts(arguments["query"], arguments.get("top_k", 10)))

                case "concept_update":
                    onto.concept_update(
                        uri=arguments["uri"],
                        label=arguments.get("label"),
                        definition=arguments.get("definition"),
                        aliases_add=arguments.get("aliases_add"),
                        aliases_remove=arguments.get("aliases_remove"),
                        lang=arguments.get("language", "en"),
                    )
                    if embed and (arguments.get("label") or arguments.get("definition")):
                        concept = onto.concept_get(arguments["uri"])
                        if concept:
                            label = concept.get("label", "")
                            definition = concept.get("definition", "")
                            if label:
                                embed.upsert_concept(arguments["uri"], label, definition)
                    return _text("ok")

                case "concept_merge":
                    onto.concept_merge(arguments["keep_uri"], arguments["merge_uri"])
                    if embed:
                        embed.delete_concept(arguments["merge_uri"])
                    return _text("ok")

                case "property_create":
                    return _text(onto.property_create(
                        label=arguments["label"],
                        prop_type=arguments["type"],
                        domain_uri=arguments.get("domain_uri"),
                        range_uri=arguments.get("range_uri"),
                        parent_uri=arguments.get("parent_uri"),
                        lang=arguments.get("language", "en"),
                    ))

                case "property_get":
                    result = onto.property_get(arguments["uri"])
                    return _text(result) if result else _err(f"Property not found: {arguments['uri']}")

                case "property_search":
                    return _text(onto.property_search(arguments["query"], arguments.get("top_k", 10)))

                case "relation_add":
                    onto.relation_add(
                        subject_uri=arguments["subject_uri"],
                        property_uri=arguments["property_uri"],
                        object_value=arguments["object_value"],
                        is_literal=arguments.get("is_literal", False),
                        datatype=arguments.get("datatype"),
                    )
                    return _text("ok")

                case "relation_search":
                    return _text(onto.relation_search(
                        subject_uri=arguments.get("subject_uri"),
                        property_uri=arguments.get("property_uri"),
                        object_uri=arguments.get("object_uri"),
                        limit=arguments.get("limit", 20),
                    ))

                case "restriction_add":
                    bnode = onto.restriction_add(
                        class_uri=arguments["class_uri"],
                        property_uri=arguments["property_uri"],
                        restriction_type=arguments["restriction_type"],
                        value=arguments["value"],
                        cardinality=arguments.get("cardinality"),
                        is_literal_value=arguments.get("is_literal_value", False),
                    )
                    return _text({"restriction_bnode": bnode})

                case "seed_list":
                    return _text(onto.seed_list())

                case "seed_load":
                    return _text(onto.seed_load(
                        graph_id=arguments.get("graph_id"),
                        url=arguments.get("url"),
                        content=arguments.get("content"),
                    ))

                case "ontology_list":
                    return _text(onto.ontology_list())

                case "ontology_create":
                    return _text(onto.ontology_create(
                        ontology_id=arguments["ontology_id"],
                        name=arguments["name"],
                        base_uri=arguments.get("base_uri"),
                    ))

                case "ontology_switch":
                    result = onto.ontology_switch(arguments["ontology_id"])
                    if embed:
                        embed.switch_collection(f"{config.qdrant.concepts_collection}_{arguments['ontology_id']}")
                    return _text(result)

                case "ontology_summary":
                    summary = onto.summary()
                    try:
                        summary["chunks_total"] = chunks.count_total()
                        summary["chunks_processed"] = chunks.count_processed()
                    except Exception:
                        pass
                    return _text(summary)

                case "ontology_export":
                    return _text(onto.export_ttl(include_seeds=arguments.get("include_seeds", False)))

                case _:
                    return _err(f"Unknown tool: {name}")

        except Exception as e:
            return _err(str(e))

    return server


def _make_sse_app(server: Server):
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    return Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=transport.handle_post_message),
    ])


async def _run() -> None:
    config = Config.load()
    server = create_server(config)

    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "sse":
        import uvicorn
        host = os.environ.get("MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_PORT", "8000"))
        app = _make_sse_app(server)
        uv_config = uvicorn.Config(app, host=host, port=port, log_level="info")
        uv_server = uvicorn.Server(uv_config)
        await uv_server.serve()
    else:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


def _make_onto(config: Config) -> OntologyStore:
    return OntologyStore(
        url=config.oxigraph.url,
        base_uri=config.ontology.base_uri,
        name=config.ontology.name,
        ontology_id=config.ontology.ontology_id,
    )


def _cmd_drop(ontology_id: str) -> None:
    onto = _make_onto(Config.load())
    result = onto.drop_ontology(ontology_id)
    print(f"Ontologie '{result['dropped']}' supprimée.")


def _cmd_drop_seed(seed_id: str) -> None:
    onto = _make_onto(Config.load())
    result = onto.seed_drop(seed_id)
    print(f"Seed '{result['dropped']}' supprimée.")


def main() -> None:
    if len(sys.argv) >= 2:
        match sys.argv[1]:
            case "drop":
                if len(sys.argv) < 3:
                    print("Usage: olaf drop <ontology_id>", file=sys.stderr)
                    sys.exit(1)
                _cmd_drop(sys.argv[2])
            case "drop-seed":
                if len(sys.argv) < 3:
                    print("Usage: olaf drop-seed <seed_id>", file=sys.stderr)
                    sys.exit(1)
                _cmd_drop_seed(sys.argv[2])
            case _:
                asyncio.run(_run())
    else:
        asyncio.run(_run())


if __name__ == "__main__":
    main()
