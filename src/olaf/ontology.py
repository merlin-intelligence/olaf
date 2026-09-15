from __future__ import annotations
import asyncio
import re
import uuid
from urllib.parse import urlparse

_PREFIXES = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""

_RESTRICTION_PROP = {
    "some":      "owl:someValuesFrom",
    "all":       "owl:allValuesFrom",
    "has_value": "owl:hasValue",
    "exactly":   "owl:qualifiedCardinality",
    "min":       "owl:minQualifiedCardinality",
    "max":       "owl:maxQualifiedCardinality",
}
_UNQUALIFIED_RESTRICTION_PROP = {
    "exactly": "owl:cardinality",
    "min":     "owl:minCardinality",
    "max":     "owl:maxCardinality",
}

_CHUNK_URI_PREFIX = "urn:olaf:chunk:"


def _slugify_class(text: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9\s]", "", text).split()
    return "".join(w.capitalize() for w in words) or "Unknown"


def _slugify_prop(text: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9\s]", "", text).split()
    if not words:
        return "unknown"
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


_INVALID_IRI = re.compile(r'[<>\s"{}|\\^`]')


def _check_uri(uri: str) -> str:
    if _INVALID_IRI.search(uri):
        raise ValueError(f"Invalid URI (contains illegal characters): {uri!r}")
    return uri


def _fmt_http(binding: dict | None) -> str | None:
    if binding is None:
        return None
    return binding["value"]


def _chunk_uri(chunk_id: str) -> str:
    return f"{_CHUNK_URI_PREFIX}{chunk_id}"


def _chunk_id_from_uri(uri: str) -> str:
    return uri[len(_CHUNK_URI_PREFIX):] if uri.startswith(_CHUNK_URI_PREFIX) else uri


# ── HTTP executor ──────────────────────────────────────────────────────────────

class _HttpExecutor:
    def __init__(self, url: str):
        import httpx
        self._url = url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30)

    async def execute_select(self, sparql: str, variables: list[str]) -> list[dict[str, str | None]]:
        resp = await self._client.post(
            f"{self._url}/query",
            content=sparql.encode(),
            headers={"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"},
        )
        resp.raise_for_status()
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])
        return [{v: _fmt_http(b.get(v)) for v in variables} for b in bindings]

    async def execute_ask(self, sparql: str) -> bool:
        resp = await self._client.post(
            f"{self._url}/query",
            content=sparql.encode(),
            headers={"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"},
        )
        resp.raise_for_status()
        return bool(resp.json().get("boolean", False))

    async def execute_update(self, sparql: str) -> None:
        resp = await self._client.post(
            f"{self._url}/update",
            content=sparql.encode(),
            headers={"Content-Type": "application/sparql-update"},
        )
        resp.raise_for_status()

    async def load_graph(self, data: bytes, graph_uri: str) -> None:
        resp = await self._client.put(
            f"{self._url}/store",
            params={"graph": graph_uri},
            content=data,
            headers={"Content-Type": "text/turtle"},
        )
        resp.raise_for_status()

    async def dump_graph(self, graph_uri: str) -> bytes:
        resp = await self._client.get(
            f"{self._url}/store",
            params={"graph": graph_uri},
            headers={"Accept": "text/turtle"},
        )
        resp.raise_for_status()
        return resp.content


# ── OntologyStore ──────────────────────────────────────────────────────────────

class OntologyStore:
    """
    Stateless RDF store — all methods accept ontology_id explicitly.
    A single instance can safely serve multiple concurrent sessions.
    Per-session active ontology is tracked in server.py's session dict.
    """

    def __init__(self, url: str):
        self._ex = _HttpExecutor(url)
        self._meta: dict[str, tuple[str, str]] = {}

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _graph(ontology_id: str) -> str:
        graph = f"urn:olaf:{ontology_id}"
        _check_uri(graph)
        return graph

    async def _resolve_base(self, ontology_id: str) -> str:
        if ontology_id not in self._meta:
            rows = await self._ex.execute_select(f"""
            {_PREFIXES}
            SELECT ?base ?name WHERE {{
                GRAPH <urn:olaf:{ontology_id}> {{
                    ?base a owl:Ontology .
                    OPTIONAL {{ ?base rdfs:label ?name }}
                }}
            }}
            """, ["base", "name"])
            if not rows or not rows[0]["base"]:
                raise ValueError(
                    f"Ontology '{ontology_id}' not found. "
                    "Call ontology_create first, or check the configured ontology_id."
                )
            self._meta[ontology_id] = (rows[0]["base"].rstrip("#/"), rows[0]["name"] or "")
        return self._meta[ontology_id][0]

    async def _uri(self, local: str, ontology_id: str) -> str:
        return f"{await self._resolve_base(ontology_id)}#{local}"

    # ── Startup ────────────────────────────────────────────────────────────

    async def bootstrap(self, ontology_id: str, base_uri: str, name: str = "Ontology") -> None:
        """Ensure the default ontology declaration exists. Call once at server startup."""
        base = _check_uri(base_uri.rstrip("#/"))
        graph = self._graph(ontology_id)
        if not await self._ex.execute_ask(
            f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{base}> a owl:Ontology }} }}"
        ):
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{
                GRAPH <{graph}> {{
                    <{base}> a owl:Ontology ;
                        rdfs:label "{_esc(name)}" .
                }}
            }}
            """)
        self._meta[ontology_id] = (base, name)

    # ── Multi-ontology management ──────────────────────────────────────────

    async def ontology_list(self, current_id: str) -> list[dict]:
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?g ?base ?name (COUNT(DISTINCT ?cls) AS ?classes) WHERE {{
            GRAPH ?g {{
                ?base a owl:Ontology .
                OPTIONAL {{ ?base rdfs:label ?name }}
                OPTIONAL {{ ?cls a owl:Class }}
            }}
            FILTER(STRSTARTS(STR(?g), "urn:olaf:"))
            FILTER(!CONTAINS(STR(?g), ":seed:"))
        }}
        GROUP BY ?g ?base ?name
        """, ["g", "base", "name", "classes"])
        return [
            {
                "id": (row["g"] or "")[len("urn:olaf:"):],
                "name": row["name"] or "",
                "base_uri": (row["base"] or "") + "#",
                "classes": int(row["classes"] or 0),
                "active": (row["g"] or "") == self._graph(current_id),
            }
            for row in rows
        ]

    async def ontology_create(self, ontology_id: str, name: str, base_uri: str) -> dict:
        base = _check_uri(base_uri.rstrip("#/"))
        graph = self._graph(ontology_id)

        if await self._ex.execute_ask(f"ASK {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"):
            return {"id": ontology_id, "created": False}

        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{
                <{base}> a owl:Ontology ;
                    rdfs:label "{_esc(name)}" .
            }}
        }}
        """)
        self._meta[ontology_id] = (base, name)
        return {"id": ontology_id, "created": True, "base_uri": base + "#"}

    async def ontology_switch(self, ontology_id: str) -> dict:
        graph = self._graph(ontology_id)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?base ?name WHERE {{
            GRAPH <{graph}> {{
                ?base a owl:Ontology .
                OPTIONAL {{ ?base rdfs:label ?name }}
            }}
        }}
        """, ["base", "name"])
        if not rows:
            raise ValueError(f"Ontology '{ontology_id}' not found. Use ontology_create first.")

        base = rows[0]["base"].rstrip("#/") if rows[0]["base"] else await self._resolve_base(ontology_id)
        name = rows[0]["name"] or ""
        self._meta[ontology_id] = (base, name)
        return {"active_id": ontology_id, "name": name, "base_uri": base + "#"}

    async def drop_ontology(self, ontology_id: str) -> dict:
        """Drop an ontology graph. Seeds are independent and unaffected. Intended for CLI use only."""
        graph = self._graph(ontology_id)
        await self._ex.execute_update(f"DROP SILENT GRAPH <{graph}>")
        self._meta.pop(ontology_id, None)
        return {"dropped": ontology_id}

    async def seed_list(self) -> list[dict]:
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?g (COUNT(*) AS ?triples) WHERE {{
            GRAPH ?g {{ ?s ?p ?o }}
            FILTER(STRSTARTS(STR(?g), "urn:olaf:seed:"))
        }}
        GROUP BY ?g
        """, ["g", "triples"])
        return [
            {
                "id": (row["g"] or "")[len("urn:olaf:seed:"):],
                "graph_uri": row["g"] or "",
                "triples": int(row["triples"] or 0),
            }
            for row in rows
        ]

    async def seed_drop(self, seed_id: str) -> dict:
        """Drop a global seed graph. Intended for CLI use only."""
        graph = f"urn:olaf:seed:{seed_id}"
        _check_uri(graph)
        await self._ex.execute_update(f"DROP SILENT GRAPH <{graph}>")
        return {"dropped": seed_id}

    # ── Concepts (owl:Class) ───────────────────────────────────────────────

    async def concept_create(
        self,
        ontology_id: str,
        label: str,
        definition: str,
        parent_uri: str | None = None,
        aliases: list[str] | None = None,
        lang: str = "en",
        source_chunk_id: str | None = None,
    ) -> dict:
        graph = self._graph(ontology_id)
        uri = await self._uri(_slugify_class(label), ontology_id)
        # Guard against any pre-existing entity (class or individual) at this URI.
        if await self._ex.execute_ask(f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{uri}> ?p ?o }} }}"):
            return {"uri": uri, "created": False}

        lines = [
            f"<{uri}> a owl:Class",
            f'<{uri}> rdfs:label "{_esc(label)}"@{lang}',
            f'<{uri}> skos:definition "{_esc(definition)}"@{lang}',
        ]
        if parent_uri:
            lines.append(f"<{uri}> rdfs:subClassOf <{_check_uri(parent_uri)}>")
        for alias in (aliases or []):
            lines.append(f'<{uri}> rdfs:altLabel "{_esc(alias)}"@{lang}')
        if source_chunk_id is not None:
            lines.append(f"<{uri}> <urn:olaf:extractedFrom> <{_chunk_uri(source_chunk_id)}>")

        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{
                {" .\n                ".join(lines)} .
            }}
        }}
        """)
        return {"uri": uri, "created": True}

    async def concept_get(self, ontology_id: str, uri: str) -> dict | None:
        _check_uri(uri)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?p ?o WHERE {{ GRAPH <{self._graph(ontology_id)}> {{ <{uri}> ?p ?o }} }}
        """, ["p", "o"])
        if not rows:
            return None

        result: dict = {"uri": uri, "triples": [], "source_chunk_ids": []}
        for row in rows:
            result["triples"].append({"predicate": row["p"] or "", "object": row["o"] or ""})

        for t in result["triples"]:
            if t["predicate"] == "http://www.w3.org/2000/01/rdf-schema#label" and "label" not in result:
                result["label"] = t["object"]
            if t["predicate"] == "http://www.w3.org/2004/02/skos/core#definition" and "definition" not in result:
                result["definition"] = t["object"]
            if t["predicate"] == "urn:olaf:extractedFrom":
                result["source_chunk_ids"].append(_chunk_id_from_uri(t["object"]))

        return result

    async def concept_search_sparql(self, ontology_id: str, query_str: str, top_k: int = 10) -> list[dict]:
        q = _esc(query_str.lower())
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT DISTINCT ?c ?label ?definition ?chunk WHERE {{
            GRAPH <{self._graph(ontology_id)}> {{
                ?c a owl:Class ; rdfs:label ?label .
                OPTIONAL {{ ?c skos:definition ?definition }}
                OPTIONAL {{ ?c <urn:olaf:extractedFrom> ?chunk }}
                FILTER(CONTAINS(LCASE(STR(?label)), "{q}"))
            }}
        }}
        LIMIT {top_k}
        """, ["c", "label", "definition", "chunk"])
        return [
            {
                "uri": row["c"] or "",
                "label": row["label"] or "",
                "definition": row["definition"] or "",
                "source_chunk_id": _chunk_id_from_uri(row["chunk"]) if row["chunk"] else None,
                "match_type": "sparql",
            }
            for row in rows
        ]

    async def concept_update(
        self,
        ontology_id: str,
        uri: str,
        label: str | None = None,
        definition: str | None = None,
        aliases_add: list[str] | None = None,
        aliases_remove: list[str] | None = None,
        lang: str = "en",
    ) -> None:
        _check_uri(uri)
        graph = self._graph(ontology_id)
        if label is not None:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{uri}> rdfs:label ?o }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{uri}> rdfs:label ?o }} }}
            """)
            if label:
                await self._ex.execute_update(f"""
                {_PREFIXES}
                INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:label "{_esc(label)}"@{lang} }} }}
                """)
        if definition is not None:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{uri}> skos:definition ?o }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{uri}> skos:definition ?o }} }}
            """)
            if definition:
                await self._ex.execute_update(f"""
                {_PREFIXES}
                INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> skos:definition "{_esc(definition)}"@{lang} }} }}
                """)
        for alias in (aliases_add or []):
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:altLabel "{_esc(alias)}"@{lang} }} }}
            """)
        for alias in (aliases_remove or []):
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:altLabel "{_esc(alias)}"@{lang} }} }}
            """)

    async def concept_merge(self, ontology_id: str, keep_uri: str, merge_uri: str) -> None:
        _check_uri(keep_uri)
        _check_uri(merge_uri)
        graph = self._graph(ontology_id)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT {{
            GRAPH <{graph}> {{ <{keep_uri}> ?p ?o }}
        }}
        WHERE {{
            GRAPH <{graph}> {{
                <{merge_uri}> ?p ?o .
                FILTER NOT EXISTS {{ GRAPH <{graph}> {{ <{keep_uri}> ?p ?o }} }}
            }}
        }}
        """)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{graph}> {{ ?s ?p <{merge_uri}> }} }}
        INSERT {{ GRAPH <{graph}> {{ ?s ?p <{keep_uri}> }} }}
        WHERE  {{ GRAPH <{graph}> {{ ?s ?p <{merge_uri}> }} }}
        """)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{graph}> {{ <{merge_uri}> ?p ?o }} }}
        WHERE  {{ GRAPH <{graph}> {{ <{merge_uri}> ?p ?o }} }}
        """)

    async def concept_list(self, ontology_id: str, root_only: bool = False, limit: int = 100) -> list[dict]:
        root_filter = ""
        if root_only:
            root_filter = """
                FILTER NOT EXISTS {
                    ?c rdfs:subClassOf ?anyParent .
                    FILTER(!isBlank(?anyParent))
                    FILTER(?anyParent != owl:Thing)
                }"""

        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?c (SAMPLE(?lbl) AS ?label) (SAMPLE(?def) AS ?definition) (SAMPLE(?par) AS ?parent) WHERE {{
            GRAPH <{self._graph(ontology_id)}> {{
                ?c a owl:Class .
                OPTIONAL {{ ?c rdfs:label ?lbl }}
                OPTIONAL {{ ?c skos:definition ?def }}
                OPTIONAL {{
                    ?c rdfs:subClassOf ?par .
                    FILTER(!isBlank(?par))
                    FILTER(?par != owl:Thing)
                }}
                {root_filter}
            }}
        }}
        GROUP BY ?c
        LIMIT {limit}
        """, ["c", "label", "definition", "parent"])

        return [
            {
                "uri": row["c"] or "",
                "label": row["label"] or "",
                "definition": row["definition"] or "",
                "parent_uri": row["parent"],
            }
            for row in rows
        ]

    # ── Individuals (owl:NamedIndividual) ─────────────────────────────────

    async def individual_create(
        self,
        ontology_id: str,
        label: str,
        class_uri: str,
        definition: str | None = None,
        aliases: list[str] | None = None,
        lang: str = "en",
        source_chunk_id: str | None = None,
    ) -> dict:
        _check_uri(class_uri)
        graph = self._graph(ontology_id)
        uri = await self._uri(_slugify_class(label), ontology_id)

        # Guard against any pre-existing entity (class or individual) at this URI.
        if await self._ex.execute_ask(
            f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{uri}> ?p ?o }} }}"
        ):
            return {"uri": uri, "created": False}

        lines = [
            f"<{uri}> a owl:NamedIndividual",
            f"<{uri}> a <{class_uri}>",
            f'<{uri}> rdfs:label "{_esc(label)}"@{lang}',
        ]
        if definition:
            lines.append(f'<{uri}> skos:definition "{_esc(definition)}"@{lang}')
        for alias in (aliases or []):
            lines.append(f'<{uri}> rdfs:altLabel "{_esc(alias)}"@{lang}')
        if source_chunk_id is not None:
            lines.append(f"<{uri}> <urn:olaf:extractedFrom> <{_chunk_uri(source_chunk_id)}>")

        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{
                {" .\n                ".join(lines)} .
            }}
        }}
        """)
        return {"uri": uri, "created": True}

    # ── Properties ────────────────────────────────────────────────────────

    async def property_create(
        self,
        ontology_id: str,
        label: str,
        prop_type: str,
        domain_uri: str | None = None,
        range_uri: str | None = None,
        parent_uri: str | None = None,
        lang: str = "en",
    ) -> dict:
        graph = self._graph(ontology_id)
        uri = await self._uri(_slugify_prop(label), ontology_id)
        owl_type = "owl:ObjectProperty" if prop_type == "object" else "owl:DatatypeProperty"

        if await self._ex.execute_ask(f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{uri}> a {owl_type} }} }}"):
            return {"uri": uri, "created": False}

        lines = [
            f"<{uri}> a {owl_type}",
            f'<{uri}> rdfs:label "{_esc(label)}"@{lang}',
        ]
        if domain_uri:
            lines.append(f"<{uri}> rdfs:domain <{_check_uri(domain_uri)}>")
        if range_uri:
            lines.append(f"<{uri}> rdfs:range <{_check_uri(range_uri)}>")
        if parent_uri:
            lines.append(f"<{uri}> rdfs:subPropertyOf <{_check_uri(parent_uri)}>")

        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{
                {" .\n                ".join(lines)} .
            }}
        }}
        """)
        return {"uri": uri, "created": True}

    async def property_update(
        self,
        ontology_id: str,
        uri: str,
        domain_uri: str | None = None,
        range_uri: str | None = None,
        parent_uri: str | None = None,
    ) -> None:
        _check_uri(uri)
        graph = self._graph(ontology_id)
        if domain_uri is not None:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{uri}> rdfs:domain ?o }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{uri}> rdfs:domain ?o }} }}
            """)
            if domain_uri:
                await self._ex.execute_update(f"""
                {_PREFIXES}
                INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:domain <{_check_uri(domain_uri)}> }} }}
                """)
        if range_uri is not None:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{uri}> rdfs:range ?o }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{uri}> rdfs:range ?o }} }}
            """)
            if range_uri:
                await self._ex.execute_update(f"""
                {_PREFIXES}
                INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:range <{_check_uri(range_uri)}> }} }}
                """)
        if parent_uri is not None:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{uri}> rdfs:subPropertyOf ?o }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{uri}> rdfs:subPropertyOf ?o }} }}
            """)
            if parent_uri:
                await self._ex.execute_update(f"""
                {_PREFIXES}
                INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:subPropertyOf <{_check_uri(parent_uri)}> }} }}
                """)

    async def property_get(self, ontology_id: str, uri: str) -> dict | None:
        _check_uri(uri)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?p ?o WHERE {{ GRAPH <{self._graph(ontology_id)}> {{ <{uri}> ?p ?o }} }}
        """, ["p", "o"])
        if not rows:
            return None
        return {"uri": uri, "triples": [{"predicate": row["p"] or "", "object": row["o"] or ""} for row in rows]}

    async def property_search(self, ontology_id: str, query_str: str, top_k: int = 10) -> list[dict]:
        q = _esc(query_str.lower())
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT DISTINCT ?p ?label ?type WHERE {{
            GRAPH <{self._graph(ontology_id)}> {{
                ?p rdfs:label ?label .
                {{
                    ?p a owl:ObjectProperty .
                    BIND("object" AS ?type)
                }}
                UNION
                {{
                    ?p a owl:DatatypeProperty .
                    BIND("datatype" AS ?type)
                }}
                FILTER(CONTAINS(LCASE(STR(?label)), "{q}"))
            }}
        }}
        LIMIT {top_k}
        """, ["p", "label", "type"])
        return [
            {"uri": row["p"] or "", "label": row["label"] or "", "type": row["type"] or ""}
            for row in rows
        ]

    # ── Relations ─────────────────────────────────────────────────────────

    async def _require_existing(self, graph: str, base: str, uri: str) -> None:
        """Reject URIs under our own base namespace that don't correspond to any existing
        entity — catches hand-typed/mis-slugified URIs before they create a silent duplicate
        or a dangling reference. External URIs (seeds, well-known vocab) are not checked."""
        if not uri.startswith(base):
            return
        exists = await self._ex.execute_ask(f"""
        {_PREFIXES}
        ASK {{ GRAPH <{graph}> {{ {{ <{uri}> ?p ?o }} UNION {{ ?s ?p2 <{uri}> }} }} }}
        """)
        if not exists:
            raise ValueError(
                f"URI not found in this ontology: {uri!r}. If this entity doesn't exist yet, "
                "create it first with concept_create/individual_create/property_create. "
                "If it should already exist, use concept_search/concept_get/property_search "
                "to find its exact URI instead of guessing it."
            )

    async def relation_add(
        self,
        ontology_id: str,
        subject_uri: str,
        property_uri: str,
        object_value: str,
        is_literal: bool = False,
        datatype: str | None = None,
    ) -> None:
        _check_uri(subject_uri)
        _check_uri(property_uri)
        graph = self._graph(ontology_id)
        base = await self._resolve_base(ontology_id)
        await self._require_existing(graph, base, subject_uri)
        await self._require_existing(graph, base, property_uri)
        if is_literal:
            obj = f'"{_esc(object_value)}"^^<{_check_uri(datatype)}>' if datatype else f'"{_esc(object_value)}"'
        else:
            _check_uri(object_value)
            await self._require_existing(graph, base, object_value)
            obj = f"<{object_value}>"
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{ <{subject_uri}> <{property_uri}> {obj} }}
        }}
        """)

    async def relation_delete(
        self,
        ontology_id: str,
        subject_uri: str,
        property_uri: str,
        object_value: str,
        is_literal: bool = False,
        datatype: str | None = None,
    ) -> None:
        _check_uri(subject_uri)
        _check_uri(property_uri)
        if is_literal:
            obj = f'"{_esc(object_value)}"^^<{_check_uri(datatype)}>' if datatype else f'"{_esc(object_value)}"'
        else:
            obj = f"<{_check_uri(object_value)}>"
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE DATA {{
            GRAPH <{self._graph(ontology_id)}> {{ <{subject_uri}> <{property_uri}> {obj} }}
        }}
        """)

    async def relation_search(
        self,
        ontology_id: str,
        subject_uri: str | None = None,
        property_uri: str | None = None,
        object_uri: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        filters = []
        if subject_uri:
            filters.append(f"FILTER(?s = <{_check_uri(subject_uri)}>)")
        if property_uri:
            filters.append(f"FILTER(?p = <{_check_uri(property_uri)}>)")
        if object_uri:
            filters.append(f"FILTER(?o = <{_check_uri(object_uri)}>)")
        filter_block = "\n                ".join(filters)

        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?s ?p ?o WHERE {{
            GRAPH <{self._graph(ontology_id)}> {{
                ?s ?p ?o .
                {filter_block}
            }}
        }}
        LIMIT {limit}
        """, ["s", "p", "o"])
        return [
            {"subject": row["s"] or "", "property": row["p"] or "", "object": row["o"] or ""}
            for row in rows
        ]

    # ── OWL Restrictions ──────────────────────────────────────────────────

    async def restriction_add(
        self,
        ontology_id: str,
        class_uri: str,
        property_uri: str,
        restriction_type: str,
        value: str,
        cardinality: int | None = None,
        is_literal_value: bool = False,
    ) -> str:
        if restriction_type not in _RESTRICTION_PROP:
            raise ValueError(f"Unknown restriction_type '{restriction_type}'. Valid: {list(_RESTRICTION_PROP)}")
        _check_uri(class_uri)
        _check_uri(property_uri)
        if value and not is_literal_value:
            _check_uri(value)

        graph = self._graph(ontology_id)
        # INSERT DATA forbids blank nodes (SPARQL 1.1 §3.1.1).
        # INSERT { } WHERE {} creates a fresh blank node per execution.
        bnode = "_:r"

        if restriction_type in ("exactly", "min", "max"):
            if cardinality is None:
                raise ValueError(f"restriction_type='{restriction_type}' requires cardinality")
            if value:
                prop_name = _RESTRICTION_PROP[restriction_type]
                extra = f"{bnode} owl:onClass <{value}> ."
            else:
                prop_name = _UNQUALIFIED_RESTRICTION_PROP[restriction_type]
                extra = ""
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT {{
                GRAPH <{graph}> {{
                    <{class_uri}> rdfs:subClassOf {bnode} .
                    {bnode} a owl:Restriction ;
                             owl:onProperty <{property_uri}> ;
                             {prop_name} "{cardinality}"^^xsd:nonNegativeInteger .
                    {extra}
                }}
            }}
            WHERE {{}}
            """)
        elif is_literal_value:
            prop_name = _RESTRICTION_PROP[restriction_type]
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT {{
                GRAPH <{graph}> {{
                    <{class_uri}> rdfs:subClassOf {bnode} .
                    {bnode} a owl:Restriction ;
                             owl:onProperty <{property_uri}> ;
                             {prop_name} "{_esc(value)}" .
                }}
            }}
            WHERE {{}}
            """)
        else:
            prop_name = _RESTRICTION_PROP[restriction_type]
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT {{
                GRAPH <{graph}> {{
                    <{class_uri}> rdfs:subClassOf {bnode} .
                    {bnode} a owl:Restriction ;
                             owl:onProperty <{property_uri}> ;
                             {prop_name} <{value}> .
                }}
            }}
            WHERE {{}}
            """)
        return bnode

    # ── Validation ────────────────────────────────────────────────────────

    async def orphans(self, ontology_id: str, limit: int = 100) -> list[dict]:
        """Classes/individuals with no relation to the rest of the graph beyond their own
        rdf:type/label/altLabel/definition/extractedFrom bookkeeping triples."""
        graph = self._graph(ontology_id)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?e (SAMPLE(?lbl) AS ?label) (SAMPLE(?knd) AS ?kind) WHERE {{
            GRAPH <{graph}> {{
                {{ ?e a owl:Class . BIND("class" AS ?knd) }}
                UNION
                {{ ?e a owl:NamedIndividual . BIND("individual" AS ?knd) }}
                OPTIONAL {{ ?e rdfs:label ?lbl }}
                FILTER NOT EXISTS {{
                    ?e ?p ?o .
                    FILTER(?p NOT IN (rdf:type, rdfs:label, rdfs:altLabel, skos:definition, <urn:olaf:extractedFrom>))
                }}
                FILTER NOT EXISTS {{ ?s ?p2 ?e }}
            }}
        }}
        GROUP BY ?e
        LIMIT {limit}
        """, ["e", "label", "kind"])
        return [
            {"uri": row["e"] or "", "label": row["label"] or "", "kind": row["kind"] or ""}
            for row in rows
        ]

    # ── Seeds ─────────────────────────────────────────────────────────────

    async def seed_load(
        self,
        graph_id: str | None = None,
        url: str | None = None,
        content: str | None = None,
    ) -> dict:
        import httpx

        if url is None and content is None:
            raise ValueError("Provide either 'url' or 'content'.")

        if graph_id is None:
            graph_id = f"urn:olaf:seed:{uuid.uuid4().hex[:8]}"
        else:
            _check_uri(graph_id)

        if url is not None:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"seed URL must use http or https (got {parsed.scheme!r})")
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                data = resp.content
        else:
            data = content.encode()

        await self._ex.load_graph(data, graph_id)

        rows = await self._ex.execute_select(
            f"SELECT (COUNT(*) AS ?n) WHERE {{ GRAPH <{graph_id}> {{ ?s ?p ?o }} }}",
            ["n"],
        )
        count = int(rows[0]["n"]) if rows and rows[0]["n"] is not None else 0
        source = url or "(inline content)"
        return {"graph_id": graph_id, "source": source, "triples_loaded": count}

    # ── Export ────────────────────────────────────────────────────────────

    async def export_ttl(self, ontology_id: str, include_seeds: bool = False) -> str:
        result = (await self._ex.dump_graph(self._graph(ontology_id))).decode()

        if include_seeds:
            rows = await self._ex.execute_select(
                "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }", ["g"]
            )
            for row in rows:
                g = row["g"] or ""
                if g.startswith("urn:olaf:seed:"):
                    result += f"\n# Seed graph: {g}\n" + (await self._ex.dump_graph(g)).decode()

        return result

    # ── Summary ───────────────────────────────────────────────────────────

    async def summary(self, ontology_id: str) -> dict:
        graph = self._graph(ontology_id)
        root_rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?c ?label WHERE {{
            GRAPH <{graph}> {{
                ?c a owl:Class .
                OPTIONAL {{ ?c rdfs:label ?label }}
                FILTER NOT EXISTS {{
                    ?c rdfs:subClassOf ?parent .
                    FILTER(!isBlank(?parent))
                    FILTER(?parent != owl:Thing)
                }}
            }}
        }}
        LIMIT 20
        """, ["c", "label"])
        root_classes = [
            {"uri": row["c"] or "", "label": row["label"] or row["c"] or ""}
            for row in root_rows
        ]

        counts_rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?classes ?object_properties ?datatype_properties ?individuals ?restrictions WHERE {{
            {{ SELECT (COUNT(DISTINCT ?x) AS ?classes)            WHERE {{ GRAPH <{graph}> {{ ?x a owl:Class }} }} }}
            {{ SELECT (COUNT(DISTINCT ?x) AS ?object_properties)  WHERE {{ GRAPH <{graph}> {{ ?x a owl:ObjectProperty }} }} }}
            {{ SELECT (COUNT(DISTINCT ?x) AS ?datatype_properties) WHERE {{ GRAPH <{graph}> {{ ?x a owl:DatatypeProperty }} }} }}
            {{ SELECT (COUNT(DISTINCT ?x) AS ?individuals)         WHERE {{ GRAPH <{graph}> {{ ?x a owl:NamedIndividual }} }} }}
            {{ SELECT (COUNT(DISTINCT ?x) AS ?restrictions)        WHERE {{ GRAPH <{graph}> {{ ?x a owl:Restriction }} }} }}
        }}
        """, ["classes", "object_properties", "datatype_properties", "individuals", "restrictions"])
        counts = {k: int(v or 0) for k, v in (counts_rows[0] if counts_rows else {}).items()}

        return {
            "active_ontology": ontology_id,
            **counts,
            "root_classes": root_classes,
        }
