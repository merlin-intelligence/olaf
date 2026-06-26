from __future__ import annotations
import re
import uuid

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


def _fmt_http(binding: dict | None) -> str | None:
    if binding is None:
        return None
    if binding["type"] == "literal":
        lang = binding.get("xml:lang", "")
        return f"{binding['value']}@{lang}" if lang else binding["value"]
    return binding["value"]


# ── HTTP executor ──────────────────────────────────────────────────────────────

class _HttpExecutor:
    def __init__(self, url: str):
        import httpx
        self._url = url.rstrip("/")
        self._client = httpx.Client(timeout=30)

    def execute_select(self, sparql: str, variables: list[str]) -> list[dict[str, str | None]]:
        resp = self._client.post(
            f"{self._url}/query",
            content=sparql.encode(),
            headers={"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"},
        )
        resp.raise_for_status()
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])
        return [{v: _fmt_http(b.get(v)) for v in variables} for b in bindings]

    def execute_ask(self, sparql: str) -> bool:
        resp = self._client.post(
            f"{self._url}/query",
            content=sparql.encode(),
            headers={"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"},
        )
        resp.raise_for_status()
        return bool(resp.json().get("boolean", False))

    def execute_update(self, sparql: str) -> None:
        resp = self._client.post(
            f"{self._url}/update",
            content=sparql.encode(),
            headers={"Content-Type": "application/sparql-update"},
        )
        resp.raise_for_status()

    def load_graph(self, data: bytes, graph_uri: str) -> None:
        resp = self._client.put(
            f"{self._url}/store",
            params={"graph": graph_uri},
            content=data,
            headers={"Content-Type": "text/turtle"},
        )
        resp.raise_for_status()

    def dump_graph(self, graph_uri: str) -> bytes:
        resp = self._client.get(
            f"{self._url}/store",
            params={"graph": graph_uri},
            headers={"Accept": "text/turtle"},
        )
        resp.raise_for_status()
        return resp.content


# ── OntologyStore ──────────────────────────────────────────────────────────────

class OntologyStore:
    def __init__(self, url: str, base_uri: str, name: str = "Ontology", ontology_id: str = "main"):
        self._ex = _HttpExecutor(url)
        self.base = base_uri.rstrip("#/")
        self.name = name
        self._id = ontology_id
        self._bootstrap()

    # Named graph helpers — depend on active ontology

    @property
    def _main(self) -> str:
        return f"urn:olaf:{self._id}"

    def _uri(self, local: str) -> str:
        return f"{self.base}#{local}"

    def _count(self, pattern: str) -> int:
        q = f"{_PREFIXES}\nSELECT (COUNT(DISTINCT ?x) AS ?n) WHERE {{ GRAPH <{self._main}> {{ {pattern} }} }}"
        rows = self._ex.execute_select(q, ["n"])
        if rows and rows[0]["n"] is not None:
            return int(rows[0]["n"])
        return 0

    def _bootstrap(self) -> None:
        if not self._ex.execute_ask(
            f"{_PREFIXES}\nASK {{ GRAPH <{self._main}> {{ <{self.base}> a owl:Ontology }} }}"
        ):
            self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{
                GRAPH <{self._main}> {{
                    <{self.base}> a owl:Ontology ;
                        rdfs:label "{_esc(self.name)}" .
                }}
            }}
            """)

    # ── Multi-ontology management ──────────────────────────────────────────

    def ontology_list(self) -> list[dict]:
        rows = self._ex.execute_select(f"""
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
                "active": (row["g"] or "") == self._main,
            }
            for row in rows
        ]

    def ontology_create(self, ontology_id: str, name: str, base_uri: str | None = None) -> dict:
        graph = f"urn:olaf:{ontology_id}"
        actual_base = (base_uri.rstrip("#/") if base_uri else self.base)

        if self._ex.execute_ask(f"ASK {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"):
            return {"id": ontology_id, "created": False}

        self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{
                <{actual_base}> a owl:Ontology ;
                    rdfs:label "{_esc(name)}" .
            }}
        }}
        """)
        return {"id": ontology_id, "created": True, "base_uri": actual_base + "#"}

    def ontology_switch(self, ontology_id: str) -> dict:
        graph = f"urn:olaf:{ontology_id}"
        rows = self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?base ?name WHERE {{
            GRAPH <{graph}> {{
                ?base a owl:Ontology .
                OPTIONAL {{ ?base rdfs:label ?name }}
            }}
        }}
        """, ["base", "name"])
        if not rows:
            raise ValueError(f"Ontologie '{ontology_id}' introuvable. Utilisez ontology_create d'abord.")

        self._id = ontology_id
        if rows[0]["base"]:
            self.base = rows[0]["base"].rstrip("#/")
        if rows[0]["name"]:
            self.name = rows[0]["name"]
        return {"active_id": self._id, "name": self.name, "base_uri": self.base + "#"}

    def drop_ontology(self, ontology_id: str) -> dict:
        """Drop an ontology graph. Seeds are independent and unaffected. Intended for CLI use only."""
        self._ex.execute_update(f"DROP SILENT GRAPH <urn:olaf:{ontology_id}>")
        return {"dropped": ontology_id}

    def seed_list(self) -> list[dict]:
        rows = self._ex.execute_select(f"""
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

    def seed_drop(self, seed_id: str) -> dict:
        """Drop a global seed graph. Intended for CLI use only."""
        graph = f"urn:olaf:seed:{seed_id}"
        self._ex.execute_update(f"DROP SILENT GRAPH <{graph}>")
        return {"dropped": seed_id}

    # ── Concepts (owl:Class) ───────────────────────────────────────────────

    def concept_create(
        self,
        label: str,
        definition: str,
        parent_uri: str | None = None,
        aliases: list[str] | None = None,
        lang: str = "en",
        source_chunk_id: str | None = None,
    ) -> dict:
        uri = self._uri(_slugify_class(label))
        if self._ex.execute_ask(f"{_PREFIXES}\nASK {{ GRAPH <{self._main}> {{ <{uri}> a owl:Class }} }}"):
            return {"uri": uri, "created": False}

        lines = [
            f"<{uri}> a owl:Class",
            f'<{uri}> rdfs:label "{_esc(label)}"@{lang}',
            f'<{uri}> skos:definition "{_esc(definition)}"@{lang}',
        ]
        if parent_uri:
            lines.append(f"<{uri}> rdfs:subClassOf <{parent_uri}>")
        for alias in (aliases or []):
            lines.append(f'<{uri}> rdfs:altLabel "{_esc(alias)}"@{lang}')
        if source_chunk_id is not None:
            lines.append(f'<{uri}> <urn:olaf:extractedFrom> "{_esc(source_chunk_id)}"')

        self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{self._main}> {{
                {" .\n                ".join(lines)} .
            }}
        }}
        """)
        return {"uri": uri, "created": True}

    def concept_get(self, uri: str) -> dict | None:
        rows = self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?p ?o WHERE {{ GRAPH <{self._main}> {{ <{uri}> ?p ?o }} }}
        """, ["p", "o"])
        if not rows:
            return None

        result: dict = {"uri": uri, "triples": [], "source_chunk_ids": []}
        for row in rows:
            result["triples"].append({"predicate": row["p"] or "", "object": row["o"] or ""})

        for t in result["triples"]:
            if t["predicate"].endswith("label") and "label" not in result:
                result["label"] = t["object"]
            if t["predicate"].endswith("definition") and "definition" not in result:
                result["definition"] = t["object"]
            if t["predicate"] == "urn:olaf:extractedFrom":
                result["source_chunk_ids"].append(t["object"])

        return result

    def concept_search_sparql(self, query_str: str, top_k: int = 10) -> list[dict]:
        q = _esc(query_str.lower())
        rows = self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT DISTINCT ?c ?label ?definition ?chunk WHERE {{
            GRAPH <{self._main}> {{
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
                "source_chunk_id": row["chunk"],
                "match_type": "sparql",
            }
            for row in rows
        ]

    def concept_update(
        self,
        uri: str,
        label: str | None = None,
        definition: str | None = None,
        aliases_add: list[str] | None = None,
        aliases_remove: list[str] | None = None,
        lang: str = "en",
    ) -> None:
        if label:
            self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{self._main}> {{ <{uri}> rdfs:label ?o }} }}
            WHERE  {{ GRAPH <{self._main}> {{ <{uri}> rdfs:label ?o }} }}
            """)
            self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{ GRAPH <{self._main}> {{ <{uri}> rdfs:label "{_esc(label)}"@{lang} }} }}
            """)
        if definition:
            self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{self._main}> {{ <{uri}> skos:definition ?o }} }}
            WHERE  {{ GRAPH <{self._main}> {{ <{uri}> skos:definition ?o }} }}
            """)
            self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{ GRAPH <{self._main}> {{ <{uri}> skos:definition "{_esc(definition)}"@{lang} }} }}
            """)
        for alias in (aliases_add or []):
            self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{ GRAPH <{self._main}> {{ <{uri}> rdfs:altLabel "{_esc(alias)}"@{lang} }} }}
            """)
        for alias in (aliases_remove or []):
            self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE DATA {{ GRAPH <{self._main}> {{ <{uri}> rdfs:altLabel "{_esc(alias)}"@{lang} }} }}
            """)

    def concept_merge(self, keep_uri: str, merge_uri: str) -> None:
        self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT {{
            GRAPH <{self._main}> {{ <{keep_uri}> ?p ?o }}
        }}
        WHERE {{
            GRAPH <{self._main}> {{
                <{merge_uri}> ?p ?o .
                FILTER(?p != rdf:type)
                FILTER NOT EXISTS {{ <{keep_uri}> rdfs:label ?existing . FILTER(?p = rdfs:label) }}
            }}
        }}
        """)
        self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{self._main}> {{ ?s ?p <{merge_uri}> }} }}
        INSERT {{ GRAPH <{self._main}> {{ ?s ?p <{keep_uri}> }} }}
        WHERE  {{ GRAPH <{self._main}> {{ ?s ?p <{merge_uri}> }} }}
        """)
        self._ex.execute_update(f"""
        DELETE {{ GRAPH <{self._main}> {{ <{merge_uri}> ?p ?o }} }}
        WHERE  {{ GRAPH <{self._main}> {{ <{merge_uri}> ?p ?o }} }}
        """)

    # ── Properties ────────────────────────────────────────────────────────

    def property_create(
        self,
        label: str,
        prop_type: str,
        domain_uri: str | None = None,
        range_uri: str | None = None,
        parent_uri: str | None = None,
        lang: str = "en",
    ) -> dict:
        uri = self._uri(_slugify_prop(label))
        owl_type = "owl:ObjectProperty" if prop_type == "object" else "owl:DatatypeProperty"

        if self._ex.execute_ask(f"{_PREFIXES}\nASK {{ GRAPH <{self._main}> {{ <{uri}> a {owl_type} }} }}"):
            return {"uri": uri, "created": False}

        lines = [
            f"<{uri}> a {owl_type}",
            f'<{uri}> rdfs:label "{_esc(label)}"@{lang}',
        ]
        if domain_uri:
            lines.append(f"<{uri}> rdfs:domain <{domain_uri}>")
        if range_uri:
            lines.append(f"<{uri}> rdfs:range <{range_uri}>")
        if parent_uri:
            lines.append(f"<{uri}> rdfs:subPropertyOf <{parent_uri}>")

        self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{self._main}> {{
                {" .\n                ".join(lines)} .
            }}
        }}
        """)
        return {"uri": uri, "created": True}

    def property_get(self, uri: str) -> dict | None:
        rows = self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?p ?o WHERE {{ GRAPH <{self._main}> {{ <{uri}> ?p ?o }} }}
        """, ["p", "o"])
        if not rows:
            return None
        return {"uri": uri, "triples": [{"predicate": row["p"] or "", "object": row["o"] or ""} for row in rows]}

    def property_search(self, query_str: str, top_k: int = 10) -> list[dict]:
        q = _esc(query_str.lower())
        rows = self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT DISTINCT ?p ?label ?type WHERE {{
            GRAPH <{self._main}> {{
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

    def relation_add(
        self,
        subject_uri: str,
        property_uri: str,
        object_value: str,
        is_literal: bool = False,
        datatype: str | None = None,
    ) -> None:
        if is_literal:
            obj = f'"{_esc(object_value)}"^^<{datatype}>' if datatype else f'"{_esc(object_value)}"'
        else:
            obj = f"<{object_value}>"
        self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{self._main}> {{ <{subject_uri}> <{property_uri}> {obj} }}
        }}
        """)

    def relation_search(
        self,
        subject_uri: str | None = None,
        property_uri: str | None = None,
        object_uri: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        filters = []
        if subject_uri:
            filters.append(f"FILTER(?s = <{subject_uri}>)")
        if property_uri:
            filters.append(f"FILTER(?p = <{property_uri}>)")
        if object_uri:
            filters.append(f"FILTER(?o = <{object_uri}>)")
        filter_block = "\n                ".join(filters)

        rows = self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?s ?p ?o WHERE {{
            GRAPH <{self._main}> {{
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

    def restriction_add(
        self,
        class_uri: str,
        property_uri: str,
        restriction_type: str,
        value: str,
        cardinality: int | None = None,
        is_literal_value: bool = False,
    ) -> str:
        if restriction_type not in _RESTRICTION_PROP:
            raise ValueError(f"Unknown restriction_type '{restriction_type}'. Valid: {list(_RESTRICTION_PROP)}")

        bnode = f"_:r{uuid.uuid4().hex}"

        if restriction_type in ("exactly", "min", "max"):
            if cardinality is None:
                raise ValueError(f"restriction_type='{restriction_type}' requires cardinality")
            if value:
                prop_name = _RESTRICTION_PROP[restriction_type]
                extra = f"{bnode} owl:onClass <{value}> ."
            else:
                prop_name = _UNQUALIFIED_RESTRICTION_PROP[restriction_type]
                extra = ""
            self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{
                GRAPH <{self._main}> {{
                    <{class_uri}> rdfs:subClassOf {bnode} .
                    {bnode} a owl:Restriction ;
                             owl:onProperty <{property_uri}> ;
                             {prop_name} "{cardinality}"^^xsd:nonNegativeInteger .
                    {extra}
                }}
            }}
            """)
        elif is_literal_value:
            prop_name = _RESTRICTION_PROP[restriction_type]
            self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{
                GRAPH <{self._main}> {{
                    <{class_uri}> rdfs:subClassOf {bnode} .
                    {bnode} a owl:Restriction ;
                             owl:onProperty <{property_uri}> ;
                             {prop_name} "{_esc(value)}" .
                }}
            }}
            """)
        else:
            prop_name = _RESTRICTION_PROP[restriction_type]
            self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{
                GRAPH <{self._main}> {{
                    <{class_uri}> rdfs:subClassOf {bnode} .
                    {bnode} a owl:Restriction ;
                             owl:onProperty <{property_uri}> ;
                             {prop_name} <{value}> .
                }}
            }}
            """)
        return bnode

    # ── Seeds ─────────────────────────────────────────────────────────────

    def seed_load(
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

        if url is not None:
            resp = httpx.get(url, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            data = resp.content
        else:
            data = content.encode()

        self._ex.load_graph(data, graph_id)

        rows = self._ex.execute_select(
            f"SELECT (COUNT(*) AS ?n) WHERE {{ GRAPH <{graph_id}> {{ ?s ?p ?o }} }}",
            ["n"],
        )
        count = int(rows[0]["n"]) if rows and rows[0]["n"] is not None else 0
        source = url or "(inline content)"
        return {"graph_id": graph_id, "source": source, "triples_loaded": count}

    # ── Export ────────────────────────────────────────────────────────────

    def export_ttl(self, include_seeds: bool = False) -> str:
        result = self._ex.dump_graph(self._main).decode()

        if include_seeds:
            rows = self._ex.execute_select(
                "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }", ["g"]
            )
            for row in rows:
                g = row["g"] or ""
                if g.startswith("urn:olaf:seed:"):
                    result += f"\n# Seed graph: {g}\n" + self._ex.dump_graph(g).decode()

        return result

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        root_rows = self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?c ?label WHERE {{
            GRAPH <{self._main}> {{
                ?c a owl:Class .
                OPTIONAL {{ ?c rdfs:label ?label }}
                FILTER NOT EXISTS {{
                    ?c rdfs:subClassOf ?parent .
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

        return {
            "active_ontology": self._id,
            "classes": self._count("?x a owl:Class"),
            "object_properties": self._count("?x a owl:ObjectProperty"),
            "datatype_properties": self._count("?x a owl:DatatypeProperty"),
            "individuals": self._count("?x a owl:NamedIndividual"),
            "restrictions": self._count("?x a owl:Restriction"),
            "root_classes": root_classes,
        }
