from __future__ import annotations
import asyncio
import importlib.util
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Pellet 2.3.1 (with its explanation module) ships inside the owlready2 wheel. It is
# run as a separate JVM per check rather than through owlready2's Python API: the
# server is async and multi-session, and owlready2 keeps a process-global quadstore.

_IRI = re.compile(r"<([^>]+)>")
_WORD = re.compile(r"[A-Za-z_][\w.-]*")
_EXPLANATION_START = re.compile(r"^\s*\d+\)\s+(.*)$")
_AXIOM = re.compile(r"^Axiom:\s+(.*?)\s+subClassOf\s+Nothing\s*$")

_BUILTIN_NAMESPACES = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
)
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDF_TYPE = _RDF + "type"
_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
_NT_TRIPLE = re.compile(r"^<([^>]+)> <([^>]+)> <([^>]+)> \.\s*$")

# Entailments `pellet extract` computes. Direct* variants would only give back the asserted
# hierarchy; these include what follows from it and from domains, ranges, equivalences,
# restrictions and property axioms.
_EXTRACT_STATEMENTS = (
    "SubClassOf EquivalentClasses SubPropertyOf ClassAssertion ObjectPropertyAssertion SameIndividual"
)


def _pellet_classpath() -> str:
    spec = importlib.util.find_spec("owlready2")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("owlready2 is not installed (it provides the Pellet reasoner jars).")
    pellet_dir = Path(spec.submodule_search_locations[0]) / "pellet"
    if not pellet_dir.is_dir():
        raise RuntimeError(f"Pellet jars not found in {pellet_dir}")
    return f"{pellet_dir}{os.sep}*"


def _local_name(iri: str) -> str:
    return re.split(r"[#/]", iri.rstrip("#/"))[-1]


def _local_index(ntriples: bytes) -> dict[str, list[str]]:
    """Local name → IRIs, so the local names Pellet prints can be mapped back to the
    exact URIs the agent needs for relation_delete & co."""
    index: dict[str, set[str]] = {}
    for iri in set(_IRI.findall(ntriples.decode(errors="replace"))):
        if iri.startswith(_BUILTIN_NAMESPACES):
            continue
        index.setdefault(_local_name(iri), set()).add(iri)
    return {k: sorted(v) for k, v in index.items()}


def _parse_extract(rdfxml: str, ntriples: bytes) -> list[tuple[str, str, str]]:
    """New entailments in `pellet extract`'s RDF/XML output, as (subject, predicate, object)
    IRIs: what is already asserted, trivial (X ⊑ X, X ⊑ owl:Thing, X a owl:Class…), about
    built-in terms, or about literals and blank nodes is left out. So is the type a class
    receives when used as an individual (punning: a relation between two classes gives its
    subject the property's domain) — it says nothing about the class's instances."""
    asserted: set[tuple[str, str, str]] = set()
    terms: set[str] = set()  # classes and properties, as declared in the input
    for line in ntriples.decode(errors="replace").splitlines():
        if m := _NT_TRIPLE.match(line):
            asserted.add(m.groups())
            if m.group(2) == _RDF_TYPE and m.group(3).startswith("http://www.w3.org/2002/07/owl#") and \
                    m.group(3).rsplit("#", 1)[1] in ("Class", "ObjectProperty", "DatatypeProperty"):
                terms.add(m.group(1))

    about, resource = f"{{{_RDF}}}about", f"{{{_RDF}}}resource"
    found: set[tuple[str, str, str]] = set()
    for node in ET.fromstring(rdfxml):
        subject = node.get(about)
        if not subject or subject.startswith(_BUILTIN_NAMESPACES):
            continue
        statements = [(child.tag[1:].replace("}", ""), child.get(resource)) for child in node]
        if node.tag != f"{{{_RDF}}}Description":  # typed node: <owl:Class rdf:about=…>
            statements.append((_RDF_TYPE, node.tag[1:].replace("}", "")))
        for predicate, obj in statements:
            if not obj or obj == subject or obj.startswith(_BUILTIN_NAMESPACES):
                continue
            if predicate == _RDF_TYPE and subject in terms:
                continue
            if (subject, predicate, obj) not in asserted:
                found.add((subject, predicate, obj))
    return sorted(found)


def _taxonomic_chain(edges: dict[str, dict[str, list[str]]], s: str, p: str, o: str) -> list[str] | None:
    """Shortest chain of asserted rdf:type / rdfs:subClassOf axioms from which `s p o`
    follows, in Pellet's notation ("A subClassOf B", "i type C"), or None if there is none.
    `edges[predicate][subject]` lists the asserted objects."""
    if p not in (_RDF_TYPE, _SUBCLASS_OF):
        return None
    start = [(c, [f"{_local_name(s)} {'type' if p == _RDF_TYPE else 'subClassOf'} {_local_name(c)}"])
             for c in edges[p].get(s, [])]
    seen, queue = {s}, list(start)
    while queue:
        node, chain = queue.pop(0)
        if node == o:
            return chain
        if node in seen:
            continue
        seen.add(node)
        queue.extend((parent, chain + [f"{_local_name(node)} subClassOf {_local_name(parent)}"])
                     for parent in edges[_SUBCLASS_OF].get(node, []))
    return None


def _parse_explanations(output: str) -> list[dict]:
    """Parse `pellet explain` output into [{axiom, explanations: [[axiom, ...], ...]}].

        Axiom: Employee subClassOf Nothing
        Explanation(s):
        1)   Organisation disjointWith Person
             Employee subClassOf Organisation
    """
    blocks: list[dict] = []
    in_explanations = False
    for raw in output.splitlines():
        line = raw.rstrip()
        if line.startswith("Axiom:"):
            blocks.append({"axiom": line[len("Axiom:"):].strip(), "explanations": []})
            in_explanations = False
        elif line.startswith("Explanation(s):"):
            in_explanations = True
        elif in_explanations and blocks and line.strip():
            if m := _EXPLANATION_START.match(line):
                blocks[-1]["explanations"].append([m.group(1).strip()])
            elif blocks[-1]["explanations"]:
                blocks[-1]["explanations"][-1].append(line.strip())
    return blocks


class Reasoner:
    """OWL 2 DL reasoning with Pellet: consistency / satisfiability checks and the
    entailments of the ontology, each with its justification (the minimal set of axioms
    causing a problem, or from which an entailment follows)."""

    def __init__(self, java: str = "java", memory_mb: int = 2048, timeout_seconds: int = 120):
        self._java = java
        self._memory_mb = memory_mb
        self._timeout = timeout_seconds

    async def _pellet(self, *args: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._java, f"-Xmx{self._memory_mb}m", "-cp", _pellet_classpath(), "pellet.Pellet", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Java not found ({self._java!r}): install a Java 11+ runtime or set [reasoner] java in config.toml."
            )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"Reasoner timed out after {self._timeout}s (pellet {args[0]}).")
        out = stdout.decode(errors="replace")
        if proc.returncode != 0:
            detail = (stderr.decode(errors="replace").strip() or out.strip())[-2000:]
            raise RuntimeError(f"Reasoner failed (pellet {args[0]}, exit {proc.returncode}): {detail}")
        return out

    async def check(self, ntriples: bytes, max_explanations: int = 1) -> dict:
        index = _local_index(ntriples)
        with tempfile.TemporaryDirectory(prefix="olaf-reasoner-") as tmp:
            path = str(Path(tmp) / "ontology.nt")
            Path(path).write_bytes(ntriples)

            consistency = await self._pellet("consistency", path)
            if "Consistent: No" in consistency:
                reason = next(
                    (l.split(":", 1)[1].strip() for l in consistency.splitlines() if l.startswith("Reason:")), ""
                )
                blocks = _parse_explanations(
                    await self._pellet("explain", "--inconsistent", "--max", str(max_explanations), path)
                )
                explanations = blocks[0]["explanations"] if blocks else []
                return {
                    "consistent": False,
                    "inconsistency": {"reason": reason, "explanations": explanations},
                    "unsatisfiable_classes": [],
                    "entities": self._entities(explanations, index),
                }
            if "Consistent: Yes" not in consistency:
                raise RuntimeError(f"Unexpected reasoner output: {consistency.strip()[-500:]}")

            blocks = _parse_explanations(
                await self._pellet("explain", "--all-unsat", "--max", str(max_explanations), path)
            )

        unsat = []
        for block in blocks:
            m = _AXIOM.match(f"Axiom: {block['axiom']}")
            if not m:
                continue
            name = m.group(1)
            uris = index.get(name, [])
            unsat.append({
                "class": name,
                "uri": uris[0] if len(uris) == 1 else uris or None,
                "explanations": block["explanations"],
            })
        all_explanations = [e for u in unsat for e in u["explanations"]]
        return {
            "consistent": True,
            "inconsistency": None,
            "unsatisfiable_classes": unsat,
            "entities": self._entities(all_explanations, index),
        }

    async def infer(self, ntriples: bytes) -> dict:
        """New entailments of a consistent ontology: {consistent, inferences: [(s, p, o)]}.
        Nothing is inferred from an inconsistent ontology (everything would follow)."""
        with tempfile.TemporaryDirectory(prefix="olaf-reasoner-") as tmp:
            path = str(Path(tmp) / "ontology.nt")
            Path(path).write_bytes(ntriples)
            if "Consistent: Yes" not in await self._pellet("consistency", path):
                return {"consistent": False, "inferences": []}
            output = await self._pellet("extract", "--statements", _EXTRACT_STATEMENTS, path)
        return {"consistent": True, "inferences": _parse_extract(output, ntriples)}

    async def explain_entailments(
        self, ntriples: bytes, triples: list[tuple[str, str, str]], max_explanations: int = 1
    ) -> list[dict]:
        """Why each entailment holds: [{explanations, entities}] in the order of `triples`.
        Pellet explains subclass and instance entailments only; others get no explanation."""
        index = _local_index(ntriples)
        out = []
        with tempfile.TemporaryDirectory(prefix="olaf-reasoner-") as tmp:
            path = str(Path(tmp) / "ontology.nt")
            Path(path).write_bytes(ntriples)
            for s, p, o in triples:
                option = {_SUBCLASS_OF: "--subclass", _RDF_TYPE: "--instance"}.get(p)
                if not option:
                    out.append({"explanations": [], "entities": {}})
                    continue
                blocks = _parse_explanations(
                    await self._pellet("explain", option, f"{s},{o}", "--max", str(max_explanations), path)
                )
                explanations = blocks[0]["explanations"] if blocks else []
                out.append({"explanations": explanations, "entities": self._entities(explanations, index)})
        return out

    async def explain_inferences(
        self, ntriples: bytes, triples: list[tuple[str, str, str]], max_explained: int = 20
    ) -> list[dict]:
        """Each entailment with why it holds. Those that follow from the asserted class
        hierarchy alone (`taxonomic`) get their chain of rdf:type / rdfs:subClassOf axioms,
        computed here; Pellet explains the others — the ones involving domains, ranges,
        equivalences, restrictions… — up to `max_explained` (one JVM run each)."""
        edges: dict[str, dict[str, list[str]]] = {_RDF_TYPE: {}, _SUBCLASS_OF: {}}
        for line in ntriples.decode(errors="replace").splitlines():
            if (m := _NT_TRIPLE.match(line)) and m.group(2) in edges and not m.group(3).startswith(_BUILTIN_NAMESPACES):
                edges[m.group(2)].setdefault(m.group(1), []).append(m.group(3))
        index = _local_index(ntriples)

        items, to_explain = [], []
        for s, p, o in triples:
            chain = _taxonomic_chain(edges, s, p, o)
            item = {"subject": s, "predicate": p, "object": o, "taxonomic": chain is not None,
                    "explanations": [chain] if chain else [], "entities": {}}
            if chain:
                item["entities"] = self._entities(item["explanations"], index)
            elif len(to_explain) < max_explained:
                to_explain.append(item)
            items.append(item)
        explained = await self.explain_entailments(
            ntriples, [(i["subject"], i["predicate"], i["object"]) for i in to_explain]
        )
        for item, e in zip(to_explain, explained):
            item.update(e)
        return items

    @staticmethod
    def _entities(explanations: list[list[str]], index: dict[str, list[str]]) -> dict:
        """Map every local name appearing in the explanations to its URI (or URIs, when
        the local name is ambiguous across namespaces)."""
        names = {w for expl in explanations for axiom in expl for w in _WORD.findall(axiom)}
        return {
            n: (index[n][0] if len(index[n]) == 1 else index[n])
            for n in sorted(names) if n in index
        }
