from __future__ import annotations
import asyncio
import importlib.util
import os
import re
import tempfile
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
    """OWL 2 DL consistency / satisfiability checks with Pellet, with justifications
    (the minimal set of axioms causing each problem)."""

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

    @staticmethod
    def _entities(explanations: list[list[str]], index: dict[str, list[str]]) -> dict:
        """Map every local name appearing in the explanations to its URI (or URIs, when
        the local name is ambiguous across namespaces)."""
        names = {w for expl in explanations for axiom in expl for w in _WORD.findall(axiom)}
        return {
            n: (index[n][0] if len(index[n]) == 1 else index[n])
            for n in sorted(names) if n in index
        }
