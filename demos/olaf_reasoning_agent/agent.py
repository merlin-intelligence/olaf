"""
olaf_reasoning_agent — LLM agent that curates an existing OWL/RDFS ontology: merges duplicates,
repairs its logical problems, adds missing disjointness, then reviews and materializes what it
entails, using the OLAF reasoner (`ontology_check`, `ontology_infer`) + LiteLLM.

The code drives the work; the LLM only decides and applies the changes:
  0. Deduplication (`dedup = true`) — classes whose concept embeddings are closer than
     `dedup_threshold` are reviewed by pairs: merge or keep.
  1. `ontology_check` (Pellet: inconsistency, unsatisfiable classes; SPARQL: subclass cycles,
     domain/range violations, untyped individuals, property misuse) and `ontology_orphans` run.
  2. The problems are split into groups of `problems_per_task`. For each group the code gathers
     the context itself — the axioms involved, the entities with their definitions and ancestors,
     for domain/range violations the classes the property could be widened to, and the text of
     the chunks they were extracted from — and hands it to a fresh LLM conversation, which fixes
     them with a few write tools (relation_delete, property_update, …) and searches only for what
     the context misses. The ontology is backed up to `backup_dir` before the first change.
  3. The check runs again, for up to `max_rounds` rounds, until nothing is left or nothing changes.
  4. Enrichment (`enrich_disjointness = true`, once the ontology is consistent) — groups of sibling
     classes with pairs not declared disjoint are handed to the LLM, which declares those that
     exclude each other. The check runs again: a disjointness that contradicts an axiom shows up.
  5. Once the ontology is consistent (`infer = true`): the inferences of the reasoner not yet
     materialized are reviewed by the LLM by groups of `inferences_per_task`, each with why it
     holds — an absurd one reveals a wrong axiom, which it fixes. The check runs again, then the
     inferences are written into the ontology, marked as inferred.

Usage:
    python agent.py                      # uses config.toml in current directory
    python agent.py --config my.toml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession
from mcp.client.sse import sse_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_common import LoopSettings, ToolLoop, logger, run, select_context, setup_logging  # noqa: E402

from prompts import DEDUP_PROMPT, ENRICH_PROMPT, FIX_PROMPT, REVIEW_PROMPT, SYSTEM_PROMPT  # noqa: E402

# Tools the LLM may use to fix problems: the writes, plus targeted reads for what the task
# context does not cover (e.g. finding where to attach an orphan). Everything else is hidden
# and refused.
FIX_TOOLS = {
    "relation_delete", "relation_add", "property_update", "concept_update", "concept_merge",
    "restriction_delete", "entity_delete", "disjoint_add",
    "concept_get", "property_get", "relation_search", "concept_search",
}

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
# Types that say what kind of term an entity is, not which class it belongs to.
_META_TYPES = (
    "http://www.w3.org/2002/07/owl#Class", "http://www.w3.org/2002/07/owl#NamedIndividual",
    "http://www.w3.org/2002/07/owl#ObjectProperty", "http://www.w3.org/2002/07/owl#DatatypeProperty",
    "http://www.w3.org/2002/07/owl#Thing",
)


def _local(uri: str) -> str:
    return re.split(r"[#/]", uri.rstrip("#/"))[-1]


# ─── Problems ─────────────────────────────────────────────────────────────────


@dataclass
class Problem:
    kind: str
    text: str                                                       # markdown description for the LLM
    entities: list[str] = field(default_factory=list)               # URIs to describe
    triples: list[tuple[str, str, str]] = field(default_factory=list)  # axioms whose sources to show
    violation: dict | None = None                                   # domain/range: the check's row

    @property
    def key(self) -> tuple:
        """Identity across rounds, to tell whether a round changed anything."""
        return (self.kind, self.text)


# ─── Agent ────────────────────────────────────────────────────────────────────


class ReasoningAgent:
    def __init__(self, session: ClientSession, config: dict, mcp_tools: list):
        self.session = session
        self.ontology_id = config["olaf"]["ontology_id"]
        self.graph = f"urn:olaf:{self.ontology_id}"
        agent_cfg = config.get("agent", {})
        self.max_rounds = agent_cfg.get("max_rounds", 3)
        self.problems_per_task = agent_cfg.get("problems_per_task", 5)
        self.fix_orphans = agent_cfg.get("fix_orphans", True)
        self.include_seeds = agent_cfg.get("include_seeds", False)
        self.max_source_chunks = agent_cfg.get("max_source_chunks", 8)
        self.max_chunk_chars = agent_cfg.get("max_chunk_chars", 2500)
        self.max_entities = agent_cfg.get("max_entities_per_task", 25)
        self.export_path: str | None = agent_cfg.get("export_path")
        self.backup_dir: str | None = agent_cfg.get("backup_dir", "backups")
        self.infer = agent_cfg.get("infer", True)
        self.review_inferences = agent_cfg.get("review_inferences", True)
        self.inferences_per_task = agent_cfg.get("inferences_per_task", 10)
        self.max_explained = agent_cfg.get("max_explained_inferences", 20)
        self.dedup = agent_cfg.get("dedup", True)
        self.dedup_threshold = agent_cfg.get("dedup_threshold", 0.9)
        self.dedup_pairs_per_task = agent_cfg.get("dedup_pairs_per_task", 20)
        self.enrich_disjointness = agent_cfg.get("enrich_disjointness", True)
        self.sibling_groups_per_task = agent_cfg.get("sibling_groups_per_task", 4)
        self.max_siblings_per_group = agent_cfg.get("max_siblings_per_group", 15)
        self._backed_up = False
        # Problems the last check left unfixed: not handed over again unless something changed.
        self._unresolved: set | None = None
        # Disjointness axioms the enrichment declared in this run, as stored (a disjointWith b):
        # flagged as the likely culprit when they appear in an explanation.
        self._added_disjoint: set[tuple[str, str]] = set()

        self.loop = ToolLoop(
            session, config["litellm"], mcp_tools, allowed=FIX_TOOLS,
            settings=LoopSettings.from_config(agent_cfg, max_iterations=12, keep_recent_turns=6),
        )
        self.system_prompt = SYSTEM_PROMPT.format(graph=self.graph)
        self._entity_cache: dict[str, dict | None] = {}

    # ── Tool calls ────────────────────────────────────────────────────────

    async def _call(self, tool_name: str, args: dict | None = None) -> str:
        """Tool call made by the code itself; an error stops the run."""
        result = await self.session.call_tool(tool_name, args or {})
        text = result.content[0].text if result.content else ""
        if text.startswith("Error"):
            raise RuntimeError(f"{tool_name} failed: {text}")
        return text

    async def _try_json(self, tool_name: str, args: dict):
        """Tool call made to gather context: an error only means less context."""
        try:
            return json.loads(await self._call(tool_name, args))
        except (RuntimeError, json.JSONDecodeError) as exc:
            logger.debug("Context lookup %s(%s) failed: %s", tool_name, args, exc)
            return None

    # ── Problems from the check report ────────────────────────────────────

    async def _cycle_edges(self, classes: list[str]) -> list[tuple[str, str, str]]:
        members = ", ".join(f"<{c}>" for c in classes)
        rows = await self._try_json("sparql_query", {"query": f"""
            PREFIX rdfs: <{RDFS}>
            SELECT ?a ?b WHERE {{ GRAPH <{self.graph}> {{
                ?a rdfs:subClassOf ?b . FILTER(?a IN ({members}) && ?b IN ({members}))
            }} }}"""})
        return [(r["a"], RDFS + "subClassOf", r["b"]) for r in (rows or {}).get("rows", [])]

    async def problems(self, report: dict, orphans: list[dict]) -> list[Problem]:
        out: list[Problem] = []
        reasoner = report.get("reasoner", {})

        def entity_uris(entities: dict) -> list[str]:
            uris = []
            for v in entities.values():
                uris.extend(v if isinstance(v, list) else [v])
            return uris

        added = {frozenset((_local(a), _local(b))): (a, b) for a, b in self._added_disjoint}

        def explanation(expl: list[list[str]]) -> str:
            lines = []
            for axiom in (a for axioms in expl[:1] for a in axioms):
                lines.append(f"  - `{axiom}`")
                m = re.fullmatch(r"(\S+) disjointWith (\S+)", axiom.strip())
                if m and (pair := added.get(frozenset(m.groups()))):
                    lines.append(
                        f"    ⚠ declared by the enrichment of this run — the likely wrong axiom: undo with "
                        f"relation_delete(subject_uri=`{pair[0]}`, property_uri=`{OWL}disjointWith`, "
                        f"object_value=`{pair[1]}`) unless the text says they exclude each other"
                    )
            return "\n".join(lines)

        if inc := reasoner.get("inconsistency"):
            out.append(Problem(
                "inconsistency",
                f"**Inconsistency** — {inc.get('reason', '')}\nExplanation (minimal axioms causing it):\n"
                + explanation(inc.get("explanations", [])),
                entities=entity_uris(reasoner.get("entities", {})),
            ))
        for u in reasoner.get("unsatisfiable_classes", []):
            names = {w for axioms in u["explanations"][:1] for a in axioms for w in re.findall(r"[A-Za-z_][\w.-]*", a)}
            out.append(Problem(
                "unsatisfiable",
                f"**Unsatisfiable class** `{u['uri']}` — it can have no instance.\nExplanation:\n"
                + explanation(u["explanations"]),
                entities=[uri for n, v in reasoner.get("entities", {}).items() if n in names
                          for uri in (v if isinstance(v, list) else [v])],
            ))

        integrity = report.get("integrity", {})
        if cycle := integrity.get("subclass_cycles"):
            edges = await self._cycle_edges(cycle)
            out.append(Problem(
                "cycle",
                "**Subclass cycle** between: " + ", ".join(f"`{c}`" for c in cycle)
                + "\nSubclass axioms among them:\n" + "\n".join(f"  - `{a}` rdfs:subClassOf `{b}`" for a, _, b in edges),
                entities=list(cycle),
                triples=edges,
            ))
        for kind, key in (("domain", "domain_violations"), ("range", "range_violations")):
            for v in integrity.get(key, []):
                side = "subject" if kind == "domain" else "object"
                out.append(Problem(
                    kind,
                    f"**{kind.capitalize()} violation** — `{v['subject']}` `{v['property']}` `{v['object']}`: "
                    f"the {side} is not a `{v['expected']}` ({kind} of the property).",
                    entities=[v["subject"], v["property"], v["object"], v["expected"]],
                    triples=[(v["subject"], v["property"], v["object"])],
                    violation=v,
                ))
        for v in integrity.get("property_kind_mismatches", []):
            out.append(Problem(
                "kind",
                f"**Property kind mismatch** — `{v['subject']}` `{v['property']}` `{v['object']}`: {v['problem']}.",
                entities=[v["subject"], v["property"]],
                triples=[(v["subject"], v["property"], v["object"])],
            ))
        for v in integrity.get("unknown_terms", []):
            hint = f" — the closest real term is `{v['suggestion']}`" if v.get("suggestion") else ""
            out.append(Problem(
                "unknown_term",
                f"**Unknown vocabulary term** — `{v['subject']}` `{v['property']}` `{v['object']}` uses "
                f"`{v['term']}`, which its vocabulary does not define{hint}.",
                entities=[v["subject"], v["object"]] if v["object"].startswith("http") else [v["subject"]],
                triples=[(v["subject"], v["property"], v["object"])],
            ))
        for uri in integrity.get("untyped_individuals", []):
            out.append(Problem("untyped", f"**Untyped individual** `{uri}` — it has no class.", entities=[uri]))

        if self.fix_orphans:
            for o in orphans:
                out.append(Problem(
                    "orphan",
                    f"**Orphan {o['kind']}** `{o['uri']}` (\"{o['label']}\") — no relation to the rest. Connect it "
                    "only if the source text says explicitly what it is; otherwise leave it.",
                    entities=[o["uri"]],
                ))
        return out

    # ── Context gathering ─────────────────────────────────────────────────

    async def _entity(self, uri: str) -> dict | None:
        if uri not in self._entity_cache:
            self._entity_cache[uri] = await self._try_json("concept_get", {"uri": uri})
        return self._entity_cache[uri]

    async def _ancestors(self, uris: list[str]) -> dict[str, list[str]]:
        """Every class each entity is (transitively) a subclass or an instance of."""
        meta = ", ".join(f"<{t}>" for t in _META_TYPES)
        rows = await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX rdf: <{RDF_TYPE.rsplit('#', 1)[0]}#>
            PREFIX rdfs: <{RDFS}>
            SELECT DISTINCT ?e ?a WHERE {{ GRAPH <{self.graph}> {{
                VALUES ?e {{ {" ".join(f"<{u}>" for u in uris)} }}
                ?e (rdf:type|rdfs:subClassOf)/rdfs:subClassOf* ?a .
                FILTER(isIRI(?a) && ?a != ?e && ?a NOT IN ({meta}))
            }} }}"""})
        out: dict[str, list[str]] = {}
        for r in (rows or {}).get("rows", []):
            out.setdefault(r["e"], []).append(r["a"])
        return out

    async def _property_uses(self, prop: str, side: str) -> list[str]:
        var = "?s" if side == "subject" else "?o"
        rows = await self._try_json("sparql_query", {"limit": 15, "query": f"""
            SELECT DISTINCT {var} WHERE {{ GRAPH <{self.graph}> {{ ?s <{prop}> ?o . FILTER(isIRI(?o)) }} }}"""})
        return [r[var[1:]] for r in (rows or {}).get("rows", [])]

    async def _violation_hint(self, v: dict, kind: str, ancestors: dict[str, list[str]]) -> str:
        """What the LLM needs to choose between fixing the relation and widening the property."""
        side = "subject" if kind == "domain" else "object"
        offender, expected = v[side], v["expected"]
        shared = [a for a in [expected, *ancestors.get(expected, [])]
                  if a == offender or a in ancestors.get(offender, [])]
        uses = await self._property_uses(v["property"], side)
        lines = [
            f"  - Classes `{_local(offender)}` and `{_local(expected)}` have in common: "
            + (", ".join(f"`{a}`" for a in shared) if shared else "none (widening means clearing the "
               f"{kind} with an empty string)"),
            f"  - Current {side}s of `{_local(v['property'])}`: " + ", ".join(f"`{_local(u)}`" for u in uses),
        ]
        return "\n".join(lines)

    @staticmethod
    def _describe(uri: str, e: dict | None, ancestors: list[str]) -> str:
        if not e:
            return f"- `{uri}` — (not in this ontology: seed or external term)"
        by_pred: dict[str, list[str]] = {}
        for t in e["triples"]:
            by_pred.setdefault(t["predicate"], []).append(t["object"])
        parts = [f"- `{uri}`"]
        if label := e.get("label"):
            parts.append(f'"{label}"')
        if definition := e.get("definition"):
            parts.append("— " + (definition[:300] + "…" if len(definition) > 300 else definition))
        details = []
        for pred, name in ((RDF_TYPE, "type"), (RDFS + "subClassOf", "subClassOf"),
                           (RDFS + "domain", "domain"), (RDFS + "range", "range")):
            values = [_local(v) for v in by_pred.get(pred, []) if v and not v.startswith("_:")]
            if values:
                details.append(f"{name}: {', '.join(values)}")
        if ancestors:
            details.append(f"ancestors: {', '.join(_local(a) for a in ancestors)}")
        for r in e.get("restrictions", []):
            target = r.get("value") or ""
            card = f" {r['cardinality']}" if "cardinality" in r else ""
            details.append(f"restriction: {_local(r['property'])} {r['restriction_type']}{card} {_local(target)}".rstrip())
        return " ".join(parts) + (f" [{'; '.join(details)}]" if details else "")

    async def context(self, group: list[Problem], heading: str = "Problem", with_sources: bool = True) -> tuple[str, str]:
        """Problems text with the involved entities described, and the source chunk texts."""
        entities: list[str] = []
        for p in group:
            entities.extend(u for u in p.entities if u not in entities)
        entities = entities[: self.max_entities]

        # Sources: those of the axioms at stake first, then those of the entities.
        chunk_ids: list[str] = []
        for p in group:
            for s, prop, o in p.triples:
                ids = await self._try_json("relation_sources", {"subject_uri": s, "property_uri": prop, "object_value": o})
                chunk_ids.extend(i for i in ids or [] if i not in chunk_ids)
        for uri in entities:
            e = await self._entity(uri)
            chunk_ids.extend(i for i in (e or {}).get("source_chunk_ids", []) if i not in chunk_ids)
        chunk_ids = chunk_ids[: self.max_source_chunks] if with_sources else []

        ancestors = await self._ancestors(entities) if entities else {}
        blocks = []
        for i, p in enumerate(group, 1):
            text = f"### {heading} {i}\n{p.text}"
            if p.violation:
                text += "\n" + await self._violation_hint(p.violation, p.kind, ancestors)
            blocks.append(text)
        problems_text = "\n\n".join(blocks) + "\n\n## Entities involved\n" + "\n".join(
            [self._describe(u, await self._entity(u), ancestors.get(u, [])) for u in entities]
        )

        sources = "(no source chunk recorded)"
        if chunk_ids:
            chunks = await self._try_json("chunk_read_batch", {"chunk_ids": chunk_ids})
            if chunks is None:
                sources = "(source text unavailable — check [olaf] collection)"
            else:
                texts = {c["id"]: re.sub(r"[ \t]+", " ", c["text"]) for c in chunks if c.get("text")}
                sources = "\n\n".join(
                    f"### Chunk {cid}\n" + (
                        text[: self.max_chunk_chars] + " …" if len(text) > self.max_chunk_chars else text
                    )
                    for cid, text in texts.items()
                )
        return problems_text, sources

    # ── Run ───────────────────────────────────────────────────────────────

    async def check(self) -> tuple[dict, list[dict]]:
        report = json.loads(await self._call("ontology_check", {"include_seeds": self.include_seeds}))
        if error := report.get("reasoner", {}).get("error"):
            logger.warning("Reasoner unavailable, only the SPARQL checks ran: %s", error)
        orphans = json.loads(await self._call("ontology_orphans")) if self.fix_orphans else []
        return report, orphans

    async def backup(self) -> None:
        """Turtle dump of the whole ontology graph (provenance included) before any change,
        so that a run can be undone — see the README for how to restore it."""
        os.makedirs(self.backup_dir, exist_ok=True)
        path = os.path.join(self.backup_dir, f"{self.ontology_id}-{time.strftime('%Y%m%d-%H%M%S')}.ttl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(await self._call("ontology_export"))
        logger.info("Backup of the ontology before any change: %s", path)

    async def _ensure_backup(self) -> None:
        if self.backup_dir and not self._backed_up:
            await self.backup()
            self._backed_up = True

    async def repair(self, label: str = "Round") -> bool:
        """Check → fix → check again, for up to max_rounds rounds. Returns whether the
        ontology is consistent at the end."""
        for round_number in range(1, self.max_rounds + 1):
            self._entity_cache.clear()
            report, orphans = await self.check()
            problems = await self.problems(report, orphans)
            _log_report(f"{label} {round_number}", report, orphans)
            keys = {p.key for p in problems}
            if not problems:
                logger.info("Nothing left to fix.")
                self._unresolved = keys
                break
            if keys == self._unresolved:
                logger.warning("Nothing changed since the last fixes — stopping with %d problem(s) left.", len(problems))
                break
            self._unresolved = keys
            await self._ensure_backup()

            groups = [problems[i : i + self.problems_per_task] for i in range(0, len(problems), self.problems_per_task)]
            for n, group in enumerate(groups, 1):
                problems_text, sources = await self.context(group)
                await self.loop.task(
                    self.system_prompt,
                    FIX_PROMPT.format(count=len(group), problems=problems_text, sources=sources),
                    f"{label.lower()} {round_number} · task {n}/{len(groups)}",
                )
        else:
            report, orphans = await self.check()
            _log_report("Final", report, orphans)
            self._unresolved = {p.key for p in await self.problems(report, orphans)}
        return report.get("reasoner", {}).get("consistent") is not False

    # ── Inferences ────────────────────────────────────────────────────────

    async def _preview_inferences(self) -> list[dict] | None:
        try:
            preview = json.loads(await self._call("ontology_infer", {
                "action": "preview", "include_seeds": self.include_seeds, "max_explained": self.max_explained,
            }))
        except RuntimeError as exc:
            logger.warning("Inference skipped: %s", exc)
            return None
        items = preview["inferences"]
        logger.info(
            "Inferences: %d entailed but not asserted — %d from the class hierarchy alone, %d already materialized by an earlier run.",
            len(items), sum(i["taxonomic"] for i in items), sum(i["materialized"] for i in items),
        )
        return items

    async def _parent_counts(self, uris: list[str]) -> dict[str, int]:
        """Number of named superclasses each class has (asserted)."""
        if not uris:
            return {}
        rows = (await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX rdfs: <{RDFS}>
            SELECT ?c (COUNT(DISTINCT ?p) AS ?n) WHERE {{ GRAPH <{self.graph}> {{
                VALUES ?c {{ {" ".join(f"<{u}>" for u in uris)} }}
                ?c rdfs:subClassOf ?p . FILTER(isIRI(?p))
            }} }} GROUP BY ?c"""}) or {}).get("rows", [])
        return {r["c"]: int(r["n"] or 0) for r in rows}

    @staticmethod
    def _inference_problem(item: dict, parent_counts: dict[str, int]) -> Problem:
        s, p, o = item["subject"], item["predicate"], item["object"]
        names = item.get("entities", {})
        lines = []
        for axiom in (a for axioms in item["explanations"][:1] for a in axioms):
            note = ""
            m = re.fullmatch(r"(\S+) subClassOf (\S+)", axiom.strip())
            if m and isinstance(sub := names.get(m.group(1)), str) and parent_counts.get(sub) == 1:
                note = f" — the only parent of {m.group(1)}: deleting it leaves {m.group(1)} without any parent"
            lines.append(f"  - `{axiom}`{note}")
        why = "\n".join(lines)
        origin = "from the class hierarchy alone" if item["taxonomic"] else "involving domains, ranges, equivalences…"
        text = (f"**Inference** `{s}` `{p}` `{o}` ({origin})\n"
                + (f"Because of:\n{why}" if why else "(no explanation computed)"))
        entities = [s, o]
        for v in item.get("entities", {}).values():
            entities.extend(u for u in (v if isinstance(v, list) else [v]) if u not in entities)
        return Problem("inference", text, entities=entities)

    async def review(self, items: list[dict]) -> None:
        """Have the LLM read the new inferences: an absurd one reveals a wrong axiom in its
        explanation, which it fixes. Inferences with the same object are kept together, as
        they often come from the same axiom."""
        items = sorted(items, key=lambda i: (i["object"], i["predicate"], i["subject"]))
        groups = [items[i : i + self.inferences_per_task] for i in range(0, len(items), self.inferences_per_task)]
        for n, group in enumerate(groups, 1):
            self._entity_cache.clear()
            subjects = sorted({u for i in group for name, u in i.get("entities", {}).items()
                               if isinstance(u, str) and any(
                                   a.strip().startswith(f"{name} subClassOf ") for ax in i["explanations"][:1] for a in ax)})
            counts = await self._parent_counts(subjects)
            problems_text, sources = await self.context([self._inference_problem(i, counts) for i in group], "Inference")
            await self.loop.task(
                    self.system_prompt,
                REVIEW_PROMPT.format(count=len(group), inferences=problems_text, sources=sources),
                f"review · task {n}/{len(groups)}",
            )

    async def infer_and_materialize(self) -> None:
        items = await self._preview_inferences()
        if items is None:
            return
        to_review = [i for i in items if not i["materialized"]]
        if self.review_inferences and to_review:
            await self._ensure_backup()
            await self.review(to_review)
            # The review may have changed axioms: check again before materializing.
            if not await self.repair(label="After review"):
                logger.warning("The ontology is inconsistent after the review — inferences not materialized.")
                return
        await self._ensure_backup()
        result = json.loads(await self._call("ontology_infer", {"action": "materialize", "include_seeds": self.include_seeds}))
        logger.info("Inferences materialized: %d (replacing %d).", result["stored"], result["replaced"])

    # ── Deduplication ─────────────────────────────────────────────────────

    async def deduplicate(self) -> None:
        """Pairs of classes whose concept embeddings are closer than dedup_threshold are
        reviewed by the LLM, dedup_pairs_per_task at a time: merge or keep."""
        classes = json.loads(await self._call("concept_list", {"limit": 100_000}))
        by_uri = {c["uri"]: c for c in classes}
        pairs: dict[tuple[str, str], float] = {}
        for c in classes:
            query = c["label"] + (f": {c['definition']}" if c.get("definition") else "")
            try:
                hits = json.loads(await self._call("concept_semantic_search", {"query": query, "top_k": 5}))
            except RuntimeError as exc:
                logger.warning("Deduplication skipped: %s", exc)
                return
            for h in hits:
                if h["uri"] != c["uri"] and h["uri"] in by_uri and h["score"] >= self.dedup_threshold:
                    key = tuple(sorted((c["uri"], h["uri"])))
                    pairs[key] = max(pairs.get(key, 0.0), h["score"])

        if not pairs:
            logger.info("Deduplication: no candidate pairs above %.2f.", self.dedup_threshold)
            return
        logger.info("Deduplication: %d candidate pairs above %.2f.", len(pairs), self.dedup_threshold)
        await self._ensure_backup()

        def describe(uri: str) -> str:
            c = by_uri[uri]
            definition = (c.get("definition") or "")[:200]
            return f'"{c["label"]}" <{uri}>' + (f" — {definition}" if definition else "")

        ordered = sorted(pairs.items(), key=lambda kv: -kv[1])
        for start in range(0, len(ordered), self.dedup_pairs_per_task):
            group = ordered[start : start + self.dedup_pairs_per_task]
            text = "\n".join(f"- score {score:.3f}\n  A: {describe(a)}\n  B: {describe(b)}" for (a, b), score in group)
            await self.loop.task(
                self.system_prompt, DEDUP_PROMPT.format(pairs=text),
                f"dedup · task {start // self.dedup_pairs_per_task + 1}",
            )

    # ── Enrichment: disjointness ──────────────────────────────────────────

    async def _sibling_groups(self) -> list[tuple[str, list[str]]]:
        """(parent, children) of the classes that share a parent (or are all roots) and have
        at least one pair neither disjoint nor in a subclass relation."""
        rows = (await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX owl: <{OWL}> PREFIX rdfs: <{RDFS}>
            SELECT ?c ?parent WHERE {{ GRAPH <{self.graph}> {{
                ?c a owl:Class . FILTER(isIRI(?c))
                OPTIONAL {{ ?c rdfs:subClassOf ?parent . FILTER(isIRI(?parent) && ?parent != owl:Thing && ?parent != ?c) }}
            }} }}"""}) or {}).get("rows", [])
        related = (await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX owl: <{OWL}> PREFIX rdfs: <{RDFS}>
            SELECT ?a ?b WHERE {{ GRAPH <{self.graph}> {{
                {{ ?a owl:disjointWith ?b }} UNION {{ ?a rdfs:subClassOf+ ?b . FILTER(isIRI(?b)) }}
            }} }}"""}) or {}).get("rows", [])
        excluded = {frozenset((r["a"], r["b"])) for r in related}

        children: dict[str, list[str]] = {}
        for r in rows:
            parent = r["parent"] or "(root classes)"
            if r["c"] not in children.setdefault(parent, []):
                children[parent].append(r["c"])
        groups = []
        for parent, kids in sorted(children.items()):
            for start in range(0, len(kids), self.max_siblings_per_group):
                chunk = sorted(kids)[start : start + self.max_siblings_per_group]
                if any(frozenset((a, b)) not in excluded for i, a in enumerate(chunk) for b in chunk[i + 1:]):
                    groups.append((parent, chunk))
        return groups

    async def _disjoint_pairs(self) -> set[tuple[str, str]]:
        rows = (await self._try_json("sparql_query", {"limit": 1000, "query": f"""
            PREFIX owl: <{OWL}>
            SELECT ?a ?b WHERE {{ GRAPH <{self.graph}> {{ ?a owl:disjointWith ?b }} }}"""}) or {}).get("rows", [])
        return {(r["a"], r["b"]) for r in rows}

    async def enrich(self) -> bool:
        """Have the LLM declare disjoint sibling classes. Returns whether it was asked."""
        groups = [g for g in await self._sibling_groups() if len(g[1]) >= 2]
        if not groups:
            logger.info("Enrichment: no sibling classes left to examine.")
            return False
        logger.info("Enrichment: %d groups of sibling classes to examine for disjointness.", len(groups))
        await self._ensure_backup()
        before = await self._disjoint_pairs()
        per_task = self.sibling_groups_per_task
        for n, start in enumerate(range(0, len(groups), per_task), 1):
            self._entity_cache.clear()
            blocks = []
            for parent, kids in groups[start : start + per_task]:
                parent_name = parent if parent.startswith("(") else f"subclasses of `{parent}`"
                problem = Problem("siblings", f"Siblings — {parent_name}", entities=kids)
                text, _ = await self.context([problem], heading="Group", with_sources=False)
                blocks.append(text.replace("### Group 1", "###", 1))
            await self.loop.task(
                self.system_prompt, ENRICH_PROMPT.format(groups="\n\n".join(blocks)),
                f"enrich · task {n}/{-(-len(groups) // per_task)}",
            )
        added = await self._disjoint_pairs() - before
        self._added_disjoint |= added
        logger.info("Enrichment: %d disjoint pairs declared.", len(added))
        return True

    # ── Run ───────────────────────────────────────────────────────────────

    async def run(self) -> None:
        if self.dedup:
            await self.deduplicate()
        consistent = await self.repair()
        if self.enrich_disjointness and consistent and await self.enrich():
            consistent = await self.repair(label="After enrichment")
        if self.infer:
            if consistent:
                await self.infer_and_materialize()
            else:
                logger.warning("The ontology is still inconsistent — nothing can be inferred from it.")

        if self.export_path:
            with open(self.export_path, "w", encoding="utf-8") as f:
                f.write(await self._call("ontology_export"))
            logger.info("Ontology exported to %s", self.export_path)


def _log_report(label: str, report: dict, orphans: list[dict]) -> None:
    r, it = report.get("reasoner", {}), report.get("integrity", {})
    logger.info(
        "%s — consistent: %s, unsatisfiable: %d, classes in cycles: %d, domain: %d, range: %d, untyped: %d, "
        "kind: %d, unknown terms: %d, orphans: %d",
        label, r.get("consistent", "?"), len(r.get("unsatisfiable_classes") or []),
        len(it.get("subclass_cycles", [])), len(it.get("domain_violations", [])),
        len(it.get("range_violations", [])), len(it.get("untyped_individuals", [])),
        len(it.get("property_kind_mismatches", [])), len(it.get("unknown_terms", [])), len(orphans),
    )


# ─── Entry point ──────────────────────────────────────────────────────────────


async def main_async(config: dict) -> None:
    olaf_url = config["olaf"]["url"]
    logger.info("Connecting to OLAF at %s", olaf_url)

    # The reasoner can take a while on a large ontology: don't let the SSE stream time out.
    async with sse_client(olaf_url, sse_read_timeout=900) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await select_context(session, config["olaf"])
            tools_result = await session.list_tools()
            await ReasoningAgent(session, config, tools_result.tools).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="olaf_reasoning_agent — curate an ontology: dedup, repair, enrich, infer"
    )
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    setup_logging(args.log_level or config.get("agent", {}).get("log_level", "INFO"))
    run(main_async(config))


if __name__ == "__main__":
    main()
