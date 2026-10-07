"""
olaf_reasoning_agent — LLM agent that repairs the logical problems of an existing OWL/RDFS ontology,
using the OLAF reasoner (`ontology_check`) + LiteLLM.

The code drives the repair; the LLM only decides and applies the fixes:
  1. `ontology_check` (Pellet: inconsistency, unsatisfiable classes; SPARQL: subclass cycles,
     domain/range violations, untyped individuals, property misuse) and `ontology_orphans` run.
  2. The problems are split into groups of `problems_per_task`. For each group the code gathers
     the context itself — the axioms involved, the entities with their definitions and ancestors,
     for domain/range violations the classes the property could be widened to, and the text of
     the chunks they were extracted from — and hands it to a fresh LLM conversation, which fixes
     them with a few write tools (relation_delete, property_update, …) and searches only for what
     the context misses. The ontology is backed up to `backup_dir` before the first change.
  3. The check runs again, for up to `max_rounds` rounds, until nothing is left or nothing changes.

Usage:
    python agent.py                      # uses config.toml in current directory
    python agent.py --config my.toml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import tomllib
from dataclasses import dataclass, field

import litellm
from mcp import ClientSession
from mcp.client.sse import sse_client

from prompts import FIX_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Tools the LLM may use to fix problems: the writes, plus targeted reads for what the task
# context does not cover (e.g. finding where to attach an orphan). Everything else is hidden
# and refused.
FIX_TOOLS = {
    "relation_delete", "relation_add", "property_update", "concept_update", "concept_merge",
    "concept_get", "property_get", "relation_search", "concept_search",
}

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
# Types that say what kind of term an entity is, not which class it belongs to.
_META_TYPES = (
    "http://www.w3.org/2002/07/owl#Class", "http://www.w3.org/2002/07/owl#NamedIndividual",
    "http://www.w3.org/2002/07/owl#ObjectProperty", "http://www.w3.org/2002/07/owl#DatatypeProperty",
    "http://www.w3.org/2002/07/owl#Thing",
)


def _endpoint_kwargs(llm_cfg: dict) -> dict:
    """Optional custom endpoint for OpenAI-compatible providers (Scaleway, vLLM, …).
    The key is read from the env var named by api_key_env, never stored in the config."""
    kwargs = {}
    if llm_cfg.get("api_base"):
        kwargs["api_base"] = llm_cfg["api_base"]
    if llm_cfg.get("api_key_env"):
        key = os.environ.get(llm_cfg["api_key_env"])
        if not key:
            raise SystemExit(f"Environment variable {llm_cfg['api_key_env']} is not set.")
        kwargs["api_key"] = key
    return kwargs


# Errors worth retrying as-is: the provider was slow or briefly unavailable.
_TRANSIENT_ERRORS = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)


def _completion(llm_cfg: dict, **kwargs):
    """litellm.completion, retried on rate limits (HTTP 429) with a growing wait — quotas
    are usually per minute, so the wait goes up to a full minute — and on transient errors
    (timeout, connection, 5xx), which are retried a few times after a short pause."""
    rate_limit_retries = llm_cfg.get("rate_limit_retries", 6)
    transient_retries = llm_cfg.get("transient_retries", 2)
    rate_limited = transient = 0
    while True:
        try:
            return litellm.completion(**kwargs, **_endpoint_kwargs(llm_cfg))
        except litellm.RateLimitError:
            if rate_limited == rate_limit_retries:
                raise
            wait = min(15 * 2 ** rate_limited, 60)
            rate_limited += 1
            logger.warning(
                "Rate limited by the LLM provider — retrying in %ds (%d/%d).", wait, rate_limited, rate_limit_retries
            )
            time.sleep(wait)
        except _TRANSIENT_ERRORS as exc:
            if transient == transient_retries:
                raise
            transient += 1
            logger.warning(
                "LLM call failed (%s) — retrying in 10s (%d/%d).", type(exc).__name__, transient, transient_retries
            )
            time.sleep(10)


def mcp_tools_to_litellm(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions to the OpenAI function-calling format used by LiteLLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


def prune_history(messages: list[dict], keep_recent_turns: int, pruned_chars: int) -> list[dict]:
    """Copy of `messages` to send to the LLM: tool results older than the last
    `keep_recent_turns` turns (assistant message + its tool results) are cut to a
    `pruned_chars` preview. The task message, which holds the context, is never pruned."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("tool_calls")]
    keep = max(1, keep_recent_turns)
    cutoff = starts[-keep] if len(starts) > keep else 0
    pruned = []
    for i, m in enumerate(messages):
        content = m.get("content") or ""
        if i < cutoff and m["role"] == "tool" and len(content) > pruned_chars:
            m = {**m, "content": content[:pruned_chars] + f"\n… [pruned from context: {len(content)} chars]"}
        pruned.append(m)
    return pruned


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
        self.llm_cfg = config["litellm"]
        self.ontology_id = config["olaf"]["ontology_id"]
        self.graph = f"urn:olaf:{self.ontology_id}"
        agent_cfg = config.get("agent", {})
        self.max_rounds = agent_cfg.get("max_rounds", 3)
        self.problems_per_task = agent_cfg.get("problems_per_task", 5)
        self.fix_orphans = agent_cfg.get("fix_orphans", True)
        self.include_seeds = agent_cfg.get("include_seeds", False)
        self.max_iterations = agent_cfg.get("max_iterations", 12)
        self.max_source_chunks = agent_cfg.get("max_source_chunks", 8)
        self.max_chunk_chars = agent_cfg.get("max_chunk_chars", 2500)
        self.max_entities = agent_cfg.get("max_entities_per_task", 25)
        self.max_tool_result_chars = agent_cfg.get("max_tool_result_chars", 20_000)
        self.keep_recent_turns = agent_cfg.get("keep_recent_turns", 6)
        self.pruned_result_chars = agent_cfg.get("pruned_result_chars", 500)
        self.export_path: str | None = agent_cfg.get("export_path")
        self.backup_dir: str | None = agent_cfg.get("backup_dir", "backups")

        self.tools = mcp_tools_to_litellm([t for t in mcp_tools if t.name in FIX_TOOLS])
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

    async def _llm_tool_call(self, tool_name: str, raw_args: str) -> str:
        """Tool call requested by the LLM; errors are returned to it so it can adapt."""
        try:
            tool_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            return f"Error: invalid JSON arguments: {raw_args}"
        if tool_name not in FIX_TOOLS:
            return f"Error: {tool_name} is not available for this task."

        logger.info("Tool call: %s(%s)", tool_name, _summarise(tool_args))
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            content = result.content[0].text if result.content else ""
        except Exception as exc:
            logger.warning("Tool %s failed: %s", tool_name, exc)
            return f"Error: {exc}"

        logger.debug("  → %s", content[:200])
        if len(content) > self.max_tool_result_chars:
            content = content[: self.max_tool_result_chars] + f"\n… [truncated: {len(content)} chars in total]"
        return content

    async def converse(self, task: str, label: str) -> str:
        """One task = one fresh conversation: the LLM calls tools until it replies with a
        plain-text summary. Returns that summary."""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        logger.debug("[%s] task:\n%s", label, task)
        for iteration in range(1, self.max_iterations + 1):
            logger.info("[%s] iteration %d", label, iteration)
            response = await asyncio.to_thread(
                _completion,
                self.llm_cfg,
                model=self.llm_cfg["model"],
                messages=prune_history(messages, self.keep_recent_turns, self.pruned_result_chars),
                tools=self.tools,
                tool_choice="auto",
                max_tokens=self.llm_cfg.get("max_tokens", 4096),
                temperature=self.llm_cfg.get("temperature", 0),
                timeout=self.llm_cfg.get("timeout", 120),
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                summary = (msg.content or "").strip()
                logger.info("[%s] done: %s", label, summary[:1000])
                return summary
            if msg.content and msg.content.strip():
                logger.info("[%s] %s", label, msg.content.strip()[:500])

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": await self._llm_tool_call(tc.function.name, tc.function.arguments),
                })

        logger.warning("[%s] reached max_iterations (%d) — task stopped.", label, self.max_iterations)
        return ""

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

        def explanation(expl: list[list[str]]) -> str:
            return "\n".join(f"  - `{axiom}`" for axioms in expl[:1] for axiom in axioms)

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
        for uri in integrity.get("untyped_individuals", []):
            out.append(Problem("untyped", f"**Untyped individual** `{uri}` — it has no class.", entities=[uri]))

        if self.fix_orphans:
            for o in orphans:
                out.append(Problem(
                    "orphan", f"**Orphan {o['kind']}** `{o['uri']}` (\"{o['label']}\") — no relation to the rest.",
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
        return " ".join(parts) + (f" [{'; '.join(details)}]" if details else "")

    async def context(self, group: list[Problem]) -> tuple[str, str]:
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
        chunk_ids = chunk_ids[: self.max_source_chunks]

        ancestors = await self._ancestors(entities) if entities else {}
        blocks = []
        for i, p in enumerate(group, 1):
            text = f"### Problem {i}\n{p.text}"
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

    async def run(self) -> None:
        previous: set | None = None
        backed_up = False
        for round_number in range(1, self.max_rounds + 1):
            self._entity_cache.clear()
            report, orphans = await self.check()
            problems = await self.problems(report, orphans)
            _log_report(f"Round {round_number}", report, orphans)
            if not problems:
                logger.info("Nothing left to fix.")
                break
            keys = {p.key for p in problems}
            if keys == previous:
                logger.warning("The last round changed nothing — stopping with %d problem(s) left.", len(problems))
                break
            previous = keys
            if self.backup_dir and not backed_up:
                await self.backup()
                backed_up = True

            groups = [problems[i : i + self.problems_per_task] for i in range(0, len(problems), self.problems_per_task)]
            for n, group in enumerate(groups, 1):
                problems_text, sources = await self.context(group)
                await self.converse(
                    FIX_PROMPT.format(count=len(group), problems=problems_text, sources=sources),
                    f"round {round_number} · task {n}/{len(groups)}",
                )
        else:
            report, orphans = await self.check()
            _log_report("Final", report, orphans)

        if self.export_path:
            with open(self.export_path, "w", encoding="utf-8") as f:
                f.write(await self._call("ontology_export"))
            logger.info("Ontology exported to %s", self.export_path)


def _log_report(label: str, report: dict, orphans: list[dict]) -> None:
    r, it = report.get("reasoner", {}), report.get("integrity", {})
    logger.info(
        "%s — consistent: %s, unsatisfiable: %d, classes in cycles: %d, domain: %d, range: %d, untyped: %d, "
        "kind: %d, orphans: %d",
        label, r.get("consistent", "?"), len(r.get("unsatisfiable_classes") or []),
        len(it.get("subclass_cycles", [])), len(it.get("domain_violations", [])),
        len(it.get("range_violations", [])), len(it.get("untyped_individuals", [])),
        len(it.get("property_kind_mismatches", [])), len(orphans),
    )


def _summarise(args: dict) -> str:
    """One-line representation of tool arguments for logging. URIs are kept whole, so that
    every change can be read (and undone) from the log; long texts are shortened."""
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:120]!r}…" if len(s) > 120 and not s.startswith("http") else f"{k}={s!r}")
    return ", ".join(parts)


# ─── Entry point ──────────────────────────────────────────────────────────────


async def _call_or_exit(session: ClientSession, tool_name: str, args: dict) -> str:
    result = await session.call_tool(tool_name, args)
    text = result.content[0].text if result.content else ""
    if text.startswith("Error"):
        raise SystemExit(f"{tool_name} failed: {text}")
    return text


async def select_context(session: ClientSession, olaf_cfg: dict) -> None:
    """Activate the configured ontology (it must exist) and select the chunk collection the
    source texts are read from. Session state only: other agents on the same server are unaffected."""
    await _call_or_exit(session, "ontology_switch", {"ontology_id": olaf_cfg["ontology_id"]})
    logger.info("Active ontology: %s", olaf_cfg["ontology_id"])

    if collection := olaf_cfg.get("collection"):
        args: dict = {"collection": collection}
        if olaf_cfg.get("field_mapping"):
            args["field_mapping"] = olaf_cfg["field_mapping"]
        await _call_or_exit(session, "chunk_collection_switch", args)
        logger.info("Chunk collection: %s", collection)
    else:
        logger.info("Chunk collection: server default ([qdrant].collection)")


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


def _run(coro) -> None:
    """asyncio.run, but a SystemExit raised inside the MCP session (wrapped by anyio in an
    exception group) exits with its message instead of a long traceback."""
    def first_exit(group: BaseExceptionGroup) -> SystemExit | None:
        for exc in group.exceptions:
            found = exc if isinstance(exc, SystemExit) else (
                first_exit(exc) if isinstance(exc, BaseExceptionGroup) else None
            )
            if found:
                return found
        return None

    try:
        asyncio.run(coro)
    except BaseExceptionGroup as group:
        if exit_exc := first_exit(group):
            raise exit_exc from None
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="olaf_reasoning_agent — repair the logical problems of an ontology"
    )
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    log_level = args.log_level or config.get("agent", {}).get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Library debug output (full LLM requests, HTTP traces) drowns the agent's own log.
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore", "mcp", "openai"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, logging.getLogger().level))

    _run(main_async(config))


if __name__ == "__main__":
    main()
