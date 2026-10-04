"""
olaf_building_agent — LLM agent that builds an OWL/RDFS ontology using OLAF MCP tools + LiteLLM.

The code drives the pipeline; the LLM does the ontology work through tool calls:
  1. Extraction — pending chunks are processed by batches of `batch_size`; each batch is a fresh
     LLM conversation given the chunks and a digest of the current ontology. The code marks
     the chunks processed once the batch is done.
  2. Consolidation (`consolidate = true`) — near-duplicate classes found with the concept
     embeddings are reviewed by the LLM (merge or keep), then `ontology_check` + orphans are
     run and the LLM fixes what they report.
  3. Export — the ontology is written to `export_path`.

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
import sys
import time
import tomllib

import litellm
from mcp import ClientSession
from mcp.client.sse import sse_client

from prompts import BATCH_PROMPT, CHECK_PROMPT, DEDUP_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Session context tools: the ontology and chunk collection are selected once from the
# config before the pipeline starts.
SESSION_TOOLS = {"ontology_create", "ontology_switch", "chunk_collection_switch"}


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


def _completion(llm_cfg: dict, **kwargs):
    """litellm.completion, retried on rate limits (HTTP 429) with a growing wait.
    Quotas are usually per minute, so the wait goes up to a full minute."""
    retries = llm_cfg.get("rate_limit_retries", 6)
    for attempt in range(retries + 1):
        try:
            return litellm.completion(**kwargs, **_endpoint_kwargs(llm_cfg))
        except litellm.RateLimitError:
            if attempt == retries:
                raise
            wait = min(15 * 2 ** attempt, 60)
            logger.warning("Rate limited by the LLM provider — retrying in %ds (%d/%d).", wait, attempt + 1, retries)
            time.sleep(wait)


# ─── Tool conversion ──────────────────────────────────────────────────────────


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


# ─── Context pruning ──────────────────────────────────────────────────────────


def prune_history(
    messages: list[dict], keep_recent_turns: int, pruned_chars: int, max_turns: int | None = None
) -> list[dict]:
    """Copy of `messages` to send to the LLM. A turn is an assistant message with tool calls
    plus its tool results. Tool results older than the last `keep_recent_turns` turns are cut
    to a `pruned_chars` preview; with `max_turns`, turns beyond it are dropped altogether
    (oldest first, the leading system/user messages are kept). Whole turns are always kept or
    dropped together, so every tool call keeps its result, as the API requires."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("tool_calls")]
    if max_turns is not None and len(starts) > max_turns:
        head_end, first_kept = starts[0], starts[-max_turns]
        messages = messages[:head_end] + messages[first_kept:]
        starts = [s - (first_kept - head_end) for s in starts[-max_turns:]]

    keep = max(1, keep_recent_turns)
    cutoff = starts[-keep] if len(starts) > keep else 0
    pruned = []
    for i, m in enumerate(messages):
        content = m.get("content") or ""
        if i < cutoff and m["role"] == "tool" and len(content) > pruned_chars:
            m = {**m, "content": (
                content[:pruned_chars]
                + f"\n… [pruned from context: {len(content)} chars — call the tool again if you need it]"
            )}
        pruned.append(m)
    return pruned


# ─── Agent ────────────────────────────────────────────────────────────────────

# Pipeline steps done by the code itself (reading/marking chunks, export) or not part of any
# task: hidden from the LLM, and refused if called anyway.
PIPELINE_TOOLS = SESSION_TOOLS | {"chunk_list", "chunk_mark_processed", "ontology_export", "seed_load"}
# Consistency checking belongs to the consolidation phase, not to extraction.
EXTRACTION_HIDDEN = PIPELINE_TOOLS | {"ontology_check", "ontology_orphans"}

_PROPERTIES_QUERY = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?p (SAMPLE(?l) AS ?label) (SAMPLE(?t) AS ?type) (SAMPLE(?d) AS ?domain) (SAMPLE(?r) AS ?range) WHERE {{
    GRAPH <urn:olaf:{ontology_id}> {{
        ?p a ?t . FILTER(?t IN (owl:ObjectProperty, owl:DatatypeProperty))
        OPTIONAL {{ ?p rdfs:label ?l }} OPTIONAL {{ ?p rdfs:domain ?d }} OPTIONAL {{ ?p rdfs:range ?r }}
    }}
}} GROUP BY ?p
"""

_SEED_CLASSES_QUERY = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?c (SAMPLE(?l) AS ?label) WHERE {
    GRAPH ?g { ?c a owl:Class . FILTER(isIRI(?c)) OPTIONAL { ?c rdfs:label ?l } }
    FILTER(STRSTARTS(STR(?g), "urn:olaf:seed:"))
} GROUP BY ?c
"""


class BuildingAgent:
    """Code-driven pipeline: extraction by batches of chunks, then consolidation
    (deduplication + consistency check), then export. Each batch and each consolidation
    task is a fresh LLM conversation, so the context size does not grow with the corpus."""

    def __init__(self, session: ClientSession, config: dict, mcp_tools: list):
        self.session = session
        self.llm_cfg = config["litellm"]
        self.ontology_id = config["olaf"]["ontology_id"]
        agent_cfg = config.get("agent", {})
        self.batch_size = agent_cfg.get("batch_size", 5)
        self.max_batches = agent_cfg.get("max_batches", 0)
        self.consolidate = agent_cfg.get("consolidate", True)
        self.dedup_threshold = agent_cfg.get("dedup_threshold", 0.9)
        self.dedup_pairs_per_task = agent_cfg.get("dedup_pairs_per_task", 20)
        self.digest_max_entries = agent_cfg.get("digest_max_entries", 150)
        self.max_iterations = agent_cfg.get("max_iterations", 30)
        self.export_path: str | None = agent_cfg.get("export_path")
        self.max_tool_result_chars = agent_cfg.get("max_tool_result_chars", 20_000)
        self.keep_recent_turns = agent_cfg.get("keep_recent_turns", 3)
        self.pruned_result_chars = agent_cfg.get("pruned_result_chars", 500)
        self.max_history_turns = agent_cfg.get("max_history_turns", 20)

        self.extraction_tools = mcp_tools_to_litellm([t for t in mcp_tools if t.name not in EXTRACTION_HIDDEN])
        self.consolidation_tools = mcp_tools_to_litellm([t for t in mcp_tools if t.name not in PIPELINE_TOOLS])

    # ── Tool calls ────────────────────────────────────────────────────────

    async def _call(self, tool_name: str, args: dict | None = None) -> str:
        """Tool call made by the pipeline itself; an error stops the run."""
        result = await self.session.call_tool(tool_name, args or {})
        text = result.content[0].text if result.content else ""
        if text.startswith("Error"):
            raise RuntimeError(f"{tool_name} failed: {text}")
        return text

    async def _llm_tool_call(self, tool_name: str, raw_args: str, hidden: set[str]) -> str:
        """Tool call requested by the LLM; errors are returned to it so it can adapt."""
        try:
            tool_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            return f"Error: invalid JSON arguments: {raw_args}"
        if tool_name in hidden:
            return f"Error: {tool_name} is not available for this task — the pipeline handles it."

        logger.info("Tool call: %s(%s)", tool_name, _summarise(tool_args))
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            content = result.content[0].text if result.content else ""
        except Exception as exc:
            logger.warning("Tool %s failed: %s", tool_name, exc)
            return f"Error: {exc}"

        logger.debug("  → %s", content[:200])
        if len(content) > self.max_tool_result_chars:
            content = (
                content[: self.max_tool_result_chars]
                + f"\n… [truncated: {len(content)} chars in total — narrow your query]"
            )
        return content

    async def converse(self, task: str, tools: list[dict], hidden: set[str], label: str) -> str:
        """One task = one fresh conversation: the LLM calls tools until it replies with a
        plain-text summary. Returns that summary."""
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        for iteration in range(1, self.max_iterations + 1):
            logger.info("[%s] iteration %d", label, iteration)
            response = await asyncio.to_thread(
                _completion,
                self.llm_cfg,
                model=self.llm_cfg["model"],
                messages=prune_history(
                    messages, self.keep_recent_turns, self.pruned_result_chars, self.max_history_turns
                ),
                tools=tools,
                tool_choice="auto",
                max_tokens=self.llm_cfg.get("max_tokens", 4096),
                temperature=self.llm_cfg.get("temperature", 0),
                timeout=120,
            )
            msg = response.choices[0].message

            usage = getattr(response, "usage", None)
            if usage:
                logger.info(
                    "Tokens — in: %s, out: %s",
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                )

            if not msg.tool_calls:
                summary = (msg.content or "").strip()
                logger.info("[%s] done: %s", label, summary[:1000])
                return summary

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
                    "content": await self._llm_tool_call(tc.function.name, tc.function.arguments, hidden),
                })

        logger.warning("[%s] reached max_iterations (%d) — task stopped.", label, self.max_iterations)
        return ""

    # ── Phase 1: extraction by batches ────────────────────────────────────

    async def digest(self) -> str:
        """Compact view of the current ontology, given to each batch so the LLM reuses
        existing classes and properties instead of re-creating them."""
        n = self.digest_max_entries
        summary = json.loads(await self._call("ontology_summary"))
        classes = json.loads(await self._call("concept_list", {"limit": n}))
        labels = {c["uri"]: c["label"] for c in classes}
        props = json.loads(await self._call(
            "sparql_query", {"query": _PROPERTIES_QUERY.format(ontology_id=self.ontology_id), "limit": n}
        ))
        seeds = json.loads(await self._call("sparql_query", {"query": _SEED_CLASSES_QUERY, "limit": n}))

        lines = [
            "## Current ontology",
            f"{summary.get('classes', 0)} classes, {summary.get('object_properties', 0)} object properties, "
            f"{summary.get('datatype_properties', 0)} datatype properties, {summary.get('individuals', 0)} individuals.",
        ]
        if classes:
            lines.append(f"\n### Classes ({len(classes)} of {summary.get('classes', len(classes))} — "
                         "use concept_search for the others)")
            for c in classes:
                parent = c.get("parent_uri")
                lines.append(f"- {c['label']} <{c['uri']}>"
                             + (f" ⊂ {labels.get(parent) or f'<{parent}>'}" if parent else ""))
        if props["rows"]:
            lines.append(f"\n### Properties ({len(props['rows'])} of {props['row_count']} — "
                         "use property_search for the others)")
            for p in props["rows"]:
                kind = "object" if (p["type"] or "").endswith("ObjectProperty") else "datatype"
                lines.append(f"- {p['label'] or p['p']} <{p['p']}> [{kind}]"
                             + (f" domain <{p['domain']}>" if p["domain"] else "")
                             + (f" range <{p['range']}>" if p["range"] else ""))
        if seeds["rows"]:
            lines.append(f"\n### Seed classes — reference ontologies ({len(seeds['rows'])} of {seeds['row_count']})")
            for s in seeds["rows"]:
                lines.append(f"- {s['label'] or s['c']} <{s['c']}>")
        return "\n".join(lines)

    async def extract(self) -> None:
        batch_number = 0
        while not self.max_batches or batch_number < self.max_batches:
            pending = json.loads(await self._call("chunk_list", {"status": "pending", "limit": self.batch_size}))
            if not pending:
                logger.info("No pending chunks left.")
                return
            batch_number += 1
            ids = [c["id"] for c in pending]
            chunks = json.loads(await self._call("chunk_read_batch", {"chunk_ids": ids}))
            chunks_text = "\n\n".join(
                f"### Chunk {c['id']} (document {c.get('doc_id', '')}, #{c.get('chunk_index', '')})\n{c.get('text', '')}"
                for c in chunks if "error" not in c
            )
            task = BATCH_PROMPT.format(batch_number=batch_number, digest=await self.digest(), chunks=chunks_text)
            await self.converse(task, self.extraction_tools, EXTRACTION_HIDDEN, f"batch {batch_number}")

            # If the batch failed (LLM error), the exception stops the run before this point:
            # its chunks stay pending and are picked up again on the next run.
            for chunk_id in ids:
                if await self._call("chunk_mark_processed", {"chunk_id": chunk_id}) != "ok":
                    raise RuntimeError(f"Could not mark chunk {chunk_id} as processed.")
            summary = json.loads(await self._call("ontology_summary"))
            logger.info(
                "Batch %d done — chunks %s/%s processed, %s classes, %s object properties, %s individuals.",
                batch_number, summary.get("chunks_processed", "?"), summary.get("chunks_total", "?"),
                summary.get("classes"), summary.get("object_properties"), summary.get("individuals"),
            )
        logger.info("Reached max_batches (%d) — remaining chunks stay pending.", self.max_batches)

    # ── Phase 2: consolidation ────────────────────────────────────────────

    async def deduplicate(self) -> None:
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

        def describe(uri: str) -> str:
            c = by_uri[uri]
            definition = (c.get("definition") or "")[:200]
            return f'"{c["label"]}" <{uri}>' + (f" — {definition}" if definition else "")

        ordered = sorted(pairs.items(), key=lambda kv: -kv[1])
        for start in range(0, len(ordered), self.dedup_pairs_per_task):
            group = ordered[start : start + self.dedup_pairs_per_task]
            text = "\n".join(f"- score {score:.3f}\n  A: {describe(a)}\n  B: {describe(b)}" for (a, b), score in group)
            await self.converse(
                DEDUP_PROMPT.format(pairs=text), self.consolidation_tools, PIPELINE_TOOLS,
                f"dedup {start // self.dedup_pairs_per_task + 1}",
            )

    async def check(self) -> None:
        report = json.loads(await self._call("ontology_check"))
        orphans = json.loads(await self._call("ontology_orphans"))
        if error := report.get("reasoner", {}).get("error"):
            logger.warning("Reasoner unavailable, only the SPARQL checks ran: %s", error)
        if report["issues"] == 0 and not orphans:
            logger.info("Consistency check: no issues, no orphans.")
            return

        logger.info("Consistency check: %d issues, %d orphans — asking the LLM to fix them.", report["issues"], len(orphans))
        await self.converse(
            CHECK_PROMPT.format(report=json.dumps({**report, "orphans": orphans}, indent=1, ensure_ascii=False)),
            self.consolidation_tools, PIPELINE_TOOLS, "check",
        )
        final = json.loads(await self._call("ontology_check"))
        (logger.info if final["issues"] == 0 else logger.warning)(
            "Consistency check after fixes: %d issues left.", final["issues"]
        )

    # ── Run ───────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.extract()
        if self.consolidate:
            await self.deduplicate()
            await self.check()

        if self.export_path:
            with open(self.export_path, "w", encoding="utf-8") as f:
                f.write(await self._call("ontology_export"))
            logger.info("Ontology exported to %s", self.export_path)

        summary = json.loads(await self._call("ontology_summary"))
        logger.info(
            "Finished — %s classes, %s object properties, %s datatype properties, %s individuals; "
            "chunks %s/%s processed.",
            summary.get("classes"), summary.get("object_properties"), summary.get("datatype_properties"),
            summary.get("individuals"), summary.get("chunks_processed", "?"), summary.get("chunks_total", "?"),
        )


def _summarise(args: dict) -> str:
    """Compact one-line representation of tool arguments for logging."""
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:40]!r}" if len(s) > 40 else f"{k}={s!r}")
    return ", ".join(parts)


# ─── Entry point ──────────────────────────────────────────────────────────────


async def _call_or_exit(session: ClientSession, tool_name: str, args: dict) -> str:
    result = await session.call_tool(tool_name, args)
    text = result.content[0].text if result.content else ""
    if text.startswith("Error"):
        raise SystemExit(f"{tool_name} failed: {text}")
    return text


async def select_context(session: ClientSession, olaf_cfg: dict) -> None:
    """Create (if needed) and activate the configured ontology, then select the chunk
    collection. Session state only: other agents on the same server are unaffected."""
    ontology_id = olaf_cfg["ontology_id"]
    create_args = {"ontology_id": ontology_id, "name": olaf_cfg.get("ontology_name", ontology_id)}
    if olaf_cfg.get("base_uri"):
        create_args["base_uri"] = olaf_cfg["base_uri"]
    created = json.loads(await _call_or_exit(session, "ontology_create", create_args))
    await _call_or_exit(session, "ontology_switch", {"ontology_id": ontology_id})
    logger.info("Active ontology: %s%s", ontology_id, " (created)" if created.get("created") else "")

    if collection := olaf_cfg.get("collection"):
        args: dict = {"collection": collection}
        if olaf_cfg.get("field_mapping"):
            args["field_mapping"] = olaf_cfg["field_mapping"]
        counts = json.loads(await _call_or_exit(session, "chunk_collection_switch", args))
        logger.info(
            "Chunk collection: %s (%s/%s processed)",
            collection, counts["chunks_processed"], counts["chunks_total"],
        )
    else:
        logger.info("Chunk collection: server default ([qdrant].collection)")


async def main_async(config: dict) -> None:
    olaf_url = config["olaf"]["url"]
    logger.info("Connecting to OLAF at %s", olaf_url)

    async with sse_client(olaf_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await select_context(session, config["olaf"])
            tools_result = await session.list_tools()
            await BuildingAgent(session, config, tools_result.tools).run()


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
        description="olaf_building_agent — autonomous OWL/RDFS ontology construction"
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

    _run(main_async(config))


if __name__ == "__main__":
    main()
