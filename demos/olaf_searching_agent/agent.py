"""
olaf_searching_agent — LLM agent that answers natural-language questions over an OLAF ontology.

The LLM searches the ontology and its source chunks with the read-only OLAF MCP tools,
writes and runs SPARQL queries (sparql_query) when needed, and answers in natural language.
Python just: connects to OLAF, exposes the read-only tools to LiteLLM, and runs the loop.

Usage:
    python agent.py                                   # interactive session
    python agent.py --question "Which bonds fund renewable projects?"
    python agent.py --config my.toml --show-sparql
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

from prompts import WRAP_UP_PROMPT, build_system_prompt

logger = logging.getLogger(__name__)

# Tools that only read the store. Anything else (create/update/merge/delete/load/mark…) is
# hidden from the LLM, and refused if it is called anyway.
READ_ONLY_TOOLS = {
    "chunk_list",
    "chunk_read",
    "chunk_read_batch",
    "concept_list",
    "concept_get",
    "concept_search",
    "concept_semantic_search",
    "property_get",
    "property_search",
    "relation_search",
    "relation_sources",
    "seed_list",
    "ontology_list",
    "ontology_summary",
    "ontology_export",
    "sparql_query",
}

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}
RESET_COMMANDS = {"/reset", "/new"}


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


# ─── Agent ────────────────────────────────────────────────────────────────────


class SearchingAgent:
    """Holds the conversation, so follow-up questions can refer to previous answers."""

    def __init__(self, session: ClientSession, config: dict, tools: list[dict], show_sparql: bool):
        self.session = session
        self.llm_cfg = config["litellm"]
        agent_cfg = config.get("agent", {})
        self.max_iterations = agent_cfg.get("max_iterations", 20)
        self.max_tool_result_chars = agent_cfg.get("max_tool_result_chars", 20_000)
        self.keep_recent_turns = agent_cfg.get("keep_recent_turns", 4)
        self.pruned_result_chars = agent_cfg.get("pruned_result_chars", 500)
        self.show_sparql = show_sparql
        self.tools = tools
        self.system_prompt = build_system_prompt(config["olaf"]["ontology_id"])
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

    async def _complete(self, tool_choice: str = "auto"):
        return await asyncio.to_thread(
            _completion,
            self.llm_cfg,
            model=self.llm_cfg["model"],
            messages=prune_history(self.messages, self.keep_recent_turns, self.pruned_result_chars),
            tools=self.tools,
            tool_choice=tool_choice,
            max_tokens=self.llm_cfg.get("max_tokens", 4096),
            temperature=self.llm_cfg.get("temperature", 0),
            timeout=self.llm_cfg.get("timeout", 120),
        )

    async def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})

        for iteration in range(1, self.max_iterations + 1):
            logger.debug("--- Iteration %d ---", iteration)
            response = await self._complete()
            msg = response.choices[0].message

            usage = getattr(response, "usage", None)
            if usage:
                logger.debug(
                    "Tokens — in: %s, out: %s",
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                )

            # No tool calls → this is the answer
            if not msg.tool_calls:
                answer = msg.content or ""
                self.messages.append({"role": "assistant", "content": answer})
                return answer

            self.messages.append({
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
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": await self._call_tool(tc.function.name, tc.function.arguments),
                })

        # Budget exhausted: force a final answer from what was gathered.
        logger.warning("Reached max_iterations (%d) — asking for a final answer.", self.max_iterations)
        self.messages.append({"role": "user", "content": WRAP_UP_PROMPT})
        response = await self._complete(tool_choice="none")
        answer = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    async def _call_tool(self, tool_name: str, raw_args: str) -> str:
        try:
            tool_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            return f"Error: invalid JSON arguments: {raw_args}"

        if tool_name not in READ_ONLY_TOOLS:
            logger.warning("Refused non read-only tool: %s", tool_name)
            return f"Error: tool '{tool_name}' is not available — this agent has read-only access."

        logger.info("Tool call: %s(%s)", tool_name, _summarise(tool_args))
        if tool_name == "sparql_query" and self.show_sparql:
            _print_sparql(tool_args.get("query", ""))

        try:
            result = await self.session.call_tool(tool_name, tool_args)
            content = result.content[0].text if result.content else ""
        except Exception as exc:
            logger.warning("Tool %s failed: %s", tool_name, exc)
            return f"Error: {exc}"

        logger.debug("  → %s", content[:300])
        if len(content) > self.max_tool_result_chars:
            content = (
                content[: self.max_tool_result_chars]
                + f"\n… [truncated: {len(content)} chars in total — narrow your query]"
            )
        return content


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


def _summarise(args: dict) -> str:
    """Compact one-line representation of tool arguments for logging."""
    parts = []
    for k, v in args.items():
        s = " ".join(str(v).split())
        parts.append(f"{k}={s[:60]!r}…" if len(s) > 60 else f"{k}={s!r}")
    return ", ".join(parts)


def _print_sparql(query: str) -> None:
    print("┌── SPARQL " + "─" * 50, file=sys.stderr)
    for line in query.strip().splitlines():
        print(f"│ {line}", file=sys.stderr)
    print("└" + "─" * 60, file=sys.stderr)


# ─── Entry point ──────────────────────────────────────────────────────────────


async def interactive_loop(agent: SearchingAgent) -> None:
    print("OLAF searching agent — ask a question ('/reset' to start over, 'exit' to quit).")
    while True:
        try:
            # input() runs in a thread so the SSE connection keeps being served meanwhile.
            question = (await asyncio.to_thread(input, "\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            return
        if question.lower() in RESET_COMMANDS:
            agent.reset()
            print("Conversation reset.")
            continue
        try:
            print("\n" + await agent.ask(question))
        except Exception as exc:
            logger.error("Failed to answer: %s", exc)


async def main_async(config: dict, question: str | None, show_sparql: bool) -> None:
    olaf_url = config["olaf"]["url"]
    ontology_id = config["olaf"]["ontology_id"]
    logger.info("Connecting to OLAF at %s", olaf_url)

    async with sse_client(olaf_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Select the ontology for this connection (session state only, nothing is written).
            switch = await session.call_tool("ontology_switch", {"ontology_id": ontology_id})
            switch_text = switch.content[0].text if switch.content else ""
            if switch_text.startswith("Error"):
                raise SystemExit(f"Cannot use ontology '{ontology_id}': {switch_text}")
            logger.info("Active ontology: %s", ontology_id)

            # Source chunks must be read from the collection the ontology was built from.
            if collection := config["olaf"].get("collection"):
                args: dict = {"collection": collection}
                if config["olaf"].get("field_mapping"):
                    args["field_mapping"] = config["olaf"]["field_mapping"]
                switch = await session.call_tool("chunk_collection_switch", args)
                switch_text = switch.content[0].text if switch.content else ""
                if switch_text.startswith("Error"):
                    raise SystemExit(f"Cannot use collection '{collection}': {switch_text}")
                logger.info("Chunk collection: %s", collection)

            tools_result = await session.list_tools()
            read_only = [t for t in tools_result.tools if t.name in READ_ONLY_TOOLS]
            missing = READ_ONLY_TOOLS - {t.name for t in read_only}
            if "sparql_query" in missing:
                logger.warning("The OLAF server has no sparql_query tool — update the server.")
            logger.info("Loaded %d read-only OLAF tools.", len(read_only))

            agent = SearchingAgent(session, config, mcp_tools_to_litellm(read_only), show_sparql)
            if question:
                print(await agent.ask(question))
            else:
                await interactive_loop(agent)


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
        description="olaf_searching_agent — natural-language Q&A over an OLAF ontology"
    )
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--question", "-q", default=None, help="Ask one question and exit")
    parser.add_argument(
        "--show-sparql", action=argparse.BooleanOptionalAction, default=None,
        help="Print generated SPARQL queries (default: [agent] show_sparql in config)",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    agent_cfg = config.get("agent", {})
    log_level = args.log_level or agent_cfg.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    show_sparql = args.show_sparql if args.show_sparql is not None else agent_cfg.get("show_sparql", True)

    _run(main_async(config, args.question, show_sparql))


if __name__ == "__main__":
    main()
