"""
olaf_building_agent — LLM agent that builds an OWL/RDFS ontology using OLAF MCP tools + LiteLLM.

The LLM drives the entire workflow via tool use (tool-calling / function-calling).
Python just: connects to OLAF, converts MCP tools to the LiteLLM format, and runs the loop.

Usage:
    python agent.py                      # uses config.toml in current directory
    python agent.py --config my.toml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tomllib
from typing import Any

import litellm
from mcp import ClientSession
from mcp.client.sse import sse_client

from prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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


# ─── Agent loop ───────────────────────────────────────────────────────────────


async def run_agent(session: ClientSession, config: dict) -> None:
    llm_cfg = config["litellm"]
    max_iterations = config["agent"].get("max_iterations", 50)
    export_path: str | None = config["agent"].get("export_path")

    # Discover available OLAF tools and convert them
    tools_result = await session.list_tools()
    tools = mcp_tools_to_litellm(tools_result.tools)
    logger.info("Loaded %d OLAF tools.", len(tools))

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Build the ontology from the available Qdrant chunks."},
    ]

    for iteration in range(1, max_iterations + 1):
        logger.info("--- Iteration %d ---", iteration)

        response = await asyncio.to_thread(
            litellm.completion,
            model=llm_cfg["model"],
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=llm_cfg.get("max_tokens", 4096),
            temperature=llm_cfg.get("temperature", 0),
            timeout=120,
        )

        msg = response.choices[0].message

        # Log token usage
        usage = getattr(response, "usage", None)
        if usage:
            logger.debug(
                "Tokens — in: %s, out: %s",
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
            )

        # No tool calls → agent is done
        if not msg.tool_calls:
            logger.info("Agent finished.")
            if msg.content:
                print(msg.content)
            break

        # Add the assistant message (with its tool_calls) to history
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

        # Execute all tool calls (sequentially — results are independent but order matters for context)
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            logger.info("Tool call: %s(%s)", tool_name, _summarise(tool_args))

            try:
                result = await session.call_tool(tool_name, tool_args)
                content = result.content[0].text if result.content else ""
            except Exception as exc:
                content = f"Error: {exc}"
                logger.warning("Tool %s failed: %s", tool_name, exc)

            logger.debug("  → %s", content[:200])

            if tool_name == "ontology_export" and export_path and content:
                with open(export_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info("Ontology exported to %s", export_path)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": content,
            })

    else:
        logger.warning("Reached max_iterations (%d) — agent stopped.", max_iterations)


def _summarise(args: dict) -> str:
    """Compact one-line representation of tool arguments for logging."""
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:40]!r}" if len(s) > 40 else f"{k}={s!r}")
    return ", ".join(parts)


# ─── Entry point ──────────────────────────────────────────────────────────────


async def main_async(config: dict) -> None:
    olaf_url = config["olaf"]["url"]
    logger.info("Connecting to OLAF at %s", olaf_url)

    async with sse_client(olaf_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await run_agent(session, config)


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

    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
