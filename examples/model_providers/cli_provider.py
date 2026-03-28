from __future__ import annotations

import argparse
import asyncio

from agents import Agent, CLIProvider, RunConfig, Runner, function_tool, set_tracing_disabled

"""Run an agent through a locally installed CLI runtime such as Codex, Gemini, or Copilot.

Examples:
uv run examples/model_providers/cli_provider.py --model codex
uv run examples/model_providers/cli_provider.py --model gemini:gemini-2.5-pro --execution-mode sdk_controlled
uv run examples/model_providers/cli_provider.py --model copilot:gpt-4.1 --execution-mode cli_autonomous
uv run examples/model_providers/cli_provider.py --model gemini:gemini-2.5-flash --transport acp

Requirements:
- A supported local CLI must be installed: codex, gemini, or copilot
- That CLI must already be authenticated locally
"""

set_tracing_disabled(disabled=True)


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


async def main(model: str, execution_mode: str, transport: str) -> None:
    provider = CLIProvider(
        default_model_name=model,
        execution_mode=execution_mode,  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )
    agent = Agent(
        name="CLI Assistant",
        instructions="Answer concisely. Use tools when useful.",
        tools=[get_weather],
    )

    result = await Runner.run(
        agent,
        "What's the weather in Tokyo? Mention which runtime path you used.",
        run_config=RunConfig(model_provider=provider),
    )
    print(result.final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="codex",
        help="Model name in '<vendor>' or '<vendor>:<model>' form.",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["sdk_controlled", "cli_autonomous"],
        default="cli_autonomous",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["auto", "acp", "json", "jsonl", "stream_json"],
        default="auto",
    )
    args = parser.parse_args()

    asyncio.run(main(args.model, args.execution_mode, args.transport))
