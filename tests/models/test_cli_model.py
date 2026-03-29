from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

import agents.extensions.models.cli_model as cli_model_module
from agents import CLIProvider, UserError
from agents.extensions.models import CLIModel
from agents.extensions.models.cli_model import (
    CLIModelConfig,
    _build_response_obj,
    _parse_controlled_envelope,
)
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.usage import Usage


async def _get_response(model: CLIModel):
    return await model.get_response(
        system_instructions="Follow the test harness.",
        input="Test input",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )


async def _get_stream_events(model: CLIModel, *, previous_response_id: str | None = None):
    return [
        event
        async for event in model.stream_response(
            system_instructions="Follow the test harness.",
            input="Test input",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=previous_response_id,
            conversation_id=None,
            prompt=None,
        )
    ]


@pytest.mark.asyncio
async def test_gemini_cli_model_autonomous_parses_json_response(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_stream_jsonl_command(*, command, cwd, env, timeout_seconds):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["timeout_seconds"] = timeout_seconds
        payloads = [
            {"type": "init", "session_id": "gemini-session-1", "model": "gemini-2.5-pro"},
            {"type": "message", "role": "assistant", "content": "Gemini says hello", "delta": True},
            {
                "type": "result",
                "status": "success",
                "stats": {
                    "total_tokens": 15,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cached": 2,
                },
            },
        ]
        for payload in payloads:
            yield payload

    monkeypatch.setattr(cli_model_module, "_stream_jsonl_command", fake_stream_jsonl_command)
    monkeypatch.setattr(CLIModel, "_resolve_executable", lambda self, config, name: f"/bin/{name}")

    model = CLIModel(CLIModelConfig(vendor="gemini", model_name="gemini-2.5-pro"))
    response = await _get_response(model)

    assert response.response_id == "gemini-session-1"
    assert response.provider_response_id == "gemini-session-1"
    assert response.provider_session_id == "gemini-session-1"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 15
    assert isinstance(response.output[0], ResponseOutputMessage)
    assert response.output[0].content[0].text == "Gemini says hello"

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:5] == ["/bin/gemini", "-p", command[2], "-o", "stream-json"]
    assert "-m" in command
    assert "gemini-2.5-pro" in command


@pytest.mark.asyncio
async def test_gemini_cli_model_sdk_controlled_returns_function_calls(monkeypatch) -> None:
    async def fake_run_json_command(*, command, cwd, env, timeout_seconds):
        del command, cwd, env, timeout_seconds
        return {
            "session_id": "gemini-session-2",
            "response": json.dumps(
                {
                    "assistant_message": "Calling the tool now.",
                    "tool_calls": [
                        {
                            "name": "lookup_customer",
                            "arguments_json": '{"customer_id":"cust_42"}',
                        }
                    ],
                    "reasoning_summary": "Need current customer data.",
                }
            ),
        }

    monkeypatch.setattr(cli_model_module, "_run_json_command", fake_run_json_command)
    monkeypatch.setattr(CLIModel, "_resolve_executable", lambda self, config, name: f"/bin/{name}")

    model = CLIModel(CLIModelConfig(vendor="gemini", execution_mode="sdk_controlled"))
    response = await _get_response(model)

    assert isinstance(response.output[0], ResponseReasoningItem)
    assert isinstance(response.output[1], ResponseFunctionToolCall)
    assert response.output[1].name == "lookup_customer"
    assert response.output[1].arguments == '{"customer_id":"cust_42"}'
    assert isinstance(response.output[2], ResponseOutputMessage)


def test_parse_controlled_envelope_accepts_fenced_json_with_trailing_text() -> None:
    envelope = _parse_controlled_envelope(
        """```json
{
  "assistant_message": null,
  "tool_calls": [],
  "reasoning_summary": "Need current customer data."
}
```

Final answer here.
"""
    )

    assert envelope.reasoning_summary == "Need current customer data."
    assert envelope.assistant_message == "Final answer here."


@pytest.mark.asyncio
async def test_gemini_cli_model_sdk_controlled_uses_trailing_text_as_assistant_message(
    monkeypatch,
) -> None:
    async def fake_run_json_command(*, command, cwd, env, timeout_seconds):
        del command, cwd, env, timeout_seconds
        return {
            "session_id": "gemini-session-2b",
            "response": """```json
{
  "assistant_message": null,
  "tool_calls": [],
  "reasoning_summary": "Need current customer data."
}
```

Final assistant output.
""",
        }

    monkeypatch.setattr(cli_model_module, "_run_json_command", fake_run_json_command)
    monkeypatch.setattr(CLIModel, "_resolve_executable", lambda self, config, name: f"/bin/{name}")

    model = CLIModel(CLIModelConfig(vendor="gemini", execution_mode="sdk_controlled"))
    response = await _get_response(model)

    assert isinstance(response.output[0], ResponseReasoningItem)
    assert isinstance(response.output[1], ResponseOutputMessage)
    assert response.output[1].content[0].text == "Final assistant output."


@pytest.mark.asyncio
async def test_copilot_cli_model_uses_model_flag_and_parses_jsonl(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_stream_jsonl_command(*, command, cwd, env, timeout_seconds):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["timeout_seconds"] = timeout_seconds
        payloads = [
            {
                "type": "assistant.reasoning_delta",
                "data": {"reasoningId": "reason-1", "deltaContent": "Plan: "},
            },
            {
                "type": "assistant.reasoning",
                "data": {"reasoningId": "reason-1", "content": "Plan: inspect repo."},
            },
            {
                "type": "assistant.message",
                "data": {"messageId": "msg-1", "content": "Copilot final answer"},
            },
            {"type": "result", "sessionId": "copilot-session-1"},
        ]
        for payload in payloads:
            yield payload

    monkeypatch.setattr(cli_model_module, "_stream_jsonl_command", fake_stream_jsonl_command)
    monkeypatch.setattr(
        CLIModel, "_resolve_copilot_command_prefix", lambda self, config: ["copilot"]
    )

    model = CLIModel(CLIModelConfig(vendor="copilot", model_name="gpt-4.1"))
    response = await model.get_response(
        system_instructions=None,
        input="Test input",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id="copilot-session-1",
        conversation_id=None,
        prompt=None,
    )

    assert response.response_id == "copilot-session-1"
    assert response.provider_session_id == "copilot-session-1"
    assert isinstance(response.output[0], ResponseReasoningItem)
    assert response.output[0].summary[0].text == "Plan: inspect repo."
    assert isinstance(response.output[1], ResponseOutputMessage)
    assert response.output[1].content[0].text == "Copilot final answer"

    command = captured["command"]
    assert isinstance(command, list)
    assert "--stream" in command
    assert "on" in command
    assert "--model" in command
    assert "gpt-4.1" in command
    assert any(part.startswith("--resume=") for part in command)


@pytest.mark.asyncio
async def test_cli_model_uses_runtime_overrides_from_model_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run_json_command(*, command, cwd, env, timeout_seconds):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout_seconds"] = timeout_seconds
        return {
            "session_id": "gemini-session-3",
            "response": json.dumps(
                {
                    "assistant_message": "Done.",
                    "tool_calls": [],
                    "reasoning_summary": "Planning only.",
                }
            ),
        }

    monkeypatch.setattr(cli_model_module, "_run_json_command", fake_run_json_command)
    monkeypatch.setattr(CLIModel, "_resolve_executable", lambda self, config, name: f"/bin/{name}")

    model = CLIModel(CLIModelConfig(vendor="gemini"))
    response = await model.get_response(
        system_instructions="Follow the test harness.",
        input="Test input",
        model_settings=ModelSettings(
            extra_args={
                "cli": {
                    "execution_mode": "sdk_controlled",
                    "cwd": "/tmp/cli-runtime",
                    "env": {"CLI_ENV_TEST": "yes"},
                    "timeout_seconds": 42,
                    "extra_args": ["--raw-output"],
                    "model_name": "gemini-2.5-flash",
                }
            }
        ),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )

    assert response.provider_session_id == "gemini-session-3"
    assert isinstance(response.output[0], ResponseReasoningItem)
    command = captured["command"]
    env = captured["env"]
    assert isinstance(command, list)
    assert "-m" in command
    assert "gemini-2.5-flash" in command
    assert "--approval-mode" in command
    assert "--raw-output" in command
    assert captured["cwd"] == str(Path("/tmp/cli-runtime").resolve())
    assert captured["timeout_seconds"] == 42.0
    assert isinstance(env, dict)
    assert env["CLI_ENV_TEST"] == "yes"


def test_cli_provider_accepts_default_model_name() -> None:
    provider = CLIProvider(default_model_name="gemini:gemini-2.5-pro")
    model = provider.get_model(None)
    assert isinstance(model, CLIModel)
    assert model.config.vendor == "gemini"
    assert model.config.model_name == "gemini-2.5-pro"


def test_cli_provider_accepts_transport_configuration() -> None:
    provider = CLIProvider(default_model_name="copilot:gpt-4.1", transport="acp")
    model = provider.get_model(None)
    assert isinstance(model, CLIModel)
    assert model.config.vendor == "copilot"
    assert model.config.transport == "acp"


def test_cli_provider_accepts_command_prefix_configuration() -> None:
    provider = CLIProvider(
        default_model_name="codex:gpt-5.4/high",
        transport="acp",
        command_prefix=["npx", "-y", "@zed-industries/codex-acp"],
    )
    model = provider.get_model(None)
    assert isinstance(model, CLIModel)
    assert model.config.vendor == "codex"
    assert model.config.command_prefix == ("npx", "-y", "@zed-industries/codex-acp")


def test_controlled_envelope_schema_is_strict() -> None:
    schema = cli_model_module._controlled_envelope_json_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    tool_call_schema = schema["$defs"]["_ControlledToolCall"]
    assert tool_call_schema["type"] == "object"
    assert tool_call_schema["additionalProperties"] is False


@pytest.mark.asyncio
async def test_cli_model_rejects_invalid_runtime_overrides() -> None:
    model = CLIModel(CLIModelConfig(vendor="gemini"))

    with pytest.raises(UserError, match="timeout_seconds"):
        await model.get_response(
            system_instructions=None,
            input="Test input",
            model_settings=ModelSettings(extra_args={"cli_timeout_seconds": 0}),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )

    with pytest.raises(UserError, match="CLI transport"):
        await model.get_response(
            system_instructions=None,
            input="Test input",
            model_settings=ModelSettings(extra_args={"cli_transport": "bogus"}),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )

    with pytest.raises(UserError, match="command_prefix"):
        await model.get_response(
            system_instructions=None,
            input="Test input",
            model_settings=ModelSettings(extra_args={"cli_command_prefix": []}),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )


@pytest.mark.asyncio
async def test_codex_cli_model_rejects_sdk_controlled_mode() -> None:
    model = CLIModel(
        CLIModelConfig(vendor="codex", transport="acp", execution_mode="sdk_controlled")
    )

    with pytest.raises(UserError, match="sdk_controlled thread runtime was removed"):
        await model.get_response(
            system_instructions=None,
            input="Test input",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )


@pytest.mark.asyncio
async def test_codex_cli_model_streams_native_acp(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_stream_acp_command(
        *,
        command,
        cwd,
        env,
        timeout_seconds,
        prompt,
        previous_response_id,
    ):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout_seconds"] = timeout_seconds
        captured["prompt"] = prompt
        captured["previous_response_id"] = previous_response_id
        payloads = [
            {
                "type": "acp.session_initialized",
                "session_id": "codex-acp-session-1",
                "initialize": {"protocolVersion": 1},
                "session": {"sessionId": "codex-acp-session-1"},
            },
            {
                "type": "acp.session_update",
                "session_id": "codex-acp-session-1",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-acp-codex-1",
                    "title": "run tests",
                    "kind": "exec",
                    "status": "pending",
                    "rawInput": {"command": "pytest"},
                },
            },
            {
                "type": "acp.session_update",
                "session_id": "codex-acp-session-1",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tool-acp-codex-1",
                    "status": "completed",
                    "rawOutput": {"exitCode": 0},
                },
            },
            {
                "type": "acp.session_update",
                "session_id": "codex-acp-session-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Codex ACP OK"},
                },
            },
            {
                "type": "acp.prompt_result",
                "session_id": "codex-acp-session-1",
                "result": {"stopReason": "end_turn"},
            },
        ]
        for payload in payloads:
            yield payload

    monkeypatch.setattr(cli_model_module, "_stream_acp_command", fake_stream_acp_command)
    monkeypatch.setattr(
        CLIModel,
        "_resolve_codex_acp_adapter_command",
        lambda self, config: ["codex-acp"],
    )

    model = CLIModel(CLIModelConfig(vendor="codex", model_name="gpt-5.4/high"))
    events = await _get_stream_events(model)

    completed = events[-1]
    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.id == "codex-acp-session-1"
    assert completed.response.output[0].type == "mcp_call"
    assert completed.response.output[0].name == "run tests"
    assert completed.response.output[1].content[0].text == "Codex ACP OK"

    command = captured["command"]
    assert isinstance(command, list)
    assert command == ["codex-acp"]
    assert captured["previous_response_id"] is None


@pytest.mark.asyncio
async def test_codex_cli_model_get_response_uses_acp_stream_for_auto_transport(monkeypatch) -> None:
    async def fake_stream_acp_command(
        *,
        command,
        cwd,
        env,
        timeout_seconds,
        prompt,
        previous_response_id,
    ):
        del command, cwd, env, timeout_seconds, prompt, previous_response_id
        for payload in [
            {
                "type": "acp.session_initialized",
                "session_id": "codex-acp-session-2",
                "initialize": {"protocolVersion": 1},
                "session": {"sessionId": "codex-acp-session-2"},
            },
            {
                "type": "acp.session_update",
                "session_id": "codex-acp-session-2",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Codex ACP OK"},
                },
            },
            {
                "type": "acp.prompt_result",
                "session_id": "codex-acp-session-2",
                "result": {"stopReason": "end_turn"},
            },
        ]:
            yield payload

    monkeypatch.setattr(cli_model_module, "_stream_acp_command", fake_stream_acp_command)
    monkeypatch.setattr(
        CLIModel,
        "_resolve_codex_acp_adapter_command",
        lambda self, config: ["codex-acp"],
    )

    response = await _get_response(CLIModel(CLIModelConfig(vendor="codex")))

    assert response.response_id == "codex-acp-session-2"
    assert isinstance(response.output[0], ResponseOutputMessage)
    assert response.output[0].content[0].text == "Codex ACP OK"


@pytest.mark.asyncio
async def test_gemini_cli_model_streams_native_jsonl(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_stream_jsonl_command(*, command, cwd, env, timeout_seconds):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout_seconds"] = timeout_seconds
        payloads = [
            {
                "type": "init",
                "session_id": "gemini-stream-1",
                "model": "gemini-2.5-flash",
            },
            {"type": "message", "role": "user", "content": "Test input"},
            {
                "type": "tool_use",
                "tool_name": "list_directory",
                "tool_id": "tool-gemini-1",
                "parameters": {"dir_path": "."},
            },
            {
                "type": "tool_result",
                "tool_id": "tool-gemini-1",
                "status": "success",
                "output": "Listed 7 item(s).",
            },
            {"type": "message", "role": "assistant", "content": "Gemini ", "delta": True},
            {"type": "message", "role": "assistant", "content": "final answer", "delta": True},
            {
                "type": "result",
                "status": "success",
                "stats": {
                    "total_tokens": 12,
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cached": 1,
                },
            },
        ]
        for payload in payloads:
            yield payload

    monkeypatch.setattr(cli_model_module, "_stream_jsonl_command", fake_stream_jsonl_command)
    monkeypatch.setattr(CLIModel, "_resolve_executable", lambda self, config, name: f"/bin/{name}")

    model = CLIModel(CLIModelConfig(vendor="gemini", model_name="gemini-2.5-flash"))
    events = await _get_stream_events(model)

    assert events[0].type == "response.created"
    assert events[1].type == "response.in_progress"
    text_deltas = [event.delta for event in events if isinstance(event, ResponseTextDeltaEvent)]
    assert text_deltas == ["Gemini ", "final answer"]

    completed = events[-1]
    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.id == "gemini-stream-1"
    assert completed.response.usage.input_tokens == 10
    assert completed.response.usage.output_tokens == 2
    assert completed.response.usage.total_tokens == 12
    assert completed.response.output[0].type == "mcp_call"
    assert completed.response.output[0].name == "list_directory"
    assert completed.response.output[0].output == "Listed 7 item(s)."
    assert completed.response.output[1].content[0].text == "Gemini final answer"

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:5] == ["/bin/gemini", "-p", command[2], "-o", "stream-json"]
    assert "-m" in command
    assert "gemini-2.5-flash" in command


@pytest.mark.asyncio
async def test_gemini_cli_model_segments_messages_around_tool_use(monkeypatch) -> None:
    async def fake_stream_jsonl_command(*, command, cwd, env, timeout_seconds):
        del command, cwd, env, timeout_seconds
        payloads = [
            {"type": "init", "session_id": "gemini-segment-1", "model": "gemini-2.5-flash"},
            {"type": "message", "role": "assistant", "content": "Planning...", "delta": True},
            {
                "type": "tool_use",
                "tool_name": "list_directory",
                "tool_id": "tool-gemini-segment-1",
                "parameters": {"dir_path": "."},
            },
            {
                "type": "tool_result",
                "tool_id": "tool-gemini-segment-1",
                "status": "success",
                "output": "Listed 3 item(s).",
            },
            {"type": "message", "role": "assistant", "content": "Final answer", "delta": True},
            {
                "type": "result",
                "status": "success",
                "stats": {"total_tokens": 8, "input_tokens": 6, "output_tokens": 2, "cached": 0},
            },
        ]
        for payload in payloads:
            yield payload

    monkeypatch.setattr(cli_model_module, "_stream_jsonl_command", fake_stream_jsonl_command)
    monkeypatch.setattr(CLIModel, "_resolve_executable", lambda self, config, name: f"/bin/{name}")

    model = CLIModel(CLIModelConfig(vendor="gemini", model_name="gemini-2.5-flash"))
    response = await _get_response(model)

    assert response.response_id == "gemini-segment-1"
    assert response.output[0].content[0].text == "Planning..."
    assert response.output[1].type == "mcp_call"
    assert response.output[2].content[0].text == "Final answer"


@pytest.mark.asyncio
async def test_gemini_cli_model_streams_native_acp(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_stream_acp_command(
        *,
        command,
        cwd,
        env,
        timeout_seconds,
        prompt,
        previous_response_id,
    ):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout_seconds"] = timeout_seconds
        captured["prompt"] = prompt
        captured["previous_response_id"] = previous_response_id
        payloads = [
            {
                "type": "acp.session_initialized",
                "session_id": "gemini-acp-session-1",
                "initialize": {"protocolVersion": 1},
                "session": {"sessionId": "gemini-acp-session-1"},
            },
            {
                "type": "acp.session_update",
                "session_id": "gemini-acp-session-1",
                "update": {
                    "sessionUpdate": "plan",
                    "entries": [
                        {
                            "content": "Inspect repository",
                            "status": "in_progress",
                            "priority": "high",
                        }
                    ],
                },
            },
            {
                "type": "acp.session_update",
                "session_id": "gemini-acp-session-1",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-acp-1",
                    "title": "list_directory",
                    "kind": "read",
                    "status": "pending",
                    "rawInput": {"dir_path": "."},
                },
            },
            {
                "type": "acp.session_update",
                "session_id": "gemini-acp-session-1",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tool-acp-1",
                    "status": "completed",
                    "rawOutput": {"entries": ["README.md"]},
                },
            },
            {
                "type": "acp.session_update",
                "session_id": "gemini-acp-session-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Gemini ACP "},
                },
            },
            {
                "type": "acp.session_update",
                "session_id": "gemini-acp-session-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "final answer"},
                },
            },
            {
                "type": "acp.prompt_result",
                "session_id": "gemini-acp-session-1",
                "result": {"stopReason": "end_turn"},
            },
        ]
        for payload in payloads:
            yield payload

    monkeypatch.setattr(cli_model_module, "_stream_acp_command", fake_stream_acp_command)
    monkeypatch.setattr(CLIModel, "_resolve_executable", lambda self, config, name: f"/bin/{name}")

    model = CLIModel(
        CLIModelConfig(vendor="gemini", model_name="gemini-2.5-flash", transport="acp")
    )
    events = await _get_stream_events(model)

    completed = events[-1]
    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.id == "gemini-acp-session-1"
    assert completed.response.output[0].summary[0].text == "[in_progress] Inspect repository"
    assert completed.response.output[1].type == "mcp_call"
    assert completed.response.output[1].name == "list_directory"
    assert completed.response.output[1].output == '{"entries": ["README.md"]}'
    assert completed.response.output[2].content[0].text == "Gemini ACP final answer"

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:4] == ["/bin/gemini", "-m", "gemini-2.5-flash", "--acp"]
    assert captured["previous_response_id"] is None


@pytest.mark.asyncio
async def test_copilot_cli_model_streams_native_jsonl(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_stream_jsonl_command(*, command, cwd, env, timeout_seconds):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout_seconds"] = timeout_seconds
        payloads = [
            {"type": "assistant.reasoning_delta", "data": {"deltaContent": "Plan: "}},
            {"type": "assistant.reasoning", "data": {"content": "Plan: inspect repo."}},
            {
                "type": "tool.execution_start",
                "data": {
                    "toolCallId": "tool-copilot-1",
                    "toolName": "github-mcp-server-get_file_contents",
                    "arguments": {"owner": "openai", "repo": "openai-agents-python", "path": "/"},
                    "mcpServerName": "github-mcp-server",
                    "mcpToolName": "get_file_contents",
                },
            },
            {
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "tool-copilot-1",
                    "success": True,
                    "result": {"content": "[]"},
                    "mcpServerName": "github-mcp-server",
                    "mcpToolName": "get_file_contents",
                },
            },
            {"type": "assistant.message_delta", "data": {"deltaContent": "Copilot "}},
            {"type": "assistant.message", "data": {"content": "Copilot final answer"}},
            {"type": "result", "sessionId": "copilot-session-stream-1"},
        ]
        for payload in payloads:
            yield payload

    monkeypatch.setattr(cli_model_module, "_stream_jsonl_command", fake_stream_jsonl_command)
    monkeypatch.setattr(
        CLIModel, "_resolve_copilot_command_prefix", lambda self, config: ["copilot"]
    )

    model = CLIModel(CLIModelConfig(vendor="copilot", model_name="gpt-4.1"))
    events = await _get_stream_events(model)

    assert events[0].type == "response.created"
    assert events[1].type == "response.in_progress"
    assert any(
        isinstance(event, ResponseReasoningSummaryTextDeltaEvent) and event.delta == "Plan: "
        for event in events
    )
    text_deltas = [event.delta for event in events if isinstance(event, ResponseTextDeltaEvent)]
    assert text_deltas == ["Copilot ", "final answer"]

    completed = events[-1]
    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.id == "copilot-session-stream-1"
    assert completed.response.output[0].summary[0].text == "Plan: inspect repo."
    assert completed.response.output[1].type == "mcp_call"
    assert completed.response.output[1].name == "get_file_contents"
    assert completed.response.output[1].server_label == "github-mcp-server"
    assert completed.response.output[2].content[0].text == "Copilot final answer"

    command = captured["command"]
    assert isinstance(command, list)
    assert "--stream" in command
    assert "on" in command
    assert "--model" in command
    assert "gpt-4.1" in command
    resume_arg = next(part for part in command if part.startswith("--resume="))
    uuid.UUID(resume_arg.split("=", 1)[1])


@pytest.mark.asyncio
async def test_copilot_cli_model_streams_native_acp(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_stream_acp_command(
        *,
        command,
        cwd,
        env,
        timeout_seconds,
        prompt,
        previous_response_id,
    ):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout_seconds"] = timeout_seconds
        captured["prompt"] = prompt
        captured["previous_response_id"] = previous_response_id
        payloads = [
            {
                "type": "acp.session_initialized",
                "session_id": "copilot-acp-session-1",
                "initialize": {"protocolVersion": 1},
                "session": {"sessionId": "copilot-acp-session-1"},
            },
            {
                "type": "acp.session_update",
                "session_id": "copilot-acp-session-1",
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "Plan: inspect repo."},
                },
            },
            {
                "type": "acp.session_update",
                "session_id": "copilot-acp-session-1",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-acp-2",
                    "title": "github search",
                    "kind": "search",
                    "status": "pending",
                    "rawInput": {"query": "openai-agents-python"},
                },
            },
            {
                "type": "acp.session_update",
                "session_id": "copilot-acp-session-1",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tool-acp-2",
                    "status": "completed",
                    "rawOutput": {"result": "[]"},
                },
            },
            {
                "type": "acp.session_update",
                "session_id": "copilot-acp-session-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Copilot ACP final answer"},
                },
            },
            {
                "type": "acp.prompt_result",
                "session_id": "copilot-acp-session-1",
                "result": {"stopReason": "end_turn"},
            },
        ]
        for payload in payloads:
            yield payload

    monkeypatch.setattr(cli_model_module, "_stream_acp_command", fake_stream_acp_command)
    monkeypatch.setattr(
        CLIModel, "_resolve_copilot_command_prefix", lambda self, config: ["copilot"]
    )

    model = CLIModel(CLIModelConfig(vendor="copilot", model_name="gpt-4.1", transport="acp"))
    events = await _get_stream_events(model)

    completed = events[-1]
    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.id == "copilot-acp-session-1"
    assert completed.response.output[0].summary[0].text == "Plan: inspect repo."
    assert completed.response.output[1].type == "mcp_call"
    assert completed.response.output[1].name == "github search"
    assert completed.response.output[2].content[0].text == "Copilot ACP final answer"

    command = captured["command"]
    assert isinstance(command, list)
    assert command == ["copilot", "--model", "gpt-4.1", "--acp"]


def test_build_response_obj_sanitizes_provider_managed_output_items() -> None:
    response_obj = _build_response_obj(
        output=[
            {
                "type": "local_shell_call",
                "id": "ls_1",
                "call_id": "call_ls_1",
                "status": "completed",
                "action": {
                    "type": "exec",
                    "command": ["sh", "-lc", "pwd"],
                    "env": {},
                    "working_directory": "/tmp/workspace",
                },
                "managed_by": "provider",
                "provider_data": {"vendor": "codex"},
            },
            {
                "type": "shell_call_output",
                "id": "ls_out_1",
                "call_id": "call_ls_1",
                "status": "completed",
                "output": [
                    {
                        "stdout": "/tmp/workspace\n",
                        "stderr": "",
                        "outcome": {"type": "exit", "exit_code": 0},
                        "created_by": "provider",
                    }
                ],
                "managed_by": "provider",
                "provider_data": {"vendor": "codex"},
                "created_by": "provider",
            },
        ],
        response_id="resp-cli-1",
        usage=Usage(),
    )

    assert response_obj.id == "resp-cli-1"
    assert response_obj.output[0].type == "local_shell_call"
    assert response_obj.output[1].type == "shell_call_output"
