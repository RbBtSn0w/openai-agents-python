from __future__ import annotations

from openai.types.responses import ResponseCompletedEvent, ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from agents.extensions.experimental.codex import CommandExecutionItem
from agents.extensions.models.cli_model import _CLIStreamingSession
from agents.extensions.models.cli_protocol_adapter import (
    CopilotStreamAdapter,
    GeminiStreamAdapter,
    codex_provider_outputs_for_item,
)
from agents.usage import Usage


def test_copilot_stream_adapter_normalizes_payloads() -> None:
    session = _CLIStreamingSession(response_id="resp-copilot-initial")
    adapter = CopilotStreamAdapter(provisional_response_id="resp-copilot-initial")

    for payload in [
        {"type": "assistant.reasoning_delta", "data": {"deltaContent": "Plan: "}},
        {"type": "assistant.reasoning", "data": {"content": "Plan: inspect repo."}},
        {
            "type": "tool.execution_start",
            "data": {
                "toolCallId": "tool-copilot-1",
                "toolName": "github-mcp-server-get_file_contents",
                "arguments": {"path": "/"},
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
        {"type": "result", "sessionId": "copilot-session-1"},
    ]:
        adapter.consume_payload(session=session, payload=payload)

    adapter.finish(session=session)
    completed = session.complete(response_id=adapter.final_response_id, usage=Usage())

    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.id == "copilot-session-1"
    assert isinstance(completed.response.output[0], ResponseReasoningItem)
    assert completed.response.output[0].summary[0].text == "Plan: inspect repo."
    assert completed.response.output[1].type == "mcp_call"
    assert completed.response.output[1].name == "get_file_contents"
    assert isinstance(completed.response.output[2], ResponseOutputMessage)
    assert completed.response.output[2].content[0].text == "Copilot final answer"


def test_gemini_stream_adapter_segments_messages_around_tool_use() -> None:
    session = _CLIStreamingSession(response_id="resp-gemini-initial")
    adapter = GeminiStreamAdapter(
        initial_response_id="resp-gemini-initial",
        usage_from_payload=lambda payload: Usage(
            requests=1,
            input_tokens=int(payload["stats"]["input_tokens"]),
            output_tokens=int(payload["stats"]["output_tokens"]),
            total_tokens=int(payload["stats"]["total_tokens"]),
        ),
    )

    for payload in [
        {"type": "init", "session_id": "gemini-session-1"},
        {"type": "message", "role": "assistant", "content": "Planning...", "delta": True},
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
            "output": "Listed 3 item(s).",
        },
        {"type": "message", "role": "assistant", "content": "Final answer", "delta": True},
        {
            "type": "result",
            "status": "success",
            "stats": {"input_tokens": 6, "output_tokens": 2, "total_tokens": 8},
        },
    ]:
        adapter.consume_payload(session=session, payload=payload)

    adapter.finish(session=session)
    completed = session.complete(
        response_id=adapter.final_response_id,
        usage=adapter.final_usage,
    )

    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.id == "gemini-session-1"
    assert completed.response.output[0].content[0].text == "Planning..."
    assert completed.response.output[1].type == "mcp_call"
    assert completed.response.output[1].name == "list_directory"
    assert completed.response.output[2].content[0].text == "Final answer"
    assert completed.response.usage.input_tokens == 6
    assert completed.response.usage.output_tokens == 2
    assert completed.response.usage.total_tokens == 8


def test_codex_provider_outputs_map_command_execution() -> None:
    outputs = codex_provider_outputs_for_item(
        CommandExecutionItem(
            id="cmd-1",
            command="pwd",
            aggregated_output="/tmp/workspace\n",
            exit_code=0,
            status="completed",
        ),
        cwd="/tmp/workspace",
    )

    assert outputs[0]["type"] == "local_shell_call"
    assert outputs[0]["action"]["command"] == ["sh", "-lc", "pwd"]
    assert outputs[1]["type"] == "shell_call_output"
    assert outputs[1]["output"][0]["stdout"] == "/tmp/workspace\n"
