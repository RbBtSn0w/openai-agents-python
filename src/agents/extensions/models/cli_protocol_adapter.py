from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from ...exceptions import ModelBehaviorError
from ...items import TResponseOutputItem, TResponseStreamEvent
from ...usage import Usage


class CLIStreamingSessionProtocol(Protocol):
    response_id: str
    message_states: Mapping[str, Any]
    reasoning_states: Mapping[str, Any]
    tool_states: Mapping[str, Any]

    def emit_start_events(self) -> list[TResponseStreamEvent]: ...

    def append_message_delta(self, item_id: str, delta: str) -> list[TResponseStreamEvent]: ...

    def sync_message_text(
        self, item_id: str, text: str, *, label: str
    ) -> list[TResponseStreamEvent]: ...

    def finish_message(self, item_id: str) -> list[TResponseStreamEvent]: ...

    def append_reasoning_delta(
        self, item_id: str, delta: str
    ) -> list[TResponseStreamEvent]: ...

    def sync_reasoning_text(
        self, item_id: str, text: str, *, label: str
    ) -> list[TResponseStreamEvent]: ...

    def finish_reasoning(self, item_id: str) -> list[TResponseStreamEvent]: ...

    def start_tool_item(
        self, item_id: str, item: TResponseOutputItem
    ) -> list[TResponseStreamEvent]: ...

    def finish_tool_item(
        self, item_id: str, item: TResponseOutputItem
    ) -> list[TResponseStreamEvent]: ...

    def append_instant_output_item(
        self, item: TResponseOutputItem
    ) -> list[TResponseStreamEvent]: ...


def stringify_json_value(*, label: str, value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError as exc:
        raise ModelBehaviorError(f"{label} is not JSON-serializable: {value!r}") from exc


def make_provider_mcp_call_item(
    *,
    item_id: str,
    server_label: str,
    name: str,
    arguments: Any,
    status: str,
    provider_data: Mapping[str, Any],
    output: Any | None = None,
    error: str | None = None,
) -> TResponseOutputItem:
    return {
        "type": "mcp_call",
        "id": item_id,
        "arguments": stringify_json_value(
            label=f"CLI tool arguments for {server_label}/{name}",
            value=arguments if arguments is not None else {},
        ),
        "name": name,
        "server_label": server_label,
        "status": status,
        "output": None
        if output is None
        else stringify_json_value(
            label=f"CLI tool output for {server_label}/{name}",
            value=output,
        ),
        "error": error,
        "managed_by": "provider",
        "provider_data": dict(provider_data),
    }


def extract_cli_tool_error_message(value: Any) -> str:
    if isinstance(value, Mapping):
        message = value.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)
    if isinstance(value, str):
        return value
    return str(value)


@dataclass
class CopilotStreamAdapter:
    provisional_response_id: str
    fallback_reasoning_item_id: str = field(
        default_factory=lambda: f"rs_{uuid.uuid4().hex}"
    )
    fallback_message_item_id: str = field(
        default_factory=lambda: f"msg_{uuid.uuid4().hex}"
    )
    latest_message_item_id: str | None = None
    tool_starts: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_response_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.final_response_id = self.provisional_response_id

    def consume_payload(
        self,
        *,
        session: CLIStreamingSessionProtocol,
        payload: Mapping[str, Any],
    ) -> list[TResponseStreamEvent]:
        event_type = payload.get("type")
        data = payload.get("data")
        events: list[TResponseStreamEvent] = []

        if event_type == "assistant.reasoning_delta" and isinstance(data, Mapping):
            delta = data.get("deltaContent")
            if isinstance(delta, str):
                reasoning_item_id = cast(
                    str | None,
                    data.get("reasoningId"),
                ) or self.fallback_reasoning_item_id
                events.extend(session.append_reasoning_delta(reasoning_item_id, delta))
        elif event_type == "assistant.message_delta" and isinstance(data, Mapping):
            delta = data.get("deltaContent")
            if isinstance(delta, str):
                message_item_id = cast(
                    str | None,
                    data.get("messageId"),
                ) or self.fallback_message_item_id
                self.latest_message_item_id = message_item_id
                events.extend(session.append_message_delta(message_item_id, delta))
        elif event_type == "assistant.reasoning" and isinstance(data, Mapping):
            content = data.get("content")
            reasoning_item_id = cast(
                str | None,
                data.get("reasoningId"),
            ) or self.fallback_reasoning_item_id
            if isinstance(content, str) and content:
                events.extend(
                    session.sync_reasoning_text(
                        reasoning_item_id,
                        content,
                        label="Copilot reasoning stream",
                    )
                )
            if reasoning_item_id in session.reasoning_states:
                events.extend(session.finish_reasoning(reasoning_item_id))
        elif event_type == "assistant.message" and isinstance(data, Mapping):
            content = data.get("content")
            message_item_id = cast(
                str | None,
                data.get("messageId"),
            ) or self.fallback_message_item_id
            if isinstance(content, str) and content:
                self.latest_message_item_id = message_item_id
                events.extend(
                    session.sync_message_text(
                        message_item_id,
                        content,
                        label="Copilot message stream",
                    )
                )
            if message_item_id in session.message_states:
                events.extend(session.finish_message(message_item_id))
        elif event_type == "tool.execution_start" and isinstance(data, Mapping):
            tool_call_id = data.get("toolCallId")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                self.tool_starts[tool_call_id] = dict(data)
                events.extend(
                    session.start_tool_item(
                        tool_call_id,
                        make_provider_mcp_call_item(
                            item_id=tool_call_id,
                            server_label=cast(
                                str | None,
                                data.get("mcpServerName") or data.get("toolName"),
                            )
                            or "copilot-cli",
                            name=cast(
                                str | None,
                                data.get("mcpToolName") or data.get("toolName"),
                            )
                            or "tool",
                            arguments=data.get("arguments"),
                            status="calling",
                            provider_data={
                                "vendor": "copilot",
                                "source_type": event_type,
                            },
                        ),
                    )
                )
        elif event_type == "tool.execution_complete" and isinstance(data, Mapping):
            tool_call_id = data.get("toolCallId")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                start_payload = self.tool_starts.get(tool_call_id, {})
                success = bool(data.get("success"))
                result = data.get("result")
                events.extend(
                    session.finish_tool_item(
                        tool_call_id,
                        make_provider_mcp_call_item(
                            item_id=tool_call_id,
                            server_label=cast(
                                str | None,
                                data.get("mcpServerName")
                                or start_payload.get("mcpServerName")
                                or data.get("toolName")
                                or start_payload.get("toolName"),
                            )
                            or "copilot-cli",
                            name=cast(
                                str | None,
                                data.get("mcpToolName")
                                or start_payload.get("mcpToolName")
                                or data.get("toolName")
                                or start_payload.get("toolName"),
                            )
                            or "tool",
                            arguments=data.get("arguments") or start_payload.get("arguments"),
                            status="completed" if success else "failed",
                            output=result,
                            error=None if success else extract_cli_tool_error_message(result),
                            provider_data={
                                "vendor": "copilot",
                                "source_type": event_type,
                            },
                        ),
                    )
                )
        elif event_type == "result":
            session_id = payload.get("sessionId") or payload.get("session_id")
            if isinstance(session_id, str) and session_id.strip():
                self.final_response_id = session_id.strip()

        return events

    def finish(
        self,
        *,
        session: CLIStreamingSessionProtocol,
    ) -> list[TResponseStreamEvent]:
        events: list[TResponseStreamEvent] = []
        for reasoning_item_id, reasoning_state in list(session.reasoning_states.items()):
            if not _state_done(reasoning_state) and _state_text(reasoning_state):
                events.extend(session.finish_reasoning(reasoning_item_id))

        if self.latest_message_item_id is None:
            raise ModelBehaviorError("Copilot stream did not produce an assistant message.")
        latest_message_state = session.message_states.get(self.latest_message_item_id)
        if latest_message_state is None or not _state_text(latest_message_state):
            raise ModelBehaviorError("Copilot stream did not produce an assistant message.")

        for message_item_id, message_state in list(session.message_states.items()):
            if not _state_done(message_state) and _state_text(message_state):
                events.extend(session.finish_message(message_item_id))

        for tool_call_id, tool_state in list(session.tool_states.items()):
            if not _state_done(tool_state):
                start_payload = self.tool_starts.get(tool_call_id, {})
                events.extend(
                    session.finish_tool_item(
                        tool_call_id,
                        make_provider_mcp_call_item(
                            item_id=tool_call_id,
                            server_label=cast(
                                str | None,
                                start_payload.get("mcpServerName")
                                or start_payload.get("toolName"),
                            )
                            or "copilot-cli",
                            name=cast(
                                str | None,
                                start_payload.get("mcpToolName")
                                or start_payload.get("toolName"),
                            )
                            or "tool",
                            arguments=start_payload.get("arguments"),
                            status="incomplete",
                            provider_data={
                                "vendor": "copilot",
                                "source_type": "tool.execution_start",
                            },
                        ),
                    )
                )
        return events


@dataclass
class GeminiStreamAdapter:
    initial_response_id: str
    usage_from_payload: Callable[[Mapping[str, Any]], Usage]
    current_message_item_id: str = field(
        default_factory=lambda: f"msg_{uuid.uuid4().hex}"
    )
    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    start_emitted: bool = False
    final_response_id: str = field(init=False)
    final_usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self.final_response_id = self.initial_response_id

    def consume_payload(
        self,
        *,
        session: CLIStreamingSessionProtocol,
        payload: Mapping[str, Any],
    ) -> list[TResponseStreamEvent]:
        payload_type = payload.get("type")
        events: list[TResponseStreamEvent] = []

        if payload_type == "init":
            session_id = payload.get("session_id")
            if isinstance(session_id, str) and session_id.strip():
                self.final_response_id = session_id.strip()
                session.response_id = self.final_response_id

        if not self.start_emitted:
            self.start_emitted = True
            events.extend(session.emit_start_events())

        if payload_type == "message":
            role = payload.get("role")
            if role != "assistant":
                return events
            content = payload.get("content")
            if not isinstance(content, str) or not content:
                return events
            if payload.get("delta") is True:
                events.extend(session.append_message_delta(self.current_message_item_id, content))
            else:
                events.extend(
                    session.sync_message_text(
                        self.current_message_item_id,
                        content,
                        label="Gemini message stream",
                    )
                )
        elif payload_type == "tool_use":
            current_message_state = session.message_states.get(self.current_message_item_id)
            if (
                current_message_state is not None
                and _state_text(current_message_state)
                and not _state_done(current_message_state)
            ):
                events.extend(session.finish_message(self.current_message_item_id))
                self.current_message_item_id = f"msg_{uuid.uuid4().hex}"
            tool_id = payload.get("tool_id")
            if isinstance(tool_id, str) and tool_id.strip():
                tool_payload = {
                    "tool_name": payload.get("tool_name"),
                    "parameters": payload.get("parameters"),
                }
                self.tool_calls[tool_id] = tool_payload
                events.extend(
                    session.start_tool_item(
                        tool_id,
                        make_provider_mcp_call_item(
                            item_id=tool_id,
                            server_label="gemini-cli",
                            name=cast(str | None, payload.get("tool_name")) or "tool",
                            arguments=payload.get("parameters"),
                            status="calling",
                            provider_data={
                                "vendor": "gemini",
                                "source_type": payload_type,
                            },
                        ),
                    )
                )
        elif payload_type == "tool_result":
            tool_id = payload.get("tool_id")
            if isinstance(tool_id, str) and tool_id.strip():
                tool_payload = self.tool_calls.get(tool_id, {})
                status = payload.get("status")
                is_success = status == "success"
                output = payload.get("output")
                error_payload = payload.get("error")
                error_message = extract_cli_tool_error_message(error_payload or output)
                events.extend(
                    session.finish_tool_item(
                        tool_id,
                        make_provider_mcp_call_item(
                            item_id=tool_id,
                            server_label="gemini-cli",
                            name=cast(str | None, tool_payload.get("tool_name")) or "tool",
                            arguments=tool_payload.get("parameters"),
                            status="completed" if is_success else "failed",
                            output=output,
                            error=None if is_success else error_message,
                            provider_data={
                                "vendor": "gemini",
                                "source_type": payload_type,
                            },
                        ),
                    )
                )
        elif payload_type == "result":
            status = payload.get("status")
            if isinstance(status, str) and status not in {"success", "completed"}:
                raise RuntimeError(f"Gemini stream ended with non-success status: {status}")
            self.final_usage = self.usage_from_payload(payload)

        return events

    def finish(
        self,
        *,
        session: CLIStreamingSessionProtocol,
    ) -> list[TResponseStreamEvent]:
        events: list[TResponseStreamEvent] = []
        if not self.start_emitted:
            self.start_emitted = True
            events.extend(session.emit_start_events())

        message_state = session.message_states.get(self.current_message_item_id)
        if (
            message_state is not None
            and _state_text(message_state)
            and not _state_done(message_state)
        ):
            events.extend(session.finish_message(self.current_message_item_id))

        if not any(_state_text(message_state) for message_state in session.message_states.values()):
            raise ModelBehaviorError("Gemini stream did not produce an assistant message.")

        for tool_call_id, tool_state in list(session.tool_states.items()):
            if not _state_done(tool_state):
                tool_payload = self.tool_calls.get(tool_call_id, {})
                events.extend(
                    session.finish_tool_item(
                        tool_call_id,
                        make_provider_mcp_call_item(
                            item_id=tool_call_id,
                            server_label="gemini-cli",
                            name=cast(str | None, tool_payload.get("tool_name")) or "tool",
                            arguments=tool_payload.get("parameters"),
                            status="incomplete",
                            provider_data={
                                "vendor": "gemini",
                                "source_type": "tool_use",
                            },
                        ),
                    )
                )
        return events


@dataclass
class ACPStreamAdapter:
    vendor: str
    initial_response_id: str
    fallback_reasoning_item_id: str = field(
        default_factory=lambda: f"rs_{uuid.uuid4().hex}"
    )
    fallback_plan_item_id: str = field(
        default_factory=lambda: f"plan_{uuid.uuid4().hex}"
    )
    fallback_message_item_id: str = field(
        default_factory=lambda: f"msg_{uuid.uuid4().hex}"
    )
    tool_starts: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_response_id: str = field(init=False)
    final_stop_reason: str | None = None

    def __post_init__(self) -> None:
        self.final_response_id = self.initial_response_id

    def consume_payload(
        self,
        *,
        session: CLIStreamingSessionProtocol,
        payload: Mapping[str, Any],
    ) -> list[TResponseStreamEvent]:
        payload_type = payload.get("type")
        events: list[TResponseStreamEvent] = []

        if payload_type == "acp.session_initialized":
            session_id = payload.get("session_id")
            if isinstance(session_id, str) and session_id.strip():
                self.final_response_id = session_id.strip()
                session.response_id = self.final_response_id
            return events

        if payload_type == "acp.prompt_result":
            result = payload.get("result")
            if isinstance(result, Mapping):
                stop_reason = result.get("stopReason")
                if isinstance(stop_reason, str) and stop_reason.strip():
                    self.final_stop_reason = stop_reason.strip()
            return events

        if payload_type != "acp.session_update":
            return events

        update = payload.get("update")
        if not isinstance(update, Mapping):
            return events

        update_type = update.get("sessionUpdate")
        if update_type == "agent_message_chunk":
            content_text = _acp_content_block_to_text(update.get("content"))
            if content_text:
                events.extend(
                    session.append_message_delta(self.fallback_message_item_id, content_text)
                )
        elif update_type == "agent_thought_chunk":
            content_text = _acp_content_block_to_text(update.get("content"))
            if content_text:
                events.extend(
                    session.append_reasoning_delta(self.fallback_reasoning_item_id, content_text)
                )
        elif update_type == "plan":
            plan_text = _acp_plan_to_text(update.get("entries"))
            if plan_text:
                events.extend(
                    session.sync_reasoning_text(
                        self.fallback_plan_item_id,
                        plan_text,
                        label=f"{self.vendor} ACP plan stream",
                    )
                )
        elif update_type == "tool_call":
            tool_call_id = update.get("toolCallId")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                self.tool_starts[tool_call_id] = dict(update)
                events.extend(
                    session.start_tool_item(
                        tool_call_id,
                        _acp_tool_update_to_output(
                            vendor=self.vendor,
                            tool_update=update,
                            fallback_start=None,
                        ),
                    )
                )
        elif update_type == "tool_call_update":
            tool_call_id = update.get("toolCallId")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                start_payload = self.tool_starts.get(tool_call_id)
                output_item = _acp_tool_update_to_output(
                    vendor=self.vendor,
                    tool_update=update,
                    fallback_start=start_payload,
                )
                status = output_item.get("status") if isinstance(output_item, dict) else None
                if status in {"completed", "failed"}:
                    events.extend(session.finish_tool_item(tool_call_id, output_item))
                else:
                    if tool_call_id not in session.tool_states:
                        events.extend(session.start_tool_item(tool_call_id, output_item))
                    else:
                        events.extend(session.finish_tool_item(tool_call_id, output_item))
        return events

    def finish(
        self,
        *,
        session: CLIStreamingSessionProtocol,
    ) -> list[TResponseStreamEvent]:
        events: list[TResponseStreamEvent] = []

        for reasoning_item_id, reasoning_state in list(session.reasoning_states.items()):
            if not _state_done(reasoning_state) and _state_text(reasoning_state):
                events.extend(session.finish_reasoning(reasoning_item_id))

        if not any(_state_text(message_state) for message_state in session.message_states.values()):
            raise ModelBehaviorError("ACP stream did not produce an assistant message.")

        for message_item_id, message_state in list(session.message_states.items()):
            if not _state_done(message_state) and _state_text(message_state):
                events.extend(session.finish_message(message_item_id))

        for tool_call_id, tool_state in list(session.tool_states.items()):
            if not _state_done(tool_state):
                events.extend(
                    session.finish_tool_item(
                        tool_call_id,
                        _acp_tool_update_to_output(
                            vendor=self.vendor,
                            tool_update={},
                            fallback_start=self.tool_starts.get(tool_call_id),
                            force_status="incomplete",
                        ),
                    )
                )
        return events


def _state_text(state: Any) -> str:
    text = getattr(state, "text", "")
    return text if isinstance(text, str) else ""


def _state_done(state: Any) -> bool:
    return bool(getattr(state, "done", False))
def _acp_content_block_to_text(content: Any) -> str:
    if isinstance(content, Mapping):
        content_type = content.get("type")
        if content_type == "text":
            text = content.get("text")
            if isinstance(text, str):
                return text
    if content is None:
        return ""
    return stringify_json_value(label="ACP content", value=content)


def _acp_plan_to_text(entries: Any) -> str:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ""
    lines: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        status = entry.get("status")
        prefix = f"[{status}]" if isinstance(status, str) and status.strip() else "[plan]"
        lines.append(f"{prefix} {content.strip()}")
    return "\n".join(lines)


def _acp_tool_update_to_output(
    *,
    vendor: str,
    tool_update: Mapping[str, Any],
    fallback_start: Mapping[str, Any] | None,
    force_status: str | None = None,
) -> TResponseOutputItem:
    source = fallback_start or {}
    tool_call_id = cast(
        str | None,
        tool_update.get("toolCallId") or source.get("toolCallId"),
    ) or f"tool_{uuid.uuid4().hex}"
    raw_status = force_status or cast(
        str | None,
        tool_update.get("status") or source.get("status"),
    )
    status = _normalize_acp_tool_status(raw_status)
    title = cast(str | None, tool_update.get("title") or source.get("title")) or "tool"
    tool_kind = cast(str | None, tool_update.get("kind") or source.get("kind"))
    raw_input = tool_update.get("rawInput", source.get("rawInput"))
    raw_output = tool_update.get("rawOutput", source.get("rawOutput"))
    content = tool_update.get("content", source.get("content"))

    return make_provider_mcp_call_item(
        item_id=tool_call_id,
        server_label=f"{vendor}-acp",
        name=title,
        arguments=raw_input if raw_input is not None else {},
        status=status,
        output=raw_output if raw_output is not None else _acp_tool_content_to_output(content),
        error=None if status != "failed" else _extract_acp_tool_error(content, raw_output),
        provider_data={
            "vendor": vendor,
            "transport": "acp",
            "source_type": "session/update",
            "tool_kind": tool_kind,
        },
    )


def _normalize_acp_tool_status(status: str | None) -> str:
    if status in {"completed", "failed", "incomplete"}:
        return status
    if status in {"pending", "in_progress"}:
        return "calling"
    return "incomplete"


def _acp_tool_content_to_output(content: Any) -> Any | None:
    if content is None:
        return None
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        normalized: list[Any] = []
        for item in content:
            if not isinstance(item, Mapping):
                normalized.append(item)
                continue
            item_type = item.get("type")
            if item_type == "content":
                normalized.append(item.get("content"))
            else:
                normalized.append(dict(item))
        return normalized
    return content


def _extract_acp_tool_error(content: Any, raw_output: Any) -> str:
    rendered_content = _acp_tool_content_to_output(content)
    if rendered_content is not None:
        return extract_cli_tool_error_message(rendered_content)
    return extract_cli_tool_error_message(raw_output)
