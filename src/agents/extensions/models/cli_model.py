from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from openai._models import construct_type
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)
from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary
from openai.types.responses.response_reasoning_summary_part_added_event import (
    Part as AddedEventPart,
)
from openai.types.responses.response_reasoning_summary_part_done_event import Part as DoneEventPart
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel, Field, ValidationError

from ...agent_output import AgentOutputSchemaBase
from ...exceptions import ModelBehaviorError, UserError
from ...handoffs import Handoff
from ...items import ModelResponse, TResponseInputItem, TResponseOutputItem, TResponseStreamEvent
from ...model_settings import ModelSettings
from ...models.interface import Model, ModelProvider, ModelTracing
from ...tool import FunctionTool, Tool
from ...usage import Usage
from ..experimental.codex import (
    Codex,
    ItemCompletedEvent,
    ItemStartedEvent,
    ItemUpdatedEvent,
    ThreadErrorEvent,
    ThreadItem,
    ThreadOptions,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnOptions,
)
from .cli_acp_adapter import (
    CLIAcpInvocation,
    build_acp_invocation,
    stream_acp_prompt_invocation,
)
from .cli_protocol_adapter import (
    ACPStreamAdapter,
    CopilotStreamAdapter,
    GeminiStreamAdapter,
    codex_provider_outputs_for_item,
    stream_codex_item_event,
)
from .cli_subprocess_adapter import (
    CLISubprocessInvocation,
    build_copilot_invocation,
    build_gemini_invocation,
    parse_copilot_jsonl,
    run_json_invocation,
    run_jsonl_invocation,
    stream_jsonl_invocation,
)

if TYPE_CHECKING:
    from openai.types.responses.response_prompt_param import ResponsePromptParam

__all__ = [
    "CLIExecutionMode",
    "CLITransport",
    "CLIModel",
    "CLIProvider",
    "CodexCLIProvider",
    "GeminiCLIProvider",
    "CopilotCLIProvider",
]


CLIExecutionMode = Literal["sdk_controlled", "cli_autonomous"]
CLITransport = Literal["auto", "acp", "json", "jsonl", "stream_json"]
CLIVendor = Literal["codex", "gemini", "copilot"]


class _ControlledToolCall(BaseModel):
    name: str
    arguments_json: str = "{}"


class _ControlledEnvelope(BaseModel):
    assistant_message: str | None = None
    tool_calls: list[_ControlledToolCall] = Field(default_factory=list)
    reasoning_summary: str | None = None


@dataclass(frozen=True)
class CLIModelConfig:
    vendor: CLIVendor
    model_name: str | None = None
    executable_path: str | None = None
    execution_mode: CLIExecutionMode = "cli_autonomous"
    transport: CLITransport = "auto"
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    timeout_seconds: float = 300.0
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CLIProviderConfig:
    execution_mode: CLIExecutionMode = "cli_autonomous"
    transport: CLITransport = "auto"
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    timeout_seconds: float = 300.0
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    executable_path: str | None = None
    default_model_name: str | None = None


@dataclass
class _StreamingMessageState:
    item_id: str
    output_index: int
    text: str = ""
    done: bool = False


@dataclass
class _StreamingReasoningState:
    item_id: str
    output_index: int
    text: str = ""
    done: bool = False


@dataclass
class _StreamingToolState:
    item_id: str
    output_index: int
    done: bool = False


@dataclass
class _CLIStreamingSession:
    response_id: str
    usage: Usage = field(default_factory=Usage)
    sequence_number: int = 0
    output_items: list[TResponseOutputItem] = field(default_factory=list)
    message_states: dict[str, _StreamingMessageState] = field(default_factory=dict)
    reasoning_states: dict[str, _StreamingReasoningState] = field(default_factory=dict)
    tool_states: dict[str, _StreamingToolState] = field(default_factory=dict)

    def emit_start_events(self) -> list[TResponseStreamEvent]:
        response = _build_response_obj(self.output_items, self.response_id, self.usage)
        return [
            ResponseCreatedEvent(
                type="response.created",
                response=response,
                sequence_number=self._next_sequence_number(),
            ),
            ResponseInProgressEvent(
                type="response.in_progress",
                response=response,
                sequence_number=self._next_sequence_number(),
            ),
        ]

    def append_message_delta(self, item_id: str, delta: str) -> list[TResponseStreamEvent]:
        if not delta:
            return []
        events: list[TResponseStreamEvent] = []
        state = self.message_states.get(item_id)
        if state is None:
            state = _StreamingMessageState(
                item_id=item_id,
                output_index=len(self.output_items),
            )
            self.message_states[item_id] = state
            self.output_items.append(
                _make_message_output_with_id(
                    item_id=item_id,
                    text="",
                    status="in_progress",
                )
            )
            message_item = cast(
                ResponseOutputMessage,
                _coerce_output_item_for_response(self.output_items[state.output_index]),
            )
            events.append(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    item=message_item,
                    output_index=state.output_index,
                    sequence_number=self._next_sequence_number(),
                )
            )
            events.append(
                ResponseContentPartAddedEvent(
                    type="response.content_part.added",
                    item_id=item_id,
                    output_index=state.output_index,
                    content_index=0,
                    part=ResponseOutputText(
                        text="",
                        type="output_text",
                        annotations=[],
                    ),
                    sequence_number=self._next_sequence_number(),
                )
            )
        elif state.done:
            raise ModelBehaviorError(
                f"CLI stream emitted text for completed message item: {item_id}"
            )

        state.text += delta
        self.output_items[state.output_index] = _make_message_output_with_id(
            item_id=item_id,
            text=state.text,
            status="in_progress",
        )
        events.append(
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                item_id=item_id,
                output_index=state.output_index,
                content_index=0,
                delta=delta,
                logprobs=[],
                sequence_number=self._next_sequence_number(),
            )
        )
        return events

    def sync_message_text(
        self, item_id: str, text: str, *, label: str
    ) -> list[TResponseStreamEvent]:
        if not text:
            return []
        current = self.message_states.get(item_id)
        current_text = current.text if current is not None else ""
        delta = _compute_stream_delta(label=label, current_text=current_text, new_text=text)
        return self.append_message_delta(item_id, delta)

    def finish_message(self, item_id: str) -> list[TResponseStreamEvent]:
        state = self.message_states.get(item_id)
        if state is None:
            return []
        if state.done:
            raise ModelBehaviorError(f"CLI stream finished the same message item twice: {item_id}")
        state.done = True
        self.output_items[state.output_index] = _make_message_output_with_id(
            item_id=item_id,
            text=state.text,
            status="completed",
        )
        message_item = cast(
            ResponseOutputMessage,
            _coerce_output_item_for_response(self.output_items[state.output_index]),
        )
        text_part = cast(ResponseOutputText, message_item.content[0])
        return [
            ResponseTextDoneEvent(
                type="response.output_text.done",
                item_id=item_id,
                output_index=state.output_index,
                content_index=0,
                text=state.text,
                logprobs=[],
                sequence_number=self._next_sequence_number(),
            ),
            ResponseContentPartDoneEvent(
                type="response.content_part.done",
                item_id=item_id,
                output_index=state.output_index,
                content_index=0,
                part=text_part,
                sequence_number=self._next_sequence_number(),
            ),
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=message_item,
                output_index=state.output_index,
                sequence_number=self._next_sequence_number(),
            ),
        ]

    def append_reasoning_delta(self, item_id: str, delta: str) -> list[TResponseStreamEvent]:
        if not delta:
            return []
        events: list[TResponseStreamEvent] = []
        state = self.reasoning_states.get(item_id)
        if state is None:
            state = _StreamingReasoningState(
                item_id=item_id,
                output_index=len(self.output_items),
            )
            self.reasoning_states[item_id] = state
            self.output_items.append(
                _make_reasoning_item_with_id(
                    item_id=item_id,
                    text="",
                    status="in_progress",
                )
            )
            reasoning_item = cast(
                ResponseReasoningItem,
                _coerce_output_item_for_response(self.output_items[state.output_index]),
            )
            events.append(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    item=reasoning_item,
                    output_index=state.output_index,
                    sequence_number=self._next_sequence_number(),
                )
            )
            events.append(
                ResponseReasoningSummaryPartAddedEvent(
                    type="response.reasoning_summary_part.added",
                    item_id=item_id,
                    output_index=state.output_index,
                    summary_index=0,
                    part=AddedEventPart(text="", type="summary_text"),
                    sequence_number=self._next_sequence_number(),
                )
            )
        elif state.done:
            raise ModelBehaviorError(f"CLI stream emitted reasoning for completed item: {item_id}")

        state.text += delta
        self.output_items[state.output_index] = _make_reasoning_item_with_id(
            item_id=item_id,
            text=state.text,
            status="in_progress",
        )
        events.append(
            ResponseReasoningSummaryTextDeltaEvent(
                type="response.reasoning_summary_text.delta",
                item_id=item_id,
                output_index=state.output_index,
                summary_index=0,
                delta=delta,
                sequence_number=self._next_sequence_number(),
            )
        )
        return events

    def sync_reasoning_text(
        self, item_id: str, text: str, *, label: str
    ) -> list[TResponseStreamEvent]:
        if not text:
            return []
        current = self.reasoning_states.get(item_id)
        current_text = current.text if current is not None else ""
        delta = _compute_stream_delta(label=label, current_text=current_text, new_text=text)
        return self.append_reasoning_delta(item_id, delta)

    def finish_reasoning(self, item_id: str) -> list[TResponseStreamEvent]:
        state = self.reasoning_states.get(item_id)
        if state is None:
            return []
        if state.done:
            raise ModelBehaviorError(
                f"CLI stream finished the same reasoning item twice: {item_id}"
            )
        state.done = True
        self.output_items[state.output_index] = _make_reasoning_item_with_id(
            item_id=item_id,
            text=state.text,
            status="completed",
        )
        reasoning_item = cast(
            ResponseReasoningItem,
            _coerce_output_item_for_response(self.output_items[state.output_index]),
        )
        summary = reasoning_item.summary[0]
        part = DoneEventPart(text=summary.text, type=summary.type)
        return [
            ResponseReasoningSummaryTextDoneEvent(
                type="response.reasoning_summary_text.done",
                item_id=item_id,
                output_index=state.output_index,
                summary_index=0,
                text=summary.text,
                sequence_number=self._next_sequence_number(),
            ),
            ResponseReasoningSummaryPartDoneEvent(
                type="response.reasoning_summary_part.done",
                item_id=item_id,
                output_index=state.output_index,
                summary_index=0,
                part=part,
                sequence_number=self._next_sequence_number(),
            ),
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=reasoning_item,
                output_index=state.output_index,
                sequence_number=self._next_sequence_number(),
            ),
        ]

    def append_instant_output_item(self, item: TResponseOutputItem) -> list[TResponseStreamEvent]:
        output_index = len(self.output_items)
        self.output_items.append(item)
        response_item = _coerce_output_item_for_response(item)
        return [
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=response_item,
                output_index=output_index,
                sequence_number=self._next_sequence_number(),
            ),
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=response_item,
                output_index=output_index,
                sequence_number=self._next_sequence_number(),
            ),
        ]

    def start_tool_item(
        self, item_id: str, item: TResponseOutputItem
    ) -> list[TResponseStreamEvent]:
        state = self.tool_states.get(item_id)
        if state is not None:
            raise ModelBehaviorError(f"CLI stream started the same tool item twice: {item_id}")
        output_index = len(self.output_items)
        self.output_items.append(item)
        self.tool_states[item_id] = _StreamingToolState(item_id=item_id, output_index=output_index)
        response_item = _coerce_output_item_for_response(item)
        return [
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=response_item,
                output_index=output_index,
                sequence_number=self._next_sequence_number(),
            )
        ]

    def finish_tool_item(
        self, item_id: str, item: TResponseOutputItem
    ) -> list[TResponseStreamEvent]:
        state = self.tool_states.get(item_id)
        if state is None:
            return self.append_instant_output_item(item)
        if state.done:
            raise ModelBehaviorError(f"CLI stream finished the same tool item twice: {item_id}")
        state.done = True
        self.output_items[state.output_index] = item
        response_item = _coerce_output_item_for_response(item)
        return [
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=response_item,
                output_index=state.output_index,
                sequence_number=self._next_sequence_number(),
            )
        ]

    def complete(
        self,
        *,
        response_id: str | None = None,
        usage: Usage | None = None,
    ) -> ResponseCompletedEvent:
        if response_id is not None:
            self.response_id = response_id
        if usage is not None:
            self.usage = usage
        return ResponseCompletedEvent(
            type="response.completed",
            response=_build_response_obj(self.output_items, self.response_id, self.usage),
            sequence_number=self._next_sequence_number(),
        )

    def _next_sequence_number(self) -> int:
        sequence_number = self.sequence_number
        self.sequence_number += 1
        return sequence_number


class CLIModel(Model):
    def __init__(self, config: CLIModelConfig):
        self.config = config

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        del tracing, conversation_id, prompt
        effective_config = _resolve_cli_model_config(self.config, model_settings)

        if effective_config.vendor == "codex":
            return await self._get_codex_response(
                config=effective_config,
                system_instructions=system_instructions,
                input=input,
                model_settings=model_settings,
                tools=tools,
                output_schema=output_schema,
                handoffs=handoffs,
                previous_response_id=previous_response_id,
            )

        if effective_config.vendor == "gemini":
            return await self._get_gemini_response(
                config=effective_config,
                system_instructions=system_instructions,
                input=input,
                model_settings=model_settings,
                tools=tools,
                output_schema=output_schema,
                handoffs=handoffs,
                previous_response_id=previous_response_id,
            )

        if effective_config.vendor == "copilot":
            return await self._get_copilot_response(
                config=effective_config,
                system_instructions=system_instructions,
                input=input,
                model_settings=model_settings,
                tools=tools,
                output_schema=output_schema,
                handoffs=handoffs,
                previous_response_id=previous_response_id,
            )

        raise UserError(f"Unsupported CLI vendor: {effective_config.vendor}")

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        del tracing, conversation_id, prompt
        effective_config = _resolve_cli_model_config(self.config, model_settings)

        if (
            effective_config.vendor == "codex"
            and effective_config.execution_mode == "cli_autonomous"
        ):
            async for event in self._stream_codex_response(
                config=effective_config,
                system_instructions=system_instructions,
                input=input,
                previous_response_id=previous_response_id,
            ):
                yield event
            return

        if (
            effective_config.vendor == "gemini"
            and effective_config.execution_mode == "cli_autonomous"
        ):
            async for event in self._stream_gemini_response(
                config=effective_config,
                system_instructions=system_instructions,
                input=input,
                previous_response_id=previous_response_id,
            ):
                yield event
            return

        if (
            effective_config.vendor == "copilot"
            and effective_config.execution_mode == "cli_autonomous"
        ):
            async for event in self._stream_copilot_response(
                config=effective_config,
                system_instructions=system_instructions,
                input=input,
                previous_response_id=previous_response_id,
            ):
                yield event
            return

        response = await self.get_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            ModelTracing.DISABLED,
            previous_response_id=previous_response_id,
            conversation_id=None,
            prompt=None,
        )
        async for event in _stream_events_from_model_response(response):
            yield event

    async def _get_codex_response(
        self,
        *,
        config: CLIModelConfig,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        previous_response_id: str | None,
    ) -> ModelResponse:
        _resolve_cli_transport(config)
        if config.extra_args:
            raise UserError(
                "Codex CLI provider does not support cli_extra_args yet. "
                "Use explicit provider settings such as model_name, cwd, env, or execution_mode."
            )

        if config.execution_mode == "cli_autonomous":
            return await _collect_model_response_from_stream(
                self._stream_codex_response(
                    config=config,
                    system_instructions=system_instructions,
                    input=input,
                    previous_response_id=previous_response_id,
                )
            )

        codex = Codex(
            codex_path_override=self._resolve_executable(config, "codex"),
            env=config.env,
        )
        thread_options = ThreadOptions(
            model=config.model_name,
            working_directory=self._resolve_cwd(config),
        )
        thread = (
            codex.resume_thread(previous_response_id, thread_options)
            if previous_response_id
            else codex.start_thread(thread_options)
        )

        if config.execution_mode == "sdk_controlled":
            prompt_text = _build_sdk_controlled_prompt(
                vendor=config.vendor,
                system_instructions=system_instructions,
                input=input,
                tools=tools,
                handoffs=handoffs,
                output_schema=output_schema,
                tool_choice=model_settings.tool_choice,
            )
            turn = await thread.run(
                prompt_text,
                TurnOptions(
                    output_schema=_ControlledEnvelope.model_json_schema(),
                    idle_timeout_seconds=config.timeout_seconds,
                ),
            )
            envelope = _parse_controlled_envelope(turn.final_response)
            outputs = _controlled_envelope_to_outputs(
                envelope=envelope,
                output_schema=output_schema,
            )
        return ModelResponse(
            output=outputs,
            usage=_codex_usage_to_usage(getattr(turn, "usage", None)),
            response_id=thread.id,
            provider_response_id=thread.id,
            provider_session_id=thread.id,
        )

    async def _get_gemini_response(
        self,
        *,
        config: CLIModelConfig,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        previous_response_id: str | None,
    ) -> ModelResponse:
        transport = _resolve_cli_transport(config)
        if config.execution_mode == "cli_autonomous":
            return await _collect_model_response_from_stream(
                self._stream_gemini_response(
                    config=config,
                    system_instructions=system_instructions,
                    input=input,
                    previous_response_id=previous_response_id,
                )
            )

        prompt_text = (
            _build_sdk_controlled_prompt(
                vendor=config.vendor,
                system_instructions=system_instructions,
                input=input,
                tools=tools,
                handoffs=handoffs,
                output_schema=output_schema,
                tool_choice=model_settings.tool_choice,
            )
            if config.execution_mode == "sdk_controlled"
            else _build_autonomous_prompt(system_instructions=system_instructions, input=input)
        )
        invocation = build_gemini_invocation(
            executable=self._resolve_executable(config, "gemini"),
            prompt_text=prompt_text,
            model_name=config.model_name,
            previous_response_id=previous_response_id,
            execution_mode=config.execution_mode,
            extra_args=config.extra_args,
            cwd=self._resolve_cwd(config),
            env=self._resolve_env(config),
            timeout_seconds=config.timeout_seconds,
            output_format=cast(Literal["json"], transport),
        )

        payload = await _run_json_command(
            command=invocation.command,
            cwd=invocation.cwd,
            env=invocation.env,
            timeout_seconds=invocation.timeout_seconds,
        )
        response_text = require_non_empty_str(payload.get("response"), "Gemini response")
        session_id = cast(str | None, payload.get("session_id"))
        if config.execution_mode == "sdk_controlled":
            envelope = _parse_controlled_envelope(response_text)
            outputs = _controlled_envelope_to_outputs(
                envelope=envelope, output_schema=output_schema
            )
        else:
            outputs = _basic_autonomous_outputs(response_text, reasoning_summary=None)

        return ModelResponse(
            output=outputs,
            usage=_gemini_usage_to_usage(payload),
            response_id=session_id,
            provider_response_id=session_id,
            provider_session_id=session_id,
        )

    async def _get_copilot_response(
        self,
        *,
        config: CLIModelConfig,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        previous_response_id: str | None,
    ) -> ModelResponse:
        transport = _resolve_cli_transport(config)
        if config.execution_mode == "cli_autonomous":
            return await _collect_model_response_from_stream(
                self._stream_copilot_response(
                    config=config,
                    system_instructions=system_instructions,
                    input=input,
                    previous_response_id=previous_response_id,
                )
            )

        prompt_text = (
            _build_sdk_controlled_prompt(
                vendor=config.vendor,
                system_instructions=system_instructions,
                input=input,
                tools=tools,
                handoffs=handoffs,
                output_schema=output_schema,
                tool_choice=model_settings.tool_choice,
            )
            if config.execution_mode == "sdk_controlled"
            else _build_autonomous_prompt(system_instructions=system_instructions, input=input)
        )
        session_id = previous_response_id or str(uuid.uuid4())
        invocation = build_copilot_invocation(
            command_prefix=self._resolve_copilot_command_prefix(config),
            prompt_text=prompt_text,
            model_name=config.model_name,
            response_id=session_id,
            extra_args=config.extra_args,
            cwd=self._resolve_cwd(config),
            env=self._resolve_env(config),
            timeout_seconds=config.timeout_seconds,
            output_format="json",
        )
        lines = await _run_jsonl_command(
            command=invocation.command,
            cwd=invocation.cwd,
            env=invocation.env,
            timeout_seconds=invocation.timeout_seconds,
        )
        if transport != "jsonl":
            raise UserError(
                "Copilot CLI sdk_controlled mode currently only supports transport='jsonl'."
            )
        parsed = _parse_copilot_jsonl(lines)
        if config.execution_mode == "sdk_controlled":
            envelope = _parse_controlled_envelope(parsed["message"])
            outputs = _controlled_envelope_to_outputs(
                envelope=envelope, output_schema=output_schema
            )
        else:
            outputs = _basic_autonomous_outputs(
                parsed["message"],
                reasoning_summary=parsed["reasoning_summary"],
            )
        return ModelResponse(
            output=outputs,
            usage=Usage(),
            response_id=session_id,
            provider_response_id=session_id,
            provider_session_id=session_id,
        )

    async def _stream_codex_response(
        self,
        *,
        config: CLIModelConfig,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        previous_response_id: str | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        if config.extra_args:
            raise UserError(
                "Codex CLI provider does not support cli_extra_args yet. "
                "Use explicit provider settings such as model_name, cwd, env, or execution_mode."
            )

        codex = Codex(
            codex_path_override=self._resolve_executable(config, "codex"),
            env=config.env,
        )
        thread_options = ThreadOptions(
            model=config.model_name,
            working_directory=self._resolve_cwd(config),
        )
        thread = (
            codex.resume_thread(previous_response_id, thread_options)
            if previous_response_id
            else codex.start_thread(thread_options)
        )
        prompt_text = _build_autonomous_prompt(system_instructions=system_instructions, input=input)
        stream_result = await thread.run_streamed(
            prompt_text,
            TurnOptions(idle_timeout_seconds=config.timeout_seconds),
        )

        session = _CLIStreamingSession(
            response_id=previous_response_id or thread.id or f"resp_{uuid.uuid4().hex}"
        )
        start_emitted = False
        final_response_id = session.response_id
        final_usage = Usage()
        cwd = self._resolve_cwd(config)

        async for event in stream_result.events:
            if isinstance(event, ThreadStartedEvent):
                final_response_id = event.thread_id
                session.response_id = event.thread_id

            if not start_emitted:
                start_emitted = True
                for stream_event in session.emit_start_events():
                    yield stream_event

            if isinstance(event, (ItemStartedEvent, ItemUpdatedEvent, ItemCompletedEvent)):
                for stream_event in stream_codex_item_event(
                    session=session,
                    item=event.item,
                    event=event,
                    cwd=cwd,
                ):
                    yield stream_event
            elif isinstance(event, TurnCompletedEvent):
                final_usage = _codex_usage_to_usage(event.usage)
            elif isinstance(event, TurnFailedEvent):
                error = event.error.message.strip()
                raise RuntimeError(f"Codex turn failed{(': ' + error) if error else ''}")
            elif isinstance(event, ThreadErrorEvent):
                raise RuntimeError(f"Codex stream error: {event.message}")

        if not start_emitted:
            for stream_event in session.emit_start_events():
                yield stream_event

        for state in list(session.reasoning_states.values()):
            if not state.done and state.text:
                for stream_event in session.finish_reasoning(state.item_id):
                    yield stream_event
        for state in list(session.message_states.values()):
            if not state.done and state.text:
                for stream_event in session.finish_message(state.item_id):
                    yield stream_event

        yield session.complete(response_id=final_response_id, usage=final_usage)

    async def _stream_copilot_response(
        self,
        *,
        config: CLIModelConfig,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        previous_response_id: str | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        transport = _resolve_cli_transport(config)
        prompt_text = _build_autonomous_prompt(system_instructions=system_instructions, input=input)
        if transport == "acp":
            async for event in self._stream_acp_response(
                vendor="copilot",
                config=config,
                command_prefix=self._resolve_copilot_command_prefix(config)
                + (["--model", config.model_name] if config.model_name else []),
                prompt_text=prompt_text,
                previous_response_id=previous_response_id,
            ):
                yield event
            return
        provisional_response_id = previous_response_id or str(uuid.uuid4())
        session = _CLIStreamingSession(response_id=provisional_response_id)
        for stream_event in session.emit_start_events():
            yield stream_event

        adapter = CopilotStreamAdapter(provisional_response_id=provisional_response_id)

        invocation = build_copilot_invocation(
            command_prefix=self._resolve_copilot_command_prefix(config),
            prompt_text=prompt_text,
            model_name=config.model_name,
            response_id=provisional_response_id,
            extra_args=config.extra_args,
            cwd=self._resolve_cwd(config),
            env=self._resolve_env(config),
            timeout_seconds=config.timeout_seconds,
            output_format="json",
            stream=True,
        )

        async for payload in _stream_jsonl_command(
            command=invocation.command,
            cwd=invocation.cwd,
            env=invocation.env,
            timeout_seconds=invocation.timeout_seconds,
        ):
            for stream_event in adapter.consume_payload(session=session, payload=payload):
                yield stream_event

        for stream_event in adapter.finish(session=session):
            yield stream_event

        yield session.complete(response_id=adapter.final_response_id, usage=Usage())

    async def _stream_gemini_response(
        self,
        *,
        config: CLIModelConfig,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        previous_response_id: str | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        transport = _resolve_cli_transport(config)
        prompt_text = _build_autonomous_prompt(system_instructions=system_instructions, input=input)
        if transport == "acp":
            async for event in self._stream_acp_response(
                vendor="gemini",
                config=config,
                command_prefix=[self._resolve_executable(config, "gemini")]
                + (["-m", config.model_name] if config.model_name else []),
                prompt_text=prompt_text,
                previous_response_id=previous_response_id,
            ):
                yield event
            return
        session = _CLIStreamingSession(
            response_id=previous_response_id or f"resp_{uuid.uuid4().hex}"
        )
        adapter = GeminiStreamAdapter(
            initial_response_id=session.response_id,
            usage_from_payload=_gemini_usage_to_usage,
        )
        invocation = build_gemini_invocation(
            executable=self._resolve_executable(config, "gemini"),
            prompt_text=prompt_text,
            model_name=config.model_name,
            previous_response_id=previous_response_id,
            execution_mode=config.execution_mode,
            extra_args=config.extra_args,
            cwd=self._resolve_cwd(config),
            env=self._resolve_env(config),
            timeout_seconds=config.timeout_seconds,
            output_format="stream-json",
        )

        async for payload in _stream_jsonl_command(
            command=invocation.command,
            cwd=invocation.cwd,
            env=invocation.env,
            timeout_seconds=invocation.timeout_seconds,
        ):
            for stream_event in adapter.consume_payload(session=session, payload=payload):
                yield stream_event

        for stream_event in adapter.finish(session=session):
            yield stream_event

        yield session.complete(
            response_id=adapter.final_response_id,
            usage=adapter.final_usage,
        )

    async def _stream_acp_response(
        self,
        *,
        vendor: Literal["gemini", "copilot"],
        config: CLIModelConfig,
        command_prefix: Sequence[str],
        prompt_text: str,
        previous_response_id: str | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        session = _CLIStreamingSession(
            response_id=previous_response_id or f"resp_{uuid.uuid4().hex}"
        )
        adapter = ACPStreamAdapter(
            vendor=vendor,
            initial_response_id=session.response_id,
        )
        invocation = build_acp_invocation(
            command_prefix=command_prefix,
            extra_args=config.extra_args,
            cwd=self._resolve_cwd(config),
            env=self._resolve_env(config),
            timeout_seconds=config.timeout_seconds,
        )

        async for payload in _stream_acp_command(
            command=invocation.command,
            cwd=invocation.cwd,
            env=invocation.env,
            timeout_seconds=invocation.timeout_seconds,
            prompt=prompt_text,
            previous_response_id=previous_response_id,
        ):
            for stream_event in adapter.consume_payload(session=session, payload=payload):
                yield stream_event

        for stream_event in adapter.finish(session=session):
            yield stream_event

        if adapter.final_stop_reason == "cancelled":
            raise RuntimeError(f"{vendor} ACP prompt was cancelled.")

        yield session.complete(response_id=adapter.final_response_id, usage=Usage())

    def _resolve_executable(self, config: CLIModelConfig, fallback_name: str) -> str:
        if config.executable_path:
            return config.executable_path
        executable_path = shutil.which(fallback_name)
        if executable_path is None:
            raise UserError(f"Could not find required CLI executable: {fallback_name}")
        return executable_path

    def _resolve_copilot_command_prefix(self, config: CLIModelConfig) -> list[str]:
        if config.executable_path:
            return [config.executable_path]
        executable_path = shutil.which("copilot")
        if executable_path is not None:
            return [executable_path]
        gh_path = shutil.which("gh")
        if gh_path is not None:
            return [gh_path, "copilot", "--"]
        raise UserError("Could not find required CLI executable: copilot or gh")

    def _resolve_env(self, config: CLIModelConfig) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if value is not None}
        if config.env:
            env.update({str(key): str(value) for key, value in config.env.items()})
        return env

    def _resolve_cwd(self, config: CLIModelConfig) -> str:
        return str(Path(config.cwd).expanduser().resolve()) if config.cwd else os.getcwd()


class CLIProvider(ModelProvider):
    def __init__(
        self,
        *,
        execution_mode: CLIExecutionMode = "cli_autonomous",
        transport: CLITransport = "auto",
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 300.0,
        extra_args: Sequence[str] | None = None,
        executable_path: str | None = None,
        default_model_name: str | None = None,
    ) -> None:
        self._config = CLIProviderConfig(
            execution_mode=execution_mode,
            transport=transport,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
            extra_args=tuple(extra_args or ()),
            executable_path=executable_path,
            default_model_name=default_model_name,
        )

    def get_model(self, model_name: str | None) -> Model:
        resolved_model_name = model_name or self._config.default_model_name
        if resolved_model_name is None:
            raise UserError(
                "CLIProvider requires model_name in the form '<vendor>' or '<vendor>:<model>', "
                "or a default_model_name in the provider config."
            )
        vendor, vendor_model_name = _parse_cli_model_name(resolved_model_name)
        return CLIModel(
            CLIModelConfig(
                vendor=vendor,
                model_name=vendor_model_name,
                execution_mode=self._config.execution_mode,
                transport=self._config.transport,
                cwd=self._config.cwd,
                env=self._config.env,
                timeout_seconds=self._config.timeout_seconds,
                extra_args=self._config.extra_args,
                executable_path=self._config.executable_path,
            )
        )


class _SingleVendorCLIProvider(ModelProvider):
    def __init__(
        self,
        *,
        vendor: CLIVendor,
        execution_mode: CLIExecutionMode = "cli_autonomous",
        transport: CLITransport = "auto",
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 300.0,
        extra_args: Sequence[str] | None = None,
        executable_path: str | None = None,
        default_model_name: str | None = None,
    ) -> None:
        self._vendor = vendor
        self._config = CLIProviderConfig(
            execution_mode=execution_mode,
            transport=transport,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
            extra_args=tuple(extra_args or ()),
            executable_path=executable_path,
            default_model_name=default_model_name,
        )

    def get_model(self, model_name: str | None) -> Model:
        resolved_model_name = model_name or self._config.default_model_name
        return CLIModel(
            CLIModelConfig(
                vendor=self._vendor,
                model_name=resolved_model_name,
                execution_mode=self._config.execution_mode,
                transport=self._config.transport,
                cwd=self._config.cwd,
                env=self._config.env,
                timeout_seconds=self._config.timeout_seconds,
                extra_args=self._config.extra_args,
                executable_path=self._config.executable_path,
            )
        )


class CodexCLIProvider(_SingleVendorCLIProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(vendor="codex", **kwargs)


class GeminiCLIProvider(_SingleVendorCLIProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(vendor="gemini", **kwargs)


class CopilotCLIProvider(_SingleVendorCLIProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(vendor="copilot", **kwargs)


def _parse_cli_model_name(model_name: str) -> tuple[CLIVendor, str | None]:
    raw = require_non_empty_str(model_name, "CLI model_name")
    if raw.startswith("cli:") or raw.startswith("cli/"):
        raw = raw[4:]
    if ":" in raw:
        vendor_name, vendor_model_name = raw.split(":", 1)
    else:
        vendor_name, vendor_model_name = raw, None
    vendor = cast(CLIVendor, vendor_name.strip().lower())
    if vendor not in {"codex", "gemini", "copilot"}:
        raise UserError(f"Unsupported CLI vendor: {vendor_name}")
    cleaned_model_name = vendor_model_name.strip() if vendor_model_name else None
    return vendor, cleaned_model_name or None


def _resolve_cli_model_config(
    base_config: CLIModelConfig,
    model_settings: ModelSettings,
) -> CLIModelConfig:
    if not model_settings.extra_args:
        return base_config

    overrides = _extract_cli_overrides(model_settings.extra_args)
    if not overrides:
        return base_config

    execution_mode = overrides.get("execution_mode", base_config.execution_mode)
    if execution_mode not in {"sdk_controlled", "cli_autonomous"}:
        raise UserError("CLI execution_mode must be either 'sdk_controlled' or 'cli_autonomous'.")
    transport = overrides.get("transport", base_config.transport)
    if transport not in {"auto", "acp", "json", "jsonl", "stream_json"}:
        raise UserError(
            "CLI transport must be one of 'auto', 'acp', 'json', 'jsonl', or 'stream_json'."
        )

    model_name = _coerce_optional_str(overrides.get("model_name"), "CLI model_name")
    executable_path = _coerce_optional_str(
        overrides.get("executable_path"),
        "CLI executable_path",
    )
    cwd = _coerce_optional_str(overrides.get("cwd"), "CLI cwd")
    timeout_seconds = _coerce_timeout_seconds(
        overrides.get("timeout_seconds"),
        base_config.timeout_seconds,
    )
    env = _coerce_cli_env(overrides.get("env"), base_config.env)
    extra_args = _coerce_cli_extra_args(overrides.get("extra_args"), base_config.extra_args)

    return CLIModelConfig(
        vendor=base_config.vendor,
        model_name=model_name if model_name is not None else base_config.model_name,
        executable_path=executable_path
        if executable_path is not None
        else base_config.executable_path,
        execution_mode=cast(CLIExecutionMode, execution_mode),
        transport=cast(CLITransport, transport),
        cwd=cwd if cwd is not None else base_config.cwd,
        env=env,
        timeout_seconds=timeout_seconds,
        extra_args=extra_args,
    )


def _extract_cli_overrides(extra_args: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    nested = extra_args.get("cli")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise UserError("ModelSettings.extra_args['cli'] must be a mapping when provided.")
        overrides.update(dict(nested))

    alias_map = {
        "cli_execution_mode": "execution_mode",
        "cli_transport": "transport",
        "cli_model_name": "model_name",
        "cli_executable_path": "executable_path",
        "cli_cwd": "cwd",
        "cli_env": "env",
        "cli_timeout_seconds": "timeout_seconds",
        "cli_extra_args": "extra_args",
    }
    for source_key, target_key in alias_map.items():
        if source_key in extra_args:
            overrides[target_key] = extra_args[source_key]
    return overrides


def _coerce_optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise UserError(f"{name} must be a string when provided.")
    trimmed = value.strip()
    if not trimmed:
        raise UserError(f"{name} must not be empty when provided.")
    return trimmed


def _coerce_timeout_seconds(value: Any, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise UserError("CLI timeout_seconds override must be a number when provided.")
    timeout_seconds = float(value)
    if timeout_seconds <= 0:
        raise UserError("CLI timeout_seconds override must be greater than zero.")
    return timeout_seconds


def _coerce_cli_env(
    value: Any,
    default: Mapping[str, str] | None,
) -> Mapping[str, str] | None:
    if value is None:
        return default
    if not isinstance(value, Mapping):
        raise UserError("CLI env override must be a mapping of string keys and values.")
    env: dict[str, str] = dict(default or {})
    for key, env_value in value.items():
        if not isinstance(key, str) or not isinstance(env_value, str):
            raise UserError("CLI env override must contain only string keys and string values.")
        env[key] = env_value
    return env


def _coerce_cli_extra_args(value: Any, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise UserError("CLI extra_args override must be a sequence of strings.")
    args: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise UserError("CLI extra_args override must contain only strings.")
        trimmed = entry.strip()
        if not trimmed:
            raise UserError("CLI extra_args override must not contain empty strings.")
        args.append(trimmed)
    return (*tuple(default), *tuple(args))


def _compute_stream_delta(*, label: str, current_text: str, new_text: str) -> str:
    if not new_text.startswith(current_text):
        raise ModelBehaviorError(
            f"{label} must be cumulative. Expected prefix {current_text!r}, got {new_text!r}."
        )
    return new_text[len(current_text) :]


def require_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ModelBehaviorError(f"{name} must be a non-empty string.")
    trimmed = value.strip()
    if not trimmed:
        raise ModelBehaviorError(f"{name} must be a non-empty string.")
    return trimmed


def _normalize_input_text(input: str | list[TResponseInputItem]) -> str:
    if isinstance(input, str):
        return input
    return json.dumps(input, ensure_ascii=False, indent=2)


def _build_autonomous_prompt(
    *,
    system_instructions: str | None,
    input: str | list[TResponseInputItem],
) -> str:
    parts: list[str] = []
    if system_instructions:
        parts.append("## System Instructions")
        parts.append(system_instructions)
    parts.append("## User Input")
    parts.append(_normalize_input_text(input))
    return "\n\n".join(parts)


def _build_sdk_controlled_prompt(
    *,
    vendor: CLIVendor,
    system_instructions: str | None,
    input: str | list[TResponseInputItem],
    tools: list[Tool],
    handoffs: list[Handoff],
    output_schema: AgentOutputSchemaBase | None,
    tool_choice: Any | None,
) -> str:
    del vendor
    prompt_sections: list[str] = [
        "You are running inside the OpenAI Agents SDK compatibility bridge.",
        "You must respond with JSON only. Do not wrap the JSON in markdown fences.",
    ]
    if system_instructions:
        prompt_sections.extend(["## System Instructions", system_instructions])

    available_actions: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            available_actions.append(
                {
                    "type": "function_tool",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.params_json_schema,
                }
            )
    for handoff in handoffs:
        available_actions.append(
            {
                "type": "handoff",
                "name": handoff.tool_name,
                "description": handoff.tool_description,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            }
        )
    if output_schema is not None:
        available_actions.append(
            {
                "type": "structured_final_output",
                "name": "json_tool_call",
                "description": "Use this when you are ready to return the final structured output.",
                "parameters": output_schema.json_schema(),
            }
        )

    prompt_sections.extend(
        [
            "## Available Actions",
            json.dumps(available_actions, ensure_ascii=False, indent=2),
            "## Required JSON Output Shape",
            json.dumps(_ControlledEnvelope.model_json_schema(), ensure_ascii=False, indent=2),
        ]
    )
    if tool_choice == "required":
        prompt_sections.append(
            "At least one tool_calls entry is required. "
            "If you are returning final structured output, use json_tool_call."
        )
    prompt_sections.extend(["## User Input", _normalize_input_text(input)])
    return "\n\n".join(prompt_sections)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def _extract_controlled_envelope_payload(text: str) -> tuple[dict[str, Any], str]:
    cleaned = require_non_empty_str(text, "controlled CLI output").strip()
    decoder = json.JSONDecoder()
    candidates: list[tuple[str, str]] = []

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE):
        block = match.group(1).strip()
        suffix = cleaned[match.end() :].strip()
        candidates.append((block, suffix))

    candidates.append((_strip_code_fences(cleaned), ""))

    for candidate, suffix in candidates:
        if not candidate:
            continue

        for start_char in ("{", "["):
            start_index = candidate.find(start_char)
            if start_index < 0:
                continue
            try:
                payload, end_index = decoder.raw_decode(candidate[start_index:])
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            trailing = candidate[start_index + end_index :].strip()
            extra_text = "\n\n".join(part for part in (trailing, suffix) if part).strip()
            return payload, extra_text

    raise ModelBehaviorError(f"CLI output is not valid JSON envelope: {cleaned}")


def _parse_controlled_envelope(text: str) -> _ControlledEnvelope:
    payload, extra_text = _extract_controlled_envelope_payload(text)
    try:
        envelope = _ControlledEnvelope.model_validate(payload)
    except ValidationError as exc:
        raise ModelBehaviorError(
            f"CLI output does not match controlled envelope schema: {payload}"
        ) from exc
    if extra_text and not envelope.assistant_message:
        return envelope.model_copy(update={"assistant_message": extra_text})
    return envelope


def _controlled_envelope_to_outputs(
    *,
    envelope: _ControlledEnvelope,
    output_schema: AgentOutputSchemaBase | None,
) -> list[TResponseOutputItem]:
    outputs: list[TResponseOutputItem] = []
    if envelope.reasoning_summary:
        outputs.append(_make_reasoning_item(envelope.reasoning_summary))
    for tool_call in envelope.tool_calls:
        tool_name = tool_call.name.strip()
        if output_schema is None and tool_name == "json_tool_call":
            raise ModelBehaviorError(
                "CLI returned json_tool_call but no output schema is configured."
            )
        outputs.append(
            ResponseFunctionToolCall(
                type="function_call",
                name=tool_name,
                arguments=tool_call.arguments_json,
                call_id=f"cli-tool-{uuid.uuid4()}",
                id=f"fc_{uuid.uuid4().hex}",
            )
        )
    if envelope.assistant_message:
        outputs.append(_make_message_output(envelope.assistant_message))
    return outputs


def _make_message_output(text: str) -> ResponseOutputMessage:
    return _make_message_output_with_id(
        item_id=f"msg_{uuid.uuid4().hex}",
        text=text,
        status="completed",
    )


def _make_message_output_with_id(
    *,
    item_id: str,
    text: str,
    status: Literal["in_progress", "completed", "incomplete"],
) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=item_id,
        type="message",
        role="assistant",
        status=status,
        content=[
            ResponseOutputText(
                text=text,
                type="output_text",
                annotations=[],
            )
        ],
    )


def _make_reasoning_item(text: str) -> ResponseReasoningItem:
    return _make_reasoning_item_with_id(
        item_id=f"rs_{uuid.uuid4().hex}",
        text=text,
        status="completed",
    )


def _make_reasoning_item_with_id(
    *,
    item_id: str,
    text: str,
    status: Literal["in_progress", "completed", "incomplete"],
) -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id=item_id,
        type="reasoning",
        summary=[Summary(text=text, type="summary_text")],
        status=status,
    )


def _basic_autonomous_outputs(
    message_text: str,
    *,
    reasoning_summary: str | None,
) -> list[TResponseOutputItem]:
    outputs: list[TResponseOutputItem] = []
    if reasoning_summary:
        outputs.append(_make_reasoning_item(reasoning_summary))
    outputs.append(_make_message_output(message_text))
    return outputs


def _codex_turn_to_outputs(
    thread_items: list[Any],
    final_response: str,
    *,
    cwd: str,
) -> list[TResponseOutputItem]:
    outputs: list[TResponseOutputItem] = []
    saw_message = False
    for item in thread_items:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            reasoning_text = getattr(item, "text", "")
            if reasoning_text:
                outputs.append(_make_reasoning_item(reasoning_text))
        elif item_type == "agent_message":
            message_text = getattr(item, "text", "")
            if message_text:
                outputs.append(_make_message_output(message_text))
                saw_message = True
        else:
            outputs.extend(codex_provider_outputs_for_item(cast(ThreadItem, item), cwd=cwd))
    if final_response and not saw_message:
        outputs.append(_make_message_output(final_response))
    return outputs


def _codex_usage_to_usage(raw_usage: Any) -> Usage:
    if raw_usage is None:
        return Usage()
    return Usage(
        requests=1,
        input_tokens=getattr(raw_usage, "input_tokens", 0),
        output_tokens=getattr(raw_usage, "output_tokens", 0),
        total_tokens=getattr(raw_usage, "input_tokens", 0) + getattr(raw_usage, "output_tokens", 0),
        input_tokens_details=InputTokensDetails(
            cached_tokens=getattr(raw_usage, "cached_input_tokens", 0)
        ),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _gemini_usage_to_usage(payload: Mapping[str, Any]) -> Usage:
    stats = payload.get("stats")
    if not isinstance(stats, Mapping):
        return Usage()

    # `gemini -o stream-json` returns flat token fields directly under `stats`.
    if "input_tokens" in stats or "output_tokens" in stats or "total_tokens" in stats:
        input_tokens = int(stats.get("input_tokens", stats.get("input", 0)) or 0)
        output_tokens = int(stats.get("output_tokens", 0) or 0)
        total_tokens = int(stats.get("total_tokens", input_tokens + output_tokens) or 0)
        cached_tokens = int(stats.get("cached", 0) or 0)
        return Usage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )

    models = stats.get("models")
    if not isinstance(models, Mapping) or not models:
        return Usage()
    first_model = next(iter(models.values()))
    if not isinstance(first_model, Mapping):
        return Usage()
    if (
        "input_tokens" in first_model
        or "output_tokens" in first_model
        or "total_tokens" in first_model
    ):
        input_tokens = int(first_model.get("input_tokens", first_model.get("input", 0)) or 0)
        output_tokens = int(first_model.get("output_tokens", 0) or 0)
        total_tokens = int(first_model.get("total_tokens", input_tokens + output_tokens) or 0)
        cached_tokens = int(first_model.get("cached", 0) or 0)
        return Usage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
    tokens = first_model.get("tokens")
    if not isinstance(tokens, Mapping):
        return Usage()
    input_tokens = int(tokens.get("input", 0) or 0)
    output_tokens = int(tokens.get("candidates", 0) or 0)
    total_tokens = int(tokens.get("total", input_tokens + output_tokens) or 0)
    cached_tokens = int(tokens.get("cached", 0) or 0)
    reasoning_tokens = int(tokens.get("thoughts", 0) or 0)
    return Usage(
        requests=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
    )


def _parse_copilot_jsonl(lines: list[dict[str, Any]]) -> dict[str, str]:
    parsed = parse_copilot_jsonl(lines)
    return {
        "message": require_non_empty_str(parsed.get("message"), "Copilot message"),
        "reasoning_summary": cast(str, parsed.get("reasoning_summary") or ""),
    }


async def _run_json_command(
    *,
    command: list[str],
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    return await run_json_invocation(
        CLISubprocessInvocation(
            command=command,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
        )
    )


async def _stream_jsonl_command(
    *,
    command: list[str],
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> AsyncIterator[dict[str, Any]]:
    async for payload in stream_jsonl_invocation(
        CLISubprocessInvocation(
            command=command,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
        )
    ):
        yield payload


async def _stream_acp_command(
    *,
    command: list[str],
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
    prompt: str,
    previous_response_id: str | None,
) -> AsyncIterator[dict[str, Any]]:
    async for payload in stream_acp_prompt_invocation(
        CLIAcpInvocation(
            command=command,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
        ),
        prompt=prompt,
        previous_session_id=previous_response_id,
    ):
        yield payload


async def _collect_model_response_from_stream(
    stream: AsyncIterator[TResponseStreamEvent],
) -> ModelResponse:
    completed_event: ResponseCompletedEvent | None = None
    async for event in stream:
        if isinstance(event, ResponseCompletedEvent):
            completed_event = event
    if completed_event is None:
        raise ModelBehaviorError("CLI stream ended without a response.completed event.")
    response = completed_event.response
    response_usage = response.usage
    usage = Usage(
        requests=1,
        input_tokens=response_usage.input_tokens if response_usage is not None else 0,
        output_tokens=response_usage.output_tokens if response_usage is not None else 0,
        total_tokens=response_usage.total_tokens if response_usage is not None else 0,
        input_tokens_details=response_usage.input_tokens_details
        if response_usage is not None
        else None,
        output_tokens_details=(
            response_usage.output_tokens_details if response_usage is not None else None
        ),
    )
    return ModelResponse(
        output=cast(list[TResponseOutputItem], response.output),
        usage=usage,
        response_id=response.id,
        provider_response_id=response.id,
        provider_session_id=response.id,
    )


async def _run_jsonl_command(
    *,
    command: list[str],
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    return await run_jsonl_invocation(
        CLISubprocessInvocation(
            command=command,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
        )
    )


def _resolve_cli_transport(config: CLIModelConfig) -> str:
    if config.vendor == "codex":
        if config.transport != "auto":
            raise UserError("Codex CLI only supports transport='auto'.")
        return "thread"

    if config.vendor == "gemini":
        if config.execution_mode == "sdk_controlled":
            allowed = {"auto", "json"}
            default_transport = "json"
        else:
            allowed = {"auto", "acp", "stream_json"}
            default_transport = "stream_json"
    elif config.vendor == "copilot":
        allowed = {"auto", "acp", "jsonl"} if config.execution_mode == "cli_autonomous" else {
            "auto",
            "jsonl",
        }
        default_transport = "jsonl"
    else:
        raise UserError(f"Unsupported CLI vendor: {config.vendor}")

    if config.transport not in allowed:
        raise UserError(
            f"{config.vendor} CLI does not support transport={config.transport!r} "
            f"for execution_mode={config.execution_mode!r}."
        )
    return default_transport if config.transport == "auto" else config.transport


def _build_response_obj(
    output: list[TResponseOutputItem],
    response_id: str | None,
    usage: Usage,
) -> Response:
    sanitized_output = [_coerce_output_item_for_response(item) for item in output]
    return Response(
        id=response_id or f"resp_{uuid.uuid4().hex}",
        created_at=123,
        model="cli_model",
        object="response",
        output=sanitized_output,
        tool_choice="none",
        tools=[],
        top_p=None,
        parallel_tool_calls=False,
        usage=ResponseUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            input_tokens_details=usage.input_tokens_details
            if usage.input_tokens_details is not None
            else InputTokensDetails(cached_tokens=0),
            output_tokens_details=usage.output_tokens_details
            if usage.output_tokens_details is not None
            else OutputTokensDetails(reasoning_tokens=0),
        ),
    )


def _coerce_output_item_for_response(item: TResponseOutputItem) -> ResponseOutputItem:
    if isinstance(item, Mapping):
        payload = dict(item)
        payload.pop("managed_by", None)
        payload.pop("provider_data", None)
        payload.pop("created_by", None)
        if payload.get("type") == "shell_call_output":
            payload.pop("shell_output", None)
            outputs = payload.get("output")
            if isinstance(outputs, list):
                for entry in outputs:
                    if isinstance(entry, dict):
                        entry.pop("created_by", None)
        try:
            return cast(ResponseOutputItem, construct_type(type_=ResponseOutputItem, value=payload))
        except Exception as exc:
            raise ModelBehaviorError(
                f"Could not coerce CLI output item into a Responses output item: {payload}"
            ) from exc
    return cast(ResponseOutputItem, item)


async def _stream_events_from_model_response(
    response: ModelResponse,
) -> AsyncIterator[TResponseStreamEvent]:
    response_obj = _build_response_obj(response.output, response.response_id, response.usage)
    sequence_number = 0
    yield ResponseCreatedEvent(
        type="response.created",
        response=response_obj,
        sequence_number=sequence_number,
    )
    sequence_number += 1
    yield ResponseInProgressEvent(
        type="response.in_progress",
        response=response_obj,
        sequence_number=sequence_number,
    )
    sequence_number += 1

    for output_index, output_item in enumerate(response_obj.output):
        yield ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=output_item,
            output_index=output_index,
            sequence_number=sequence_number,
        )
        sequence_number += 1

        if isinstance(output_item, ResponseReasoningItem):
            for summary_index, summary in enumerate(output_item.summary):
                yield ResponseReasoningSummaryPartAddedEvent(
                    type="response.reasoning_summary_part.added",
                    item_id=output_item.id,
                    output_index=output_index,
                    summary_index=summary_index,
                    part=AddedEventPart(text=summary.text, type=summary.type),
                    sequence_number=sequence_number,
                )
                sequence_number += 1
                yield ResponseReasoningSummaryTextDeltaEvent(
                    type="response.reasoning_summary_text.delta",
                    item_id=output_item.id,
                    output_index=output_index,
                    summary_index=summary_index,
                    delta=summary.text,
                    sequence_number=sequence_number,
                )
                sequence_number += 1
                yield ResponseReasoningSummaryTextDoneEvent(
                    type="response.reasoning_summary_text.done",
                    item_id=output_item.id,
                    output_index=output_index,
                    summary_index=summary_index,
                    text=summary.text,
                    sequence_number=sequence_number,
                )
                sequence_number += 1
                yield ResponseReasoningSummaryPartDoneEvent(
                    type="response.reasoning_summary_part.done",
                    item_id=output_item.id,
                    output_index=output_index,
                    summary_index=summary_index,
                    part=DoneEventPart(text=summary.text, type=summary.type),
                    sequence_number=sequence_number,
                )
                sequence_number += 1
        elif isinstance(output_item, ResponseFunctionToolCall):
            yield ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                item_id=output_item.call_id,
                output_index=output_index,
                delta=output_item.arguments,
                sequence_number=sequence_number,
            )
            sequence_number += 1
            yield ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                item_id=output_item.call_id,
                output_index=output_index,
                arguments=output_item.arguments,
                name=output_item.name,
                sequence_number=sequence_number,
            )
            sequence_number += 1
        elif isinstance(output_item, ResponseOutputMessage):
            for content_index, content_part in enumerate(output_item.content or []):
                if isinstance(content_part, ResponseOutputText):
                    yield ResponseContentPartAddedEvent(
                        type="response.content_part.added",
                        item_id=output_item.id,
                        output_index=output_index,
                        content_index=content_index,
                        part=content_part,
                        sequence_number=sequence_number,
                    )
                    sequence_number += 1
                    yield ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        item_id=output_item.id,
                        output_index=output_index,
                        content_index=content_index,
                        delta=content_part.text,
                        logprobs=[],
                        sequence_number=sequence_number,
                    )
                    sequence_number += 1
                    yield ResponseTextDoneEvent(
                        type="response.output_text.done",
                        item_id=output_item.id,
                        output_index=output_index,
                        content_index=content_index,
                        text=content_part.text,
                        logprobs=[],
                        sequence_number=sequence_number,
                    )
                    sequence_number += 1
                    yield ResponseContentPartDoneEvent(
                        type="response.content_part.done",
                        item_id=output_item.id,
                        output_index=output_index,
                        content_index=content_index,
                        part=content_part,
                        sequence_number=sequence_number,
                    )
                    sequence_number += 1

        yield ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=output_item,
            output_index=output_index,
            sequence_number=sequence_number,
        )
        sequence_number += 1

    yield ResponseCompletedEvent(
        type="response.completed",
        response=response_obj,
        sequence_number=sequence_number,
    )
