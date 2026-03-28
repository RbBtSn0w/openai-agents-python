from __future__ import annotations

from types import SimpleNamespace

import pytest

import agents.extensions.models.cli_subprocess_adapter as adapter_module
from agents.extensions.models.cli_subprocess_adapter import (
    CLISubprocessInvocation,
    parse_copilot_jsonl,
    run_json_invocation,
    run_jsonl_invocation,
    stream_jsonl_invocation,
)


@pytest.mark.asyncio
async def test_run_json_invocation_parses_object(monkeypatch) -> None:
    async def fake_to_thread(func, *args, **kwargs):
        del func, args, kwargs
        return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(adapter_module.asyncio, "to_thread", fake_to_thread)

    payload = await run_json_invocation(
        CLISubprocessInvocation(
            command=["gemini", "-p", "hi", "-o", "json"],
            cwd="/tmp",
            env={},
            timeout_seconds=5,
        )
    )

    assert payload == {"ok": True}


@pytest.mark.asyncio
async def test_run_json_invocation_raises_on_nonzero_exit(monkeypatch) -> None:
    async def fake_to_thread(func, *args, **kwargs):
        del func, args, kwargs
        return SimpleNamespace(returncode=7, stdout="", stderr="boom")

    monkeypatch.setattr(adapter_module.asyncio, "to_thread", fake_to_thread)

    with pytest.raises(RuntimeError, match="exit code 7"):
        await run_json_invocation(
            CLISubprocessInvocation(
                command=["gemini", "-p", "hi", "-o", "json"],
                cwd="/tmp",
                env={},
                timeout_seconds=5,
            )
        )


@pytest.mark.asyncio
async def test_run_jsonl_invocation_parses_multiple_events(monkeypatch) -> None:
    async def fake_to_thread(func, *args, **kwargs):
        del func, args, kwargs
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"one"}\n{"type":"two"}\n',
            stderr="",
        )

    monkeypatch.setattr(adapter_module.asyncio, "to_thread", fake_to_thread)

    payloads = await run_jsonl_invocation(
        CLISubprocessInvocation(
            command=["copilot", "--output-format", "json"],
            cwd="/tmp",
            env={},
            timeout_seconds=5,
        )
    )

    assert payloads == [{"type": "one"}, {"type": "two"}]


@pytest.mark.asyncio
async def test_stream_jsonl_invocation_yields_events(monkeypatch) -> None:
    class _FakeStdout:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = [*lines, b""]

        async def readline(self) -> bytes:
            return self._lines.pop(0)

    class _FakeStderr:
        async def read(self) -> bytes:
            return b""

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = _FakeStdout([b'{"type":"one"}\n', b'{"type":"two"}\n'])
            self.stderr = _FakeStderr()
            self.returncode = None

        async def wait(self) -> int:
            self.returncode = 0
            return 0

        def kill(self) -> None:
            self.returncode = -9

    async def fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return _FakeProcess()

    monkeypatch.setattr(
        adapter_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    payloads = [
        payload
        async for payload in stream_jsonl_invocation(
            CLISubprocessInvocation(
                command=["copilot", "--output-format", "json", "--stream", "on"],
                cwd="/tmp",
                env={},
                timeout_seconds=5,
            )
        )
    ]

    assert payloads == [{"type": "one"}, {"type": "two"}]


def test_parse_copilot_jsonl_extracts_message_and_reasoning() -> None:
    parsed = parse_copilot_jsonl(
        [
            {
                "type": "assistant.reasoning_delta",
                "data": {"reasoningId": "r1", "deltaContent": "Plan: "},
            },
            {
                "type": "assistant.reasoning_delta",
                "data": {"reasoningId": "r1", "deltaContent": "inspect repo."},
            },
            {
                "type": "assistant.message",
                "data": {"messageId": "m1", "content": "Final answer"},
            },
        ]
    )

    assert parsed == {
        "message": "Final answer",
        "reasoning_summary": "Plan: inspect repo.",
    }
