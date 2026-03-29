from __future__ import annotations

import asyncio

import pytest

import agents.extensions.models.cli_acp_adapter as adapter_module
from agents.extensions.models.cli_acp_adapter import (
    CLIAcpInvocation,
    _AcpClientProcess,
    _build_permission_outcome,
    build_acp_invocation,
    stream_acp_prompt_invocation,
)


def test_build_acp_invocation_adds_acp_flag() -> None:
    invocation = build_acp_invocation(
        command_prefix=["gemini", "-m", "gemini-2.5-flash"],
        extra_args=["--debug"],
        cwd="/tmp/workspace",
        env={"PATH": "/bin"},
        timeout_seconds=30,
    )

    assert invocation.command == [
        "gemini",
        "-m",
        "gemini-2.5-flash",
        "--acp",
        "--debug",
    ]
    assert invocation.cwd == "/tmp/workspace"
    assert invocation.env == {"PATH": "/bin"}
    assert invocation.timeout_seconds == 30


def test_build_acp_invocation_supports_direct_agent_launch() -> None:
    invocation = build_acp_invocation(
        command_prefix=["codex-acp"],
        extra_args=["-c", 'model="gpt-5.4"'],
        cwd="/tmp/workspace",
        env={"PATH": "/bin"},
        timeout_seconds=30,
        append_acp_flag=False,
        session_model="gpt-5.4/high",
    )

    assert invocation.command == ["codex-acp", "-c", 'model="gpt-5.4"']
    assert invocation.session_model == "gpt-5.4/high"


def test_build_permission_outcome_prefers_allow_option() -> None:
    outcome = _build_permission_outcome(
        [
            {"kind": "reject_once", "optionId": "reject", "name": "Reject"},
            {"kind": "allow_once", "optionId": "allow", "name": "Allow once"},
        ]
    )

    assert outcome == {
        "outcome": "selected",
        "optionId": "allow",
    }


def test_build_permission_outcome_cancels_without_valid_options() -> None:
    outcome = _build_permission_outcome([{"kind": "allow_once", "name": "Allow"}])

    assert outcome == {"outcome": "cancelled"}


@pytest.mark.asyncio
async def test_acp_client_close_does_not_hang_when_process_wait_stalls(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.kill_called = False

        def kill(self) -> None:
            self.kill_called = True
            self.returncode = -9

        async def wait(self) -> int:
            await asyncio.sleep(999)
            return -9

    invocation = CLIAcpInvocation(
        command=["copilot", "--acp"],
        cwd="/tmp/workspace",
        env={"PATH": "/bin"},
        timeout_seconds=30,
    )
    session = _AcpClientProcess(invocation)
    fake_process = FakeProcess()
    session._process = fake_process  # type: ignore[assignment]
    session._reader_task = asyncio.create_task(asyncio.sleep(999))
    monkeypatch.setattr(adapter_module, "_ACP_SHUTDOWN_TIMEOUT_SECONDS", 0.01)

    await asyncio.wait_for(session.close(), timeout=0.2)

    assert fake_process.kill_called is True


@pytest.mark.asyncio
async def test_stream_acp_prompt_invocation_sets_session_model_when_requested(monkeypatch) -> None:
    events: list[dict[str, object]] = []

    class FakeSession:
        def __init__(self, invocation: CLIAcpInvocation) -> None:
            self.invocation = invocation

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
            if method == "initialize":
                return {"protocolVersion": 1}
            if method == "session/prompt":
                return {"stopReason": "end_turn"}
            raise AssertionError(f"Unexpected request method: {method}")

        async def start_or_resume_session(
            self,
            *,
            initialize_result,
            previous_session_id,
        ) -> tuple[str, dict[str, object]]:
            del initialize_result, previous_session_id
            return "session-1", {"sessionId": "session-1", "models": {"currentModelId": "gpt-5.4"}}

        async def configure_session(
            self,
            *,
            session_id: str,
            session_payload: dict[str, object],
            model_name: str | None,
        ) -> None:
            events.append(
                {
                    "session_id": session_id,
                    "session_payload": session_payload,
                    "model_name": model_name,
                }
            )

        async def next_notification(self) -> dict[str, object] | None:
            await asyncio.sleep(999)
            return None

        def get_queued_notification(self) -> dict[str, object] | None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(adapter_module, "_AcpClientProcess", FakeSession)

    invocation = build_acp_invocation(
        command_prefix=["codex-acp"],
        extra_args=[],
        cwd="/tmp/workspace",
        env={"PATH": "/bin"},
        timeout_seconds=30,
        append_acp_flag=False,
        session_model="gpt-5.4/high",
    )

    payloads = [
        payload
        async for payload in stream_acp_prompt_invocation(
            invocation,
            prompt="Reply with OK only.",
            previous_session_id=None,
        )
    ]

    assert payloads[0]["type"] == "acp.session_initialized"
    assert payloads[-1]["type"] == "acp.prompt_result"
    assert events == [
        {
            "session_id": "session-1",
            "session_payload": {"sessionId": "session-1", "models": {"currentModelId": "gpt-5.4"}},
            "model_name": "gpt-5.4/high",
        }
    ]
