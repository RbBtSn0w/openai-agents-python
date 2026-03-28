from __future__ import annotations

import asyncio

import pytest

import agents.extensions.models.cli_acp_adapter as adapter_module
from agents.extensions.models.cli_acp_adapter import (
    CLIAcpInvocation,
    _AcpClientProcess,
    _build_permission_outcome,
    build_acp_invocation,
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
