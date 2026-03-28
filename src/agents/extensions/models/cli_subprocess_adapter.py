from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import subprocess
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ...exceptions import ModelBehaviorError

if TYPE_CHECKING:
    from .cli_model import CLIExecutionMode


@dataclass(frozen=True)
class CLISubprocessInvocation:
    command: list[str]
    cwd: str
    env: Mapping[str, str]
    timeout_seconds: float


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_gemini_invocation(
    *,
    executable: str,
    prompt_text: str,
    model_name: str | None,
    previous_response_id: str | None,
    execution_mode: CLIExecutionMode,
    extra_args: Sequence[str],
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
    output_format: str,
) -> CLISubprocessInvocation:
    command = [executable, "-p", prompt_text, "-o", output_format]
    if model_name:
        command.extend(["-m", model_name])
    if previous_response_id:
        command.extend(["--resume", previous_response_id])
    if execution_mode == "sdk_controlled" and "--approval-mode" not in extra_args:
        command.extend(["--approval-mode", "plan"])
    command.extend(extra_args)
    return CLISubprocessInvocation(
        command=command,
        cwd=cwd,
        env=env,
        timeout_seconds=timeout_seconds,
    )


def build_copilot_invocation(
    *,
    command_prefix: Sequence[str],
    prompt_text: str,
    model_name: str | None,
    response_id: str,
    extra_args: Sequence[str],
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
    output_format: str = "json",
    stream: bool = False,
) -> CLISubprocessInvocation:
    command = [
        *command_prefix,
        "--output-format",
        output_format,
    ]
    if stream:
        command.extend(["--stream", "on"])
    command.extend([f"--resume={response_id}", "-p", prompt_text])
    if model_name:
        command.extend(["--model", model_name])
    command.extend(extra_args)
    return CLISubprocessInvocation(
        command=command,
        cwd=cwd,
        env=env,
        timeout_seconds=timeout_seconds,
    )


def parse_copilot_jsonl(lines: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    reasoning_parts: list[str] = []
    final_message: str | None = None
    for payload in lines:
        event_type = payload.get("type")
        data = payload.get("data")
        if event_type == "assistant.reasoning_delta" and isinstance(data, Mapping):
            delta = data.get("deltaContent")
            if isinstance(delta, str):
                reasoning_parts.append(delta)
        elif event_type == "assistant.message" and isinstance(data, Mapping):
            content = data.get("content")
            if isinstance(content, str):
                final_message = content
    if final_message is None:
        raise ModelBehaviorError("Copilot JSONL did not contain an assistant.message event.")
    reasoning_summary = "".join(reasoning_parts).strip() or None
    return {"message": final_message, "reasoning_summary": reasoning_summary}


async def run_json_invocation(invocation: CLISubprocessInvocation) -> dict[str, Any]:
    result = await asyncio.to_thread(
        subprocess.run,
        invocation.command,
        cwd=invocation.cwd,
        env=dict(invocation.env),
        text=True,
        capture_output=True,
        timeout=invocation.timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            "CLI command failed with exit code "
            f"{result.returncode}: {_format_command(invocation.command)}"
            f" | {stderr or result.stdout.strip()}"
        )
    stdout = _require_non_empty_str(result.stdout, "CLI stdout")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ModelBehaviorError(f"CLI stdout is not valid JSON: {stdout}") from exc
    if not isinstance(payload, dict):
        raise ModelBehaviorError("CLI JSON output must be an object.")
    return cast(dict[str, Any], payload)


async def run_jsonl_invocation(invocation: CLISubprocessInvocation) -> list[dict[str, Any]]:
    result = await asyncio.to_thread(
        subprocess.run,
        invocation.command,
        cwd=invocation.cwd,
        env=dict(invocation.env),
        text=True,
        capture_output=True,
        timeout=invocation.timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            "CLI command failed with exit code "
            f"{result.returncode}: {_format_command(invocation.command)}"
            f" | {stderr or result.stdout.strip()}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ModelBehaviorError("CLI stdout did not contain any JSONL events.")
    parsed: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModelBehaviorError(f"CLI stdout line is not valid JSON: {line}") from exc
        if not isinstance(payload, dict):
            raise ModelBehaviorError("CLI JSONL event must be an object.")
        parsed.append(cast(dict[str, Any], payload))
    return parsed


async def stream_jsonl_invocation(
    invocation: CLISubprocessInvocation,
) -> AsyncIterator[dict[str, Any]]:
    process = await asyncio.create_subprocess_exec(
        *invocation.command,
        cwd=invocation.cwd,
        env=dict(invocation.env),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("CLI subprocess did not expose stdout/stderr pipes.")

    stderr_task = asyncio.create_task(process.stderr.read())

    try:
        saw_payload = False
        while True:
            try:
                raw_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=invocation.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.wait()
                stderr = (await stderr_task).decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    "CLI JSONL stream timed out after "
                    f"{invocation.timeout_seconds} seconds: {_format_command(invocation.command)}"
                    f"{' | ' + stderr if stderr else ''}"
                ) from exc

            if not raw_line:
                break

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            saw_payload = True
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ModelBehaviorError(f"CLI stdout line is not valid JSON: {line}") from exc
            if not isinstance(payload, dict):
                raise ModelBehaviorError("CLI JSONL event must be an object.")
            yield cast(dict[str, Any], payload)

        return_code = await asyncio.wait_for(process.wait(), timeout=invocation.timeout_seconds)
        stderr = (await stderr_task).decode("utf-8", errors="replace").strip()
        if return_code != 0:
            raise RuntimeError(
                f"CLI command failed with exit code {return_code}: "
                f"{_format_command(invocation.command)}"
                f"{' | ' + stderr if stderr else ''}"
            )
        if not saw_payload:
            raise ModelBehaviorError("CLI stdout did not contain any JSONL events.")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        if not stderr_task.done():
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task


def _require_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ModelBehaviorError(f"{name} must be a non-empty string.")
    trimmed = value.strip()
    if not trimmed:
        raise ModelBehaviorError(f"{name} must be a non-empty string.")
    return trimmed
