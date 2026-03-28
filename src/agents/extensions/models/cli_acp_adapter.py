from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...exceptions import ModelBehaviorError, UserError

ACP_PROTOCOL_VERSION = 1
_ACP_CLIENT_INFO = {
    "name": "openai-agents-python",
    "title": "OpenAI Agents Python",
    "version": "dev",
}
_ACP_SHUTDOWN_TIMEOUT_SECONDS = 1.0
_ACP_CLIENT_CAPABILITIES = {
    "fs": {
        "readTextFile": False,
        "writeTextFile": False,
    },
    "terminal": False,
}


@dataclass(frozen=True)
class CLIAcpInvocation:
    command: list[str]
    cwd: str
    env: Mapping[str, str]
    timeout_seconds: float


def build_acp_invocation(
    *,
    command_prefix: Sequence[str],
    extra_args: Sequence[str],
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> CLIAcpInvocation:
    command = [*command_prefix, "--acp", *extra_args]
    return CLIAcpInvocation(
        command=command,
        cwd=cwd,
        env=env,
        timeout_seconds=timeout_seconds,
    )


async def stream_acp_prompt_invocation(
    invocation: CLIAcpInvocation,
    *,
    prompt: str,
    previous_session_id: str | None,
) -> AsyncIterator[dict[str, Any]]:
    session = _AcpClientProcess(invocation)
    await session.start()
    try:
        initialize_result = await session.request(
            "initialize",
            {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": _ACP_CLIENT_CAPABILITIES,
                "clientInfo": _ACP_CLIENT_INFO,
            },
        )
        session_id, session_payload = await session.start_or_resume_session(
            initialize_result=initialize_result,
            previous_session_id=previous_session_id,
        )
        yield {
            "type": "acp.session_initialized",
            "session_id": session_id,
            "initialize": initialize_result,
            "session": session_payload,
        }

        prompt_task = asyncio.create_task(
            session.request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                },
            )
        )

        while True:
            notification_task = asyncio.create_task(session.next_notification())
            done, pending = await asyncio.wait(
                {prompt_task, notification_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if notification_task in done:
                notification = notification_task.result()
                if notification is not None:
                    yield notification
                continue

            notification_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await notification_task

            prompt_result = prompt_task.result()
            yield {
                "type": "acp.prompt_result",
                "session_id": session_id,
                "result": prompt_result,
            }

            while True:
                notification = session.get_queued_notification()
                if notification is None:
                    break
                yield notification
            break
    finally:
        await session.close()


class _AcpClientProcess:
    def __init__(self, invocation: CLIAcpInvocation) -> None:
        self._invocation = invocation
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[bytes] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._notification_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._request_counter = 0
        self._write_lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> None:
        process = await asyncio.create_subprocess_exec(
            *self._invocation.command,
            cwd=self._invocation.cwd,
            env=dict(self._invocation.env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("ACP subprocess did not expose stdin/stdout/stderr pipes.")
        self._process = process
        self._stderr_task = asyncio.create_task(process.stderr.read())
        self._reader_task = asyncio.create_task(self._read_stdout())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task

        if self._process is not None and self._process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._process.kill()
            with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                await asyncio.wait_for(
                    self._process.wait(),
                    timeout=_ACP_SHUTDOWN_TIMEOUT_SECONDS,
                )

        error = RuntimeError("ACP subprocess closed.")
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(error)
        self._pending_requests.clear()
        await self._notification_queue.put(None)

    async def request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if self._process is None:
            raise RuntimeError("ACP subprocess has not been started.")
        request_id = self._next_request_id()
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future
        await self._write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": dict(params),
            }
        )
        try:
            return await asyncio.wait_for(future, timeout=self._invocation.timeout_seconds)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RuntimeError(
                "ACP request timed out after "
                f"{self._invocation.timeout_seconds} seconds: "
                f"{_format_command(self._invocation.command)}"
            ) from exc
        finally:
            self._pending_requests.pop(request_id, None)

    async def start_or_resume_session(
        self,
        *,
        initialize_result: Mapping[str, Any],
        previous_session_id: str | None,
    ) -> tuple[str, dict[str, Any]]:
        if previous_session_id:
            agent_capabilities = initialize_result.get("agentCapabilities")
            if isinstance(agent_capabilities, Mapping) and agent_capabilities.get("loadSession"):
                payload = await self.request(
                    "session/load",
                    {
                        "sessionId": previous_session_id,
                        "cwd": self._invocation.cwd,
                        "mcpServers": [],
                    },
                )
                return previous_session_id, payload
            session_capabilities = (
                agent_capabilities.get("sessionCapabilities")
                if isinstance(agent_capabilities, Mapping)
                else None
            )
            if isinstance(session_capabilities, Mapping) and session_capabilities.get("resume"):
                payload = await self.request(
                    "session/resume",
                    {
                        "sessionId": previous_session_id,
                        "cwd": self._invocation.cwd,
                        "mcpServers": [],
                    },
                )
                return previous_session_id, payload
            raise UserError(
                "ACP session resume was requested, but the CLI agent did not advertise "
                "session/load or session/resume support."
            )

        payload = await self.request(
            "session/new",
            {
                "cwd": self._invocation.cwd,
                "mcpServers": [],
            },
        )
        session_id = _require_non_empty_str(payload.get("sessionId"), "ACP sessionId")
        return session_id, payload

    async def next_notification(self) -> dict[str, Any] | None:
        return await asyncio.wait_for(
            self._notification_queue.get(),
            timeout=self._invocation.timeout_seconds,
        )

    def get_queued_notification(self) -> dict[str, Any] | None:
        try:
            return self._notification_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def _read_stdout(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None

        try:
            while True:
                raw_line = await self._process.stdout.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ModelBehaviorError(f"ACP stdout line is not valid JSON: {line}") from exc
                if not isinstance(payload, dict):
                    raise ModelBehaviorError("ACP stream message must be a JSON object.")
                await self._handle_message(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(exc)
        finally:
            error = await self._termination_error()
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(error)
            self._pending_requests.clear()
            await self._notification_queue.put(None)

    async def _handle_message(self, payload: dict[str, Any]) -> None:
        if "method" in payload:
            method = payload.get("method")
            if not isinstance(method, str):
                raise ModelBehaviorError("ACP method field must be a string.")
            params = payload.get("params")
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise ModelBehaviorError("ACP params field must be an object when provided.")
            if "id" in payload:
                request_id = payload.get("id")
                if not isinstance(request_id, (str, int)):
                    raise ModelBehaviorError("ACP request id must be a string or integer.")
                await self._handle_agent_request(str(request_id), method, params)
                return

            if method == "session/update":
                await self._notification_queue.put(
                    {
                        "type": "acp.session_update",
                        "session_id": params.get("sessionId"),
                        "update": params.get("update"),
                    }
                )
                return

            await self._notification_queue.put(
                {
                    "type": "acp.notification",
                    "method": method,
                    "params": params,
                }
            )
            return

        request_id = payload.get("id")
        if request_id is None:
            raise ModelBehaviorError("ACP response message must include an id.")
        future = self._pending_requests.get(str(request_id))
        if future is None:
            return
        if "error" in payload:
            error_payload = payload.get("error")
            future.set_exception(_build_rpc_error(error_payload))
            return
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ModelBehaviorError("ACP response result must be an object.")
        future.set_result(result)

    async def _handle_agent_request(
        self,
        request_id: str,
        method: str,
        params: dict[str, Any],
    ) -> None:
        if method == "session/request_permission":
            await self._write_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "outcome": _build_permission_outcome(params.get("options")),
                    },
                }
            )
            return

        await self._write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
        )

    async def _write_message(self, payload: Mapping[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("ACP subprocess stdin is not available.")
        async with self._write_lock:
            self._process.stdin.write(
                (json.dumps(dict(payload), ensure_ascii=False) + "\n").encode("utf-8")
            )
            await self._process.stdin.drain()

    async def _termination_error(self) -> Exception:
        stderr = await self._stderr_text()
        return_code = None
        if self._process is not None:
            return_code = self._process.returncode
            if return_code is None:
                return_code = await self._process.wait()
        if return_code in {None, 0}:
            return RuntimeError("ACP subprocess closed unexpectedly.")
        message = (
            f"ACP command failed with exit code {return_code}: "
            f"{_format_command(self._invocation.command)}"
        )
        if stderr:
            message += f" | {stderr}"
        return RuntimeError(message)

    async def _stderr_text(self) -> str:
        if self._stderr_task is None:
            return ""
        if not self._stderr_task.done():
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            return ""
        with contextlib.suppress(asyncio.CancelledError):
            stderr_bytes = self._stderr_task.result()
            return stderr_bytes.decode("utf-8", errors="replace").strip()
        return ""

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return str(self._request_counter)


def _build_permission_outcome(options: Any) -> dict[str, Any]:
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        return {"outcome": "cancelled"}

    chosen_option_id: str | None = None
    fallback_option_id: str | None = None
    for option in options:
        if not isinstance(option, Mapping):
            continue
        option_id = option.get("optionId")
        if not isinstance(option_id, str) or not option_id.strip():
            continue
        fallback_option_id = fallback_option_id or option_id
        option_kind = option.get("kind")
        if option_kind in {"allow_once", "allow_always"}:
            chosen_option_id = option_id
            break

    selected_option_id = chosen_option_id or fallback_option_id
    if selected_option_id is None:
        return {"outcome": "cancelled"}
    return {
        "outcome": "selected",
        "optionId": selected_option_id,
    }


def _build_rpc_error(error_payload: Any) -> RuntimeError:
    if isinstance(error_payload, Mapping):
        code = error_payload.get("code")
        message = error_payload.get("message")
        if isinstance(message, str) and message.strip():
            if isinstance(code, int):
                return RuntimeError(f"ACP request failed ({code}): {message}")
            return RuntimeError(f"ACP request failed: {message}")
    return RuntimeError(f"ACP request failed: {error_payload}")


def _require_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ModelBehaviorError(f"{name} must be a non-empty string.")
    trimmed = value.strip()
    if not trimmed:
        raise ModelBehaviorError(f"{name} must be a non-empty string.")
    return trimmed


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
