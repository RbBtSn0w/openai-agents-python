from __future__ import annotations

import asyncio
import os
import shutil

import pytest
from openai.types.responses import ResponseOutputMessage

from agents.extensions.models import CLIModel
from agents.extensions.models.cli_model import CLIModelConfig
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing

LIVE_TEST_ENV = "AGENTS_RUN_CLI_LIVE_TESTS"

pytestmark = [
    pytest.mark.allow_call_model_methods,
    pytest.mark.serial,
    pytest.mark.skipif(
        os.environ.get(LIVE_TEST_ENV) != "1",
        reason=f"Set {LIVE_TEST_ENV}=1 to run live CLI runtime tests.",
    ),
]


def _vendor_available(vendor: str) -> bool:
    if vendor == "copilot":
        return shutil.which("copilot") is not None or shutil.which("gh") is not None
    return shutil.which(vendor) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vendor", "model_name"),
    [
        ("codex", None),
        ("gemini", "gemini-2.5-flash"),
        ("copilot", None),
    ],
)
async def test_cli_model_live_get_response(vendor: str, model_name: str | None) -> None:
    if not _vendor_available(vendor):
        pytest.skip(f"{vendor} CLI is not installed in PATH.")

    model = CLIModel(
        CLIModelConfig(
            vendor=vendor,
            model_name=model_name,
            timeout_seconds=60,
        )
    )
    response = await model.get_response(
        system_instructions="Reply with OK only. Do not add any other text.",
        input="Reply with OK only.",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )

    messages = [
        item.content[0].text for item in response.output if isinstance(item, ResponseOutputMessage)
    ]
    assert messages, f"{vendor} live response did not include an assistant message."
    assert messages[-1].strip().rstrip(".!") == "OK"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vendor", "model_name"),
    [
        ("gemini", "gemini-2.5-flash"),
        ("copilot", None),
    ],
)
async def test_cli_model_live_get_response_via_acp(vendor: str, model_name: str | None) -> None:
    if not _vendor_available(vendor):
        pytest.skip(f"{vendor} CLI is not installed in PATH.")

    cli_timeout_seconds = 10 if vendor == "gemini" else 30
    overall_timeout_seconds = 15 if vendor == "gemini" else 45
    model = CLIModel(
        CLIModelConfig(
            vendor=vendor,
            model_name=model_name,
            timeout_seconds=cli_timeout_seconds,
            transport="acp",
        )
    )
    try:
        response = await asyncio.wait_for(
            model.get_response(
                system_instructions="Reply with OK only. Do not add any other text.",
                input="Reply with OK only.",
                model_settings=ModelSettings(),
                tools=[],
                output_schema=None,
                handoffs=[],
                tracing=ModelTracing.DISABLED,
                previous_response_id=None,
                conversation_id=None,
                prompt=None,
            ),
            timeout=overall_timeout_seconds,
        )
    except RuntimeError as exc:
        if vendor == "gemini" and "ACP request timed out" in str(exc):
            pytest.skip("Gemini CLI ACP did not respond to initialize in this environment.")
        raise
    except asyncio.TimeoutError:
        if vendor == "gemini":
            pytest.skip("Gemini CLI ACP did not complete in this environment.")
        raise

    messages = [
        item.content[0].text for item in response.output if isinstance(item, ResponseOutputMessage)
    ]
    assert messages, f"{vendor} ACP live response did not include an assistant message."
    assert messages[-1].strip().rstrip(".!") == "OK"
