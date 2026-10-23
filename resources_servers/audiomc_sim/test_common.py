# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep API reasoning separate from final text across Gym response representations."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import ResponsesConverter
from resources_servers.audiomc_sim.common import infer, infer_with_reasoning


def client_response(*, output, status="completed"):
    body = {
        "id": "response",
        "created_at": 0,
        "model": "teacher",
        "object": "response",
        "status": status,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    http = SimpleNamespace(ok=True, status=200, read=AsyncMock(return_value=json.dumps(body).encode()))
    return SimpleNamespace(post=AsyncMock(return_value=http))


def message(text):
    return {
        "id": "message",
        "role": "assistant",
        "type": "message",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


@pytest.mark.asyncio
async def test_native_chat_reasoning_survives_gym_conversion_without_entering_final_answer():
    converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=True)
    output = converter.postprocess_assistant_message_dict(
        {"role": "assistant", "content": "<think>Provider reasoning.</think>Spoken answer."}
    )
    client = client_response(output=[item.model_dump() for item in output])
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    assert await infer_with_reasoning(client, "teacher", params) == ("Spoken answer.", "Provider reasoning.")
    assert await infer(client, "teacher", params) == "Spoken answer."


@pytest.mark.asyncio
async def test_native_content_reasoning_is_not_duplicated_with_its_summary():
    client = client_response(
        output=[
            {
                "id": "reason",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Summary"}],
                "content": [{"type": "reasoning_text", "text": "Native content reasoning."}],
            },
            message("Final answer."),
        ]
    )
    assert await infer_with_reasoning(client, "teacher", NeMoGymResponseCreateParamsNonStreaming(input=[])) == (
        "Final answer.",
        "Native content reasoning.",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", [None, "", "   "])
async def test_missing_native_reasoning_is_a_generation_failure(reasoning):
    output = [message("Final answer.")]
    if reasoning is not None:
        output.insert(0, {"id": "r", "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})
    with pytest.raises(ValueError, match="no native reasoning"):
        await infer_with_reasoning(
            client_response(output=output), "teacher", NeMoGymResponseCreateParamsNonStreaming(input=[])
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status,answer", [("incomplete", "Partial"), ("completed", "   ")])
async def test_incomplete_or_empty_final_answer_cannot_be_saved(status, answer):
    output = [
        {"id": "r", "type": "reasoning", "summary": [{"type": "summary_text", "text": "Native reasoning"}]},
        message(answer),
    ]
    with pytest.raises(ValueError, match="Incomplete inference|no assistant text"):
        await infer_with_reasoning(
            client_response(output=output, status=status), "teacher", NeMoGymResponseCreateParamsNonStreaming(input=[])
        )
