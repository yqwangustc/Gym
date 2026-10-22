# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retry only recoverable speech errors, preserving bounded attempts and cancellation."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientResponseError

from resources_servers.audiomc_sim.speech import synthesize_bytes


def options():
    return dict(
        endpoint="http://tts.invalid",
        api_key="unused",
        payload={"input": "Hello"},
        timeout_seconds=0.02,
        attempts=3,
        events=[],
        turn_index=0,
    )


@pytest.mark.asyncio
async def test_timeout_retries_and_records_each_attempt_without_changing_payload(monkeypatch):
    response = Mock(ok=True, read=AsyncMock(return_value=b"audio"))
    request = AsyncMock(side_effect=[TimeoutError(), response])
    monkeypatch.setattr("resources_servers.audiomc_sim.speech.http_request", request)
    monkeypatch.setattr("resources_servers.audiomc_sim.speech.asyncio.sleep", AsyncMock())
    values = options()
    assert await synthesize_bytes(**values) == b"audio"
    assert [e["success"] for e in values["events"]] == [False, True]
    assert [e["request_attempt"] for e in values["events"]] == [0, 1]
    assert request.await_args_list[0].kwargs["json"] == request.await_args_list[1].kwargs["json"]
    assert all(call.kwargs["_max_num_tries"] == 1 for call in request.await_args_list)
    response.release.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected_calls", [(400, 1), (429, 3), (503, 3)])
async def test_permanent_errors_stop_and_recoverable_errors_have_bounded_attempts(monkeypatch, status, expected_calls):
    request = AsyncMock(side_effect=ClientResponseError(None, (), status=status))
    monkeypatch.setattr("resources_servers.audiomc_sim.speech.http_request", request)
    monkeypatch.setattr("resources_servers.audiomc_sim.speech.asyncio.sleep", AsyncMock())
    with pytest.raises(ClientResponseError):
        await synthesize_bytes(**options())
    assert request.await_count == expected_calls


@pytest.mark.asyncio
async def test_cancellation_never_retries(monkeypatch):
    request = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr("resources_servers.audiomc_sim.speech.http_request", request)
    with pytest.raises(asyncio.CancelledError):
        await synthesize_bytes(**options())
    assert request.await_count == 1
