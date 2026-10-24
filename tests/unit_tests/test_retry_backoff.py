# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backoff policy and real HTTP behavior for opt-in API retries."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError, ClientSession, web
from pydantic import ValidationError

from nemo_gym import openai_utils, server_utils
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, PermanentEndpointError
from nemo_gym.retry import RetryBackoff


def test_exponential_delay_jitter_cap_and_provider_cooldowns(monkeypatch):
    monkeypatch.setattr("nemo_gym.retry.random.uniform", lambda low, high: high / 2)
    policy = RetryBackoff()
    assert [policy.delay(i) for i in range(7)] == [1.25, 2.5, 5, 10, 20, 25, 25]
    assert policy.delay(0, {"Retry-After": "120"}) == 120.25
    assert policy.delay(0, {"retry-after-ms": "2500"}) == 2.75
    monkeypatch.setattr("nemo_gym.retry.time.time", lambda: 0)
    assert policy.delay(0, {"Retry-After": "Thu, 01 Jan 1970 00:00:10 GMT"}) == 10.25
    monkeypatch.setattr("nemo_gym.retry.random.uniform", lambda low, high: high)
    assert policy.delay(100) == 30
    monkeypatch.setattr("nemo_gym.retry.random.uniform", lambda low, high: low)
    assert policy.delay(100) == 20
    monkeypatch.setattr("nemo_gym.retry.random.uniform", lambda low, high: high / 2)
    for value in ["bad", "nan", "inf", "-1"]:
        assert policy.delay(0, {"Retry-After": value}) == 1.25


@pytest.mark.parametrize(
    "values",
    [
        {"initial_seconds": -1},
        {"multiplier": 0},
        {"max_seconds": 0},
        {"jitter_fraction": -1},
        {"max_seconds": float("inf")},
    ],
)
def test_invalid_backoff_rejected(values):
    with pytest.raises(ValidationError):
        RetryBackoff(**values)


@asynccontextmanager
async def endpoint(monkeypatch, handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    host, port = runner.addresses[0]
    async with ClientSession() as session:
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
        try:
            yield f"http://{host}:{port}/v1"
        finally:
            await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 429, 503, 529])
async def test_transient_http_retry_keeps_payload_and_native_reasoning(monkeypatch, status):
    payloads = []
    sleeps = AsyncMock()
    monkeypatch.setattr(openai_utils, "sleep", sleeps)
    monkeypatch.setattr("nemo_gym.retry.random.uniform", lambda low, high: high / 2)
    output = {"choices": [{"message": {"content": "Hello.", "reasoning_content": "Provider reasoning"}}]}

    async def handler(request):
        payloads.append(await request.json())
        if len(payloads) < 3:
            return web.json_response(
                {"error": {"message": "temporary"}},
                status=status,
                headers={"Retry-After": "5"} if len(payloads) == 1 else {},
            )
        return web.json_response(output)

    async with endpoint(monkeypatch, handler) as base:
        client = NeMoGymAsyncOpenAI(base_url=base, api_key="unused", retry_backoff=RetryBackoff())
        result = await client.create_chat_completion(model="model", messages=[{"role": "user", "content": "Hello"}])
    assert result == output
    assert len(payloads) == 3 and payloads[0] == payloads[1] == payloads[2]
    assert [call.args[0] for call in sleeps.await_args_list] == [5.25, 2.5]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code",
    [
        (400, "bad_request"),
        (401, "invalid_api_key"),
        (403, "forbidden"),
        (404, "not_found"),
        (410, "deprecated"),
        (429, "insufficient_quota"),
    ],
)
async def test_permanent_errors_do_not_retry(monkeypatch, status, code):
    calls = []
    sleeps = AsyncMock()
    monkeypatch.setattr(openai_utils, "sleep", sleeps)

    async def handler(request):
        calls.append(1)
        return web.json_response({"error": {"code": code}}, status=status)

    async with endpoint(monkeypatch, handler) as base:
        client = NeMoGymAsyncOpenAI(base_url=base, api_key="unused", retry_backoff=RetryBackoff())
        with pytest.raises(ClientResponseError):
            await client.create_chat_completion(model="model")
        if code in {"invalid_api_key", "insufficient_quota"}:
            with pytest.raises(PermanentEndpointError):
                await client.create_chat_completion(model="model")
    assert len(calls) == 1
    sleeps.assert_not_awaited()


@pytest.mark.asyncio
async def test_exhaustion_is_bounded_and_transport_cannot_retry_inside(monkeypatch):
    calls = []
    sleeps = AsyncMock()
    monkeypatch.setattr(openai_utils, "sleep", sleeps)

    async def request(**kwargs):
        calls.append(kwargs)
        raise TimeoutError()

    monkeypatch.setattr(openai_utils, "request", request)
    client = NeMoGymAsyncOpenAI(
        base_url="https://unused.invalid", api_key="unused", max_http_attempts=4, retry_backoff=RetryBackoff()
    )
    with pytest.raises(TimeoutError):
        await client._request(method="POST", url="https://unused.invalid/v1/chat/completions")
    assert len(calls) == 4 and sleeps.await_count == 3
    assert all(call["_max_num_tries"] == 1 for call in calls)
    client.max_num_tries = 1
    sleeps.reset_mock()
    with pytest.raises(TimeoutError):
        await client._request(method="POST", url="https://unused.invalid/v1/chat/completions")
    sleeps.assert_not_awaited()
    assert len(calls) == 5


@pytest.mark.asyncio
async def test_truncated_success_body_is_retried(monkeypatch):
    calls = []
    sleeps = AsyncMock()
    monkeypatch.setattr(openai_utils, "sleep", sleeps)

    async def handler(request):
        calls.append(1)
        if len(calls) == 1:
            response = web.StreamResponse(headers={"Content-Length": "1000"})
            await response.prepare(request)
            await response.write(b"{}")
            request.transport.close()
            return response
        return web.json_response({"choices": []})

    async with endpoint(monkeypatch, handler) as base:
        client = NeMoGymAsyncOpenAI(
            base_url=base, api_key="unused", request_timeout_seconds=1, retry_backoff=RetryBackoff()
        )
        assert await client.create_chat_completion(model="model") == {"choices": []}
    assert len(calls) == 2 and sleeps.await_count == 1


@pytest.mark.asyncio
async def test_cancellation_during_backoff_stops_retries(monkeypatch):
    entered = asyncio.Event()
    calls = []

    async def request(**kwargs):
        calls.append(1)
        raise TimeoutError()

    async def sleep(_):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(openai_utils, "request", request)
    monkeypatch.setattr(openai_utils, "sleep", sleep)
    client = NeMoGymAsyncOpenAI(base_url="https://unused.invalid", api_key="unused", retry_backoff=RetryBackoff())
    task = asyncio.create_task(client._request(method="POST", url="https://unused.invalid"))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) == 1
