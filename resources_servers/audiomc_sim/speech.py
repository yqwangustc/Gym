# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded speech-request retries with separate timing evidence for each attempt."""

import asyncio
import logging

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from pydantic import JsonValue

from nemo_gym.openai_utils import _error_body_is_permanent_quota
from nemo_gym.retry import RetryBackoff
from nemo_gym.server_utils import raise_for_status
from nemo_gym.server_utils import request as http_request
from resources_servers.audiomc_sim.telemetry import trace_call


logger = logging.getLogger(__name__)


async def synthesize_bytes(
    *,
    endpoint: str,
    api_key: str,
    payload: dict[str, str],
    timeout_seconds: float,
    attempts: int,
    events: list[dict[str, JsonValue]],
    turn_index: int,
    retry_backoff: RetryBackoff | None = None,
) -> bytes:
    """Retry timeouts/connection errors/429/5xx; propagate cancellation and permanent errors."""
    retry_backoff = retry_backoff or RetryBackoff()
    for attempt in range(attempts):
        response = None
        try:
            with trace_call(events, "tts", turn_index=turn_index, request_attempt=attempt):
                async with asyncio.timeout(timeout_seconds):
                    response = await http_request(
                        method="POST",
                        url=endpoint,
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=payload,
                        _max_num_tries=1,
                    )
                    await raise_for_status(response)
                    return await response.read()
        except (TimeoutError, ClientConnectionError, ClientPayloadError, ClientResponseError) as error:
            status = getattr(error, "status", 0)
            retryable = not status or status in (408, 429) or 500 <= status < 600
            if status == 429 and _error_body_is_permanent_quota(getattr(error, "response_content", b"")):
                retryable = False
            if not retryable or attempt + 1 == attempts:
                raise
            if response is not None:
                response.release()
                response = None
            delay = retry_backoff.delay(attempt, getattr(error, "headers", None))
            logger.warning(
                "TTS attempt %d/%d failed: %s status=%s; retrying in %.2fs",
                attempt + 1,
                attempts,
                type(error).__name__,
                status,
                delay,
            )
            await asyncio.sleep(delay)
        finally:
            if response is not None:
                response.release()
    raise AssertionError("At least one TTS attempt is required")
