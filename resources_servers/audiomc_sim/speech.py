# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded speech-request retries with separate timing evidence for each attempt."""

import asyncio
import logging

from aiohttp import ClientError
from pydantic import JsonValue

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
) -> bytes:
    """Retry timeouts/connection errors/429/5xx; propagate cancellation and permanent errors."""
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
                        _max_num_tries=1 if attempts > 1 else 3,
                    )
                    await raise_for_status(response)
                    return await response.read()
        except (TimeoutError, ClientError) as error:
            status = getattr(error, "status", 0)
            retryable = not status or status == 429 or 500 <= status < 600
            if not retryable or attempt + 1 == attempts:
                raise
            logger.warning(
                "TTS attempt %d/%d failed: %s status=%s; retrying", attempt + 1, attempts, type(error).__name__, status
            )
            await asyncio.sleep(min(2**attempt, 10))
        finally:
            if response is not None:
                response.release()
    raise AssertionError("At least one TTS attempt is required")
