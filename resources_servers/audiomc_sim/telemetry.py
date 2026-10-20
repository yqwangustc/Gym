# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-session timing context; concurrent tasks never share the active call record."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter, time

from pydantic import JsonValue


CURRENT_CALL: ContextVar[dict[str, JsonValue] | None] = ContextVar("audiomc_current_call", default=None)


@contextmanager
def trace_call(events: list[dict[str, JsonValue]], role: str, **tags: JsonValue) -> Iterator[dict[str, JsonValue]]:
    """Record wall latency, errors and optional usage without retaining credentials."""
    record: dict[str, JsonValue] = {"kind": "timing", "role": role, "started_at": time(), **tags}
    token = CURRENT_CALL.set(record)
    started = perf_counter()
    try:
        yield record
        record["success"] = True
    except BaseException as error:
        record["success"] = False
        record["error_type"] = type(error).__name__
        record["http_status"] = getattr(error, "status", None)
        raise
    finally:
        record["elapsed_seconds"] = perf_counter() - started
        events.append(record)
        CURRENT_CALL.reset(token)
