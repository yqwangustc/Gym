# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capped exponential backoff with jitter and provider-directed cooldowns."""

import math
import random
import time
from collections.abc import Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime

from pydantic import BaseModel, ConfigDict, Field


class RetryBackoff(BaseModel):
    """Delay after each failed attempt; Retry-After can exceed the backoff cap.

    Like openai_model's upstream retry policy, jitter is additive and uniform.
    This model describes delays only; callers own attempt limits and timeouts.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    initial_seconds: float = Field(default=1.0, ge=0)
    multiplier: float = Field(default=2.0, ge=1)
    max_seconds: float = Field(default=30.0, gt=0)
    jitter_fraction: float = Field(default=0.5, ge=0)

    def delay(self, retry_index: int, headers: Mapping[str, str] | None = None) -> float:
        """Return a jittered delay; index zero follows the first failed attempt."""
        if retry_index < 0:
            raise ValueError("retry_index must be nonnegative")
        base = min(self.initial_seconds, self.max_seconds)
        for _ in range(retry_index):
            base = min(base * self.multiplier, self.max_seconds)
            if base == self.max_seconds:
                break
        # Bound the jitter range instead of clipping the sample: clipping would
        # synchronize all retries once the exponential delay reaches the cap.
        base = min(base, self.max_seconds / (1 + self.jitter_fraction))
        jitter = random.uniform(0, base * self.jitter_fraction)
        retry_after = _retry_after_seconds(headers)
        if retry_after >= base and retry_after > 0:
            return retry_after + jitter
        return min(base + jitter, self.max_seconds)


def _retry_after_seconds(headers: Mapping[str, str] | None) -> float:
    if not headers:
        return 0
    values = {key.lower(): value for key, value in headers.items()}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1)):
        raw = values.get(name)
        if raw is None:
            continue
        try:
            seconds = float(raw) * scale
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
        except (ValueError, OverflowError):
            pass
        if name == "retry-after":
            try:
                date = parsedate_to_datetime(raw)
                seconds = date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp() - time.time()
                if math.isfinite(seconds):
                    return max(0, seconds)
            except (ValueError, TypeError, OverflowError):
                pass
    return 0
