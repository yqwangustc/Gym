# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row schema for Tau-Voice Simple: which Tau episode to run and with which voice."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Domain = Literal["airline", "retail", "telecom"]
DOMAINS: tuple[Domain, ...] = ("airline", "retail", "telecom")


class TauVoiceSimpleMetadata(BaseModel):
    """One episode; the task itself (scenario, criteria) is loaded from Tau's data."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1, description="Unique episode id, e.g. 'retail__40__trial_0'.")
    domain: str = Field(min_length=1)
    task_pack: Literal["tau", "eva", "mivas"] = "tau"
    task_id: str = Field(min_length=1)
    task_split: str = "base"
    trial: int = Field(default=0, ge=0)
    seed: int = Field(description="Seed for Tau's orchestrator and user simulator.")
    tts_voice: str | None = Field(default=None, description="Fixed TTS voice for the whole session (s2t).")


TaskData = TauVoiceSimpleMetadata
