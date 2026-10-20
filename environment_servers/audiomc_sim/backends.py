# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated per-role backends; credential interpolations stay unresolved on disk."""

from pathlib import Path
from urllib.parse import urlsplit

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_BACKEND_CONFIG = Path("environment_servers/audiomc_sim/configs/backends.yaml")
TEXT_ROLES = ("planner", "rubric_builder", "critiquer", "user", "teacher", "assistant")


class Endpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_base: str
    model_name: str = Field(min_length=1)
    api_key: str = "unused"

    @field_validator("api_base")
    @classmethod
    def valid_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("api_base must be an HTTP(S) base URL")
        return value.rstrip("/")


class ModelBackend(Endpoint):
    uses_reasoning_parser: bool = True


class AssistantBackend(ModelBackend):
    checkpoint: str = ""


class SpeechBackend(Endpoint):
    voices: list[str] = Field(default_factory=lambda: ["alloy"], min_length=1)

    @model_validator(mode="before")
    @classmethod
    def legacy_voice(cls, values: object) -> object:
        if isinstance(values, dict) and "voice" in values:
            values = dict(values)
            if "voices" in values:
                raise ValueError("Specify tts.voices or legacy tts.voice, not both")
            values["voices"] = [values.pop("voice")]
        return values

    @field_validator("voices")
    @classmethod
    def nonblank_voices(cls, voices: list[str]) -> list[str]:
        if any(not voice.strip() for voice in voices):
            raise ValueError("TTS voice names must not be blank")
        return voices


class Backends(BaseModel):
    model_config = ConfigDict(extra="forbid")
    planner: ModelBackend
    rubric_builder: ModelBackend
    critiquer: ModelBackend
    user: ModelBackend
    teacher: ModelBackend
    assistant: AssistantBackend
    tts: SpeechBackend


def load_backends(path: Path = DEFAULT_BACKEND_CONFIG) -> dict[str, dict[str, object]]:
    """Load per-role YAML, resolve noncredential fields and validate the settings."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(raw, dict) or any(not isinstance(value, dict) for value in raw.values()):
        raise ValueError("Backend YAML must map role names to settings dictionaries")
    config = OmegaConf.create(raw)
    resolved = {
        role: {key: value if key == "api_key" else config[role][key] for key, value in settings.items()}
        for role, settings in raw.items()
    }
    return Backends.model_validate(resolved).model_dump()
