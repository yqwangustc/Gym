# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated per-role backends; credential interpolations stay unresolved on disk."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_BACKEND_CONFIG = Path("environment_servers/audiomc_sim/configs/backends.yaml")
TEXT_ROLES = ("planner", "rubric_builder", "critiquer", "user", "teacher", "assistant")


class LocalServing(BaseModel):
    """Optional managed vLLM process; omitted for externally managed endpoints."""

    model_config = ConfigDict(extra="forbid")
    entrypoint: str = "vllm.entrypoints.openai.api_server"
    interface: Literal["vllm", "nemo_audio"] = "vllm"
    python: str = "python3"
    checkpoint: str = Field(min_length=1)
    cuda_visible_devices: list[int] | None = None
    tp: int = Field(default=1, ge=1)
    dp: int = Field(default=1, ge=1)
    ep: int = Field(default=1, ge=1)
    max_num_seqs_per_dp_rank: int = Field(default=32, ge=1)
    prefix_cache: bool = True
    audio_limit: int | None = Field(default=None, ge=1)
    max_model_len: int = Field(default=16384, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, lt=1)
    reasoning_parser: str | None = None
    startup_timeout_seconds: float = Field(default=1200, gt=0)
    extra_args: list[str] = Field(default_factory=list)
    engine_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_parallelism(self) -> "LocalServing":
        if self.ep > 1 and self.ep != self.tp * self.dp:
            raise ValueError("vLLM EP size is TP * DP; ep must be 1 (disabled) or tp * dp")
        if self.cuda_visible_devices is not None:
            if len(set(self.cuda_visible_devices)) != len(self.cuda_visible_devices):
                raise ValueError("cuda_visible_devices must not contain duplicates")
            if any(index < 0 for index in self.cuda_visible_devices):
                raise ValueError("cuda_visible_devices must contain nonnegative GPU indices")
            if len(self.cuda_visible_devices) != self.tp * self.dp:
                raise ValueError("cuda_visible_devices must select exactly tp * dp GPUs")
        return self


class Endpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_base: str
    model_name: str = Field(min_length=1)
    api_key: str = "unused"
    serving: LocalServing | None = None

    @field_validator("api_base")
    @classmethod
    def valid_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("api_base must be an HTTP(S) base URL")
        return value.rstrip("/")


class ModelBackend(Endpoint):
    uses_reasoning_parser: bool = True
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None


class AssistantBackend(ModelBackend):
    checkpoint: str = ""


class SpeechBackend(Endpoint):
    request_timeout_seconds: float = Field(default=180, gt=0)
    request_attempts: int = Field(default=1, ge=1)
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
        role: {
            key: value
            if key == "api_key"
            else OmegaConf.to_container(config[role][key], resolve=True)
            if OmegaConf.is_config(config[role][key])
            else config[role][key]
            for key, value in settings.items()
        }
        for role, settings in raw.items()
    }
    backends = Backends.model_validate(resolved).model_dump()
    for settings in backends.values():
        serving = settings.get("serving")
        if serving:
            # Checkpoints can also be HF model IDs; only explicit paths are rebased.
            for key in ("entrypoint", "checkpoint"):
                value = serving[key]
                if value.startswith(("./", "../", "~", "/")):
                    candidate = Path(value).expanduser()
                    serving[key] = str((path.parent / candidate).resolve())
    if backends["assistant"]["serving"]:
        backends["assistant"]["checkpoint"] = backends["assistant"]["serving"]["checkpoint"]
    return backends
