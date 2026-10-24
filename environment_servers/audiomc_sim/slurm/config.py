# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task settings shared by login-node preflight and the container controller."""

from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from environment_servers.audiomc_sim.collect import load_env


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Repositories(SettingsModel):
    nemo_gym: Path


class Resources(SettingsModel):
    persona: Path
    eval_axis: Path
    topics: Path
    assistant_system_prompt: Path


class SimulationTasks(SettingsModel):
    target_trajectory: int = Field(ge=1)
    seed: int = 44
    max_turns: int = Field(default=6, ge=1)
    max_teacher_trials: int = Field(default=5, ge=1)
    keep_generation_until_max_turn_reached: bool = False
    cleanup_finished_trajectories: bool = True
    concurrency: int = Field(default=64, ge=1)
    max_rollouts: int | None = Field(default=None, ge=1)
    export_every_min: float = Field(default=230, gt=0)
    export_reserve_min: float = Field(default=10, gt=0)
    max_audio_files: int = Field(default=50_000, ge=1)
    base_port: int = Field(default=18240, ge=1024, le=65520)

    @model_validator(mode="after")
    def attempt_budget(self) -> "SimulationTasks":
        if self.max_rollouts is None:
            self.max_rollouts = self.target_trajectory * 2
        if self.max_rollouts < self.target_trajectory:
            raise ValueError("max_rollouts must be >= target_trajectory")
        return self


class SlurmSettings(SettingsModel):
    partition: str | None = None
    account: str | None = None
    time: str = "04:00:00"
    gpus: int = Field(default=8, ge=0)
    omit_gpus_per_node: bool = False
    cpus_per_task: int = Field(default=16, ge=1)
    exclusive: bool = True
    job_name: str = "audiomc-sim"
    container_mounts: list[str] = Field(default_factory=list)
    python: str = "python3"


class TaskSettings(SettingsModel):
    repos: Repositories
    container: str = Field(min_length=1)
    mc_resources: Resources
    environment: Path | None = None
    simulation_tasks: SimulationTasks
    backend: Path
    output: Path
    slurm: SlurmSettings = Field(default_factory=SlurmSettings)


def load_settings(path: Path) -> TaskSettings:
    """Load environment first, then resolve task YAML; paths are relative to its file."""
    path = path.expanduser().resolve()
    config = OmegaConf.load(path)

    def absolute(value: str | Path) -> Path:
        return (path.parent / Path(value).expanduser()).resolve()

    if config.get("environment"):
        load_env(absolute(config.environment))
    settings = TaskSettings.model_validate(OmegaConf.to_container(config, resolve=True))
    settings.repos.nemo_gym = absolute(settings.repos.nemo_gym)
    for name in Resources.model_fields:
        setattr(settings.mc_resources, name, absolute(getattr(settings.mc_resources, name)))
    settings.backend = absolute(settings.backend)
    settings.output = absolute(settings.output)
    if settings.environment:
        settings.environment = absolute(settings.environment)
    # A .sqsh path is a file; registry image names retain their original syntax.
    if settings.container.endswith(".sqsh") or settings.container.startswith(("./", "../", "/", "~")):
        settings.container = str(absolute(settings.container))
    return settings
