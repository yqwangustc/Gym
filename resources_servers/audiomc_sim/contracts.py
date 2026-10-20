# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire and saved-data contracts for AudioMC simulation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, model_validator
from pydantic.dataclasses import dataclass

from nemo_gym.base_resources_server import ResourcesSeedSessionResponse
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskInput(StrictModel):
    persona_uuid: str
    persona: str
    axis_id: str
    evaluation: dict[str, str | list[str]]
    topic: str
    subtopic: str
    subtopic_description: str
    seed: int
    tts_voice: str | None = Field(default=None, min_length=1, pattern=r"\S")
    max_turns: int = Field(default=6, ge=1)
    max_teacher_trials: int = Field(default=3, ge=1)
    assistant_instructions: str
    developer_instructions: str | None = None
    source_hashes: dict[str, str] = Field(default_factory=dict)


class Message(StrictModel):
    turn_id: str
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1)


class AudioRef(StrictModel):
    archive: str
    member: str
    sha256: str
    voice: str
    model: str
    sample_rate: int
    duration_seconds: float


class Rubric(StrictModel):
    rubric_id: str
    requirement: str = Field(min_length=1)
    category: Literal["target_axis", "general_quality"]
    supporting_turn_ids: list[str]
    instruction_evidence: str | None = None


class BuilderRubric(StrictModel):
    index: int = Field(ge=0, strict=True)
    requirement: str = Field(min_length=1)
    category: Literal["target_axis", "general_quality"]
    supporting_turn_ids: list[str]
    instruction_evidence: str | None = None


class BuilderOutput(StrictModel):
    rubrics: list[BuilderRubric]


class Grade(StrictModel):
    rubric_id: str
    passed: StrictBool
    explanation: str = Field(min_length=1)


class Judgment(StrictModel):
    grades: list[Grade]
    passed: StrictBool | None

    @model_validator(mode="after")
    def consistent_verdict(self) -> "Judgment":
        if not self.grades:
            if self.passed is not None:
                raise ValueError("Empty grades must have passed=null")
        elif self.passed is None or self.passed != all(g.passed for g in self.grades):
            raise ValueError("Aggregate verdict disagrees with rubric grades")
        return self


class StepRequest(StrictModel):
    resources_session_id: str
    turn_index: int = Field(ge=0)
    history: list[Message]


class TraceRequest(StrictModel):
    resources_session_id: str
    input_prefix: list[Message]
    invocations: list[dict[str, JsonValue]]


class SynthesisRequest(StrictModel):
    resources_session_id: str
    turn_index: int = Field(ge=0)
    text: str = Field(min_length=1)


class GradeRequest(StepRequest):
    candidate: str = Field(min_length=1)


class TurnEvaluation(StrictModel):
    turn_index: int
    user_turn_id: str
    user_text: str
    audio: AudioRef
    rubrics: list[Rubric]
    policy_response: str
    judgment: Judgment


class Candidate(StrictModel):
    text: str
    judgment: Judgment


Outcome = Literal[
    "accepted_correction", "teacher_correction_exhausted", "max_turn_reached_without_failing", "ungradable"
]


class FinalizeRequest(StrictModel):
    resources_session_id: str
    outcome: Outcome
    input_prefix: list[Message]
    turn_evaluations: list[TurnEvaluation]
    teacher_attempts: list[Candidate] = Field(default_factory=list)
    invocations: list[dict[str, JsonValue]] = Field(default_factory=list)


class EpisodeResult(StrictModel):
    outcome: Outcome
    artifact_path: str
    turns: int
    teacher_trials: int
    axis_id: str


class EpisodeRequest(BaseEpisodeRequest[TaskInput]):
    pass


class EpisodeResponse(BaseEpisodeResponse[EpisodeResult]):
    pass


class SeedResponse(ResourcesSeedSessionResponse):
    pass


@dataclass(config=ConfigDict(extra="forbid"))
class SavedTrainingExample:
    """Accepted teacher replacement with all turn-level evaluations and provenance."""

    example_id: str
    rollout_id: str
    input_prefix: dict[str, JsonValue]
    turn_evaluations: list[TurnEvaluation]
    corrected_response: Candidate
    rejected_teacher_attempts: list[Candidate]
    metadata: dict[str, JsonValue]
    schema_version: str = "1.2"
    responses_create_params: dict[str, JsonValue] | None = None
    supervised_target: str | None = None
