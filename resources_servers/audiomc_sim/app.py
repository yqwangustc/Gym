# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources, grading and durable artifacts for generated AudioMC conversations."""

import asyncio
import hashlib
import io
import json
import random
import tarfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter, ValidationError, field_validator, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.server_utils import raise_for_status
from nemo_gym.server_utils import request as http_request
from resources_servers.audiomc_sim.common import Prompts, digest, infer, parse_json, prompt_values
from resources_servers.audiomc_sim.contracts import (
    AudioRef,
    BuilderOutput,
    EpisodeResult,
    FinalizeRequest,
    GradeRequest,
    Judgment,
    Rubric,
    SavedTrainingExample,
    SeedResponse,
    StepRequest,
    SynthesisRequest,
    TaskInput,
    TraceRequest,
)
from resources_servers.audiomc_sim.telemetry import trace_call


class AudioMCSimConfig(BaseResourcesServerConfig):
    helper_model: ModelServerRef | None = None
    rubric_builder_model: ModelServerRef | None = None
    critiquer_model: ModelServerRef | None = None
    artifact_root: str
    prompts_dir: str = "resources_servers/audiomc_sim/prompts"
    tts_endpoint: str
    tts_api_key: str
    tts_model: str = "openai/openai/gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    tts_voices: list[str] | None = Field(default=None, min_length=1)
    json_attempts: int = Field(default=2, ge=1)
    session_ttl_seconds: float = Field(default=7200, gt=0)

    @field_validator("tts_voices")
    @classmethod
    def nonblank_voices(cls, voices: list[str] | None) -> list[str] | None:
        if voices is not None and any(not voice.strip() for voice in voices):
            raise ValueError("TTS voice names must not be blank")
        return voices

    def model_for(self, role: str) -> ModelServerRef:
        model = {"rubric_builder": self.rubric_builder_model, "critiquer": self.critiquer_model}[role]
        model = model or self.helper_model
        if model is None:
            raise ValueError(f"Configure {role}_model or legacy helper_model")
        return model

    @model_validator(mode="after")
    def validate_role_models(self) -> "AudioMCSimConfig":
        for role in ("rubric_builder", "critiquer"):
            self.model_for(role)
        return self


@dataclass
class Session:
    seed_digest: str
    task: TaskInput
    episode: str
    episode_identity: dict[str, str | int]
    task_identity: dict[str, str]
    voice: str
    created: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cache: dict[str, tuple[str, EpisodeResult]] = field(default_factory=dict)
    prefixes: dict[int, list[dict[str, str]]] = field(default_factory=dict)
    rubrics: dict[int, list[Rubric]] = field(default_factory=dict)
    audio: dict[int, tuple[str, AudioRef]] = field(default_factory=dict)
    judgments: dict[tuple[int, str], Judgment] = field(default_factory=dict)
    invocations: list[dict[str, object]] = field(default_factory=list)
    environment_trace: dict[str, object] = field(default_factory=dict)


class AudioMCSimResources(SimpleResourcesServer):
    config: AudioMCSimConfig
    _sessions: dict[str, Session] = PrivateAttr(default_factory=dict)
    _closed: dict[str, float] = PrivateAttr(default_factory=dict)
    _prompts: Prompts = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("AudioMC Resources requires one worker for session isolation")
        self._prompts = Prompts(Path(self.config.prompts_dir), ("rubric_builder", "critiquer"))
        Path(self.config.artifact_root).mkdir(parents=True, exist_ok=True)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/build_rubrics")(self.build_rubrics)
        app.post("/synthesize")(self.synthesize)
        app.post("/grade")(self.grade)
        app.post("/checkpoint")(self.checkpoint)
        return app

    async def checkpoint(self, request: Request, body: TraceRequest) -> dict[str, bool]:
        session = self._session(request, body.resources_session_id)
        async with session.lock:
            session.environment_trace = body.model_dump(mode="json")
        return {"saved": True}

    def _session(self, request: Request, session_id: str) -> Session:
        if request.session.get("audiomc_session") != session_id:
            raise HTTPException(409, "Resources session cookie does not match")
        session = self._sessions.get(session_id)
        if session is None or time.monotonic() - session.created > self.config.session_ttl_seconds:
            self._sessions.pop(session_id, None)
            self._closed[session_id] = time.monotonic()
            raise HTTPException(409, "Unknown or expired Resources session")
        return session

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> SeedResponse:
        now = time.monotonic()
        self._closed = {
            key: when for key, when in self._closed.items() if now - when < self.config.session_ttl_seconds
        }
        if body.resources_session_id in self._closed:
            raise HTTPException(409, "Resources session already closed")
        task = TaskInput.model_validate(body.task_data)
        key = digest(body.model_dump(mode="json"))
        existing = self._sessions.get(body.resources_session_id)
        if existing is not None and existing.seed_digest != key:
            raise HTTPException(409, "Session already bound to another task")
        if existing is None:
            # New tasks already contain their speaker. Legacy tasks use a stable task-level fallback.
            voice_seed = digest({"seed": task.seed, "task_id": body.task_id.model_dump()})
            voice = task.tts_voice or random.Random(voice_seed).choice(
                self.config.tts_voices or [self.config.tts_voice]
            )
            self._sessions[body.resources_session_id] = Session(
                seed_digest=key,
                task=task,
                episode=body.episode_id.capture_key,
                episode_identity=body.episode_id.model_dump(),
                task_identity=body.task_id.model_dump(),
                voice=voice,
            )
        request.session["audiomc_session"] = body.resources_session_id
        return SeedResponse(resources_session_id=body.resources_session_id)

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        current = self._sessions.get(body.resources_session_id)
        if current is not None and current.episode_identity != body.episode_id.model_dump():
            raise HTTPException(409, "Close episode does not match seeded identity")
        if current is not None and "final" not in current.cache:
            partial = {
                "episode_id": current.episode_identity,
                "task_id": current.task_identity,
                "task": current.task.model_dump(mode="json"),
                "tts_voice": current.voice,
                "environment_trace": current.environment_trace,
                "prefixes": current.prefixes,
                "rubrics": {str(i): [r.model_dump() for r in rs] for i, rs in current.rubrics.items()},
                "audio": [a.model_dump() for _, a in current.audio.values()],
                "judgments": [
                    {"turn_index": i, "candidate_sha256": h, "judgment": g.model_dump()}
                    for (i, h), g in current.judgments.items()
                ],
                "resource_invocations": current.invocations,
            }
            target = Path(self.config.artifact_root) / "partial" / f"{current.episode}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(partial, ensure_ascii=False, indent=2))
            temporary.replace(target)
        self._sessions.pop(body.resources_session_id, None)
        self._closed[body.resources_session_id] = time.monotonic()
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def _json_call[T: BaseModel](
        self, session: Session, role: str, values: dict[str, object], schema: type[T]
    ) -> T:
        errors = []
        model = self.config.model_for(role).name
        for _ in range(self.config.json_attempts):
            with trace_call(session.invocations, role, model_server=model, turn_index=values["turn_index"]):
                text = await infer(self.server_client, model, self._prompts.request(role, values))
            session.invocations.append({"role": role, "text": text})
            try:
                return schema.model_validate(parse_json(text))
            except (ValueError, ValidationError) as error:
                errors.append(str(error)[:300])
        raise HTTPException(422, f"{role} returned invalid JSON after retries: {errors}")

    async def build_rubrics(self, request: Request, body: StepRequest) -> list[Rubric]:
        session = self._session(request, body.resources_session_id)
        async with session.lock:
            prefix = [m.model_dump() for m in body.history]
            expected_roles = ["user" if i % 2 == 0 else "assistant" for i in range(2 * body.turn_index + 1)]
            if [m.role for m in body.history] != expected_roles or body.turn_index >= session.task.max_turns:
                raise HTTPException(422, "Invalid conversation prefix boundary")
            ids = [m.turn_id for m in body.history]
            if len(ids) != len(set(ids)):
                raise HTTPException(422, "Duplicate conversation IDs")
            if body.turn_index in session.prefixes:
                if session.prefixes[body.turn_index] != prefix:
                    raise HTTPException(409, "Cannot change a frozen prefix")
                return session.rubrics[body.turn_index]
            if body.turn_index:
                previous = session.prefixes.get(body.turn_index - 1)
                if previous != prefix[:-2]:
                    raise HTTPException(409, "History changed between exchanges")
                prior = body.history[-2].text
                prior_grade = session.judgments.get((body.turn_index - 1, digest(prior)))
                if prior_grade is None or prior_grade.passed is not True:
                    raise HTTPException(409, "Previous response did not pass")
            output = await self._json_call(
                session, "rubric_builder", prompt_values(session.task, body.history, body.turn_index), BuilderOutput
            )
            if [r.index for r in output.rubrics] != list(range(len(output.rubrics))):
                raise HTTPException(422, "Rubric indices must be ordered and contiguous from zero")
            result = []
            for r in output.rubrics:
                if not set(r.supporting_turn_ids) <= set(ids):
                    raise HTTPException(422, "Rubric references unknown or future turns")
                result.append(
                    Rubric(
                        rubric_id=f"{session.episode}_t{body.turn_index}_r{r.index}", **r.model_dump(exclude={"index"})
                    )
                )
            session.prefixes[body.turn_index] = prefix
            session.rubrics[body.turn_index] = result
            return result

    async def synthesize(self, request: Request, body: SynthesisRequest) -> AudioRef:
        session = self._session(request, body.resources_session_id)
        async with session.lock:
            if body.turn_index in session.audio:
                text, audio = session.audio[body.turn_index]
                if text != body.text:
                    raise HTTPException(409, "Cannot replace audio for an existing turn")
                return audio
            if not 0 <= body.turn_index < session.task.max_turns:
                raise HTTPException(422, "Turn exceeds configured limit")
            prefix = session.prefixes.get(body.turn_index)
            if not prefix or prefix[-1]["text"] != body.text:
                raise HTTPException(409, "Speech must match the frozen latest user utterance")
            with trace_call(session.invocations, "tts", turn_index=body.turn_index):
                async with asyncio.timeout(180):
                    response = await http_request(
                        method="POST",
                        url=self.config.tts_endpoint,
                        headers={"Authorization": f"Bearer {self.config.tts_api_key}"},
                        json={
                            "model": self.config.tts_model,
                            "input": body.text,
                            "voice": session.voice,
                            "response_format": "wav",
                            "instructions": "Speak naturally. Preserve every supplied word, including corrections. Do not paraphrase.",
                        },
                        _max_num_tries=3,
                    )
                    await raise_for_status(response)
                    audio_bytes = await response.read()
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
                rate = wav.getframerate()
                frames = len(wav.readframes(wav.getnframes())) // (wav.getnchannels() * wav.getsampwidth())
                if not frames or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError("Expected nonempty mono PCM16 WAV")
            member = f"{session.episode}/user_turn_{body.turn_index:04d}.wav"
            archive = f"audio/{session.episode}/turn_{body.turn_index:04d}.tar"
            target = Path(self.config.artifact_root) / archive
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            with tarfile.open(temporary, "w") as tar:
                info = tarfile.TarInfo(member)
                info.size = len(audio_bytes)
                tar.addfile(info, io.BytesIO(audio_bytes))
            temporary.replace(target)
            result = AudioRef(
                archive=archive,
                member=member,
                sha256=hashlib.sha256(audio_bytes).hexdigest(),
                voice=session.voice,
                model=self.config.tts_model,
                sample_rate=rate,
                duration_seconds=frames / rate,
            )
            session.audio[body.turn_index] = (body.text, result)
            return result

    async def grade(self, request: Request, body: GradeRequest) -> Judgment:
        session = self._session(request, body.resources_session_id)
        async with session.lock:
            if session.prefixes.get(body.turn_index) != [m.model_dump() for m in body.history]:
                raise HTTPException(409, "Grading prefix is not the frozen prefix")
            key = (body.turn_index, digest(body.candidate))
            if key in session.judgments:
                return session.judgments[key]
            rubrics = session.rubrics[body.turn_index]
            if not rubrics:
                return Judgment(grades=[], passed=None)
            values = prompt_values(session.task, body.history, body.turn_index) | {
                "rubrics": json.dumps([r.model_dump() for r in rubrics]),
                "candidate_response": body.candidate,
            }
            result = await self._json_call(session, "critiquer", values, Judgment)
            if result.grades and [g.rubric_id for g in result.grades] != [r.rubric_id for r in rubrics]:
                raise HTTPException(422, "Critiquer must grade every frozen rubric exactly once in order")
            session.judgments[key] = result
            return result

    async def verify(self, request: Request, body: FinalizeRequest) -> EpisodeResult:
        session = self._session(request, body.resources_session_id)
        async with session.lock:
            signature = digest(body.model_dump(mode="json"))
            if "final" in session.cache:
                previous, result = session.cache["final"]
                if previous != signature:
                    raise HTTPException(409, "Episode already finalized with another result")
                return result
            turns = body.turn_evaluations
            if [t.turn_index for t in turns] != list(range(len(turns))):
                raise HTTPException(422, "Every completed exchange must be evaluated in order")
            for turn in turns:
                i = turn.turn_index
                prefix = session.prefixes.get(i)
                if not prefix or prefix[-1]["turn_id"] != turn.user_turn_id:
                    raise HTTPException(422, "Turn ID differs from frozen history")
                if turn.rubrics != session.rubrics.get(i) or session.audio.get(i) != (turn.user_text, turn.audio):
                    raise HTTPException(422, "Turn evidence differs from stored Resources records")
                if turn.judgment != session.judgments.get((i, digest(turn.policy_response))):
                    raise HTTPException(422, "Turn grades differ from stored candidate grades")
                if i < len(turns) - 1:
                    next_prefix = session.prefixes.get(i + 1, [])
                    if not next_prefix or next_prefix[-2]["text"] != turn.policy_response:
                        raise HTTPException(422, "Intermediate response differs from conversation history")
                if i < len(turns) - 1 and turn.judgment.passed is not True:
                    raise HTTPException(422, "Conversation must stop at first failure")
            if len(turns) > session.task.max_turns or len(body.teacher_attempts) > session.task.max_teacher_trials:
                raise HTTPException(422, "Configured episode limits exceeded")
            if body.outcome == "ungradable" and len(session.prefixes) == len(turns) + 1:
                if (
                    session.rubrics.get(len(turns))
                    or [m.model_dump() for m in body.input_prefix] != session.prefixes[len(turns)]
                ):
                    raise HTTPException(422, "Ungradable setup must have empty frozen rubrics")
            elif turns:
                last = turns[-1]
                if [m.model_dump() for m in body.input_prefix] != session.prefixes[last.turn_index]:
                    raise HTTPException(422, "Final prefix differs from frozen history")
                for candidate in body.teacher_attempts:
                    if candidate.judgment != session.judgments.get((last.turn_index, digest(candidate.text))):
                        raise HTTPException(422, "Unverified teacher candidate")
            if body.teacher_attempts and (not turns or turns[-1].judgment.passed is not False):
                raise HTTPException(422, "Teacher candidates require a failed original response")
            if body.outcome == "accepted_correction":
                if not turns or turns[-1].judgment.passed is not False or not body.teacher_attempts:
                    raise HTTPException(422, "Accepted example requires a failed original and a teacher correction")
                if body.teacher_attempts[-1].judgment.passed is not True:
                    raise HTTPException(422, "Correction must pass all rubrics")
            if any(c.judgment.passed is not False for c in body.teacher_attempts[:-1]):
                raise HTTPException(422, "Teacher loop must stop at first acceptance or ungradable candidate")
            if body.outcome == "max_turn_reached_without_failing":
                if len(turns) != session.task.max_turns or any(t.judgment.passed is not True for t in turns):
                    raise HTTPException(422, "Turn-limit result requires all allowed turns to pass")
                if body.teacher_attempts:
                    raise HTTPException(422, "Successful rollout cannot have teacher attempts")
            if body.outcome == "teacher_correction_exhausted":
                if (
                    not turns
                    or turns[-1].judgment.passed is not False
                    or len(body.teacher_attempts) != session.task.max_teacher_trials
                    or any(c.judgment.passed is not False for c in body.teacher_attempts)
                ):
                    raise HTTPException(422, "Exhaustion requires all allowed teacher candidates to fail")
            if body.outcome == "ungradable":
                empty_builder = len(session.prefixes) == len(turns) + 1 and not session.rubrics.get(len(turns))
                null_grade = bool(turns) and turns[-1].judgment.passed is None
                null_teacher = bool(body.teacher_attempts) and body.teacher_attempts[-1].judgment.passed is None
                if not (empty_builder or null_grade or null_teacher):
                    raise HTTPException(422, "Ungradable outcome requires an ungradable builder or grade")
            metadata = {
                "task": session.task.model_dump(),
                "episode_id": session.episode_identity,
                "task_id": session.task_identity,
                "resource_prompt_hashes": self._prompts.hashes(),
                "tts_voice": session.voice,
            }
            record = body.model_dump(mode="json") | {"metadata": metadata, "resource_invocations": session.invocations}
            if body.outcome == "accepted_correction":
                example = SavedTrainingExample(
                    example_id=session.episode,
                    rollout_id=session.episode_identity["rollout_id"],
                    input_prefix={
                        "system_prompt": session.task.assistant_instructions,
                        "developer_prompt": session.task.developer_instructions,
                        "turns": [m.model_dump() for m in body.input_prefix],
                    },
                    turn_evaluations=turns,
                    corrected_response=body.teacher_attempts[-1],
                    rejected_teacher_attempts=body.teacher_attempts[:-1],
                    metadata=metadata,
                )
                record["training_example"] = TypeAdapter(SavedTrainingExample).dump_python(example, mode="json")
            target = Path(self.config.artifact_root) / "episodes" / f"{session.episode}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            temp.replace(target)
            result = EpisodeResult(
                outcome=body.outcome,
                artifact_path=str(target),
                turns=len(turns),
                teacher_trials=len(body.teacher_attempts),
                axis_id=session.task.axis_id,
            )
            session.cache["final"] = (signature, result)
            return result

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        counts = {}
        for result in body.verify_responses:
            outcome = result.get("outcome")
            if outcome:
                counts[outcome] = counts.get(outcome, 0) + 1
        return AggregateMetrics(agent_metrics={"completed_rollouts": sum(counts.values()), **counts})


if __name__ == "__main__":
    AudioMCSimResources.run_webserver()
