# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tau-Voice Simple: Tau2 customer-service episodes with spoken user turns.

The server owns the Tau side of each episode: the domain environment and its
tools, the LLM user simulator, TTS of every spoken user turn, Tau's step,
error and timeout limits, and Tau's scorer (with the NL-assertion judge). The
paired agent owns the candidate: it sends the conversation, with every user
turn as audio, to the policy model and submits each reply through
``/agent_turn``. ``/verify`` scores the finished episode and appends its user
audio and trajectory to the run's tar archive.
"""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Literal

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.tau_voice_simple.audio_archive import AudioArchive
from resources_servers.tau_voice_simple.source import data_identity, ensure_tau2_data_dir
from resources_servers.tau_voice_simple.task_data import TauVoiceSimpleMetadata
from resources_servers.tau_voice_simple.tts import (
    DEFAULT_TTS_INSTRUCTIONS,
    GPT4O_MINI_TTS_VOICES,
    TTSClient,
    TTSConfig,
)


MetricValue = int | float | bool | None
# Terminations that end a conversation on its own terms; anything else is premature.
NATURAL_TERMINATIONS = frozenset({"agent_stop", "user_stop"})


class TauVoiceSimpleConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    interaction_mode: Literal["s2t", "t2t"] = "s2t"
    task_pack: Literal["tau", "eva", "mivas"] = "tau"
    source_root: str | None = None
    source_python: str | None = None
    task_data_root: str | None = None
    artifact_root: str | None = None
    # Tau episode limits, as in the reference run.
    max_steps: int = Field(default=100, gt=0)
    max_errors: int = Field(default=10, gt=0)
    episode_timeout_seconds: float = Field(default=1200.0, gt=0)

    # One OpenAI-compatible endpoint serves the user simulator, the NL-assertion judge, and TTS.
    llm_api_base: str
    llm_api_key: str = Field(repr=False)
    # LiteLLM model names (Tau calls LiteLLM); the leading "openai/" selects the provider.
    user_llm: str = "openai/openai/openai/gpt-5.6-luna"
    user_llm_args: dict[str, Any] = Field(default_factory=lambda: {"temperature": 1})
    nl_assertions_llm: str = "openai/openai/openai/gpt-5.4-mini"
    nl_assertions_llm_args: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.0})
    tts_model: str = "openai/openai/gpt-4o-mini-tts"
    # Defaults to ``{llm_api_base}/audio/speech``.
    tts_endpoint: str | None = None
    tts_instructions: str = DEFAULT_TTS_INSTRUCTIONS
    tts_timeout_seconds: float = Field(default=120.0, ge=0)
    tts_max_retries: int = Field(default=-1, ge=-1)

    # Prepared Tau ``data`` directory; fetched from the pinned revision when absent.
    tau2_data_dir: str
    # One tar per run with every episode's user audio and trajectory; None disables it.
    audio_archive_path: str | None = None
    expected_num_rows: int | None = Field(default=None, gt=0)
    # Blocking Tau work (user LLM, TTS, tools, scoring) runs on this many threads.
    worker_threads: int = Field(default=64, gt=0)
    # Tau logs every step at DEBUG through loguru; this is the level kept.
    tau2_log_level: str = "WARNING"


class TauVoiceSimpleSeedSessionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: TauVoiceSimpleMetadata


class TauVoiceSimpleSeedSessionResponse(BaseSeedSessionResponse):
    domain_policy: str
    tools: list[dict[str, Any]]
    done: bool
    observation: dict[str, Any] | None = None
    termination_reason: str | None = None


class ToolCallPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    arguments: dict[str, Any]


class AgentTurnRequest(BaseModel):
    """The candidate's turn: spoken text or exactly one tool call, never both."""

    model_config = ConfigDict(extra="forbid")
    content: str | None = None
    tool_call: ToolCallPayload | None = None
    raw_data: dict[str, Any] | None = None


class AgentTurnResponse(BaseModel):
    done: bool
    observation: dict[str, Any] | None = None
    termination_reason: str | None = None
    domain_policy: str | None = None
    tools: list[dict[str, Any]] | None = None


class TauVoiceSimpleVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: TauVoiceSimpleMetadata


class TauVoiceSimpleVerifyResponse(TauVoiceSimpleVerifyRequest, BaseVerifyResponse):
    sample_id: str
    domain: str
    task_id: str
    interaction_mode: str
    tts_voice: str | None
    termination_reason: str | None
    premature: bool
    duration_seconds: float | None
    reward_info: dict[str, Any] | None
    trajectory: dict[str, Any]
    audio_archive_members: list[str]
    simulation_path: str | None = None


class _Session:
    def __init__(self, episode, metadata: TauVoiceSimpleMetadata):
        self.episode = episode
        self.metadata = metadata
        self.lock = asyncio.Lock()


class TauVoiceSimpleServer(SimpleResourcesServer):
    ray_enabled = False
    config: TauVoiceSimpleConfig
    _sessions: dict[str, _Session] = PrivateAttr(default_factory=dict)
    _tasks: dict[str, dict] = PrivateAttr(default_factory=dict)
    _executor: ThreadPoolExecutor = PrivateAttr()
    _archive: AudioArchive | None = PrivateAttr(default=None)
    _episode_module: Any = PrivateAttr(default=None)
    _tau_data_identity: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object) -> None:
        # tau2 reads TAU2_DATA_DIR when first imported, so it is imported only after this.
        data_dir = ensure_tau2_data_dir(self.config.tau2_data_dir)
        self._tau_data_identity = data_identity(data_dir)
        os.environ["TAU2_DATA_DIR"] = str(data_dir.resolve())
        from resources_servers.tau_voice_simple import episode as episode_module

        logger.remove()
        logger.add(sys.stderr, level=self.config.tau2_log_level)

        self._episode_module = episode_module
        if self.config.task_pack != "tau":
            if not self.config.source_root:
                raise ValueError("External task packs require a pinned source_root")
            prefix = f"TAU_VOICE_SIMPLE_{self.config.task_pack.upper()}"
            os.environ[prefix + "_ROOT"] = self.config.source_root
            if self.config.source_python:
                os.environ[prefix + "_PYTHON"] = self.config.source_python
            if self.config.task_data_root:
                os.environ[prefix + "_DATA_ROOT"] = self.config.task_data_root
        episode_module.configure_nl_assertions_judge(
            self.config.nl_assertions_llm, self._llm_args(self.config.nl_assertions_llm_args)
        )
        self._executor = ThreadPoolExecutor(max_workers=self.config.worker_threads, thread_name_prefix="tau-voice")
        if self.config.audio_archive_path:
            self._archive = AudioArchive(self.config.audio_archive_path)
        super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/agent_turn")(self.agent_turn)
        return app

    def _llm_args(self, args: Mapping[str, Any]) -> dict[str, Any]:
        return dict(args) | {
            "api_base": self.config.llm_api_base,
            "api_key": self.config.llm_api_key,
        }

    def _task(self, metadata: TauVoiceSimpleMetadata):
        if metadata.task_pack != self.config.task_pack:
            raise HTTPException(status_code=422, detail="Row/resource task-pack mismatch")
        if self.config.task_pack != "tau":
            from resources_servers.tau_voice_simple.task_packs import TaskPack

            key = f"{metadata.domain}/{metadata.task_split}"
            if key not in self._tasks:
                self._tasks[key] = TaskPack(self.config.task_pack, metadata.domain, metadata.task_split)
            backend = self._tasks[key]
            expected = getattr(metadata, "task_contract_sha256", None)
            if expected and backend.contracts.get(metadata.task_id, {}).get("contract_sha256") != expected:
                raise HTTPException(status_code=422, detail="Prepared task contract changed")
            tasks = {str(t.id): t for t in backend.load_tasks()}
            if metadata.task_id not in tasks:
                raise HTTPException(status_code=422, detail="Unknown external task")
            return tasks[metadata.task_id]
        key = f"{metadata.domain}/{metadata.task_split}"
        if key not in self._tasks:
            self._tasks[key] = self._episode_module.load_tasks(metadata.domain, metadata.task_split)
        try:
            return self._tasks[key][metadata.task_id]
        except KeyError:
            raise HTTPException(status_code=422, detail=f"unknown Tau task {key}/{metadata.task_id}") from None

    def _tts_client(self, voice: str | None) -> TTSClient | None:
        if self.config.interaction_mode != "s2t":
            return None
        if voice is None:
            raise HTTPException(status_code=422, detail="s2t rows must set tts_voice")
        if voice not in GPT4O_MINI_TTS_VOICES:
            raise HTTPException(status_code=422, detail=f"unsupported tts_voice {voice!r}")
        return TTSClient(
            TTSConfig(
                endpoint=self.config.tts_endpoint or f"{self.config.llm_api_base.rstrip('/')}/audio/speech",
                api_key=self.config.llm_api_key,
                model=self.config.tts_model,
                voice=voice,
                instructions=self.config.tts_instructions,
                timeout_seconds=self.config.tts_timeout_seconds,
                max_retries=self.config.tts_max_retries,
            )
        )

    async def _run(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, function, *args)

    def _session(self, request: Request) -> _Session:
        session = self._sessions.get(request.session.get(SESSION_ID_KEY))
        if session is None:
            raise HTTPException(status_code=409, detail="no Tau-Voice Simple episode for this session")
        return session

    async def seed_session(
        self, request: Request, body: TauVoiceSimpleSeedSessionRequest
    ) -> TauVoiceSimpleSeedSessionResponse:
        metadata = body.verifier_metadata
        task = self._task(metadata)
        artifact_base = self.config.artifact_root or (
            str(Path(self.config.audio_archive_path).parent / "episodes") if self.config.audio_archive_path else None
        )
        if not artifact_base:
            raise ValueError("Set artifact_root or audio_archive_path to preserve trajectories")
        artifact_dir = Path(tempfile.mkdtemp(prefix="episode-", dir=_mkdir(artifact_base)))
        backend = self._tasks[f"{metadata.domain}/{metadata.task_split}"] if self.config.task_pack != "tau" else None
        environment = backend.build_environment(task, artifact_dir) if backend else None
        settings = self._episode_module.EpisodeSettings(
            interaction_mode=self.config.interaction_mode,
            max_steps=self.config.max_steps,
            max_errors=self.config.max_errors,
            timeout_seconds=self.config.episode_timeout_seconds,
            user_llm=self.config.user_llm,
            user_llm_args=self._llm_args(self.config.user_llm_args),
        )
        episode = self._episode_module.Episode(
            domain=metadata.domain,
            task=task,
            seed=metadata.seed,
            simulation_id=f"{metadata.id}__{uuid.uuid4().hex[:8]}",
            settings=settings,
            tts_client=self._tts_client(metadata.tts_voice),
            environment=environment,
            pack_backend=backend,
            artifact_dir=artifact_dir,
        )
        session_id = request.session[SESSION_ID_KEY]
        self._sessions[session_id] = _Session(episode, metadata)
        try:
            result = await self._run(episode.start)
        except BaseException:
            episode.close()
            self._sessions.pop(session_id, None)
            raise
        return TauVoiceSimpleSeedSessionResponse(
            **(
                {
                    "domain_policy": episode.policy,
                    "tools": [tool.openai_schema for tool in episode.tools],
                }
                | result
            )
        )

    async def agent_turn(self, request: Request, body: AgentTurnRequest) -> AgentTurnResponse:
        if (body.content is None) == (body.tool_call is None):
            raise HTTPException(status_code=422, detail="send exactly one of content or tool_call")
        session = self._session(request)
        reply = self._episode_module.assistant_message(
            body.content, body.tool_call.model_dump() if body.tool_call else None, body.raw_data
        )
        async with session.lock:
            result = await self._run(session.episode.submit, reply)
        return AgentTurnResponse(**result)

    def _finish(self, session: _Session) -> tuple[Any, dict[str, Any], list[str]]:
        metadata = session.metadata
        episode = session.episode
        simulation = episode.finalize(
            {
                "interaction_mode": self.config.interaction_mode,
                "candidate_interface": (
                    "speech_in_text_out_half_duplex"
                    if self.config.interaction_mode == "s2t"
                    else "text_in_text_out_half_duplex"
                ),
                "user_tts": (
                    session.episode.user.tts_client.config.model if self.config.interaction_mode == "s2t" else None
                ),
                "tts_voice": metadata.tts_voice,
                "evaluation_phase": getattr(metadata, "evaluation_phase", "scored"),
                "clean_audio": True if self.config.interaction_mode == "s2t" else None,
            }
        )
        # Persist inference before any LLM-backed judge call. A scorer outage
        # must not require a second conversation or synthesis.
        path = episode.artifact_dir / "simulation.json"
        path.write_bytes(self._episode_module.simulation_json(simulation))
        (episode.artifact_dir / "episode.json").write_text(
            json.dumps(
                {
                    "metadata": metadata.model_dump(),
                    "caller_protocol": "caller-tool-priority-v1",
                    "task_pack": self.config.task_pack,
                    "tau_data_identity": self._tau_data_identity,
                    "task_sha256": hashlib.sha256(episode.task.model_dump_json().encode()).hexdigest(),
                },
                indent=2,
            )
        )
        members = []
        for index, wav in episode.user.audio_turns:
            (episode.artifact_dir / f"user_turn_{index:04d}.wav").write_bytes(wav)
        if self._archive is not None:
            audio = [(f"user_turn_{index:04d}.wav", wav) for index, wav in episode.user.audio_turns]
            audio.append(("simulation.json", self._episode_module.simulation_json(simulation)))
            members = self._archive.append(simulation.id, audio)
        reward_info = episode.evaluate()
        (episode.artifact_dir / "reward_info.json").write_text(json.dumps(reward_info, indent=2))
        return simulation, reward_info, members

    async def verify(self, request: Request, body: TauVoiceSimpleVerifyRequest) -> TauVoiceSimpleVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        session = self._sessions.get(session_id)
        metadata = body.verifier_metadata
        base = body.model_dump() | {
            "sample_id": metadata.id,
            "domain": metadata.domain,
            "task_id": metadata.task_id,
            "interaction_mode": self.config.interaction_mode,
            "tts_voice": (
                (session.metadata.tts_voice if session else metadata.tts_voice)
                if self.config.interaction_mode == "s2t"
                else None
            ),
        }
        if session is None or not session.episode.done:
            # The agent stopped before Tau ended the episode; that is not a valid measurement.
            return TauVoiceSimpleVerifyResponse(
                **base,
                reward=0.0,
                mask_sample=True,
                failure_kind="environment_error",
                failure_reason="episode missing or not finished at verify",
                termination_reason=None,
                premature=True,
                duration_seconds=None,
                reward_info=None,
                trajectory={},
                audio_archive_members=[],
            )
        try:
            async with session.lock:
                simulation, reward_info, members = await self._run(self._finish, session)
        finally:
            session.episode.close()
            self._sessions.pop(session_id, None)
        return TauVoiceSimpleVerifyResponse(
            **base,
            reward=float(reward_info["reward"]),
            termination_reason=simulation.termination_reason,
            premature=simulation.termination_reason not in NATURAL_TERMINATIONS,
            duration_seconds=simulation.duration,
            reward_info=reward_info,
            trajectory=self._episode_module.trajectory_summary(simulation),
            audio_archive_members=members,
            simulation_path=str(session.episode.artifact_dir / "simulation.json"),
        )

    async def close_resources_session(self, request: Request, body: Any = None) -> dict[str, Any]:
        session = self._sessions.pop(request.session.get(SESSION_ID_KEY), None)
        if session:
            session.episode.close()
        return {}

    def compute_metrics(self, tasks: Sequence[Sequence[Mapping[str, object]]]) -> dict[str, MetricValue]:
        """Per-domain Tau-Voice Simple success (%, mean reward x 100) and episode diagnostics."""
        rows = [row for task in tasks for row in task]
        scored = rows  # Never drop infrastructure or model failures from denominators.

        def summarize(selected: list[Mapping[str, object]], prefix: str) -> dict[str, MetricValue]:
            rewards = [0.0 if row.get("mask_sample") else float(row.get("reward") or 0.0) for row in selected]
            fallbacks = sum(int((row.get("trajectory") or {}).get("protocol_fallbacks") or 0) for row in selected)
            return {
                f"{prefix}/success": 100.0 * sum(rewards) / len(rewards) if rewards else 0.0,
                f"{prefix}/episodes": len(selected),
                f"{prefix}/premature_episodes": sum(row.get("premature") is True for row in selected),
                f"{prefix}/tts_voices_used": len({row.get("tts_voice") for row in selected if row.get("tts_voice")}),
                f"{prefix}/protocol_fallbacks": fallbacks,
            }

        metrics = summarize(scored, "tau_voice_simple")
        domain_success = []
        for domain in sorted({row.get("domain") for row in rows if row.get("domain")}):
            selected = [row for row in scored if row.get("domain") == domain]
            if selected:
                metrics.update(summarize(selected, f"tau_voice_simple/{domain}"))
                domain_success.append(metrics[f"tau_voice_simple/{domain}/success"])
                reasons: dict[str, int] = {}
                for row in selected:
                    reason = str(row.get("termination_reason"))
                    reasons[reason] = reasons.get(reason, 0) + 1
                for reason, count in sorted(reasons.items()):
                    metrics[f"tau_voice_simple/{domain}/termination/{reason}"] = count
        metrics["tau_voice_simple/macro_success"] = (
            sum(domain_success) / len(domain_success) if domain_success else 0.0
        )
        metrics["tau_voice_simple/masked_rows"] = sum(bool(row.get("mask_sample")) for row in rows)
        if self.config.expected_num_rows is not None:
            metrics["tau_voice_simple/expected_rows"] = self.config.expected_num_rows
            metrics["tau_voice_simple/complete"] = (
                len(rows) == self.config.expected_num_rows and not metrics["tau_voice_simple/masked_rows"]
            )
        prefix = self.config.task_pack if self.config.task_pack != "tau" else "tau_voice_simple"
        return {key.replace("tau_voice_simple", prefix, 1): value for key, value in metrics.items()}

    def get_key_metrics(self, agent_metrics: Mapping[str, object]) -> dict[str, object]:
        prefix = self.config.task_pack if self.config.task_pack != "tau" else "tau_voice_simple"
        return {key: value for key, value in agent_metrics.items() if key.startswith(prefix + "/")}

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        aggregate = compute_aggregate_metrics(body.verify_responses)
        metrics = self.compute_metrics([body.verify_responses])
        aggregate.agent_metrics.update(metrics)
        aggregate.key_metrics.update(metrics)
        return aggregate


def _mkdir(path: str | Path) -> str | Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    TauVoiceSimpleServer.run_webserver()
