# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment-owned multi-turn generation and teacher correction protocol."""

import json
from pathlib import Path

from fastapi import Body
from pydantic import JsonValue, PrivateAttr, model_validator

from nemo_gym.base_environment_server import BaseEnvironmentServer, BaseEnvironmentServerConfig, CleanupContext
from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.config_types import (
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    ResourcesServerRef,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.audiomc_sim.common import Prompts, infer, prompt_values, response_text
from resources_servers.audiomc_sim.contracts import (
    AudioRef,
    Candidate,
    EpisodeRequest,
    EpisodeResponse,
    EpisodeResult,
    FinalizeRequest,
    Judgment,
    Message,
    Outcome,
    Rubric,
    TurnEvaluation,
)
from resources_servers.audiomc_sim.telemetry import trace_call


class AudioMCSimEnvironmentConfig(BaseEnvironmentServerConfig):
    assistant_agent: AgentServerRef
    planner_model: ModelServerRef
    user_teacher_model: ModelServerRef | None = None
    user_model: ModelServerRef | None = None
    teacher_model: ModelServerRef | None = None
    resources_server: ResourcesServerRef
    prompts_dir: str = "environment_servers/audiomc_sim/prompts"
    assistant_max_tokens: int = 8192

    def model_for(self, role: str) -> ModelServerRef:
        model = {
            "planner": self.planner_model,
            "user": self.user_model or self.user_teacher_model,
            "teacher": self.teacher_model or self.user_teacher_model,
        }[role]
        if model is None:
            raise ValueError(f"Configure {role}_model or legacy user_teacher_model")
        return model

    @model_validator(mode="after")
    def validate_role_models(self) -> "AudioMCSimEnvironmentConfig":
        for role in ("user", "teacher"):
            self.model_for(role)
        return self


class AudioMCSimEnvironment(BaseEnvironmentServer[EpisodeRequest, EpisodeResponse]):
    config: AudioMCSimEnvironmentConfig
    request_model = EpisodeRequest
    response_model = EpisodeResponse
    _prompts: Prompts = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._prompts = Prompts(Path(self.config.prompts_dir), ("planner", "user", "teacher", "assistant"))

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/aggregate_metrics", json=body
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def run(self, request: EpisodeRequest, cleanup: CleanupContext) -> EpisodeResponse:
        task = request.task.task_input
        episode = request.episode_id.capture_key
        resource_session = f"{episode}-resources"
        assistant_session = f"{episode}-assistant"
        resource_cookies, assistant_cookies = {}, {}

        async def post_resources(path: str, values: dict[str, object]) -> JsonValue:
            response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=path,
                json=values,
                cookies=resource_cookies,
            )
            resource_cookies.update({key: morsel.value for key, morsel in response.cookies.items()})
            await raise_for_status(response)
            return await get_response_json(response)

        async def close_resources() -> None:
            await post_resources(
                "/close_session",
                {
                    "resources_session_id": resource_session,
                    "episode_id": request.episode_id.model_dump(),
                },
            )

        async def close_assistant() -> None:
            response = await self.server_client.post(
                server_name=self.config.assistant_agent.name,
                url_path="/v1/agent_sessions/close",
                json={"agent_session_id": assistant_session, "episode_id": request.episode_id.model_dump()},
                cookies=assistant_cookies,
            )
            await raise_for_status(response)

        cleanup.register_cleanup("resources", close_resources)
        await post_resources(
            "/seed_session",
            ResourcesSeedSessionRequest(
                resources_session_id=resource_session,
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                task_data=task.model_dump(mode="json"),
            ).model_dump(mode="json"),
        )
        cleanup.register_cleanup("assistant", close_assistant)
        response = await self.server_client.post(
            server_name=self.config.assistant_agent.name,
            url_path="/v1/agent_sessions",
            json=AgentSeedSessionRequest(
                agent_session_id=assistant_session,
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                tool_accesses=[],
            ),
        )
        await raise_for_status(response)
        assistant_cookies.update({key: morsel.value for key, morsel in response.cookies.items()})
        history, assistant_history, audios, evaluations, teacher_attempts, invocations = [], [], [], [], [], []

        async def save_partial() -> None:
            await post_resources(
                "/checkpoint",
                {
                    "resources_session_id": resource_session,
                    "input_prefix": [m.model_dump() for m in history],
                    "invocations": invocations,
                },
            )

        cleanup.register_cleanup("trace_checkpoint", save_partial)

        async def helper(role: str, values: dict[str, object]) -> str:
            model = self.config.model_for(role).name
            params = self._prompts.request(role, values, max_tokens=2048 if role == "user" else 4096)
            with trace_call(invocations, role, model_server=model, turn_index=values["turn_index"]):
                text = await infer(self.server_client, model, params)
            invocations.append({"role": role, "model_server": model, "text": text})
            return text

        async def grade(text: str, index: int) -> Judgment:
            return Judgment.model_validate(
                await post_resources(
                    "/grade",
                    {
                        "resources_session_id": resource_session,
                        "turn_index": index,
                        "history": [m.model_dump() for m in history],
                        "candidate": text,
                    },
                )
            )

        async def finish(outcome: Outcome) -> EpisodeResponse:
            invocations.append({"role": "prompt_manifest", "hashes": self._prompts.hashes()})
            body = FinalizeRequest(
                resources_session_id=resource_session,
                outcome=outcome,
                input_prefix=history,
                turn_evaluations=evaluations,
                teacher_attempts=teacher_attempts,
                invocations=invocations,
            )
            result = EpisodeResult.model_validate(await post_resources("/verify", body.model_dump(mode="json")))
            return EpisodeResponse(episode_id=request.episode_id, task_id=request.task.task_id, result=result)

        for index in range(task.max_turns):
            values = prompt_values(task, history, index)
            blueprint = await helper("planner", values)
            user_text = await helper("user", values | {"blueprint": blueprint})
            history.append(Message(turn_id=f"u{index}", role="user", text=user_text))
            rubrics = [
                Rubric.model_validate(r)
                for r in await post_resources(
                    "/build_rubrics",
                    {
                        "resources_session_id": resource_session,
                        "turn_index": index,
                        "history": [m.model_dump() for m in history],
                    },
                )
            ]
            if not rubrics:
                return await finish("ungradable")
            audio = AudioRef.model_validate(
                await post_resources(
                    "/synthesize",
                    {
                        "resources_session_id": resource_session,
                        "turn_index": index,
                        "text": user_text,
                    },
                )
            )
            audios.append(audio)
            assistant_history.append({"role": "user", "content": ""})
            inputs = [{"role": "system", "content": task.assistant_instructions}]
            if task.developer_instructions:
                inputs.append({"role": "developer", "content": task.developer_instructions})
            params = NeMoGymResponseCreateParamsNonStreaming(
                input=inputs + assistant_history,
                max_output_tokens=self.config.assistant_max_tokens,
                metadata={"audio_refs": json.dumps([[{"archive": a.archive, "member": a.member}] for a in audios])},
            )
            with trace_call(
                invocations, "assistant", model_server=self.config.assistant_agent.name, turn_index=index
            ) as timing:
                response = await self.server_client.post(
                    server_name=self.config.assistant_agent.name,
                    url_path="/v1/responses",
                    json=params,
                    cookies=assistant_cookies,
                )
                await raise_for_status(response)
                raw = await get_response_json(response)
                raw.pop("_ng_trajectory", None)
                candidate = NeMoGymResponse.model_validate(raw)
                assistant_cookies.update({key: morsel.value for key, morsel in response.cookies.items()})
                text = response_text(candidate)
                if candidate.status != "completed" or not text:
                    raise ValueError("Assistant returned incomplete or empty text")
                if candidate.usage is not None:
                    timing["usage"] = candidate.usage.model_dump(mode="json")
            invocations.append(
                {
                    "role": "assistant",
                    "model_server": self.config.assistant_agent.name,
                    "response_id": candidate.id,
                    "text": text,
                }
            )
            judgment = await grade(text, index)
            evaluations.append(
                TurnEvaluation(
                    turn_index=index,
                    user_turn_id=f"u{index}",
                    user_text=user_text,
                    audio=audio,
                    rubrics=rubrics,
                    policy_response=text,
                    judgment=judgment,
                )
            )
            if judgment.passed is None:
                return await finish("ungradable")
            if judgment.passed:
                # On the final allowed exchange keep the input prefix before this response for storage.
                if index == task.max_turns - 1:
                    return await finish("max_turn_reached_without_failing")
                history.append(Message(turn_id=f"a{index}", role="assistant", text=text))
                assistant_history.append({"role": "assistant", "content": text})
                continue
            feedback = [{"text": text, "judgment": judgment.model_dump()}]
            for _ in range(task.max_teacher_trials):
                values = prompt_values(task, history, index) | {
                    "rubrics": json.dumps([r.model_dump() for r in rubrics]),
                    "failed_candidates_and_feedback": json.dumps(feedback, ensure_ascii=False),
                }
                corrected = await helper("teacher", values)
                corrected_grade = await grade(corrected, index)
                teacher_attempts.append(Candidate(text=corrected, judgment=corrected_grade))
                if corrected_grade.passed is None:
                    return await finish("ungradable")
                if corrected_grade.passed:
                    return await finish("accepted_correction")
                feedback.append({"text": corrected, "judgment": corrected_grade.model_dump()})
            return await finish("teacher_correction_exhausted")
        raise AssertionError("Nonempty turn budget must terminate")


if __name__ == "__main__":
    AudioMCSimEnvironment.run_webserver()
