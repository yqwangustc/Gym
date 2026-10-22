# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol invariants: frozen rubrics, consistent histories, and accepted labels."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from omegaconf import OmegaConf
from pydantic import ValidationError
from starlette.requests import Request

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import ServerClient
from resources_servers.audiomc_sim.app import AudioMCSimConfig, AudioMCSimResources
from resources_servers.audiomc_sim.common import Prompts, digest, prompt_values
from resources_servers.audiomc_sim.contracts import (
    AudioRef,
    Candidate,
    FinalizeRequest,
    Grade,
    GradeRequest,
    Judgment,
    Message,
    StepRequest,
    TaskInput,
    TurnEvaluation,
)


@pytest.fixture
def task() -> TaskInput:
    return TaskInput(
        persona_uuid="uuid",
        persona="A project manager.",
        axis_id="inference_memory",
        evaluation={
            "category": "Inference Memory",
            "definition": "Use earlier context.",
            "pass_criteria": "Remembers.",
            "fail_criteria": "Forgets.",
            "few_shot_examples": [],
        },
        topic="Planning",
        subtopic="A meeting",
        subtopic_description="Schedule a meeting.",
        seed=42,
        max_turns=2,
        max_teacher_trials=2,
        assistant_instructions="Be helpful.",
    )


@pytest.fixture
def resources(tmp_path):
    client = ServerClient(
        head_server_config=BaseServerConfig(host="localhost", port=1), global_config_dict=OmegaConf.create({})
    )
    return AudioMCSimResources(
        config=AudioMCSimConfig(
            name="resources",
            entrypoint="app.py",
            host="localhost",
            port=2,
            helper_model={"name": "helper", "type": "responses_api_models"},
            artifact_root=str(tmp_path),
            tts_endpoint="http://unused",
            tts_api_key="unused",
        ),
        server_client=client,
    )


async def seed(resources, task, session_id="session"):
    request = Request({"type": "http", "session": {}})
    body = ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="episode"),
        task_id=TaskId(taskset="test", task_id="task"),
        task_data=task.model_dump(),
    )
    await resources.seed_session(request, body)
    return request, body


def builder_json() -> str:
    return json.dumps(
        {
            "rubrics": [
                {
                    "index": 0,
                    "requirement": "Answer the request.",
                    "category": "general_quality",
                    "supporting_turn_ids": ["u0"],
                    "instruction_evidence": None,
                }
            ]
        }
    )


@pytest.mark.asyncio
async def test_frozen_rubrics_and_session_isolation(resources, task, monkeypatch):
    request, body = await seed(resources, task)
    infer = AsyncMock(return_value=builder_json())
    monkeypatch.setattr("resources_servers.audiomc_sim.app.infer", infer)
    prefix = [Message(turn_id="u0", role="user", text="Schedule it for Monday.")]
    step = StepRequest(resources_session_id="session", turn_index=0, history=prefix)
    rubrics = await resources.build_rubrics(request, step)
    assert rubrics[0].rubric_id == "episode_t0_r0"
    assert await resources.build_rubrics(request, step) == rubrics
    assert infer.await_count == 1
    with pytest.raises(HTTPException, match="frozen prefix"):
        await resources.build_rubrics(
            request, step.model_copy(update={"history": [prefix[0].model_copy(update={"text": "Changed"})]})
        )
    with pytest.raises(HTTPException, match="cookie"):
        await resources.build_rubrics(Request({"type": "http", "session": {}}), step)
    with pytest.raises(HTTPException, match="another task"):
        await resources.seed_session(
            request, body.model_copy(update={"task_data": task.model_copy(update={"seed": 99}).model_dump()})
        )
    await resources.close_resources_session(
        request, ResourcesCloseSessionRequest(resources_session_id="session", episode_id=body.episode_id)
    )
    with pytest.raises(HTTPException, match="already closed"):
        await resources.seed_session(request, body)


@pytest.mark.asyncio
async def test_teacher_same_prefix_and_persisted_correction(resources, task, monkeypatch):
    request, _ = await seed(resources, task)
    monkeypatch.setattr("resources_servers.audiomc_sim.app.infer", AsyncMock(return_value=builder_json()))
    prefix = [Message(turn_id="u0", role="user", text="Schedule it for Monday.")]
    rubrics = await resources.build_rubrics(
        request, StepRequest(resources_session_id="session", turn_index=0, history=prefix)
    )
    failed = Judgment(
        grades=[Grade(rubric_id=rubrics[0].rubric_id, passed=False, explanation="Wrong day.")], passed=False
    )
    passed = Judgment(grades=[Grade(rubric_id=rubrics[0].rubric_id, passed=True, explanation="Monday.")], passed=True)
    audio = AudioRef(
        archive="test.tar",
        member="test.wav",
        sha256="hash",
        voice="alloy",
        model="tts",
        sample_rate=24000,
        duration_seconds=1,
    )
    session = resources._sessions["session"]
    session.audio[0] = (prefix[0].text, audio)
    session.judgments[(0, digest("Tuesday."))] = failed
    session.judgments[(0, digest("Monday."))] = passed
    body = FinalizeRequest(
        resources_session_id="session",
        outcome="accepted_correction",
        input_prefix=prefix,
        turn_evaluations=[
            TurnEvaluation(
                turn_index=0,
                user_turn_id="u0",
                user_text=prefix[0].text,
                audio=audio,
                rubrics=rubrics,
                policy_response="Tuesday.",
                judgment=failed,
            )
        ],
        teacher_attempts=[Candidate(text="Monday.", judgment=passed)],
    )
    result = await resources.verify(request, body)
    record = json.loads(Path(result.artifact_path).read_text())
    assert record["training_example"]["corrected_response"]["text"] == "Monday."
    assert record["training_example"]["input_prefix"]["turns"] == [m.model_dump() for m in prefix]
    assert record["training_example"]["turn_evaluations"][0]["rubrics"][0]["rubric_id"] == rubrics[0].rubric_id
    assert await resources.verify(request, body) == result
    with pytest.raises(HTTPException, match="another result"):
        await resources.verify(request, body.model_copy(update={"outcome": "teacher_correction_exhausted"}))
    with pytest.raises(HTTPException, match="frozen prefix"):
        await resources.grade(
            request, GradeRequest(resources_session_id="session", turn_index=0, history=[], candidate="Monday.")
        )


@pytest.mark.asyncio
async def test_does_not_continue_after_failure(resources, task, monkeypatch):
    request, _ = await seed(resources, task)
    monkeypatch.setattr("resources_servers.audiomc_sim.app.infer", AsyncMock(return_value=builder_json()))
    prefix = [Message(turn_id="u0", role="user", text="Monday.")]
    await resources.build_rubrics(request, StepRequest(resources_session_id="session", turn_index=0, history=prefix))
    prefix += [
        Message(turn_id="a0", role="assistant", text="Wrong."),
        Message(turn_id="u1", role="user", text="Next."),
    ]
    with pytest.raises(HTTPException, match="did not pass"):
        await resources.build_rubrics(
            request, StepRequest(resources_session_id="session", turn_index=1, history=prefix)
        )


@pytest.mark.asyncio
async def test_empty_builder_is_inspection_outcome(resources, task, monkeypatch):
    request, _ = await seed(resources, task)
    monkeypatch.setattr("resources_servers.audiomc_sim.app.infer", AsyncMock(return_value='{"rubrics": []}'))
    prefix = [Message(turn_id="u0", role="user", text="Unclear context.")]
    assert (
        await resources.build_rubrics(
            request, StepRequest(resources_session_id="session", turn_index=0, history=prefix)
        )
        == []
    )
    result = await resources.verify(
        request,
        FinalizeRequest(
            resources_session_id="session", outcome="ungradable", input_prefix=prefix, turn_evaluations=[]
        ),
    )
    record = json.loads(Path(result.artifact_path).read_text())
    assert "training_example" not in record
    assert record["outcome"] == "ungradable"


def test_prompt_roles_and_no_recursive_rendering(task):
    roots = [
        ("environment_servers", ("planner", "user", "teacher", "assistant")),
        ("resources_servers", ("rubric_builder", "critiquer")),
    ]
    values = prompt_values(task, [], 0) | {
        "blueprint": "Literal {{foo}}",
        "rubrics": "[]",
        "candidate_response": "Answer",
        "failed_candidates_and_feedback": "[]",
    }
    for directory, roles in roots:
        prompts = Prompts(Path(directory) / "audiomc_sim/prompts", roles)
        for role in roles:
            assert role.replace("_", " ") in prompts.texts[f"{role}/system"].lower()
            if role != "assistant":
                params = prompts.request(role, values)
                assert len(params.input) == 2
                if role == "user":
                    assert "Literal {{foo}}" in params.input[1].content


def test_judgment_rejects_inconsistent_or_nonboolean_verdicts():
    with pytest.raises(ValidationError):
        Judgment(grades=[], passed=True)
    with pytest.raises(ValidationError):
        Judgment(grades=[Grade(rubric_id="r", passed=False, explanation="Fails")], passed=True)
    with pytest.raises(ValidationError):
        Judgment(grades=[], passed="false")


@pytest.mark.asyncio
async def test_builder_and_critiquer_route_to_independent_models(resources, task, monkeypatch):
    from resources_servers.audiomc_sim.contracts import BuilderOutput

    resources.config = AudioMCSimConfig.model_validate(
        resources.config.model_dump()
        | {
            "rubric_builder_model": {"name": "builder-local", "type": "responses_api_models"},
            "critiquer_model": {"name": "judge-api", "type": "responses_api_models"},
        }
    )
    request, _ = await seed(resources, task)
    session = resources._sessions["session"]
    infer = AsyncMock(side_effect=[builder_json(), '{"grades": [], "passed": null}'])
    monkeypatch.setattr("resources_servers.audiomc_sim.app.infer", infer)
    resources._prompts.request = lambda *args, **kwargs: None
    await resources._json_call(session, "rubric_builder", {"turn_index": 0}, BuilderOutput)
    await resources._json_call(session, "critiquer", {"turn_index": 0}, Judgment)
    assert [call.args[1] for call in infer.await_args_list] == ["builder-local", "judge-api"]


@pytest.mark.asyncio
async def test_voice_sampling_varies_by_conversation_and_is_stable_on_retry(resources, task):
    pool = ["alloy", "nova", "shimmer"]
    resources.config.tts_voices = pool
    selected = set()
    for index in range(64):
        body = ResourcesSeedSessionRequest(
            resources_session_id=f"session-{index}",
            episode_id=EpisodeId(rollout_id=f"rollout-{index}"),
            task_id=TaskId(taskset="test", task_id=f"task-{index}"),
            task_data=task.model_dump(),
        )
        request = Request({"type": "http", "session": {}})
        await resources.seed_session(request, body)
        voice = resources._sessions[body.resources_session_id].voice
        selected.add(voice)
        await resources.seed_session(request, body)
        assert resources._sessions[body.resources_session_id].voice == voice
        retry = body.model_copy(
            update={
                "resources_session_id": f"retry-{index}",
                "episode_id": EpisodeId(rollout_id=f"rollout-{index}", attempt=1),
            }
        )
        await resources.seed_session(Request({"type": "http", "session": {}}), retry)
        assert resources._sessions[retry.resources_session_id].voice == voice
    assert selected == set(pool)


@pytest.mark.asyncio
async def test_synthesis_uses_one_selected_voice_for_all_turns_and_records_it(resources, task, monkeypatch):
    import io
    import wave
    from types import SimpleNamespace

    from resources_servers.audiomc_sim.contracts import SynthesisRequest

    resources.config.tts_voices = ["alloy", "shimmer"]
    task = task.model_copy(update={"tts_voice": "nova"})
    request, body = await seed(resources, task)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 240)
    http = AsyncMock(
        return_value=SimpleNamespace(ok=True, read=AsyncMock(return_value=buffer.getvalue()), release=lambda: None)
    )
    monkeypatch.setattr("resources_servers.audiomc_sim.speech.http_request", http)
    session = resources._sessions["session"]
    for index in range(2):
        text = f"User turn {index}"
        session.prefixes[index] = [{"role": "user", "turn_id": f"u{index}", "text": text}]
        synth = SynthesisRequest(resources_session_id="session", turn_index=index, text=text)
        audio = await resources.synthesize(request, synth)
        assert audio.voice == "nova"
        assert await resources.synthesize(request, synth) == audio
    assert http.await_count == 2
    assert [call.kwargs["json"]["voice"] for call in http.await_args_list] == ["nova", "nova"]
    await resources.close_resources_session(
        request, ResourcesCloseSessionRequest(resources_session_id="session", episode_id=body.episode_id)
    )
    partial = json.loads((Path(resources.config.artifact_root) / "partial/episode.json").read_text())
    assert partial["tts_voice"] == "nova"
    assert all(ref["voice"] == "nova" for ref in partial["audio"])


@pytest.mark.asyncio
async def test_legacy_single_voice_config_still_seeds_session(resources, task):
    resources.config.tts_voice = "legacy-speaker"
    await seed(resources, task)
    assert resources._sessions["session"].voice == "legacy-speaker"


@pytest.mark.parametrize("voices", [[], [""], ["   "]])
def test_resource_config_rejects_empty_or_blank_voice_pool(resources, voices):
    with pytest.raises(ValidationError):
        AudioMCSimConfig.model_validate(resources.config.model_dump() | {"tts_voices": voices})


@pytest.mark.asyncio
async def test_saved_task_voice_is_reused_for_different_rollouts(resources, task):
    task = task.model_copy(update={"tts_voice": "nova"})
    resources.config.tts_voices = ["alloy", "shimmer"]
    for index in range(2):
        body = ResourcesSeedSessionRequest(
            resources_session_id=f"session-{index}",
            episode_id=EpisodeId(rollout_id=f"different-rollout-{index}"),
            task_id=TaskId(taskset="test", task_id="same-task"),
            task_data=task.model_dump(),
        )
        await resources.seed_session(Request({"type": "http", "session": {}}), body)
        assert resources._sessions[body.resources_session_id].voice == "nova"
