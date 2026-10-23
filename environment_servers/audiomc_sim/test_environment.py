# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise orchestration outcomes and role visibility without external models."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from environment_servers.audiomc_sim.app import AudioMCSimEnvironment, AudioMCSimEnvironmentConfig
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.audiomc_sim.contracts import EpisodeRequest, FinalizeRequest, TaskInput


def response_json(text: str) -> dict:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="mock",
        object="response",
        status="completed",
        output=[
            {
                "id": "message",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump(mode="json")


def http_response(data: object, cookie: str = "cookie") -> SimpleNamespace:
    return SimpleNamespace(
        ok=True,
        read=AsyncMock(return_value=json.dumps(data).encode()),
        cookies={"cookie": SimpleNamespace(value=cookie)},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "keep_generating,verdicts,outcome,assistant_calls,teacher_calls",
    [
        (False, [True, False, False, True], "accepted_correction", 2, 2),
        (False, [True, True], "max_turn_reached_without_failing", 2, 0),
        (False, [False, False, False], "teacher_correction_exhausted", 1, 2),
        (False, [None], "ungradable", 1, 0),
        (True, [False, True, True], "accepted_correction", 2, 1),
        (True, [False, False, True, False, False, True], "accepted_correction", 2, 4),
        (True, [False, True, False, False, False], "teacher_correction_exhausted", 2, 3),
        (True, [True, True], "max_turn_reached_without_failing", 2, 0),
        (True, [False, True, None], "ungradable", 2, 1),
        (True, [False, True, False, None], "ungradable", 2, 2),
    ],
)
@pytest.mark.parametrize("separate_backends", [False, True])
@pytest.mark.parametrize(
    "teacher_output",
    [
        "valid",
        "inference_failure",
        "missing_reasoning",
        "late_teacher_failure",
        "late_assistant_failure",
        "late_cancellation",
    ],
)
async def test_loop_outcomes_and_audio_only_prefix(
    monkeypatch, keep_generating, verdicts, outcome, assistant_calls, teacher_calls, separate_backends, teacher_output
):
    client = ServerClient(
        head_server_config=BaseServerConfig(host="localhost", port=1), global_config_dict=OmegaConf.create({})
    )
    env = AudioMCSimEnvironment(
        config=AudioMCSimEnvironmentConfig(
            name="env",
            entrypoint="app.py",
            host="localhost",
            port=2,
            assistant_agent={"name": "assistant", "type": "responses_api_agents"},
            planner_model={"name": "planner", "type": "responses_api_models"},
            **(
                {
                    "user_model": {"name": "local_user", "type": "responses_api_models"},
                    "teacher_model": {"name": "remote_teacher", "type": "responses_api_models"},
                }
                if separate_backends
                else {
                    "user_teacher_model": {"name": "user_teacher", "type": "responses_api_models"},
                }
            ),
            resources_server={"name": "resources", "type": "resources_servers"},
            cleanup_timeout_seconds=1,
        ),
        server_client=client,
    )
    task = TaskInput(
        persona_uuid="uuid",
        persona="Private biography",
        axis_id="voice_editing",
        seed=0,
        evaluation={
            "category": "Voice editing",
            "definition": "Apply edits",
            "pass_criteria": "Correct",
            "fail_criteria": "Wrong",
            "few_shot_examples": [],
        },
        topic="Tasks",
        subtopic="Edits",
        subtopic_description="Spoken updates",
        max_turns=2,
        max_teacher_trials=2,
        keep_generation_until_max_turn_reached=keep_generating,
        assistant_instructions="Be helpful.",
    )
    request = EpisodeRequest.model_validate(
        {
            "episode_id": {"rollout_id": "test"},
            "task": {"task_id": {"taskset": "test", "task_id": "task"}, "task_input": task.model_dump()},
        }
    )
    calls, helper_calls, saved = [], [], []
    judgments = iter(verdicts)
    current_turn = 0
    saved_teacher_turns = []
    teacher_inputs = {}

    async def infer(client, model, params):
        system = params.input[0].content
        perturn = params.input[1].content
        helper_calls.append((system, perturn))
        if "You are the Planner" in system:
            assert model == "planner"
            return "Strategy: private strategy. Next user move: private blueprint."
        if "You are the User Agent" in system:
            assert model == ("local_user" if separate_backends else "user_teacher")
            return "Spoken request."
        raise AssertionError("Teacher must use the reasoning-preserving inference helper")

    async def infer_with_reasoning(client, model, params):
        system = params.input[0].content
        perturn = params.input[1].content
        helper_calls.append((system, perturn))
        assert model == ("remote_teacher" if separate_backends else "user_teacher")
        assert "You are the Teacher" in system
        assert "Spoken request." in perturn and "Be helpful." in perturn
        # Earlier accepted student text remains visible; the current failed
        # response, previous Teacher trials and private grading material do not.
        assert perturn.count("wrong answer") == current_turn - sum(t < current_turn for t in saved_teacher_turns)
        for hidden in (
            "Required behavior",
            "Private biography",
            "Apply edits",
            "private blueprint",
            "Frozen turn-level rubrics",
            "Previous failed candidates",
            "Evaluation configuration",
            "Rationale based only on the visible request.",
        ):
            assert hidden not in perturn
        if current_turn in teacher_inputs:
            assert params.input == teacher_inputs[current_turn]  # Retries receive identical context.
        teacher_inputs[current_turn] = params.input
        if teacher_output == "late_teacher_failure" and current_turn > 0:
            raise ValueError("Later Teacher inference failed")
        if teacher_output == "inference_failure":
            raise ValueError("Teacher inference failed")
        if teacher_output == "missing_reasoning":
            raise ValueError("Teacher returned no native reasoning text")
        return "Corrected answer.", "Rationale based only on the visible request."

    async def post(self, *, server_name, url_path, json, **kwargs):
        nonlocal current_turn
        data = json.model_dump(exclude_unset=True) if hasattr(json, "model_dump") else json
        calls.append((url_path, data, kwargs))
        if url_path in {"/verify", "/checkpoint", "/close_session", "/v1/agent_sessions/close"}:
            assert kwargs["_max_num_tries"] == 1
            assert kwargs["timeout"].total == env.config.cleanup_timeout_seconds
        if url_path == "/build_rubrics":
            current_turn = data["turn_index"]
            assert data["history"][-1]["role"] == "user"
            assert "wrong answer" not in data["history"][-1]["text"]
            return http_response(
                [
                    {
                        "rubric_id": f"r{current_turn}",
                        "requirement": "Required behavior",
                        "category": "target_axis",
                        "supporting_turn_ids": [f"u{current_turn}"],
                    }
                ]
            )
        if url_path == "/synthesize":
            return http_response(
                {
                    "archive": "audio.tar",
                    "member": f"u{current_turn}.wav",
                    "sha256": "hash",
                    "voice": "alloy",
                    "model": "tts",
                    "sample_rate": 24000,
                    "duration_seconds": 1,
                }
            )
        if url_path == "/v1/responses":
            if teacher_output == "late_assistant_failure" and current_turn > 0:
                raise ValueError("Later Assistant inference failed")
            if teacher_output == "late_cancellation" and current_turn > 0:
                raise asyncio.CancelledError()
            assert server_name == "assistant"
            assert "Rationale based only" not in str(data)
            assert data["input"][0] == {"role": "system", "content": "Be helpful."}
            users = [m for m in data["input"] if m["role"] == "user"]
            assert len(users) == current_turn + 1
            assert all(m["content"] == "" for m in users)
            assert len(__import__("json").loads(data["metadata"]["audio_refs"])) == len(users)
            serialized = str(data)
            assert all(
                secret not in serialized
                for secret in [
                    "Private biography",
                    "private blueprint",
                    "Required behavior",
                    "Spoken request",
                ]
            )
            previous = [m["content"] for m in data["input"] if m["role"] == "assistant"]
            if previous and saved_teacher_turns:
                assert keep_generating and previous[-1] == "Corrected answer."
            return http_response(response_json("wrong answer"))
        if url_path == "/grade":
            assert data["candidate"] in {"wrong answer", "Corrected answer."}
            assert "Rationale based only" not in str(data)
            passed = next(judgments)
            if data["candidate"] == "Corrected answer." and passed:
                saved_teacher_turns.append(current_turn)
            assert data["history"][-1]["role"] == "user"
            return http_response(
                {
                    "passed": passed,
                    "grades": []
                    if passed is None
                    else [{"rubric_id": f"r{current_turn}", "passed": passed, "explanation": "Evidence"}],
                }
            )
        if url_path == "/verify":
            final = FinalizeRequest.model_validate(data)
            saved.append(final)
            return http_response(
                {
                    "outcome": final.outcome,
                    "accepted": any(t.generated_by_teacher for t in final.turn_evaluations),
                    "artifact_path": "episode.json",
                    "turns": len(final.turn_evaluations),
                    "teacher_trials": len(final.teacher_attempts),
                    "axis_id": task.axis_id,
                }
            )
        return http_response({})

    monkeypatch.setattr("environment_servers.audiomc_sim.app.infer", infer)
    monkeypatch.setattr("environment_servers.audiomc_sim.app.infer_with_reasoning", infer_with_reasoning)
    monkeypatch.setattr(ServerClient, "post", post)
    if teacher_output == "late_cancellation" and assistant_calls > 1:
        with pytest.raises(asyncio.CancelledError):
            await env.run_request(request)
        if saved_teacher_turns:
            assert saved[0].outcome == "interrupted"
            assert any(t.generated_by_teacher for t in saved[0].turn_evaluations)
        else:
            assert not saved
        return
    result = await env.run_request(request)
    if teacher_output.startswith("late_"):
        if saved and saved[0].outcome == "infrastructure_failure":
            assert result.failure is None
            assert result.result.accepted is True
            assert saved_teacher_turns
            assert any(t.generated_by_teacher for t in saved[0].turn_evaluations)
            return
        if result.failure is not None:
            assert not saved_teacher_turns and not saved
            return
    if teacher_output in {"inference_failure", "missing_reasoning"} and teacher_calls:
        assert result.failure is not None
        assert not saved  # Incomplete Teacher output cannot become accepted training data.
        assert not saved_teacher_turns  # No incomplete Teacher response reaches Critiquer.
        assert [path for path, _, _ in calls][-2:] == ["/v1/agent_sessions/close", "/close_session"]
        return
    assert result.failure is None, result.failure
    assert result.result.outcome == outcome
    assert len([path for path, _, _ in calls if path == "/v1/responses"]) == assistant_calls
    assert len([system for system, _ in helper_calls if "You are the Teacher" in system]) == teacher_calls
    final = saved[0]
    assert len(final.turn_evaluations) == assistant_calls
    assert len(final.teacher_attempts) == teacher_calls
    assert final.input_prefix[-1].role == "user"
    if not keep_generating:
        assert all(m.text != "Corrected answer." for m in final.input_prefix)
    for turn in final.turn_evaluations:
        assert turn.generated_by_teacher == (turn.turn_index in saved_teacher_turns)
        if turn.accepted_response:
            assert turn.accepted_response.judgment.passed is True
            assert turn.accepted_response.turn_index == turn.turn_index
            assert turn.accepted_response.thinking == (
                "Rationale based only on the visible request." if turn.generated_by_teacher else None
            )
    assert all(candidate.turn_index is not None for candidate in final.teacher_attempts)
    assert all(
        candidate.thinking == "Rationale based only on the visible request." for candidate in final.teacher_attempts
    )
    assert [path for path, _, _ in calls][-2:] == ["/v1/agent_sessions/close", "/close_session"]


@pytest.mark.asyncio
async def test_collector_recovers_finalized_reply_without_inference(tmp_path, monkeypatch):
    from argparse import Namespace
    from contextlib import asynccontextmanager

    from environment_servers.audiomc_sim.collect import collect

    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "test", "graph": {}}))
    (tmp_path / "tasks.jsonl").write_text(json.dumps({"task_input": {"axis_id": "voice_editing"}}) + "\n")
    episodes = tmp_path / "artifacts" / "episodes"
    episodes.mkdir(parents=True)
    (episodes / "audiomc-test-0000.json").write_text(
        json.dumps(
            {
                "outcome": "teacher_correction_exhausted",
                "training_example": {},
                "turn_evaluations": [{}, {}],
                "teacher_attempts": [{}],
                "metadata": {
                    "episode_id": {"rollout_id": "audiomc-test-0000", "attempt": 0},
                    "task_id": {"taskset": "test", "task_id": "task"},
                    "task": {"axis_id": "voice_editing"},
                },
            }
        )
    )
    client = SimpleNamespace(post=AsyncMock())

    @asynccontextmanager
    async def graph(config):
        yield client

    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setattr("environment_servers.audiomc_sim.collect.serve_graph", graph)
    monkeypatch.setattr(
        "environment_servers.audiomc_sim.collect.export_collection",
        lambda root, *, max_audio_files: {"accepted_examples": 1},
    )
    await collect(Namespace(output=tmp_path, env_file=None, max_rollouts=1, target_accepted=1))
    client.post.assert_not_awaited()
    recovered = json.loads((tmp_path / "episode_results.jsonl").read_text())
    assert recovered["result"]["outcome"] == "teacher_correction_exhausted"
    assert recovered["result"]["turns"] == 2
