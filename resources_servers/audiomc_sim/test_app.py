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
from resources_servers.audiomc_sim.common import Prompts, assistant_context, digest, prompt_values
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
        teacher_attempts=[Candidate(text="Monday.", judgment=passed, turn_index=0)],
    )
    result = await resources.verify(request, body)
    record = json.loads(Path(result.artifact_path).read_text())
    assert record["training_example"]["turn_evaluations"][-1]["accepted_response"]["text"] == "Monday."
    assert "assistant_instructions" in record["training_example"]["metadata"]["task"]
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


def test_teacher_prompt_contains_only_student_visible_instructions_and_history(task):
    task.developer_instructions = "Use concise spoken sentences."
    history = [
        Message(turn_id="u0", role="user", text="Use {{this}} literally."),
        Message(turn_id="a0", role="assistant", text="Earlier accepted reply."),
        Message(turn_id="u1", role="user", text="Continue from there."),
    ]
    context = assistant_context(task, history)
    assert set(context) == {"assistant_instructions", "conversation_history"}
    prompts = Prompts(Path("environment_servers/audiomc_sim/prompts"), ("teacher",))
    request = prompts.request("teacher", context)
    assert request.input[1].content == (
        "## Assistant instructions\nBe helpful.\nDeveloper instructions:\nUse concise spoken sentences."
        "\n\n## Conversation history through the latest user utterance\n"
        "[user, u0]: Use {{this}} literally.\n[assistant, a0]: Earlier accepted reply."
        "\n[user, u1]: Continue from there."
    )


@pytest.mark.parametrize("passed", [True, False, None])
def test_teacher_candidate_keeps_rationale_regardless_of_grade(passed):
    verdict = (
        Judgment(grades=[], passed=None)
        if passed is None
        else Judgment(grades=[Grade(rubric_id="r", passed=passed, explanation="Grade")], passed=passed)
    )
    candidate = Candidate(text="Answer", judgment=verdict, turn_index=0, thinking="Solution rationale")
    assert candidate.thinking == "Solution rationale"


@pytest.mark.parametrize("thinking", ["", "   "])
def test_thinking_annotation_must_be_nonblank(thinking):
    passed = Judgment(grades=[Grade(rubric_id="r", passed=True, explanation="Passes")], passed=True)
    assert Candidate(text="Student answer", judgment=passed, turn_index=0).thinking is None
    with pytest.raises(ValidationError):
        Candidate(text="Accepted answer", judgment=passed, turn_index=0, thinking=thinking)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["student", "teacher", "exhausted", "ungradable"])
async def test_full_trajectory_repairs_provenance_and_export(resources, task, monkeypatch, ending):
    import hashlib
    import io
    import tarfile

    from pydantic import TypeAdapter

    from resources_servers.audiomc_sim.contracts import SavedTrainingExample
    from resources_servers.audiomc_sim.export import export_collection

    root = Path(resources.config.artifact_root)
    resources.config.artifact_root = str(root / "artifacts")
    Path(resources.config.artifact_root).mkdir()
    task = task.model_copy(update={"keep_generation_until_max_turn_reached": True, "max_turns": 3})
    request, _ = await seed(resources, task)
    monkeypatch.setattr("resources_servers.audiomc_sim.app.infer", AsyncMock(return_value=builder_json()))
    session = resources._sessions["session"]
    history, turns, teachers = [], [], []
    for i in range(3):
        history.append(Message(turn_id=f"u{i}", role="user", text=f"Request {i}."))
        rubrics = await resources.build_rubrics(
            request, StepRequest(resources_session_id="session", turn_index=i, history=history)
        )

        def judgment(passed):
            return Judgment(
                passed=passed,
                grades=[]
                if passed is None
                else [
                    Grade(rubric_id=rubrics[0].rubric_id, passed=passed, explanation="Checked against frozen prefix.")
                ],
            )

        data = f"wave {i}".encode()
        archive = Path(resources.config.artifact_root) / "audio" / f"turn-{i}.tar"
        archive.parent.mkdir(exist_ok=True)
        member = f"episode/user_turn_{i:04d}.wav"
        with tarfile.open(archive, "w") as tar:
            entry = tarfile.TarInfo(member)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
        audio = AudioRef(
            archive=f"audio/turn-{i}.tar",
            member=member,
            sha256=hashlib.sha256(data).hexdigest(),
            voice="alloy",
            model="tts",
            sample_rate=24000,
            duration_seconds=1,
        )
        session.audio[i] = (history[-1].text, audio)
        original_passed = i == 1 or (i == 2 and ending == "student")
        original = f"Original response {i}."
        original_grade = judgment(original_passed)
        session.judgments[(i, digest(original))] = original_grade
        accepted = Candidate(text=original, judgment=original_grade, turn_index=i) if original_passed else None
        teacher_generated = False
        if not original_passed:
            verdicts = (
                [False, True] if i == 0 or ending == "teacher" else [False, False] if ending == "exhausted" else [None]
            )
            for trial, passed in enumerate(verdicts):
                candidate = Candidate(
                    text=f"Teacher response {i} trial {trial}.",
                    judgment=judgment(passed),
                    turn_index=i,
                    thinking=f"Visible-context rationale for turn {i}.",
                )
                teachers.append(candidate)
                session.judgments[(i, digest(candidate.text))] = candidate.judgment
                if passed:
                    accepted = candidate
                    teacher_generated = True
        turns.append(
            TurnEvaluation(
                turn_index=i,
                user_turn_id=f"u{i}",
                user_text=history[-1].text,
                audio=audio,
                rubrics=rubrics,
                policy_response=original,
                judgment=original_grade,
                accepted_response=accepted,
                generated_by_teacher=teacher_generated,
            )
        )
        if i < 2:
            history.append(Message(turn_id=f"a{i}", role="assistant", text=accepted.text))
    outcome = (
        "accepted_correction"
        if ending in {"student", "teacher"}
        else ("teacher_correction_exhausted" if ending == "exhausted" else "ungradable")
    )
    body = FinalizeRequest(
        resources_session_id="session",
        outcome=outcome,
        input_prefix=history,
        turn_evaluations=turns,
        teacher_attempts=teachers,
    )
    forged = body.model_copy(deep=True)
    forged.turn_evaluations[0].generated_by_teacher = False
    with pytest.raises(HTTPException, match="provenance"):
        await resources.verify(request, forged)
    forged = body.model_copy(deep=True)
    forged.turn_evaluations[0].accepted_response = forged.turn_evaluations[0].accepted_response.model_copy(
        update={"text": "An ungraded replacement."}
    )
    with pytest.raises(HTTPException, match="verified candidate"):
        await resources.verify(request, forged)
    forged = body.model_copy(deep=True)
    forged.turn_evaluations[0].accepted_response = forged.turn_evaluations[0].accepted_response.model_copy(
        update={"thinking": "Unmatched rationale."}
    )
    with pytest.raises(HTTPException, match="verified candidate"):
        await resources.verify(request, forged)
    if outcome == "accepted_correction":
        premature = body.model_copy(deep=True)
        premature.turn_evaluations = premature.turn_evaluations[:1]
        premature.teacher_attempts = [c for c in premature.teacher_attempts if c.turn_index == 0]
        premature.input_prefix = [Message.model_validate(m) for m in session.prefixes[0]]
        with pytest.raises(HTTPException, match="max_turns"):
            await resources.verify(request, premature)
    result = await resources.verify(request, body)
    assert await resources.verify(request, body) == result
    record = json.loads(Path(result.artifact_path).read_text())
    assert result.turns == 3 and result.teacher_trials == len(teachers)
    assert record["turn_evaluations"][0]["policy_response"] == "Original response 0."
    assert record["turn_evaluations"][0]["generated_by_teacher"] is True
    assert record["turn_evaluations"][1]["generated_by_teacher"] is False
    assert record["conversation"][1]["text"] == "Teacher response 0 trial 1."
    assert record["conversation"][1]["generated_by_teacher"] is True
    assert record["conversation"][1]["thinking"] == "Visible-context rationale for turn 0."
    assert record["conversation"][3]["generated_by_teacher"] is False
    assert record["conversation"][3]["thinking"] is None
    summary = export_collection(root, max_audio_files=1)
    if outcome == "accepted_correction":
        assert summary["accepted_examples"] == 1 and summary["accepted_turn_depths"] == {"3": 1}
        example = json.loads((root / "rollout.jsonl").read_text())
        TypeAdapter(SavedTrainingExample).validate_python(example)
        assert example["schema_version"] == "1.7"
        assert len(example["conversation"]) == 6
        teacher_replies = [t for t in example["conversation"] if t.get("generated_by_teacher")]
        expected_indices = [0, 2] if ending == "teacher" else [0]
        assert [t["turn_id"] for t in teacher_replies] == [f"a{i}" for i in expected_indices]
        for i in expected_indices:
            accepted = example["turn_evaluations"][i]["accepted_response"]
            reply = example["conversation"][2 * i + 1]
            assert reply["text"] == accepted["text"]
            assert reply["thinking"] == accepted["thinking"] == f"Visible-context rationale for turn {i}."
        for i in range(3):
            assert example["conversation"][2 * i]["audio"] == example["turn_evaluations"][i]["audio"]
        expected_index = expected_indices[-1]
        assert example["conversation"][-1]["generated_by_teacher"] is (ending == "teacher")
        assert len(example["rejected_teacher_attempts"]) == 1
        assert all(c["turn_index"] == expected_index for c in example["rejected_teacher_attempts"])
        assert len(example["teacher_attempts"]) == len(teachers)
    else:
        assert result.accepted is True
        assert summary["accepted_examples"] == 1
        assert summary["accepted_turn_depths"] == {"2": 1}
        example = json.loads((root / "rollout.jsonl").read_text())
        TypeAdapter(SavedTrainingExample).validate_python(example)
        assert len(example["conversation"]) == 4
        assert len(example["turn_evaluations"]) == 2
        assert example["conversation"][1]["generated_by_teacher"] is True
        assert example["metadata"]["termination_outcome"] == outcome
        assert all(c["turn_index"] < 2 for c in example["teacher_attempts"])
        assert len(record["turn_evaluations"]) == 3
        assert json.loads((root / "inspection.jsonl").read_text())["outcome"] == outcome
        # Recover the same verified prefix from a frozen run without its training projection.
        record.pop("training_example")
        Path(result.artifact_path).write_text(json.dumps(record))
        export_collection(root, max_audio_files=1)
        assert json.loads((root / "rollout.jsonl").read_text()) == example
