# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample the complete persona corpus reproducibly while retaining provenance."""

import json

import pytest
import yaml

from resources_servers.audiomc_sim.prepare import prepare_tasks


def test_materialization_retains_uuid_and_only_general_persona(tmp_path):
    personas = tmp_path / "personas.jsonl"
    personas.write_text(
        "".join(
            json.dumps(
                {"uuid": f"uuid-{i}", "persona": f"Persona é {i}", "professional_persona": "DO NOT USE"},
                ensure_ascii=False,
            )
            + "\n"
            for i in range(3)
        )
    )
    axes = tmp_path / "axes.yaml"
    axes.write_text(
        yaml.safe_dump(
            {
                "axes": {
                    "voice_editing": {
                        "category": "Voice Editing",
                        "definition": "Apply edits",
                        "pass_criteria": "Correct",
                        "fail_criteria": "Wrong",
                        "few_shot_examples": [],
                    }
                }
            }
        )
    )
    topics = tmp_path / "topics.yaml"
    topics.write_text(yaml.safe_dump({"category": {"voice_editing": {"Shopping": {"List edits": "Update the list"}}}}))
    kwargs = dict(
        persona_path=personas,
        axis_path=axes,
        topics_path=topics,
        assistant_prompt="Instructions",
        count=20,
        seed=42,
        max_turns=4,
        max_teacher_trials=2,
        tts_voices=["alloy", "nova", "shimmer"],
    )
    first = prepare_tasks(**kwargs)
    assert first == prepare_tasks(**kwargs)
    assert {row["task_input"]["tts_voice"] for row in first} == {"alloy", "nova", "shimmer"}
    single_voice = prepare_tasks(**(kwargs | {"tts_voices": ["nova"]}))
    assert all(row["task_input"]["tts_voice"] == "nova" for row in single_voice)
    # Voice sampling does not perturb persona/topic sampling.
    for original, changed in zip(first, single_voice):
        assert {k: v for k, v in original["task_input"].items() if k != "tts_voice"} == {
            k: v for k, v in changed["task_input"].items() if k != "tts_voice"
        }
    for invalid in ([], [""], ["   "]):
        with pytest.raises(ValueError, match="nonempty"):
            prepare_tasks(**(kwargs | {"tts_voices": invalid}))
    assert {row["task_input"]["persona_uuid"] for row in first} == {"uuid-0", "uuid-1", "uuid-2"}
    assert "DO NOT USE" not in json.dumps(first)
    for row in first:
        task = row["task_input"]
        assert task["persona"] == f"Persona é {task['persona_uuid'].split('-')[-1]}"
        assert task["max_turns"] == 4 and task["max_teacher_trials"] == 2
        assert task["axis_id"] == "voice_editing"
        assert task["subtopic"] == "List edits"
        assert len(task["source_hashes"]["personas"]) == 64
    with pytest.raises(ValueError, match="positive"):
        prepare_tasks(**(kwargs | {"count": 0}))
