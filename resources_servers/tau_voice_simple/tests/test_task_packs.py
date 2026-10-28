# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native external-domain environment and verifier integration."""

import os

import pytest


@pytest.mark.parametrize(
    "pack,domain",
    [
        ("eva", "airline"),
        ("eva", "itsm"),
        ("eva", "medical_hr"),
        ("mivas", "healthcare"),
        ("mivas", "legal"),
        ("mivas", "customer-support"),
    ],
)
def test_external_pack_native_episode_and_verifier(pack, domain, tmp_path, monkeypatch):
    if not os.environ.get(f"TAU_VOICE_SIMPLE_{pack.upper()}_ROOT"):
        pytest.skip("Configure pinned task-pack source roots for native source checks")
    from litellm import ModelResponse
    from tau2.data_model.message import AssistantMessage
    from tau2.utils import llm_utils

    from resources_servers.tau_voice_simple.episode import Episode, EpisodeSettings, simulation_json
    from resources_servers.tau_voice_simple.task_packs import TaskPack

    backend = TaskPack(pack, domain, "base")
    task = backend.load_tasks()[0]
    environment = backend.build_environment(task, tmp_path)
    responses = iter(["Hello, I need help.", "###STOP###"])

    def completion(**request):
        return ModelResponse(
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": next(responses)},
                    "finish_reason": "stop",
                }
            ],
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    episode = Episode(
        domain=domain,
        task=task,
        seed=42,
        simulation_id=f"{pack}-{domain}",
        settings=EpisodeSettings("t2t", 100, 10, 1200, "openai/test", {}),
        tts_client=None,
        environment=environment,
        pack_backend=backend,
        artifact_dir=tmp_path,
    )
    try:
        assert episode.start()["observation"]["type"] == "user_text"
        assert episode.submit(AssistantMessage.text(content="Thank you."))["done"]
        simulation = episode.finalize({})
        (tmp_path / "simulation.json").write_bytes(simulation_json(simulation))
        assert (tmp_path / "task_pack_evidence.json").is_file()
        score = episode.evaluate()
        assert score["reward"] in {0, 1} and score["components"]
        assert not episode.user.audio_turns
    finally:
        episode.close()
