# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent endpoint routing, environment interpolation and credential-safe snapshots."""

import json

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from environment_servers.audiomc_sim.app import AudioMCSimEnvironmentConfig
from environment_servers.audiomc_sim.backends import TEXT_ROLES, load_backends
from environment_servers.audiomc_sim.collect import graph_config
from resources_servers.audiomc_sim.app import AudioMCSimConfig


def write_config(tmp_path):
    config = {role: {"api_base": "http://localhost:8000/v1", "model_name": "local"} for role in (*TEXT_ROLES, "tts")}
    path = tmp_path / "backends.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    return config, path


def role_ref(graph, role):
    env = graph["environment"]["environment_servers"]["audiomc_sim"]
    resources = graph["resources"]["resources_servers"]["audiomc_sim"]
    if role in {"planner", "user", "teacher"}:
        return env[f"{role}_model"]
    if role in {"rubric_builder", "critiquer"}:
        return resources[f"{role}_model"]
    return graph["assistant"]["responses_api_agents"]["simple_agent"]["model_server"]


@pytest.mark.parametrize("role", TEXT_ROLES)
def test_each_role_can_move_independently_between_local_and_api(tmp_path, monkeypatch, role):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config, path = write_config(tmp_path)
    config[role] = {
        "api_base": "${oc.env:ROLE_API_BASE}",
        "model_name": "${oc.env:ROLE_MODEL}",
        "api_key": "${oc.env:ROLE_API_KEY}",
        "uses_reasoning_parser": False,
    }
    OmegaConf.save(OmegaConf.create(config), path)
    monkeypatch.setenv("ROLE_API_BASE", "https://remote.invalid/v1/")
    monkeypatch.setenv("ROLE_MODEL", "remote-model")
    monkeypatch.setenv("ROLE_API_KEY", "test-secret-never-persisted")
    backends = load_backends(path)
    graph = graph_config(output=tmp_path, base_port=19000, backends=backends)
    for candidate in TEXT_ROLES:
        model = graph[role_ref(graph, candidate)["name"]]["responses_api_models"]["vllm_model"]
        assert model["base_url"] == ("https://remote.invalid/v1" if candidate == role else "http://localhost:8000/v1")
        assert model["model"] == ("remote-model" if candidate == role else "local")
        assert model["uses_reasoning_parser"] is (candidate != role)
        if candidate == role:
            assert model["api_key"] == "${oc.env:ROLE_API_KEY}"
    assert "test-secret-never-persisted" not in json.dumps(graph)
    assert "test-secret-never-persisted" not in json.dumps(backends)
    resolved = OmegaConf.to_container(OmegaConf.create(graph), resolve=True)
    selected = role_ref(graph, role)["name"]
    assert resolved[selected]["responses_api_models"]["vllm_model"]["api_key"] == "test-secret-never-persisted"


def test_default_sharing_and_distinct_backend_ports(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    graph = graph_config(output=tmp_path, base_port=19000)
    assert len(graph) == 7
    assert role_ref(graph, "planner") == role_ref(graph, "rubric_builder") == role_ref(graph, "critiquer")
    assert role_ref(graph, "user") != role_ref(graph, "teacher")
    teacher = graph[role_ref(graph, "teacher")["name"]]["responses_api_models"]["vllm_model"]
    assert teacher["sampling_overrides"] == {"reasoning_effort": "minimal"}
    config, path = write_config(tmp_path)
    for role in TEXT_ROLES:
        config[role]["model_name"] = role
    OmegaConf.save(OmegaConf.create(config), path)
    graph = graph_config(output=tmp_path, base_port=19000, backends=load_backends(path))
    assert len(graph) == 9
    ports = [next(iter(next(iter(entry.values())).values()))["port"] for entry in graph.values()]
    assert ports == list(range(19000, 19009))
    env = graph["environment"]["environment_servers"]["audiomc_sim"]
    resources = graph["resources"]["resources_servers"]["audiomc_sim"]
    assert AudioMCSimEnvironmentConfig.model_validate(env).model_for("teacher").name == "teacher_model"
    assert AudioMCSimConfig.model_validate(resources).model_for("critiquer").name == "critiquer_model"


def test_local_only_config_needs_no_global_credentials_and_configures_tts(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config, path = write_config(tmp_path)
    config["tts"].update(model_name="speech-local", voice="speaker-2")
    config["assistant"]["checkpoint"] = "/models/checkpoint"
    OmegaConf.save(OmegaConf.create(config), path)
    backends = load_backends(path)
    graph = graph_config(output=tmp_path, base_port=19000, backends=backends)
    resolved = OmegaConf.to_container(OmegaConf.create(graph), resolve=True)
    resource = resolved["resources"]["resources_servers"]["audiomc_sim"]
    assert resource["tts_endpoint"] == "http://localhost:8000/v1/audio/speech"
    assert resource["tts_model"] == "speech-local" and resource["tts_voices"] == ["speaker-2"]
    assert resource["tts_api_key"] == "unused"
    assert backends["assistant"]["checkpoint"] == "/models/checkpoint"


@pytest.mark.parametrize("problem", ["unknown_role", "missing_teacher", "wrong_field", "missing_model", "bad_url"])
def test_malformed_backend_settings_fail_early(tmp_path, problem):
    config, path = write_config(tmp_path)
    if problem == "unknown_role":
        config["typo"] = config["user"]
    elif problem == "missing_teacher":
        del config["teacher"]
    elif problem == "wrong_field":
        config["teacher"]["model"] = "typo"
    elif problem == "missing_model":
        del config["teacher"]["model_name"]
    else:
        config["teacher"]["api_base"] = "not-a-url"
    OmegaConf.save(OmegaConf.create(config), path)
    with pytest.raises(ValidationError):
        load_backends(path)


@pytest.mark.parametrize(
    "flag",
    [
        "--api-base",
        "--assistant-base",
        "--planner-model",
        "--user-teacher-model",
        "--assistant-model",
        "--assistant-checkpoint",
    ],
)
def test_collector_rejects_removed_backend_flags(tmp_path, monkeypatch, capsys, flag):
    from environment_servers.audiomc_sim.collect import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "collect",
            "--output",
            str(tmp_path / "output"),
            "--personas",
            str(tmp_path / "personas.jsonl"),
            flag,
            "obsolete-value",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("voices", [[], [""], ["   "], "alloy"])
def test_invalid_voice_pools_rejected(tmp_path, voices):
    config, path = write_config(tmp_path)
    config["tts"]["voices"] = voices
    OmegaConf.save(OmegaConf.create(config), path)
    with pytest.raises(ValidationError):
        load_backends(path)


def test_voice_pool_round_trip_and_legacy_conflict(tmp_path):
    config, path = write_config(tmp_path)
    config["tts"]["voices"] = ["nova", "shimmer"]
    OmegaConf.save(OmegaConf.create(config), path)
    loaded = load_backends(path)
    assert loaded["tts"]["voices"] == ["nova", "shimmer"]
    OmegaConf.save(OmegaConf.create(loaded), path)
    assert load_backends(path) == loaded
    config["tts"]["voice"] = "alloy"
    OmegaConf.save(OmegaConf.create(config), path)
    with pytest.raises(ValidationError, match="not both"):
        load_backends(path)


def test_reasoning_effort_is_role_specific_and_controls_wrapper_sharing(tmp_path):
    config, path = write_config(tmp_path)
    config["teacher"]["reasoning_effort"] = "minimal"
    OmegaConf.save(OmegaConf.create(config), path)
    graph = graph_config(output=tmp_path, base_port=19000, backends=load_backends(path))
    teacher = role_ref(graph, "teacher")["name"]
    user = role_ref(graph, "user")["name"]
    assert teacher != user
    assert graph[teacher]["responses_api_models"]["vllm_model"]["sampling_overrides"] == {
        "reasoning_effort": "minimal"
    }
    assert graph[user]["responses_api_models"]["vllm_model"]["sampling_overrides"] is None
    config["user"]["reasoning_effort"] = "minimal"
    OmegaConf.save(OmegaConf.create(config), path)
    graph = graph_config(output=tmp_path, base_port=19000, backends=load_backends(path))
    assert role_ref(graph, "user") == role_ref(graph, "teacher")


@pytest.mark.parametrize("role", TEXT_ROLES)
def test_role_retry_options_reach_wrapper_without_sharing_different_policies(tmp_path, role):
    from responses_api_models.vllm_model.app import VLLMModelConfig

    config, path = write_config(tmp_path)
    config[role].update(request_attempts=2, request_timeout_seconds=90, retry_backoff={"initial_seconds": 3})
    config["tts"].update(retry_backoff={"initial_seconds": 4})
    OmegaConf.save(OmegaConf.create(config), path)
    graph = graph_config(output=tmp_path, base_port=19000, backends=load_backends(path))
    for candidate in TEXT_ROLES:
        wrapper = graph[role_ref(graph, candidate)["name"]]["responses_api_models"]["vllm_model"]
        parsed = VLLMModelConfig.model_validate(wrapper)
        assert parsed.upstream_request_attempts == (2 if candidate == role else 4)
        assert parsed.upstream_request_timeout_seconds == (90 if candidate == role else 180)
        assert parsed.upstream_retry_backoff.initial_seconds == (3 if candidate == role else 1)
    resources = AudioMCSimConfig.model_validate(graph["resources"]["resources_servers"]["audiomc_sim"])
    assert resources.tts_retry_backoff.initial_seconds == 4
