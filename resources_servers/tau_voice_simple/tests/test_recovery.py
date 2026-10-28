# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tau TTS transport recovery and saved inference evidence."""

import io
import json
import urllib.error
from types import SimpleNamespace

import pytest


def test_retryable_tts_preserves_request_and_redacts_auth(monkeypatch):
    from resources_servers.tau_voice_simple import tts

    wav = tts.pcm16_wav_bytes(b"\0\0" * 20, 24000)
    requests = []
    errors = [
        ConnectionResetError(),
        b"",
        urllib.error.HTTPError(
            "https://example.invalid",
            401,
            "busy",
            {},
            io.BytesIO(b"Error querying the database: too many clients already"),
        ),
    ]

    def urlopen(request, **kwargs):
        requests.append(request.data)
        if errors:
            item = errors.pop(0)
            if isinstance(item, Exception):
                raise item
            return io.BytesIO(item)
        return io.BytesIO(wav)

    monkeypatch.setattr(tts, "sleep", lambda delay: None)
    config = tts.TTSConfig("https://example.invalid", "private-key", "mini", "alloy")
    result = tts.TTSClient(config, urlopen=urlopen).synthesize("Hello")
    assert result.pcm_s16le == b"\0\0" * 20
    assert len(requests) == 4 and len(set(requests)) == 1
    assert "private-key" not in repr(config)
    unauthorized = urllib.error.HTTPError("unused", 401, "auth", {}, io.BytesIO(b"private-key"))
    with pytest.raises(RuntimeError, match="<redacted>"):
        tts.TTSClient(config, urlopen=lambda *a, **k: (_ for _ in ()).throw(unauthorized)).synthesize("Hi")


def test_judge_outage_keeps_inference_evidence(tmp_path, monkeypatch):
    from litellm import ModelResponse
    from tau2.data_model.message import AssistantMessage
    from tau2.utils import llm_utils
    from tau2.utils.utils import DATA_DIR

    from resources_servers.tau_voice_simple import app, source
    from resources_servers.tau_voice_simple import episode as native
    from resources_servers.tau_voice_simple.tts import pcm16_wav_bytes

    responses = iter(["Hello, I need help.", "###STOP###"])

    def completion(**request):
        return ModelResponse(
            model="openai/test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": next(responses)},
                    "finish_reason": "stop",
                }
            ],
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    task = next(iter(native.load_tasks("retail", "base").values()))
    tts = SimpleNamespace(
        config=SimpleNamespace(model="fixture"),
        synthesize=lambda text: SimpleNamespace(
            pcm_s16le=b"\0\0" * 240,
            sample_rate=24000,
            model="fixture",
            voice="alloy",
            wav_bytes=lambda: pcm16_wav_bytes(b"\0\0" * 240, 24000),
        ),
    )
    episode = native.Episode(
        domain="retail",
        task=task,
        seed=42,
        simulation_id="outage-test",
        settings=native.EpisodeSettings("s2t", 100, 10, 1200, "openai/test", {}),
        tts_client=tts,
        artifact_dir=tmp_path,
    )
    episode.start()
    episode.submit(AssistantMessage.text(content="Thanks.", raw_data={"reasoning_content": "trace"}))
    session = SimpleNamespace(
        episode=episode,
        metadata=app.TauVoiceSimpleMetadata(
            id="outage", domain="retail", task_id=str(task.id), seed=42, tts_voice="alloy"
        ),
    )
    server = SimpleNamespace(
        config=SimpleNamespace(interaction_mode="s2t", task_pack="tau"),
        _episode_module=native,
        _archive=None,
        _tau_data_identity=source.data_identity(DATA_DIR),
    )
    monkeypatch.setattr(episode, "evaluate", lambda: (_ for _ in ()).throw(RuntimeError("judge outage")))
    with pytest.raises(RuntimeError, match="judge outage"):
        app.TauVoiceSimpleServer._finish(server, session)
    assert (tmp_path / "user_turn_0000.wav").read_bytes().startswith(b"RIFF")
    original = (tmp_path / "simulation.json").read_bytes()
    saved = json.loads(original)
    assert saved["messages"] and (tmp_path / "episode.json").is_file()
