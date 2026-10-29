# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VoiceUserSimulator turn post-processing."""

from tau2.data_model.message import ToolCall, UserMessage
from tau2.user.user_simulator import UserSimulator

from resources_servers.tau_voice_simple.user import VoiceUserSimulator


def _simulator(monkeypatch, reply):
    monkeypatch.setattr(UserSimulator, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(UserSimulator, "_generate_next_message", lambda self, message, state: reply)
    sim = VoiceUserSimulator(tts_client=None)
    sim.tools = None
    return sim


def test_text_with_tool_call_becomes_a_tool_only_turn(monkeypatch):
    call = ToolCall(id="c1", name="check_network_status", arguments={}, requestor="user")
    reply = UserMessage(role="user", content="Okay, I'll check that now.", tool_calls=[call])
    sim = _simulator(monkeypatch, reply)
    out = sim._generate_next_message(None, None)
    assert out.is_tool_call() and out.tool_calls == [call]
    assert out.content is None
    assert out.raw_data["tau_voice_simple_text_with_tool_call"] == {"dropped_text": "Okay, I'll check that now."}


def test_tool_only_and_text_only_turns_are_unchanged(monkeypatch):
    call = ToolCall(id="c1", name="check_network_status", arguments={}, requestor="user")
    tool_only = UserMessage(role="user", content=None, tool_calls=[call])
    assert _simulator(monkeypatch, tool_only)._generate_next_message(None, None).content is None
    spoken = UserMessage(role="user", content="My data is not working.")
    out = _simulator(monkeypatch, spoken)._generate_next_message(None, None)
    assert out.content == "My data is not working." and not out.is_tool_call()


def test_empty_user_reply_is_sampled_again(monkeypatch):
    replies = iter(
        [
            UserMessage(role="user", content=""),
            UserMessage(role="user", content=None),
            UserMessage(role="user", content="Yes, go ahead."),
        ]
    )
    monkeypatch.setattr(UserSimulator, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(UserSimulator, "_generate_next_message", lambda self, message, state: next(replies))
    sim = VoiceUserSimulator(tts_client=None)
    sim.tools = None
    out = sim._generate_next_message(None, None)
    assert out.content == "Yes, go ahead."
    assert out.raw_data["tau_voice_simple_empty_user_turn"] == {"resampled": 2}
