# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tau half-duplex user simulator that speaks its turns through TTS.

Ported from VLMEvalKitMcore's ``tau_voice_simple/user.py``: Tau's voice user
guidelines, the extra stop rule for tasks with a pending agent write, the STOP
guard, and one TTS clip per spoken turn. Clips stay in memory for the episode
instead of being written next to the trajectory.
"""

import logging
import re
import time

from tau2.data_model.audio import AudioEncoding, AudioFormat, audio_bytes_to_string
from tau2.user.user_simulator import UserSimulator, get_global_user_sim_guidelines_voice

from resources_servers.tau_voice_simple.tts import TTSClient


LOG = logging.getLogger(__name__)
STOP_TOKEN = "###STOP###"
_ACTION_VERBS = (
    r"process|cancel|change|update|modify|exchange|return|refund|book|add|remove|send|issue|run|"
    r"set|switch|enable|disable|grant|refuel|reset|toggle"
)
_PENDING_ACTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        rf"\bplease\s+(?:go\s+ahead(?:\s+(?:and|with))?|proceed(?:\s+with)?|do\s+(?:it|that)|{_ACTION_VERBS})\b",
        r"\b(?:go\s+ahead|proceed)\s+(?:and|with)\b",
        rf"\bjust\s+(?:do\s+(?:it|that)|go\s+ahead|proceed|{_ACTION_VERBS})\b",
        r"\b(?:yes|yeah|yep|sure|okay|ok)\b[\s,;:—–-]{0,20}"
        r"(?:please\s+)?(?:cancel|change|update|modify|exchange|return|refund|book|add|remove|send|issue|run|"
        r"set|switch|enable|disable|grant|refuel|reset|toggle)\b",
    )
)
STOP_GUARD_GUIDELINE = (
    "\n\n## Tau-Voice Simple Stop Rule\n"
    "Never put ###STOP### in the same turn where you approve, confirm, select, "
    "or request an action for the agent to perform. Send that spoken response "
    "without ###STOP###, wait for the agent to confirm the action is complete, "
    "and only then end the call if every scenario goal is satisfied."
)


def suppress_stop_after_pending_action(content: str | None, required_agent_write_names=()) -> str | None:
    """Remove STOP when the same user turn requests an action the agent has not done yet."""
    if not required_agent_write_names or not content or STOP_TOKEN not in content:
        return content
    spoken_content = content.replace(STOP_TOKEN, "").strip()
    if not spoken_content:
        return content
    if any(pattern.search(spoken_content) for pattern in _PENDING_ACTION_PATTERNS):
        return spoken_content
    return content


class VoiceUserSimulator(UserSimulator):
    """Generate user text with Tau's user LLM and, in s2t mode, speak it."""

    def __init__(self, *args, tts_client: TTSClient | None = None, required_agent_write_names=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.tts_client = tts_client
        self.required_agent_write_names = tuple(sorted(set(required_agent_write_names)))
        # (turn index, WAV bytes) for every synthesized turn, in order.
        self.audio_turns: list[tuple[int, bytes]] = []
        self.tts_wait_seconds = 0.0

    @property
    def global_simulation_guidelines(self) -> str:
        guidelines = get_global_user_sim_guidelines_voice(use_tools=self.tools is not None)
        if self.required_agent_write_names:
            guidelines += STOP_GUARD_GUIDELINE
        return guidelines

    def _generate_next_message(self, message, state):
        user_message = super()._generate_next_message(message, state)
        if user_message.is_tool_call() and (user_message.content or "").strip():
            # The user LLM (gpt-5.6-luna through the hub, since 2026-10-26) sometimes returns a spoken preamble together
            # with its phone tool call ("Okay, I'll check that now." + check_network_status()). tau2's orchestrator treats a
            # user message with both text and tool calls as a communication error and ends the episode as user_error with
            # reward 0, which killed about half of the telecom episodes (the longer troubleshooting ones). Keep the tool
            # call as a tool-only turn, drop the text (the user speaks again after the tool result) and record it.
            dropped = user_message.content
            user_message.content = None
            user_message.raw_data = dict(user_message.raw_data or {}) | {
                "tau_voice_simple_text_with_tool_call": {"dropped_text": dropped}
            }
            LOG.warning("Tau-Voice Simple: user turn carried text with a tool call; text dropped, tool call kept")
        original_content = user_message.content
        if not user_message.is_tool_call():
            user_message.content = suppress_stop_after_pending_action(
                original_content, self.required_agent_write_names
            )
        if user_message.content != original_content:
            user_message.raw_data = dict(user_message.raw_data or {}) | {
                "tau_voice_simple_stop_guard": {
                    "reason": "pending_agent_action",
                    "original_content": original_content,
                    "reference_agent_write_names": self.required_agent_write_names,
                }
            }
            LOG.warning("Tau-Voice Simple suppressed a premature user STOP after an action request")
        if self.tts_client is None or user_message.is_tool_call() or self.is_stop(user_message):
            return user_message
        if not (user_message.content or "").strip():
            # The user LLM occasionally returns an empty turn; TTS refuses empty text and the ValueError would turn into a
            # resources-server 500 that takes the whole Gym client down. Pass the turn through as a text turn (as the t2t
            # mode would) and record it, so the episode continues and the event stays visible in the trajectory.
            user_message.raw_data = dict(user_message.raw_data or {}) | {
                "tau_voice_simple_empty_user_turn": {"tts_skipped": True}
            }
            LOG.warning("Tau-Voice Simple: empty user turn, TTS skipped (text turn)")
            return user_message

        started = time.perf_counter()
        try:
            synthesis = self.tts_client.synthesize(user_message.content or "")
        finally:
            self.tts_wait_seconds += time.perf_counter() - started
        self.audio_turns.append((len(self.audio_turns), synthesis.wav_bytes()))
        user_message.audio_content = audio_bytes_to_string(synthesis.pcm_s16le)
        user_message.audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=synthesis.sample_rate, channels=1
        )
        user_message.audio_script_gold = user_message.content
        user_message.is_audio = True
        user_message.raw_data = dict(user_message.raw_data or {}) | {
            "gpt4o_mini_tts": {"model": synthesis.model, "voice": synthesis.voice}
        }
        return user_message
