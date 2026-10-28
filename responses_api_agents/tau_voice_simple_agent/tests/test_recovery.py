# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Candidate prompt and bounded context recovery."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch


def test_native_agent_custom_prompt_and_zero_context_is_episode_local():
    from responses_api_agents.tau_voice_simple_agent.app import (
        TauVoiceSimpleAgent,
        _ModelCallError,
        _system_prompt,
    )

    assert _system_prompt("MANDATORY RULE", "VOICE STYLE") == (
        "VOICE STYLE\n\n<domain_policy>\nMANDATORY RULE\n</domain_policy>"
    )

    class Candidate:
        config = SimpleNamespace(protocol_retries=2)
        _complete = TauVoiceSimpleAgent._complete
        calls = 0

        async def _chat(self, request, cookies):
            self.calls += 1
            if self.calls == 1:
                raise _ModelCallError(
                    400,
                    "This model's maximum context length is 16384 tokens. "
                    "However, you requested 27384 tokens (11000 in the messages, "
                    "16384 in the completion).",
                )
            raise _ModelCallError(400, "max_tokens must be at least 1, got 0")

    candidate = Candidate()
    with patch(
        "responses_api_agents.tau_voice_simple_agent.app.context_length_failure",
        return_value={
            "input_tokens": 11000,
            "max_context_tokens": 16384,
            "input_tokens_lower_bound": True,
        },
    ):
        turn, _ = asyncio.run(
            TauVoiceSimpleAgent._next_turn(candidate, [], [], set(), {"max_tokens": 16384}, 42, None)
        )
    assert candidate.calls == 2
    assert turn["raw_data"]["tau_voice_simple_context_exhausted"]["remaining_output_tokens"] == 0


def test_context_lower_bound_is_delegated_to_server():
    from responses_api_agents.tau_voice_simple_agent.protocol import adjusted_output_limit

    failure = {"input_tokens": 11000, "max_context_tokens": 16384, "input_tokens_lower_bound": True}
    retry, limit, info = adjusted_output_limit(failure, 16384)
    assert retry and limit is None and info["output_limit_source"] == "server_remaining_context"
    assert adjusted_output_limit({"input_tokens": 16380, "max_context_tokens": 16384}, 8192)[0] is False
