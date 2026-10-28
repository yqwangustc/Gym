# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.tau_voice_simple_agent.app import (
    _deliver,
    _record_reply,
    _sampling,
    _split_transcript,
)
from responses_api_agents.tau_voice_simple_agent.protocol import (
    USER_TURN_INSTRUCTION,
    ProtocolError,
    adjusted_output_limit,
    build_system_prompt,
    context_length_failure,
    parse_turn,
    strip_reasoning,
)


TOOLS = {"get_user_details", "cancel_reservation"}


def completion(content=None, tool_calls=None, reasoning=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message}]}


def call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class TestParseTurn:
    def test_spoken_reply_strips_reasoning(self) -> None:
        turn = parse_turn(completion("<think>check policy</think> Could you spell your name?"), TOOLS)
        assert turn == {"content": "Could you spell your name?", "reasoning_content": "check policy"}

    def test_single_tool_call(self) -> None:
        turn = parse_turn(completion("", [call("get_user_details", '{"user_id": "a_1"}')]), TOOLS)
        assert turn["tool_call"] == {"id": "call_1", "name": "get_user_details", "arguments": {"user_id": "a_1"}}

    @pytest.mark.parametrize(
        ("output", "category"),
        [
            ({"choices": []}, "no_completion_choices"),
            (
                completion(None, [call("get_user_details", "{}"), call("get_user_details", "{}", "c2")]),
                "multiple_tool_calls",
            ),
            (completion("Sure.", [call("get_user_details", "{}")]), "mixed_text_and_tool_call"),
            (completion(None, [call("get_user_details", "{bad")]), "malformed_tool_arguments"),
            (completion(None, [call("run_speed_test", "{}")]), "unknown_tool"),
            (completion(None, [call("get_user_details", "[1]")]), "non_object_tool_arguments"),
            (completion('<tool_call>{"name": "x"}</tool_call>'), "textual_tool_call"),
            (completion("<think>only thinking</think>"), "empty_visible_content"),
            (completion("Hello <think>"), "incomplete_reasoning_fragment"),
        ],
    )
    def test_rejections_follow_reference_categories(self, output, category) -> None:
        with pytest.raises(ProtocolError) as error:
            parse_turn(output, TOOLS)
        assert error.value.category == category

    def test_stray_closing_think_keeps_text_after_it(self) -> None:
        assert strip_reasoning("reasoning without opening</think> Hello there.") == "Hello there."


class TestContextHandling:
    CHARACTERS = (
        "This model's maximum context length is 16384 tokens. However, you requested 16384 output tokens "
        "and your prompt contains 52000 characters (more than 0 characters)."
    )
    TOKENS = (
        "This model's maximum context length is 16384 tokens. However, you requested 4000 output tokens "
        "and your prompt contains at least 15000 input tokens, for a total of at least 19000 tokens."
    )

    def test_character_bound_retries_without_limit_when_output_fills_context(self) -> None:
        failure = context_length_failure(400, self.CHARACTERS)
        assert failure["max_context_tokens"] == 16384
        retry, max_tokens, adjustment = adjusted_output_limit(failure, 16384)
        assert retry and max_tokens is None
        assert adjustment["output_limit_source"] == "server_remaining_context"

    def test_character_bound_with_smaller_limit_means_history_is_too_long(self) -> None:
        failure = context_length_failure(400, self.CHARACTERS)
        assert adjusted_output_limit(failure, 4096)[0] is False

    def test_token_lower_bound_delegates_remaining_budget_to_server(self) -> None:
        failure = context_length_failure(400, self.TOKENS)
        retry, max_tokens, adjustment = adjusted_output_limit(failure, 4000)
        assert retry and max_tokens is None
        assert adjustment["output_limit_source"] == "server_remaining_context"
        exact = {"input_tokens": 15000, "max_context_tokens": 16384}
        assert adjusted_output_limit(exact, 4000)[1] == 16384 - 15000 - 128

    def test_other_errors_are_not_context_failures(self) -> None:
        assert context_length_failure(500, self.TOKENS) is None
        assert context_length_failure(400, "invalid tool schema") is None


def test_system_prompt_wraps_domain_policy() -> None:
    prompt = build_system_prompt("POLICY TEXT")
    assert prompt.startswith("You are a customer service agent on a live voice call.")
    assert prompt.endswith("<domain_policy>\nPOLICY TEXT\n</domain_policy>")


def test_history_keeps_audio_and_drops_reasoning() -> None:
    messages, transcript = [], []
    _deliver({"type": "user_audio", "turn_index": 0, "audio_wav_base64": "UklGRg=="}, messages, transcript)
    assert messages[-1]["content"][0] == {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}
    assert messages[-1]["content"][1] == {"type": "text", "text": USER_TURN_INSTRUCTION}
    _record_reply({"content": "Hello.", "raw_data": {"reasoning_content": "hidden"}}, messages, transcript)
    assert messages[-1] == {"role": "assistant", "content": "Hello."}
    _record_reply(
        {"tool_call": {"id": "c1", "name": "get_user_details", "arguments": {"user_id": "é"}}}, messages, transcript
    )
    assert json.loads(messages[-1]["tool_calls"][0]["function"]["arguments"]) == {"user_id": "é"}
    _deliver({"type": "tool_results", "results": [{"tool_call_id": "c1", "content": "{}"}]}, messages, transcript)
    assert messages[-1] == {"role": "tool", "tool_call_id": "c1", "content": "{}"}


def test_transcript_output_is_the_final_replies() -> None:
    transcript = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "[user audio turn 0000]"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        {"role": "assistant", "content": "Done."},
    ]
    inputs, outputs = _split_transcript(transcript)
    assert [item["type"] for item in inputs] == ["message", "message", "function_call", "function_call_output"]
    assert outputs[0]["content"][0]["text"] == "Done." and len(outputs) == 1


def test_sampling_maps_row_settings() -> None:
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[], temperature=1.0, top_p=0.95, max_output_tokens=16384, metadata={"extra_body": '{"top_k": 1}'}
    )
    assert _sampling(params) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16384,
        "metadata": {"extra_body": '{"top_k": 1}'},
    }
