# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Candidate-side turn protocol for Tau-Voice Simple.

Ported from VLMEvalKitMcore's ``vlmeval/vlm/tau_voice_simple_agent.py``: the
system prompt, the per-turn instruction after each user clip, reasoning
removal, the checks that reject a turn (and their order), the fallback
messages, and recognition of vLLM's context-length errors.
"""

import json
import re
from typing import Any


USER_TURN_INSTRUCTION = (
    "Listen to this user turn and respond using the audio only. Do not assume access to a transcript."
)
PROTOCOL_FALLBACK = "I’m sorry, I couldn’t complete that request. Could you try again?"
CONTEXT_EXHAUSTED_FALLBACK = "I’m sorry, this call has reached its maximum length."

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_CONTENT_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_CONTEXT_LENGTH_RE = re.compile(
    r"Input length \((?P<input_tokens>\d+)\) exceeds model's maximum context length \((?P<max_tokens>\d+)\)",
    re.IGNORECASE,
)
_CONTEXT_BUDGET_RE = re.compile(
    r"maximum context length is (?P<max_tokens>\d+) tokens.*?requested "
    r"(?P<requested_output_tokens>\d+) output tokens.*?prompt contains at least "
    r"(?P<input_tokens>\d+) input tokens.*?total of at least "
    r"(?P<total_tokens>\d+) tokens",
    re.IGNORECASE | re.DOTALL,
)
_CONTEXT_CHARACTER_BUDGET_RE = re.compile(
    r"maximum context length is (?P<max_tokens>\d+) tokens.*?requested "
    r"(?P<requested_output_tokens>\d+) output tokens.*?prompt contains "
    r"(?P<input_characters>\d+) characters",
    re.IGNORECASE | re.DOTALL,
)


def build_system_prompt(domain_policy: str) -> str:
    """The reference candidate system prompt, wrapping Tau's domain policy."""
    return (
        "You are a customer service agent on a live voice call. Every word in your "
        "response will be spoken aloud to the customer by text-to-speech.\n\n"
        "VOICE RESPONSE CONTRACT:\n"
        "- Use at most two short sentences and normally fewer than 35 words. Never "
        "exceed 50 spoken words.\n"
        "- Handle exactly one conversational step at a time: ask one question, give "
        "one small instruction, or state one result. Then stop and wait for the "
        "customer.\n"
        "- If a procedure has multiple steps, give only the single next step. Never "
        "give the whole procedure at once.\n"
        "- If required information is missing, ask only for that information. Do not "
        "also explain what you will do afterward.\n"
        "- Do not use bullets, numbered lists, headings, tables, Markdown, step "
        "numbers, menus of options, or document-style formatting.\n"
        "- Do not narrate your reasoning, plans, tool availability, or uncertainty. "
        "Do not repeat yourself or provide unsolicited background.\n"
        "Before sending a spoken reply, silently check this contract and shorten the "
        "reply if necessary.\n\n"
        "When collecting names, emails, IDs, or other information that may be "
        "unclear in audio, ask the customer to spell it letter by letter. If "
        "authentication fails, explicitly ask the customer to repeat or spell "
        "the relevant information.\n\n"
        "Follow the domain policy exactly. In each turn, either speak to the "
        "customer or make exactly one tool call; never do both. Do not claim a "
        "tool action succeeded until the tool result confirms it.\n\n"
        f"<domain_policy>\n{domain_policy}\n</domain_policy>"
    )


def strip_reasoning(text: str) -> str:
    """Remove hidden-thinking blocks from user-visible model output."""
    text = _THINK_RE.sub("", text or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text


def extract_reasoning(message: dict[str, Any]) -> str | None:
    """Parsed reasoning fields first, then ``<think>`` blocks in the content."""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    content = message.get("content")
    if isinstance(content, str):
        blocks = [block.strip() for block in _THINK_CONTENT_RE.findall(content)]
        if blocks:
            return "\n\n".join(block for block in blocks if block) or None
    return None


class ProtocolError(ValueError):
    """A retryable candidate output that violates the turn protocol."""

    def __init__(self, category: str, detail: str, rejected_output: dict[str, Any] | None = None):
        super().__init__(detail)
        self.category = category
        self.detail = detail
        self.rejected_output = rejected_output

    def as_record(self, attempt: int) -> dict[str, Any]:
        record = {"attempt": attempt + 1, "category": self.category, "detail": self.detail}
        if self.rejected_output is not None:
            record["rejected_output"] = self.rejected_output
        return record


def parse_turn(completion: dict[str, Any], tool_names: set[str]) -> dict[str, Any]:
    """Validate one chat completion as a Tau turn.

    Returns ``{"content": str}`` or ``{"tool_call": {"id", "name", "arguments"}}``
    plus ``"reasoning_content"``; raises ProtocolError in the reference's order.
    """
    choices = completion.get("choices") or []
    if not choices:
        raise ProtocolError("no_completion_choices", "vLLM returned no completion choices")
    message = choices[0].get("message") or {}
    reasoning = extract_reasoning(message)
    tool_calls = list(message.get("tool_calls") or [])
    visible = strip_reasoning(message.get("content") or "")
    rejected = {
        "content": message.get("content"),
        "visible_content": visible or None,
        "reasoning_content": reasoning,
        "tool_calls": [
            {
                "id": call.get("id"),
                "name": (call.get("function") or {}).get("name"),
                "arguments": (call.get("function") or {}).get("arguments"),
            }
            for call in tool_calls
        ],
    }

    if len(tool_calls) > 1:
        raise ProtocolError("multiple_tool_calls", "TauVoiceSimple permits exactly one tool call per turn", rejected)
    if tool_calls:
        if visible:
            raise ProtocolError(
                "mixed_text_and_tool_call",
                "Model returned text and a tool call in one turn",
                rejected,
            )
        function = tool_calls[0].get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ProtocolError(
                    "malformed_tool_arguments",
                    f"Tool arguments are not valid JSON: {error}",
                    rejected,
                ) from error
        if function.get("name") not in tool_names:
            raise ProtocolError("unknown_tool", f"Unknown tool name: {function.get('name')}", rejected)
        if not isinstance(arguments, dict):
            raise ProtocolError("non_object_tool_arguments", "Tool-call arguments must be a JSON object", rejected)
        call_id = tool_calls[0].get("id") or ""
        return {
            "tool_call": {"id": call_id, "name": function["name"], "arguments": arguments},
            "reasoning_content": reasoning,
        }

    if "<tool_call" in visible or "</tool_call>" in visible:
        raise ProtocolError("textual_tool_call", "Model returned an unparsed textual tool call", rejected)
    if not visible:
        raise ProtocolError("empty_visible_content", "Model returned empty user-visible content", rejected)
    if "<think>" in visible or "</think>" in visible:
        raise ProtocolError(
            "incomplete_reasoning_fragment",
            "Model returned an incomplete reasoning fragment",
            rejected,
        )
    return {"content": visible, "reasoning_content": reasoning}


def context_length_failure(status: int, detail: str) -> dict[str, Any] | None:
    """Recognize vLLM's context-limit HTTP 400 without masking other API errors."""
    if status != 400:
        return None
    match = _CONTEXT_LENGTH_RE.search(detail)
    if match is not None:
        return {
            "category": "context_length_exceeded",
            "detail": match.group(0),
            "input_tokens": int(match.group("input_tokens")),
            "max_context_tokens": int(match.group("max_tokens")),
        }
    match = _CONTEXT_BUDGET_RE.search(detail)
    if match is not None:
        return {
            "category": "context_length_exceeded",
            "detail": match.group(0),
            "input_tokens": int(match.group("input_tokens")),
            "max_context_tokens": int(match.group("max_tokens")),
            "requested_output_tokens": int(match.group("requested_output_tokens")),
            "input_tokens_lower_bound": True,
            "requested_total_tokens": int(match.group("total_tokens")),
        }
    match = _CONTEXT_CHARACTER_BUDGET_RE.search(detail)
    if match is not None:
        return {
            "category": "context_length_exceeded",
            "detail": match.group(0),
            "input_characters": int(match.group("input_characters")),
            "max_context_tokens": int(match.group("max_tokens")),
            "requested_output_tokens": int(match.group("requested_output_tokens")),
        }
    return None


def adjusted_output_limit(failure: dict[str, Any], max_tokens: int | None) -> tuple[bool, int | None, dict]:
    """The reference's single retry after a context rejection.

    Returns (retry, new max_tokens, adjustment record). A character-bound
    rejection retries without a limit only when the requested output alone
    fills the context; a token-count rejection retries with what remains
    minus a 128-token margin. Otherwise the history no longer fits.
    """
    max_context = failure["max_context_tokens"]
    if "input_tokens" not in failure:
        if max_tokens is None or max_tokens < max_context:
            return False, None, {}
        return (
            True,
            None,
            {
                **failure,
                "adjusted_max_tokens": None,
                "output_limit_source": "server_remaining_context",
            },
        )
    available = max_context - failure["input_tokens"]
    if max_tokens is None:
        return False, None, {}
    if failure.get("input_tokens_lower_bound"):
        if available <= 128 or available >= max_tokens:
            return False, None, {}
        return (
            True,
            None,
            {
                **failure,
                "adjusted_max_tokens": None,
                "output_limit_source": "server_remaining_context",
            },
        )
    adjusted = min(max_tokens, max(1, available - 128))
    if available <= 128 or adjusted >= max_tokens:
        return False, None, {}
    return True, adjusted, {**failure, "adjusted_max_tokens": adjusted, "safety_margin_tokens": 128}
