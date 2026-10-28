# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Candidate loop for Tau-Voice Simple.

The paired ``tau_voice_simple`` resources server runs Tau (environment, user
simulator, TTS, limits, scoring). This agent plays the customer-service agent:
it keeps the candidate's chat history, with every user turn as an inline WAV
clip, asks the policy model for one turn at a time over Chat Completions,
enforces the reference turn protocol (one tool call or one spoken reply, with
two retries and a fixed fallback), and submits the turn to ``/agent_turn``
until Tau ends the episode. It then calls ``/verify``.
"""

import copy
import json
import logging
import re
import uuid
from time import time
from typing import Any

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import (
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    ResourcesServerRef,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.tau_voice_simple_agent.protocol import (
    CONTEXT_EXHAUSTED_FALLBACK,
    PROTOCOL_FALLBACK,
    USER_TURN_INSTRUCTION,
    ProtocolError,
    adjusted_output_limit,
    build_system_prompt,
    context_length_failure,
    parse_turn,
)


LOG = logging.getLogger(__name__)


class TauVoiceSimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Rejected turns are retried this many times before the fallback reply.
    protocol_retries: int = Field(default=2, ge=0)
    # Candidate sampling seed for attempt a is the row's seed + a, as in the reference;
    # this applies only to rows without a seed.
    default_seed: int = 42
    # Safety bound only; Tau's own max_steps ends episodes long before this.
    max_agent_turns: int = Field(default=500, gt=0)


class TauVoiceSimpleRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]


class TauVoiceSimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class _ModelCallError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail[:500]}")
        self.status = status
        self.detail = detail


class TauVoiceSimpleAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: TauVoiceSimpleAgentConfig

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),  # noqa: B008
    ) -> NeMoGymResponse:
        raise NotImplementedError("Tau-Voice Simple episodes run through /run")

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:  # noqa: B008
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/aggregate_metrics", json=body
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def _post_resource(self, path: str, payload: dict[str, Any], cookies) -> tuple[dict[str, Any], Any]:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=path,
            json=payload,
            cookies=cookies,
        )
        await raise_for_status(response)
        return await get_response_json(response), response.cookies

    async def _chat(self, payload: dict[str, Any], cookies) -> tuple[dict[str, Any], Any]:
        response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/chat/completions",
            json=payload,
            cookies=cookies,
        )
        if response.status >= 400:
            raise _ModelCallError(response.status, (await response.read()).decode(errors="replace"))
        return await get_response_json(response), response.cookies

    async def _complete(self, request: dict[str, Any], cookies) -> tuple[dict[str, Any], dict | None, Any]:
        """One completion with the reference's single context-budget retry."""
        try:
            completion, cookies = await self._chat(request, cookies)
            return completion, None, cookies
        except _ModelCallError as error:
            failure = context_length_failure(error.status, error.detail)
            if failure is None:
                raise
            retry, max_tokens, adjustment = adjusted_output_limit(failure, request.get("max_tokens"))
            if not retry:
                raise
            retry_request = dict(request)
            if max_tokens is None:
                retry_request.pop("max_tokens", None)
            else:
                retry_request["max_tokens"] = max_tokens
            try:
                completion, cookies = await self._chat(retry_request, cookies)
            except _ModelCallError as retry_error:
                if (
                    max_tokens is None
                    and retry_error.status == 400
                    and re.search(r"max_tokens must be at least 1, got 0\b", retry_error.detail)
                ):
                    raise _ContextExhausted(
                        {**adjustment, "remaining_output_tokens": 0, "detail": retry_error.detail}
                    ) from retry_error
                raise
            return completion, adjustment, cookies

    async def _next_turn(self, messages, tools, tool_names, sampling, seed, cookies) -> tuple[dict[str, Any], Any]:
        """The candidate's next turn after up to ``protocol_retries`` rejected attempts."""
        failures: list[dict[str, Any]] = []
        for attempt in range(self.config.protocol_retries + 1):
            request = {"messages": copy.deepcopy(messages), "seed": seed + attempt, **sampling}
            if tools:
                request["tools"] = tools
                request["tool_choice"] = "auto"
            try:
                completion, adjustment, cookies = await self._complete(request, cookies)
                turn = parse_turn(completion, tool_names)
            except ProtocolError as error:
                failures.append(error.as_record(attempt))
                LOG.warning("Tau-Voice Simple protocol_failure %s %s", error.category, error.detail)
                continue
            except _ContextExhausted as error:
                return {
                    "content": CONTEXT_EXHAUSTED_FALLBACK,
                    "raw_data": _raw_data(None, failures + [error.failure])
                    | {"tau_voice_simple_context_exhausted": error.failure},
                }, cookies
            except _ModelCallError as error:
                failure = context_length_failure(error.status, error.detail)
                if failure is None:
                    raise
                failure["attempt"] = attempt + 1
                return {
                    "content": CONTEXT_EXHAUSTED_FALLBACK,
                    "raw_data": _raw_data(None, failures + [failure])
                    | {"tau_voice_simple_context_exhausted": failure},
                }, cookies
            raw_data = _raw_data(turn.get("reasoning_content"), failures)
            if adjustment is not None:
                raw_data["tau_voice_simple_context_budget_adjustment"] = adjustment
            if "tool_call" in turn:
                turn["tool_call"]["id"] = turn["tool_call"]["id"] or f"call_{uuid.uuid4().hex}"
                return {"tool_call": turn["tool_call"], "raw_data": raw_data}, cookies
            return {"content": turn["content"], "raw_data": raw_data}, cookies
        return {
            "content": PROTOCOL_FALLBACK,
            "raw_data": _raw_data(None, failures) | {"protocol_fallback": True},
        }, cookies

    async def run(self, request: Request, body: TauVoiceSimpleRunRequest) -> TauVoiceSimpleAgentVerifyResponse:
        resource_cookies = request.cookies
        model_cookies = None
        seeded, resource_cookies = await self._post_resource(
            "/seed_session", {"verifier_metadata": body.verifier_metadata}, resource_cookies
        )
        tools = seeded["tools"]
        tool_names = {tool["function"]["name"] for tool in tools}
        sampling = _sampling(body.responses_create_params)
        seed_value = int(body.verifier_metadata.get("seed", self.config.default_seed))

        instructions = body.responses_create_params.instructions
        system_prompt = _system_prompt(seeded["domain_policy"], instructions)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        transcript: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        result = seeded
        turns = 0
        last_context = None
        while not result["done"]:
            if turns >= self.config.max_agent_turns:
                break
            _deliver(result["observation"], messages, transcript)
            reply, model_cookies = await self._next_turn(
                messages, tools, tool_names, sampling, seed_value, model_cookies
            )
            context = {"system_prompt": messages[0]["content"], "tools": tools}
            if context != last_context:
                reply.setdefault("raw_data", {})["candidate_context"] = copy.deepcopy(context)
                last_context = copy.deepcopy(context)
            _record_reply(reply, messages, transcript)
            result, resource_cookies = await self._post_resource("/agent_turn", reply, resource_cookies)
            turns += 1

        input_items, output_items = _split_transcript(transcript)
        params = body.responses_create_params.model_dump(exclude_unset=True)
        verify_request = body.model_dump() | {
            "responses_create_params": params | {"input": input_items},
            "response": {
                "id": f"tau_voice_simple-{uuid.uuid4().hex}",
                "created_at": int(time()),
                "object": "response",
                "model": params.get("model") or "",
                "output": output_items,
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }
        verified, _ = await self._post_resource("/verify", verify_request, resource_cookies)
        return TauVoiceSimpleAgentVerifyResponse.model_validate(verified)


def _sampling(params: NeMoGymResponseCreateParamsNonStreaming) -> dict[str, Any]:
    """Map the row's Responses sampling fields onto a Chat Completions request."""
    sampling: dict[str, Any] = {}
    if params.temperature is not None:
        sampling["temperature"] = params.temperature
    if params.top_p is not None:
        sampling["top_p"] = params.top_p
    if params.max_output_tokens is not None:
        sampling["max_tokens"] = params.max_output_tokens
    if params.metadata:
        # e.g. {"extra_body": "{\"top_k\": 1}"}, merged into the request by the vLLM model server.
        sampling["metadata"] = dict(params.metadata)
    return sampling


class _ContextExhausted(Exception):
    def __init__(self, failure):
        self.failure = failure


def _system_prompt(policy, instructions=None):
    """Override the voice wrapper, never discard the task's mandatory policy."""
    if instructions is not None:
        return instructions + f"\n\n<domain_policy>\n{policy}\n</domain_policy>"
    return build_system_prompt(policy)


def _raw_data(reasoning: str | None, failures: list[dict[str, Any]]) -> dict[str, Any]:
    raw_data: dict[str, Any] = {}
    if reasoning is not None:
        raw_data["reasoning_content"] = reasoning
    if failures:
        raw_data["protocol_failures"] = copy.deepcopy(failures)
    return raw_data


def _deliver(observation: dict[str, Any], messages: list, transcript: list) -> None:
    """Append what Tau sent the candidate: a user clip (or text) or tool results."""
    kind = observation["type"]
    if kind == "user_audio":
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": observation["audio_wav_base64"], "format": "wav"},
                    },
                    {"type": "text", "text": USER_TURN_INSTRUCTION},
                ],
            }
        )
        transcript.append({"role": "user", "content": f"[user audio turn {observation['turn_index']:04d}]"})
    elif kind == "user_text":
        messages.append({"role": "user", "content": observation["text"]})
        transcript.append({"role": "user", "content": observation["text"]})
    elif kind == "tool_results":
        for result in observation["results"]:
            message = {
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["content"],
            }
            messages.append(message)
            transcript.append(message)
    else:
        raise ValueError(f"unknown observation type {kind!r}")


def _record_reply(reply: dict[str, Any], messages: list, transcript: list) -> None:
    """Keep the reply in history; reasoning is never fed back to the candidate."""
    if "tool_call" in reply:
        call = reply["tool_call"]
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                }
            ],
        }
    else:
        message = {"role": "assistant", "content": reply["content"]}
    messages.append(message)
    transcript.append(message)


def _split_transcript(transcript: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Responses items for the rollout: everything up to the last input, then the final replies."""
    items: list[dict[str, Any]] = []
    for index, message in enumerate(transcript):
        role = message["role"]
        if role in ("system", "user"):
            items.append({"type": "message", "role": role, "content": message["content"]})
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        elif message.get("tool_calls"):
            call = message["tool_calls"][0]
            items.append(
                {
                    "type": "function_call",
                    "id": f"fc_{index}",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                    "status": "completed",
                }
            )
        else:
            items.append(
                {
                    "type": "message",
                    "id": f"msg_{index}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": message["content"], "annotations": []}],
                }
            )
    split = len(items)
    while split > 0 and (items[split - 1]["type"] == "function_call" or items[split - 1].get("role") == "assistant"):
        split -= 1
    return items[:split], items[split:]


if __name__ == "__main__":
    TauVoiceSimpleAgent.run_webserver()
