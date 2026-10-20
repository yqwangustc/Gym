# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict prompt rendering and shared inference helpers."""

import asyncio
import hashlib
import json
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status
from resources_servers.audiomc_sim.contracts import Message, TaskInput
from resources_servers.audiomc_sim.telemetry import CURRENT_CALL


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class Prompts:
    """Load immutable prompt snapshots once; never template supplied values twice."""

    def __init__(self, root: Path, roles: tuple[str, ...]) -> None:
        self.texts = {}
        self.templates = {}
        renderer = Environment(undefined=StrictUndefined, autoescape=False)
        for role in roles:
            for part in ("system", "input"):
                path = root / role / f"{part}.md"
                if not path.exists() and role == "assistant" and part == "input":
                    continue
                text = path.read_text()
                self.texts[f"{role}/{part}"] = text
                self.templates[f"{role}/{part}"] = renderer.from_string(text)

    def request(
        self, role: str, values: dict[str, object], *, max_tokens: int = 4096
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": self.texts[f"{role}/system"]},
                {"role": "user", "content": self.templates[f"{role}/input"].render(**values)},
            ],
            max_output_tokens=max_tokens,
        )

    def hashes(self) -> dict[str, str]:
        return {key: hashlib.sha256(text.encode()).hexdigest() for key, text in self.texts.items()}


def prompt_values(task: TaskInput, history: list[Message], turn_index: int) -> dict[str, object]:
    return {
        "topic": task.topic,
        "subtopic": f"{task.subtopic}: {task.subtopic_description}",
        "persona": task.persona,
        "evaluation_category": task.evaluation["category"],
        "evaluation_definition": task.evaluation["definition"],
        "pass_criteria": task.evaluation["pass_criteria"],
        "fail_criteria": task.evaluation["fail_criteria"],
        "few_shot_examples": "\n".join(task.evaluation["few_shot_examples"]),
        "evaluation_configuration": json.dumps(task.evaluation, ensure_ascii=False),
        "assistant_instructions": task.assistant_instructions
        + (f"\nDeveloper instructions:\n{task.developer_instructions}" if task.developer_instructions else ""),
        "next_turn_index": turn_index,
        "turn_index": turn_index,
        "max_turns": task.max_turns,
        "conversation_history": "\n".join(f"[{m.role}, {m.turn_id}]: {m.text}" for m in history)
        or "(No previous turns)",
    }


def response_text(response: NeMoGymResponse) -> str:
    return "\n".join(
        block.text
        for item in response.output
        if item.type == "message"
        for block in item.content
        if block.type == "output_text"
    ).strip()


async def infer(client: ServerClient, model: str, params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Invoke a configured Gym Model Server with an explicit call deadline."""
    async with asyncio.timeout(300):
        http = await client.post(server_name=model, url_path="/v1/responses", json=params)
        await raise_for_status(http)
        response = NeMoGymResponse.model_validate(await get_response_json(http))
        timing = CURRENT_CALL.get()
        if timing is not None:
            timing["http_status"] = http.status
            if response.usage is not None:
                timing["usage"] = response.usage.model_dump(mode="json")
    if response.status != "completed":
        raise ValueError(f"Incomplete inference result: {response.status}")
    text = response_text(response)
    if not text:
        raise ValueError("Model returned no assistant text")
    return text


def parse_json(text: str) -> object:
    """Accept a single JSON value, optionally enclosed in a Markdown code fence."""
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[4:-4]
    return json.loads(text)
