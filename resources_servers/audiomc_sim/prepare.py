# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministically materialize persona/axis/topic combinations for Gym episodes."""

import argparse
import hashlib
import json
import random
from array import array
from pathlib import Path

import yaml

from environment_servers.audiomc_sim.backends import DEFAULT_BACKEND_CONFIG, load_backends
from resources_servers.audiomc_sim.common import digest
from resources_servers.audiomc_sim.contracts import TaskInput


PERSONA_REVISION = "5b4cd35ab46490c1da1bd2b5a2324d6f871be180"


def prepare_tasks(
    *,
    persona_path: Path,
    axis_path: Path,
    topics_path: Path,
    assistant_prompt: str,
    count: int,
    seed: int,
    max_turns: int,
    max_teacher_trials: int,
    tts_voices: list[str] | None = None,
) -> list[dict[str, object]]:
    """Use compact byte offsets to sample the entire corpus without retaining all rows."""
    if count < 1:
        raise ValueError("Task count must be positive")
    voices = tts_voices if tts_voices is not None else ["alloy"]
    if (
        not isinstance(voices, list)
        or not voices
        or any(not isinstance(voice, str) or not voice.strip() for voice in voices)
    ):
        raise ValueError("tts_voices must be a nonempty list of nonblank voice names")
    rng = random.Random(seed)
    voice_rng = random.Random(f"tts:{seed}")
    offsets = array("Q")
    hasher = hashlib.sha256()
    with persona_path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            hasher.update(line)
            if line.strip():
                offsets.append(offset)
    if not offsets:
        raise ValueError("Persona corpus is empty")
    axes = yaml.safe_load(axis_path.read_text())["axes"]
    topics = yaml.safe_load(topics_path.read_text())["category"]
    if set(axes) != set(topics):
        raise ValueError("Topic categories must match evaluation axes")
    hashes = {
        "personas": hasher.hexdigest(),
        "axes": hashlib.sha256(axis_path.read_bytes()).hexdigest(),
        "topics": hashlib.sha256(topics_path.read_bytes()).hexdigest(),
        "persona_revision": PERSONA_REVISION,
    }
    tasks = []
    with persona_path.open("rb") as stream:
        for _ in range(count):
            axis = rng.choice(list(axes))
            topic = rng.choice(list(topics[axis]))
            subtopic = rng.choice(list(topics[axis][topic]))
            stream.seek(offsets[rng.randrange(len(offsets))])
            persona = json.loads(stream.readline())
            task = TaskInput(
                persona_uuid=persona["uuid"],
                persona=persona["persona"],
                axis_id=axis,
                evaluation=axes[axis],
                topic=topic,
                subtopic=subtopic,
                subtopic_description=topics[axis][topic][subtopic],
                seed=rng.randrange(2**32),
                tts_voice=voice_rng.choice(voices),
                max_turns=max_turns,
                max_teacher_trials=max_teacher_trials,
                assistant_instructions=assistant_prompt,
                source_hashes=hashes,
            )
            tasks.append(
                {
                    "task_id": {"taskset": "audiomc_sim:generated", "task_id": digest(task.model_dump())[:24]},
                    "task_input": task.model_dump(),
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personas", type=Path, required=True)
    parser.add_argument(
        "--backend-config",
        type=Path,
        default=DEFAULT_BACKEND_CONFIG,
        help="Backend YAML containing tts.voices, sampled once per materialized task",
    )
    parser.add_argument("--axes", type=Path, default=Path("resources_servers/audiomc_sim/data/eval_axis.yaml"))
    parser.add_argument("--topics", type=Path, default=Path("resources_servers/audiomc_sim/data/topics.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-teacher-trials", type=int, default=3)
    args = parser.parse_args()
    tasks = prepare_tasks(
        persona_path=args.personas,
        axis_path=args.axes,
        topics_path=args.topics,
        assistant_prompt=Path("environment_servers/audiomc_sim/prompts/assistant/system.md").read_text(),
        count=args.count,
        seed=args.seed,
        max_turns=args.max_turns,
        max_teacher_trials=args.max_teacher_trials,
        tts_voices=load_backends(args.backend_config)["tts"]["voices"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks))


if __name__ == "__main__":
    main()
