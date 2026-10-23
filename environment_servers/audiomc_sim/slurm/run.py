# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect at the configured concurrency, reserving allocation time for export."""

import argparse
import asyncio
import json
import shutil
from time import perf_counter, time
from uuid import uuid4

from omegaconf import OmegaConf

from environment_servers.audiomc_sim.collect import collect, graph_config
from environment_servers.audiomc_sim.slurm.config import TaskSettings
from resources_servers.audiomc_sim.export import export_collection
from resources_servers.audiomc_sim.prepare import prepare_tasks
from resources_servers.audiomc_sim.storage import started_attempt_count


def prepare_collection(settings: TaskSettings, backends: dict[str, dict[str, object]]) -> None:
    """Freeze tasks, prompts and graph for this run; preserve completed work on resume."""
    output = settings.output
    output.mkdir(parents=True, exist_ok=True)
    snapshot = settings.model_dump(mode="json")
    saved = output / "task_settings.json"
    if (output / "manifest.json").exists():
        if (
            not saved.exists()
            or TaskSettings.model_validate(json.loads(saved.read_text())).model_dump(mode="json") != snapshot
        ):
            raise ValueError("Task settings differ from the saved run; use a new output directory")
        prior = OmegaConf.to_container(OmegaConf.load(output / "backend_config.yaml"), resolve=False)
        if prior != backends:
            raise ValueError("Backend settings differ from the saved run; use a new output directory")
        return
    tasks = settings.simulation_tasks
    graph = graph_config(output=output, base_port=tasks.base_port, concurrency=tasks.concurrency, backends=backends)
    for side in ("environment", "resources"):
        target = output / "prompts" / side
        shutil.copytree(settings.repos.nemo_gym / f"{side}_servers/audiomc_sim/prompts", target, dirs_exist_ok=True)
        graph[side][f"{side}_servers"]["audiomc_sim"]["prompts_dir"] = str(target)
    prompt = settings.mc_resources.assistant_system_prompt.read_text()
    (output / "prompts/environment/assistant/system.md").write_text(prompt)
    rows = prepare_tasks(
        persona_path=settings.mc_resources.persona,
        axis_path=settings.mc_resources.eval_axis,
        topics_path=settings.mc_resources.topics,
        assistant_prompt=prompt,
        count=tasks.max_rollouts,
        seed=tasks.seed,
        max_turns=tasks.max_turns,
        max_teacher_trials=tasks.max_teacher_trials,
        keep_generation_until_max_turn_reached=tasks.keep_generation_until_max_turn_reached,
        tts_voices=backends["tts"]["voices"],
    )
    (output / "tasks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    OmegaConf.save(OmegaConf.create(backends), output / "backend_config.yaml")
    OmegaConf.save(OmegaConf.create(graph), output / "gym_config.yaml")
    saved.write_text(json.dumps(snapshot, indent=2))
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": uuid4().hex[:12],
                "graph": graph,
                "seed": tasks.seed,
                "assistant_checkpoint": backends["assistant"]["checkpoint"],
                "sampling_seed_only": True,
                "concurrency": tasks.concurrency,
                "fixed_rollouts": False,
            },
            indent=2,
        )
    )


async def run_collection(
    settings: TaskSettings, backends: dict[str, dict[str, object]], *, job_end: float
) -> dict[str, object]:
    """Run the normal collector and export durable records even on deadline/cancellation."""
    tasks = settings.simulation_tasks
    deadline = job_end - tasks.export_reserve_min * 60
    if deadline <= time():
        raise RuntimeError("Allocation has no collection time left after the export reserve")
    # Task materialization counts against the allocation, not the export reserve.
    prepare_collection(settings, backends)
    started_at, started = time(), perf_counter()
    interrupted, quota_unmet, completed_normally = False, False, False
    timeout = asyncio.timeout(max(0, deadline - time()))
    try:
        async with timeout:
            await collect(
                argparse.Namespace(
                    output=settings.output,
                    env_file=None,
                    max_rollouts=tasks.max_rollouts,
                    target_accepted=tasks.target_trajectory,
                    concurrency=tasks.concurrency,
                    fixed_rollouts=False,
                    export_every=0,
                    export_every_min=tasks.export_every_min,
                    max_audio_files=tasks.max_audio_files,
                    monitor=True,
                    keep_raw_artifacts=not tasks.cleanup_finished_trajectories,
                )
            )
            completed_normally = True
    except TimeoutError:
        if not timeout.expired():
            raise
        interrupted = True
        print("Collection deadline reached; exporting completed trajectories", flush=True)
    except SystemExit as error:
        # The normal collector exports before reporting an unmet quota.
        if error.code in (None, 0):
            raise
        quota_unmet = True
        raise RuntimeError(str(error)) from error
    finally:
        collection_seconds = perf_counter() - started
        export_start = perf_counter()
        if completed_normally:
            summary = json.loads((settings.output / "summary.json").read_text())
        else:
            summary = await asyncio.to_thread(
                export_collection,
                settings.output,
                max_audio_files=tasks.max_audio_files,
                cleanup_finished_trajectories=tasks.cleanup_finished_trajectories,
            )
        started_rollouts = started_attempt_count(settings.output)
        runtime = {
            "concurrency": tasks.concurrency,
            "target_trajectory": tasks.target_trajectory,
            "started_at": started_at,
            "finished_at": time(),
            "job_end": job_end,
            "collection_deadline": deadline,
            "collection_seconds": collection_seconds,
            "export_seconds": perf_counter() - export_start,
            "deadline_interrupted": interrupted,
            "quota_unmet": quota_unmet,
            "accepted_examples": summary["accepted_examples"],
            "completed_rollouts": summary["rollout_counter"],
            "interrupted_attempts": max(0, started_rollouts - summary["rollout_counter"]),
        }
        (settings.output / "slurm_runtime.json").write_text(json.dumps(runtime, indent=2))
        print(json.dumps(runtime, indent=2), flush=True)
    return summary
