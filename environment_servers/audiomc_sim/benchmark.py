# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run sequential fixed-task concurrency experiments against one assistant backend."""

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path
from time import time
from uuid import uuid4

from omegaconf import OmegaConf

from environment_servers.audiomc_sim.backends import DEFAULT_BACKEND_CONFIG, load_backends
from environment_servers.audiomc_sim.collect import graph_config, load_env
from resources_servers.audiomc_sim.prepare import prepare_tasks


async def run(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    load_env(args.env_file)
    backend_snapshot = root / "backend_config.yaml"
    backends = load_backends(backend_snapshot if backend_snapshot.exists() else args.backend_config)
    if not backend_snapshot.exists():
        OmegaConf.save(OmegaConf.create(backends), backend_snapshot)
    manifest_path = root / "suite_manifest.json"
    if not manifest_path.exists():
        for side in ("environment", "resources"):
            shutil.copytree(Path(f"{side}_servers/audiomc_sim/prompts"), root / "prompts" / side)
        tasks = prepare_tasks(
            persona_path=args.personas,
            axis_path=args.axes,
            topics_path=args.topics,
            assistant_prompt=(root / "prompts/environment/assistant/system.md").read_text(),
            count=100,
            seed=44,
            max_turns=6,
            max_teacher_trials=3,
            tts_voices=backends["tts"]["voices"],
        )
        task_text = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
        (root / "tasks.jsonl").write_text(task_text)
        manifest_path.write_text(
            json.dumps(
                {
                    "created_at": time(),
                    "concurrencies": [4, 8, 16, 32],
                    "tasks_per_run": 100,
                    "seed": 44,
                    "max_turns": 6,
                    "max_teacher_trials": 3,
                    "tasks_sha256": hashlib.sha256(task_text.encode()).hexdigest(),
                    "assistant_checkpoint": backends["assistant"]["checkpoint"],
                    "vllm_max_num_seqs": 32,
                    "audio_limit": 50,
                },
                indent=2,
            )
        )
    manifest = json.loads(manifest_path.read_text())
    for concurrency in manifest["concurrencies"]:
        output = root / f"c{concurrency}"
        output.mkdir(exist_ok=True)
        if (output / "runtime.json").exists():
            print(f"Already finished c{concurrency}", flush=True)
            continue
        if not (output / "manifest.json").exists():
            graph = graph_config(
                output=output,
                base_port=18240,
                backends=backends,
                concurrency=concurrency,
            )
            for side in ("environment", "resources"):
                target = output / "prompts" / side
                shutil.copytree(root / "prompts" / side, target)
                graph[side][f"{side}_servers"]["audiomc_sim"]["prompts_dir"] = str(target)
            shutil.copy2(root / "tasks.jsonl", output / "tasks.jsonl")
            shutil.copy2(backend_snapshot, output / "backend_config.yaml")
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": uuid4().hex[:12],
                        "graph": graph,
                        "seed": 44,
                        "assistant_checkpoint": backends["assistant"]["checkpoint"],
                        "sampling_seed_only": True,
                        "concurrency": concurrency,
                        "fixed_rollouts": True,
                        "suite_tasks_sha256": manifest["tasks_sha256"],
                    },
                    indent=2,
                )
            )
        print(f"Starting c{concurrency}: 100 fixed rollout attempts", flush=True)
        with (output / "collection.log").open("a") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "environment_servers.audiomc_sim.collect",
                "--output",
                str(output),
                "--personas",
                str(args.personas),
                "--env-file",
                str(args.env_file),
                "--concurrency",
                str(concurrency),
                "--fixed-rollouts",
                "--max-rollouts",
                "100",
                "--target-accepted",
                "100",
                "--monitor",
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            status = await process.wait()
        if status:
            raise RuntimeError(f"c{concurrency} exited {status}; inspect {output / 'collection.log'}")
        print(f"Finished c{concurrency}: {(output / 'runtime.json').read_text()}", flush=True)
    print("All four experiments finished.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend-config",
        type=Path,
        default=DEFAULT_BACKEND_CONFIG,
        help="Per-role model backend YAML; frozen for all concurrency levels",
    )
    parser.add_argument("--personas", type=Path, required=True)
    parser.add_argument("--axes", type=Path, required=True)
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
