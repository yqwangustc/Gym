# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch a single-host Gym graph and collect accepted AudioMC corrections."""

import argparse
import asyncio
import json
import logging
import math
import os
import shlex
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

import uvicorn
from omegaconf import OmegaConf

from environment_servers.audiomc_sim.app import AudioMCSimEnvironment, AudioMCSimEnvironmentConfig
from environment_servers.audiomc_sim.backends import DEFAULT_BACKEND_CONFIG, TEXT_ROLES, load_backends
from environment_servers.audiomc_sim.monitor import CollectionMonitor
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status
from resources_servers.audiomc_sim.app import AudioMCSimConfig, AudioMCSimResources
from resources_servers.audiomc_sim.export import DEFAULT_MAX_AUDIO_FILES, export_collection
from resources_servers.audiomc_sim.prepare import prepare_tasks
from resources_servers.audiomc_sim.storage import read_episode_records, recover_export
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


LOGGER = logging.getLogger(__name__)


async def shutdown_graph(participants: list[tuple[str, uvicorn.Server, asyncio.Task]]) -> None:
    """Drain callers before their dependencies, including canceled HTTP handlers.

    Uvicorn can finish serving while canceled ASGI handlers are still unwinding.
    Keep Resources and model servers available until those handlers have finished
    finalizing and closing their sessions.
    """
    for kinds in (
        {"environment_servers"},
        {"responses_api_agents"},
        {"resources_servers"},
        {"responses_api_models"},
    ):
        stage = [(server, task) for kind, server, task in participants if kind in kinds]
        if not stage:
            continue
        for server, _ in stage:
            server.should_exit = True
        pending = {task for _, task in stage if not task.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=5)
        # ASGI tasks are owned by server_state, not by Server.serve().
        pending.update(task for server, _ in stage for task in server.server_state.tasks if not task.done())
        if pending:
            _, pending = await asyncio.wait(pending, timeout=30)
        if pending:
            LOGGER.warning("Forcing shutdown of %s remaining Gym tasks in %s", len(pending), kinds)
            for server, _ in stage:
                server.force_exit = True
            for task in pending:
                task.cancel()
            _, pending = await asyncio.wait(pending, timeout=5)
            if pending:
                LOGGER.error("%s Gym tasks did not stop within the shutdown budget", len(pending))
        for _, task in stage:
            if task.done() and not task.cancelled() and task.exception() is not None:
                LOGGER.warning("Gym server stopped with an error", exc_info=task.exception())


def load_env(path: Path) -> None:
    """Read literal KEY=value entries without executing shell expressions."""
    for line in path.read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if fields and fields[0] == "export":
            fields = fields[1:]
        if len(fields) == 1 and "=" in fields[0]:
            key, value = fields[0].split("=", 1)
            os.environ.setdefault(key, value)


def graph_config(
    *,
    output: Path,
    base_port: int,
    concurrency: int = 1,
    backends: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Wire independent role backends, sharing identical helper model wrappers."""
    if backends is None:
        backends = load_backends()
    graph, role_models, shared = {}, {}, {}
    for role in TEXT_ROLES:
        backend = backends[role]
        identity = json.dumps(
            {
                "api_base": backend["api_base"],
                "model_name": backend["model_name"],
                "api_key": backend["api_key"],
                "uses_reasoning_parser": backend["uses_reasoning_parser"],
                "reasoning_effort": backend.get("reasoning_effort"),
                "audio": role == "assistant",
            },
            sort_keys=True,
        )
        if identity in shared:
            role_models[role] = shared[identity]
            continue
        name = "user_teacher_model" if role == "user" else f"{role}_model"
        role_models[role] = shared[identity] = name
        graph[name] = {
            "responses_api_models": {
                "vllm_model": {
                    "name": name,
                    "entrypoint": "app.py",
                    "host": "127.0.0.1",
                    "port": base_port + len(graph),
                    "base_url": backend["api_base"],
                    "api_key": backend["api_key"],
                    "model": backend["model_name"],
                    "return_token_id_information": False,
                    "uses_reasoning_parser": backend["uses_reasoning_parser"],
                    "sampling_overrides": {"reasoning_effort": backend["reasoning_effort"]}
                    if backend.get("reasoning_effort") is not None
                    else None,
                    "audio_root": str(output / "artifacts") if role == "assistant" else None,
                }
            }
        }

    def model_ref(role: str) -> dict[str, str]:
        return {"name": role_models[role], "type": "responses_api_models"}

    graph["assistant"] = {
        "responses_api_agents": {
            "simple_agent": {
                "name": "assistant",
                "entrypoint": "app.py",
                "host": "127.0.0.1",
                "port": base_port + len(graph),
                "model_server": model_ref("assistant"),
                "max_steps": 1,
                "execute_tools": False,
            }
        }
    }
    tts = backends["tts"]
    graph["resources"] = {
        "resources_servers": {
            "audiomc_sim": {
                "name": "resources",
                "entrypoint": "app.py",
                "host": "127.0.0.1",
                "port": base_port + len(graph),
                "rubric_builder_model": model_ref("rubric_builder"),
                "critiquer_model": model_ref("critiquer"),
                "domain": "instruction_following",
                "artifact_root": str(output / "artifacts"),
                "tts_endpoint": tts["api_base"].rstrip("/") + "/audio/speech",
                "tts_api_key": tts["api_key"],
                "tts_model": tts["model_name"],
                "tts_voices": tts["voices"],
                "tts_request_timeout_seconds": tts.get("request_timeout_seconds", 180),
                "tts_request_attempts": tts.get("request_attempts", 1),
                "num_workers": 1,
            }
        }
    }
    graph["environment"] = {
        "environment_servers": {
            "audiomc_sim": {
                "name": "environment",
                "entrypoint": "app.py",
                "host": "127.0.0.1",
                "port": base_port + len(graph),
                "assistant_agent": {"name": "assistant", "type": "responses_api_agents"},
                "planner_model": model_ref("planner"),
                "user_model": model_ref("user"),
                "teacher_model": model_ref("teacher"),
                "resources_server": {"name": "resources", "type": "resources_servers"},
                "cleanup_timeout_seconds": 20,
                "default_episode_timeout_seconds": 2400,
                "max_concurrent_episodes": concurrency,
                "queue_timeout_seconds": 30,
            }
        }
    }
    return graph


@asynccontextmanager
async def serve_graph(graph: dict[str, object]) -> AsyncIterator[ServerClient]:
    """Run the actual Gym HTTP servers on one host, with bounded shutdown."""
    config = OmegaConf.create(graph)
    os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(graph)
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=config)
    servers, tasks, participants = [], [], []
    classes = {
        "responses_api_models": (VLLMModel, VLLMModelConfig),
        "responses_api_agents": (SimpleAgent, SimpleAgentConfig),
        "resources_servers": (AudioMCSimResources, AudioMCSimConfig),
        "environment_servers": (AudioMCSimEnvironment, AudioMCSimEnvironmentConfig),
    }
    try:
        for entry in config.values():
            kind = next(iter(entry))
            raw = next(iter(entry[kind].values()))
            cls, schema = classes[kind]
            values = OmegaConf.to_container(raw, resolve=True)
            instance = cls(config=schema.model_validate(values), server_client=client)
            server = uvicorn.Server(
                uvicorn.Config(
                    instance.setup_webserver(),
                    host=raw.host,
                    port=raw.port,
                    log_level="warning",
                    access_log=False,
                    timeout_graceful_shutdown=0.5,
                )
            )
            servers.append(server)
            tasks.append(asyncio.create_task(server.serve()))
            participants.append((kind, server, tasks[-1]))
        async with asyncio.timeout(60):
            while not all(s.started for s in servers):
                for task in tasks:
                    if task.done():
                        task.result()
                        raise RuntimeError("Gym server exited during startup")
                await asyncio.sleep(0.05)
        yield client
    finally:
        await shutdown_graph(participants)


async def collect(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    recover_export(output)
    if args.env_file:
        load_env(args.env_file)
    tasks_path = output / "tasks.jsonl"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        graph = manifest["graph"]
    else:
        backends = load_backends(getattr(args, "backend_config", DEFAULT_BACKEND_CONFIG))
        OmegaConf.save(OmegaConf.create(backends), output / "backend_config.yaml")
        graph = graph_config(
            output=output,
            base_port=args.base_port,
            backends=backends,
            concurrency=getattr(args, "concurrency", 1),
        )
        for side in ("environment", "resources"):
            source = Path(f"{side}_servers/audiomc_sim/prompts")
            target = output / "prompts" / side
            shutil.copytree(source, target, dirs_exist_ok=True)
            graph[side][f"{side}_servers"]["audiomc_sim"]["prompts_dir"] = str(target)
        tasks = prepare_tasks(
            persona_path=args.personas,
            axis_path=args.axes,
            topics_path=args.topics,
            assistant_prompt=Path("environment_servers/audiomc_sim/prompts/assistant/system.md").read_text(),
            count=args.max_rollouts,
            seed=args.seed,
            max_turns=args.max_turns,
            max_teacher_trials=args.max_teacher_trials,
            keep_generation_until_max_turn_reached=getattr(args, "keep_generation_until_max_turn_reached", False),
            tts_voices=backends["tts"]["voices"],
        )
        tasks_path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))
        manifest = {
            "run_id": uuid4().hex[:12],
            "graph": graph,
            "seed": args.seed,
            "assistant_checkpoint": backends["assistant"]["checkpoint"],
            "sampling_seed_only": True,
            "concurrency": getattr(args, "concurrency", 1),
            "fixed_rollouts": getattr(args, "fixed_rollouts", False),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        OmegaConf.save(OmegaConf.create(graph), output / "gym_config.yaml")
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    ledger = output / "episode_results.jsonl"
    previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    completed = {row["episode_id"]["rollout_id"] for row in previous}
    # Finalization is durable before its HTTP reply. Recover lost replies instead of replaying a finished episode.
    recoverable, raw_sources = read_episode_records(output)
    for key, record in recoverable.items():
        path = raw_sources.get(key, output / "episodes.jsonl")
        identity = record["metadata"]["episode_id"]
        if identity["rollout_id"] not in completed:
            row = {
                "episode_id": identity,
                "task_id": record["metadata"]["task_id"],
                "result": {
                    "outcome": record["outcome"],
                    "accepted": "training_example" in record,
                    "artifact_path": str(path),
                    "turns": len(record["turn_evaluations"]),
                    "teacher_trials": len(record["teacher_attempts"]),
                    "axis_id": record["metadata"]["task"]["axis_id"],
                },
            }
            with ledger.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            previous.append(row)
            completed.add(identity["rollout_id"])
    accepted = sum(bool((row.get("result") or {}).get("accepted")) for row in previous)
    concurrency = getattr(args, "concurrency", 1)
    fixed_rollouts = getattr(args, "fixed_rollouts", False)
    export_every = getattr(args, "export_every", 0)
    export_interval = getattr(args, "export_every_min", 0) * 60
    max_audio_files = getattr(args, "max_audio_files", DEFAULT_MAX_AUDIO_FILES)
    export_options = {"max_audio_files": max_audio_files}
    if getattr(args, "keep_raw_artifacts", False):
        export_options["cleanup_finished_trajectories"] = False
    if graph:
        configured = graph["environment"]["environment_servers"]["audiomc_sim"]["max_concurrent_episodes"]
        if configured != concurrency:
            raise ValueError(f"Resume concurrency {concurrency} differs from saved configuration {configured}")
    write_lock = asyncio.Lock()
    active = 0
    completed_count = len(previous)
    pending = iter(
        (i, task)
        for i, task in enumerate(tasks[: args.max_rollouts])
        if f"audiomc-{manifest['run_id']}-{i:04d}" not in completed
    )
    async with serve_graph(graph) as client:
        assistant_model_name = (
            graph.get("assistant", {})
            .get("responses_api_agents", {})
            .get("simple_agent", {})
            .get("model_server", {})
            .get("name", "assistant_model")
        )
        assistant_backend = graph.get(assistant_model_name, {}).get("responses_api_models", {}).get("vllm_model", {})
        monitor = CollectionMonitor(
            output=output,
            assistant_base=assistant_backend.get("base_url", "http://127.0.0.1:8000/v1"),
            active=lambda: active,
        )
        if getattr(args, "monitor", False):
            await monitor.sample()
            monitor_task = asyncio.create_task(monitor.run())
        started = perf_counter()
        export_stop = asyncio.Event()

        async def checkpoint_export(trigger: str) -> None:
            checkpoint_started = perf_counter()
            print(
                f"Checkpoint export started ({trigger}): completed={completed_count}; "
                f"elapsed_minutes={(checkpoint_started - started) / 60:.2f}",
                flush=True,
            )
            # Cancelling an asyncio waiter does not stop its worker thread. Keep the
            # export lock until publication finishes so deadline export cannot race it.
            publication = asyncio.create_task(asyncio.to_thread(export_collection, output, **export_options))
            cancellation = None
            while not publication.done():
                try:
                    await asyncio.shield(publication)
                except asyncio.CancelledError as error:
                    cancellation = error
            publication.result()
            if cancellation is not None:
                raise cancellation
            print(f"Checkpoint export finished in {perf_counter() - checkpoint_started:.2f}s", flush=True)

        async def timed_exports() -> None:
            while not export_stop.is_set():
                try:
                    await asyncio.wait_for(export_stop.wait(), timeout=export_interval)
                except TimeoutError:
                    async with write_lock:
                        if export_stop.is_set():
                            return
                        await checkpoint_export("timer")
                else:
                    return

        async def worker() -> None:
            nonlocal active, accepted, completed_count
            while fixed_rollouts or accepted < args.target_accepted:
                try:
                    index, task = next(pending)
                except StopIteration:
                    return
                episode_id = f"audiomc-{manifest['run_id']}-{index:04d}"
                pending_dir = output / "requests"
                pending_dir.mkdir(exist_ok=True)
                attempt = 0
                while (pending_dir / f"{episode_id}-attempt-{attempt}.json").exists():
                    attempt += 1
                envelope = {"episode_id": {"rollout_id": episode_id, "attempt": attempt}, "task": task}
                (pending_dir / f"{episode_id}-attempt-{attempt}.json").write_text(
                    json.dumps(envelope, ensure_ascii=False)
                )
                print(f"Starting {episode_id} attempt={attempt}: {task['task_input']['axis_id']}", flush=True)
                episode_started = perf_counter()
                active += 1
                # A fresh physical attempt gets a new capture key after interruption; existing tar files are immutable.
                try:
                    async with asyncio.timeout(2500):
                        response = await client.post(
                            server_name="environment", url_path="/run", json=envelope, _max_num_tries=1
                        )
                        await raise_for_status(response)
                        row = await get_response_json(response)
                except Exception as error:
                    row = {
                        "episode_id": envelope["episode_id"],
                        "task_id": task["task_id"],
                        "failure": {
                            "failure_reason": f"Collector transport failure: {type(error).__name__}: {error}"[:2000],
                            "terminal": False,
                        },
                    }
                elapsed = perf_counter() - episode_started
                active -= 1
                async with write_lock:
                    with ledger.open("a") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    result = row.get("result") or {}
                    accepted += bool(result.get("accepted"))
                    completed_count += 1
                    with (output / "rollout_timings.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "episode_id": envelope["episode_id"],
                                    "elapsed_seconds": elapsed,
                                    "finished_at": time(),
                                    "outcome": result.get("outcome", "infrastructure_failure"),
                                    "accepted": bool(result.get("accepted")),
                                }
                            )
                            + "\n"
                        )
                    print(
                        f"Finished {episode_id}: {result or row.get('failure')}; accepted={accepted}; completed={completed_count}",
                        flush=True,
                    )
                    if export_every and completed_count % export_every == 0:
                        await checkpoint_export("count")

        try:
            async with asyncio.TaskGroup() as background:
                if export_interval:
                    background.create_task(timed_exports())
                try:
                    async with asyncio.TaskGroup() as workers:
                        for _ in range(concurrency):
                            workers.create_task(worker())
                finally:
                    export_stop.set()
        finally:
            if getattr(args, "monitor", False):
                monitor.stop.set()
                await monitor_task
                await monitor.sample()
        collection_seconds = perf_counter() - started
    export_started = perf_counter()

    summary = export_collection(output, **export_options)
    runtime = {
        "concurrency": concurrency,
        "collection_seconds": collection_seconds,
        "export_seconds": perf_counter() - export_started,
        "completed_in_process": completed_count - len(previous),
        "accepted_in_process": accepted - sum(bool((r.get("result") or {}).get("accepted")) for r in previous),
    }
    runtime["rollouts_per_hour"] = runtime["completed_in_process"] * 3600 / max(collection_seconds, 1e-9)
    runtime["accepted_per_hour"] = runtime["accepted_in_process"] * 3600 / max(collection_seconds, 1e-9)
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print(json.dumps(summary | runtime, indent=2), flush=True)
    if not fixed_rollouts and accepted < args.target_accepted:
        raise SystemExit(f"Collected {accepted}/{args.target_accepted} accepted corrections; see inspection.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch localhost Gym servers and collect multi-turn voice conversations. "
            "Save a training example when the assistant fails and a teacher correction passes "
            "the frozen rubrics. Connect to an already running assistant model server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""Run from the repository root. Configure role backends with --backend-config;
supply their credentials through environment variables or --env-file.
The assistant endpoint must already be serving an audio-capable model; this
command starts the Gym orchestration servers, not vLLM or the checkpoint.

Examples:
  # Collect at least five accepted corrections within a budget of 20 attempts.
  .venv/bin/python -m environment_servers.audiomc_sim.collect \
    --personas /path/to/nemotron-usa.jsonl \
    --output /path/to/rollouts/quota-run \
    --env-file /path/to/credentials.env \
    --target-accepted 5 --max-rollouts 20

  # Run 100 attempts with eight concurrent conversations and record metrics.
  .venv/bin/python -m environment_servers.audiomc_sim.collect \
    --personas /path/to/nemotron-usa.jsonl \
    --output /path/to/rollouts/concurrency-8 \
    --env-file /path/to/credentials.env \
    --fixed-rollouts --max-rollouts 100 --concurrency 8 --monitor

Outputs include rollout.jsonl (accepted SFT examples), audio.tar (or audio shards), episodes.jsonl,
inspection.jsonl (excluded attempts), summary.json, and runtime.json. Acceptance
is based on automated rubric grading, not human review.

Reuse --output to resume from the saved task list, graph, prompt snapshots and
completed-attempt ledger. Saved configuration takes precedence over new resource,
model, endpoint and sampling options; use a new directory to change those settings.
Resume with the original --concurrency. Stopping and export options still apply.
""",
    )
    inputs = parser.add_argument_group("Resources and output")
    inputs.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for tasks, frozen prompts, traces, audio and exported SFT examples; reuse it to resume a run.",
    )
    inputs.add_argument(
        "--personas",
        type=Path,
        required=True,
        help="Local persona JSONL with uuid and general persona fields; scanned once to sample task personas.",
    )
    inputs.add_argument(
        "--keep-raw-artifacts",
        action="store_true",
        help="Keep per-trajectory raw artifacts and request files after successful export. By default, "
        "completed max-turn trajectories are compacted into exported JSONL/audio; unfinished traces and shared prompts stay.",
    )
    inputs.add_argument(
        "--max-audio-files",
        type=int,
        default=DEFAULT_MAX_AUDIO_FILES,
        help="Maximum WAV files per exported tar archive. Exports above this count use "
        "audio-00000-of-NNNNN.tar shards; smaller exports keep audio.tar. Applies to final and periodic exports, "
        "including audio retained for inspection. Must be >= 1. Default: %(default)s.",
    )
    inputs.add_argument(
        "--env-file",
        type=Path,
        help="Optional file of literal KEY=value or export KEY=value assignments, including OPENAI_API_KEY. "
        "Existing environment variables take precedence; shell expressions are not executed.",
    )
    inputs.add_argument(
        "--axes",
        type=Path,
        default=Path("resources_servers/audiomc_sim/data/eval_axis.yaml"),
        help="Evaluation-axis YAML used to sample each task's definition, criteria and examples. Default: %(default)s.",
    )
    inputs.add_argument(
        "--topics",
        type=Path,
        default=Path("resources_servers/audiomc_sim/data/topics.yaml"),
        help="Topic/sub-topic YAML used to sample the conversation subject. Default: %(default)s.",
    )
    collection = parser.add_argument_group("Collection budget and concurrency")
    collection.add_argument(
        "--target-accepted",
        type=int,
        default=5,
        help="Accepted-example quota. Stop scheduling new conversations when reached; in-flight work finishes, "
        "so the quota may be exceeded. Ignored for stopping in --fixed-rollouts mode. Default: %(default)s.",
    )
    collection.add_argument(
        "--max-rollouts",
        type=int,
        default=20,
        help="Maximum task attempts, including accepted and excluded outcomes, across the run and any resumes. "
        "Must be >= target-accepted. In fixed mode, process this many saved tasks; otherwise stop at the quota "
        "or this budget and exit unsuccessfully if the quota is unmet. Default: %(default)s.",
    )
    collection.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Maximum user-assistant exchanges per conversation. By default stop at the first graded failure; "
        "exclude conversations that pass all exchanges at this cap. Default: %(default)s.",
    )
    collection.add_argument(
        "--keep-generation-until-max-turn-reached",
        action="store_true",
        help="After a failed assistant reply, continue with a rubric-passing teacher correction until max-turns. "
        "Teacher trials reset each turn; earlier verified rewrites are retained if a later turn fails. Default: disabled.",
    )
    collection.add_argument(
        "--max-teacher-trials",
        type=int,
        default=5,
        help="Maximum teacher correction candidates per failed assistant turn, each graded against the same "
        "frozen rubrics. Stop the conversation if none passes; keep any earlier successful rewrites. Default: %(default)s.",
    )
    collection.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum independent conversations in flight; also sets the Environment Server admission limit. "
        "Turns within each conversation remain sequential. This does not change vLLM max_num_seqs. "
        "Must be >= 1 and match the saved setting on resume. Default: %(default)s.",
    )
    collection.add_argument(
        "--fixed-rollouts",
        action="store_true",
        help="Process the full max-rollouts budget regardless of accepted count, suitable for throughput comparisons. "
        "Does not require target-accepted to be reached for a successful exit. Default: disabled.",
    )
    collection.add_argument(
        "--export-every",
        type=int,
        default=0,
        help="Rebuild exported JSONL/audio files after every N completed attempts, including prior completions on "
        "resume. Zero disables count-based checkpoints. Raw traces and the ledger are saved throughout; frequent "
        "exports add overhead. Must be >= 0. Default: %(default)s.",
    )
    collection.add_argument(
        "--export-every-min",
        type=float,
        default=0,
        help="Export completed trajectories every N minutes after collection starts, independently of conversation "
        "completion. Use 230 to begin saving ten minutes before a four-hour job limit. The timer restarts after "
        "each timed export and on process resume. Zero disables timed checkpoints; final export always runs on "
        "normal completion. Can be combined with --export-every; exports are serialized. Does not stop collection "
        "or guarantee saving finishes before the cluster deadline. Must be finite and >= 0. Default: %(default)s.",
    )
    collection.add_argument(
        "--monitor",
        action="store_true",
        help="Record GPU utilization, vLLM metrics, collector CPU/memory and event-loop lag in system_metrics.jsonl "
        "at roughly five-second intervals. Scrapes the configured Assistant endpoint's /metrics route. Default: disabled.",
    )
    collection.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for persona/axis/topic task sampling on a new run; does not make model generation deterministic. "
        "Default: %(default)s.",
    )
    backends = parser.add_argument_group("Gym servers and model endpoints")
    backends.add_argument(
        "--backend-config",
        type=Path,
        default=DEFAULT_BACKEND_CONFIG,
        help="Per-role YAML settings (api_base, model_name, optional api_key) for Planner, Rubric Builder, "
        "Critiquer, User, Teacher, Assistant and TTS. Supports OmegaConf environment interpolation. "
        "Frozen in backend_config.yaml on a new run. Default: %(default)s.",
    )
    backends.add_argument(
        "--base-port",
        type=int,
        default=18240,
        help="First consecutive localhost port for Gym wrappers, agent, Resources and Environment. "
        "Needs one port per distinct text backend plus three; default configuration uses seven. "
        "The existing assistant endpoint uses a separate port. Default: %(default)s.",
    )
    args = parser.parse_args()
    if args.concurrency < 1 or args.export_every < 0:
        parser.error("Require concurrency >= 1 and export-every >= 0")
    if args.max_audio_files < 1:
        parser.error("Require max-audio-files >= 1")
    if not math.isfinite(args.export_every_min) or args.export_every_min < 0:
        parser.error("Require finite export-every-min >= 0")
    if args.target_accepted < 1 or args.max_rollouts < args.target_accepted:
        parser.error("Require 1 <= target-accepted <= max-rollouts")
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
