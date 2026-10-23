# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start configured local services, wait for readiness, then collect and export."""

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from time import time
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientTimeout
from omegaconf import OmegaConf

from environment_servers.audiomc_sim.backends import LocalServing, load_backends
from environment_servers.audiomc_sim.slurm.config import load_settings
from environment_servers.audiomc_sim.slurm.gpus import allocated_devices, mapped_devices
from environment_servers.audiomc_sim.slurm.preflight import managed_endpoints, require_preflight
from environment_servers.audiomc_sim.slurm.run import run_collection
from nemo_gym.server_utils import request


def server_command(backend: dict[str, object]) -> list[str]:
    """Translate serving settings for standard vLLM or the NeMo audio wrapper."""
    serving = LocalServing.model_validate(backend["serving"])
    endpoint = urlsplit(backend["api_base"])
    if endpoint.scheme != "http":
        raise ValueError("Managed vLLM endpoints must use http; put TLS at an external proxy")
    if endpoint.hostname not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        raise ValueError("Managed serving.api_base must use a local hostname")
    command = [serving.python]
    if serving.entrypoint.endswith(".py"):
        command.append(serving.entrypoint)
    else:
        command += ["-m", serving.entrypoint]
    command += [
        "--model",
        serving.checkpoint,
        "--served-model-name",
        backend["model_name"],
        "--host",
        endpoint.hostname,
        "--port",
        str(endpoint.port or 80),
        "--tensor-parallel-size",
        str(serving.tp),
        "--max-num-seqs",
        str(serving.max_num_seqs_per_dp_rank),
        "--max-model-len",
        str(serving.max_model_len),
        "--gpu-memory-utilization",
        str(serving.gpu_memory_utilization),
    ]
    if serving.reasoning_parser:
        command += ["--reasoning-parser", serving.reasoning_parser]
    if serving.audio_limit is not None:
        command += (
            ["--audio-limit", str(serving.audio_limit)]
            if serving.interface == "nemo_audio"
            else ["--limit-mm-per-prompt", json.dumps({"audio": serving.audio_limit})]
        )
    command += serving.extra_args
    if serving.interface == "nemo_audio":
        command.append("--")
    command += ["--data-parallel-size", str(serving.dp)]
    if serving.ep > 1:
        command.append("--enable-expert-parallel")
    command.append("--enable-prefix-caching" if serving.prefix_cache else "--no-enable-prefix-caching")
    return command + serving.engine_args


async def require_free_port(backend: dict[str, object]) -> None:
    """Reject every existing listener, regardless of model name or HTTP health."""
    # Also validate the managed URL before any processes start.
    server_command(backend)
    endpoint = urlsplit(backend["api_base"])
    host, port = endpoint.hostname, endpoint.port or 80
    try:
        async with asyncio.timeout(3):
            _, writer = await asyncio.open_connection(host, port, all_errors=True)
    except ConnectionRefusedError:
        return
    except ExceptionGroup as error:
        # localhost can resolve to both IPv4 and IPv6; both must be unoccupied.
        if all(isinstance(item, ConnectionRefusedError) for item in error.exceptions):
            return
        raise RuntimeError(f"Cannot determine whether managed port {host}:{port} is free") from error
    except (OSError, TimeoutError) as error:
        # Do not interpret an unreachable/ambiguous endpoint as an available port.
        raise RuntimeError(f"Cannot determine whether managed port {host}:{port} is free") from error
    writer.close()
    try:
        await writer.wait_closed()
    except ConnectionError:
        pass
    raise RuntimeError(
        f"Cannot start {backend['model_name']}: {host}:{port} is already occupied. "
        "Stop the existing service or choose another port in backend.yaml. "
        "Use serving: null only for an intentionally externally managed endpoint."
    )


async def healthy(backend: dict[str, object]) -> bool:
    """Require both server health and the expected served model name."""
    base = backend["api_base"].rstrip("/")
    key = OmegaConf.to_container(OmegaConf.create({"key": backend["api_key"]}), resolve=True)["key"]
    headers = {"Authorization": f"Bearer {key}"}
    try:
        for url in (base.removesuffix("/v1") + "/health", base + "/models"):
            response = await request("GET", url, headers=headers, timeout=ClientTimeout(total=3), _max_num_tries=1)
            try:
                if response.status != 200:
                    return False
                if url.endswith("/models"):
                    return any(model.get("id") == backend["model_name"] for model in (await response.json())["data"])
            finally:
                response.release()
    except (ClientError, TimeoutError, ValueError, KeyError):
        return False
    return False


async def stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop only a process group started by this controller, including DP workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 30)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


def allocation_end() -> float:
    """Read the actual Slurm end time so startup is charged to the allocation."""
    if value := os.environ.get("SLURM_JOB_END_TIME"):
        return float(value)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Expected a Slurm allocation with SLURM_JOB_ID")
    info = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True)
    match = re.search(r"\bEndTime=(\S+)", info)
    if not match or match[1] in {"Unknown", "N/A"}:
        raise RuntimeError("Cannot determine Slurm allocation end time")
    return datetime.fromisoformat(match[1]).timestamp()


async def run(path: Path) -> None:
    """Own service lifetimes; collection starts only after every local service is healthy."""
    os.environ.setdefault("NEMO_GYM_CONFIG_DICT", "{}")
    settings = load_settings(path)
    os.chdir(settings.repos.nemo_gym)
    backends = load_backends(settings.backend)
    services = managed_endpoints(backends)
    job_end = allocation_end()
    await require_preflight(settings)
    # Check every managed port before launching any service or querying CUDA.
    for backend in services.values():
        await require_free_port(backend)
    devices = []
    if services:
        first = next(iter(services.values()))
        devices = await allocated_devices(python=LocalServing.model_validate(first["serving"]).python)
        if len(devices) != settings.slurm.gpus:
            raise ValueError(f"Container exposes {len(devices)} GPUs but task settings request {settings.slurm.gpus}")
    selections = {
        base: mapped_devices(LocalServing.model_validate(backend["serving"]), devices)
        for base, backend in services.items()
    }
    settings.output.mkdir(parents=True, exist_ok=True)
    log_dir = settings.output / "services"
    log_dir.mkdir(exist_ok=True)
    processes = []
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        # Spawn all distinct services first; shared role endpoints launch once.
        for index, backend in enumerate(services.values()):
            serving = LocalServing.model_validate(backend["serving"])
            env = os.environ.copy()
            selected = selections[backend["api_base"]]
            if selected is not None:
                env["CUDA_VISIBLE_DEVICES"] = selected
            with (log_dir / f"vllm-{index}.log").open("a") as log:
                process = await asyncio.create_subprocess_exec(
                    *server_command(backend),
                    env=env,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            processes.append((process, backend, time() + serving.startup_timeout_seconds))
        pending = list(processes)
        while pending:
            for item in list(pending):
                process, backend, startup_end = item
                if process.returncode is not None:
                    raise RuntimeError(
                        f"Service {backend['model_name']} exited: {process.returncode}; inspect {log_dir}"
                    )
                if await healthy(backend):
                    pending.remove(item)
                    continue
                limit = min(startup_end, job_end - settings.simulation_tasks.export_reserve_min * 60)
                if time() >= limit:
                    raise TimeoutError(
                        f"Service {backend['model_name']} startup exceeded its deadline; inspect {log_dir}"
                    )
            if pending:
                await asyncio.sleep(2)
        if backends["teacher"]["api_base"] in services:
            await require_preflight(settings, managed_teacher=True)
        print("All local services healthy; starting fixed-concurrency collection", flush=True)
        collection = asyncio.create_task(run_collection(settings, backends, job_end=job_end))
        try:
            # Stop collection and preserve completed data if any owned service dies.
            while not collection.done():
                await asyncio.wait({collection}, timeout=2)
                failed = [backend["model_name"] for process, backend, _ in processes if process.returncode is not None]
                if failed:
                    raise RuntimeError(f"Local services exited during collection: {failed}")
            await collection
        finally:
            if not collection.done():
                collection.cancel()
            await asyncio.gather(collection, return_exceptions=True)
    finally:
        for process, _, _ in reversed(processes):
            await stop_process(process)
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.settings))


if __name__ == "__main__":
    main()
