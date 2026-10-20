# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-frequency GPU, vLLM, process and event-loop observations for collection."""

import asyncio
import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from time import perf_counter, time

import psutil

from nemo_gym.server_utils import raise_for_status, request


def prometheus_values(text: str) -> dict[str, float]:
    """Keep aggregate gauges/counters and histogram sums/counts, omitting buckets."""
    values = {}
    names = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
    }
    for line in text.splitlines():
        if not line.startswith("vllm:") or "_bucket{" in line:
            continue
        label, value = line.rsplit(" ", 1)
        name = label.split("{", 1)[0]
        if name not in names and not name.endswith(("_sum", "_count")):
            continue
        number = float(value)
        if math.isfinite(number):
            values[name] = values.get(name, 0) + number
    return values


class CollectionMonitor:
    """Write observations independently of the bounded rollout worker pool."""

    def __init__(self, *, output: Path, assistant_base: str, active: Callable[[], int], interval: float = 5) -> None:
        self.output = output
        self.metrics_url = assistant_base.rstrip("/").removesuffix("/v1") + "/metrics"
        self.active = active
        self.interval = interval
        self.stop = asyncio.Event()
        self.process = psutil.Process()
        self.process.cpu_percent()

    async def sample(self, *, lag: float = 0) -> None:
        row = {
            "timestamp": time(),
            "active_rollouts": self.active(),
            "event_loop_lag_seconds": lag,
            "collector_cpu_percent": self.process.cpu_percent(),
            "collector_rss_bytes": self.process.memory_info().rss,
        }
        try:
            async with asyncio.timeout(3):
                response = await request("GET", self.metrics_url, _max_num_tries=1)
                await raise_for_status(response)
                row["vllm"] = prometheus_values(await response.text())
        except Exception as error:
            row["vllm_scrape_error"] = type(error).__name__
        if shutil.which("nvidia-smi"):
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw,utilization.memory",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), 3)
                row["gpu"] = [
                    [float(x.strip()) for x in line.split(",")]
                    for line in stdout.decode(errors="replace").splitlines()
                ]
            except (ValueError, TimeoutError):
                row["gpu_sample_error"] = True
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
        with (self.output / "system_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    async def run(self) -> None:
        while not self.stop.is_set():
            due = perf_counter() + self.interval
            try:
                await asyncio.wait_for(self.stop.wait(), self.interval)
            except TimeoutError:
                await self.sample(lag=max(0, perf_counter() - due))
