# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concurrency bounds, durable single-writer records and timing-context isolation."""

import asyncio
import json
import threading
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from environment_servers.audiomc_sim.collect import collect, graph_config
from environment_servers.audiomc_sim.monitor import prometheus_values
from resources_servers.audiomc_sim.telemetry import CURRENT_CALL, trace_call


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 4, 8])
@pytest.mark.parametrize("export_every", [0, 2])
@pytest.mark.parametrize("export_every_min", [0, 230])
async def test_fixed_budget_bounded_workers_and_final_export(
    tmp_path, monkeypatch, concurrency, export_every, export_every_min
):
    graph = {"environment": {"environment_servers": {"audiomc_sim": {"max_concurrent_episodes": concurrency}}}}
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "test", "graph": graph}))
    (tmp_path / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_input": {"axis_id": "voice_editing"}, "task_id": {"taskset": "test", "task_id": str(i)}})
            + "\n"
            for i in range(7)
        )
    )
    active, peak = 0, 0
    ids = []

    async def post(**kwargs):
        nonlocal active, peak
        envelope = kwargs["json"]
        active += 1
        peak = max(active, peak)
        ids.append(envelope["episode_id"]["rollout_id"])
        await asyncio.sleep(0.01)
        active -= 1
        index = int(envelope["task"]["task_id"]["task_id"])
        if index == 3:
            raise ConnectionError("Simulated lost response")
        result = {
            "outcome": "teacher_correction_exhausted"
            if index == 4
            else "max_turn_reached_without_failing"
            if index == 5
            else "accepted_correction",
            "accepted": index != 5,
        }
        row = {"episode_id": envelope["episode_id"], "task_id": envelope["task"]["task_id"], "result": result}
        return SimpleNamespace(ok=True, read=AsyncMock(return_value=json.dumps(row).encode()))

    @asynccontextmanager
    async def serve(config):
        yield SimpleNamespace(post=post)

    exports = []

    def export(output, *, max_audio_files):
        assert max_audio_files == 2
        rows = [json.loads(line) for line in (output / "episode_results.jsonl").read_text().splitlines()]
        assert len(rows) in ({2, 4, 6, 7} if export_every else {7})
        if len(rows) == 7:
            assert active == 0
        exports.append(rows)
        return {"accepted_examples": 5}

    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setattr("environment_servers.audiomc_sim.collect.serve_graph", serve)
    monkeypatch.setattr("environment_servers.audiomc_sim.collect.export_collection", export)
    await asyncio.wait_for(
        collect(
            Namespace(
                output=tmp_path,
                env_file=None,
                max_rollouts=7,
                target_accepted=1,
                concurrency=concurrency,
                fixed_rollouts=True,
                export_every=export_every,
                export_every_min=export_every_min,
                max_audio_files=2,
            )
        ),
        timeout=2,
    )
    assert peak == min(7, concurrency)
    assert len(ids) == len(set(ids)) == 7
    assert len(exports) == (4 if export_every else 1)
    assert len(list((tmp_path / "requests").glob("*.json"))) == 7
    timings = [json.loads(line) for line in (tmp_path / "rollout_timings.jsonl").read_text().splitlines()]
    assert len(timings) == 7 and all(t["elapsed_seconds"] >= 0.01 for t in timings)
    assert sum(t["outcome"] == "infrastructure_failure" for t in timings) == 1
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["completed_in_process"] == 7 and runtime["accepted_in_process"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("export_every", [0, 2])
@pytest.mark.parametrize("timed_count", [1, 2])
async def test_timed_export_during_unfinished_conversation_is_serialized(
    tmp_path, monkeypatch, export_every, timed_count
):
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "test", "graph": {}}))
    (tmp_path / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_input": {"axis_id": "voice_editing"}}) + "\n" for _ in range(2))
    )
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    timer_started, second_started, finish_second, second_replied = (asyncio.Event() for _ in range(4))
    finish_export = threading.Event()
    exports = []
    active_exports = 0

    async def post(**kwargs):
        envelope = kwargs["json"]
        if envelope["episode_id"]["rollout_id"].endswith("0001"):
            second_started.set()
            await finish_second.wait()
            second_replied.set()
        row = {"episode_id": envelope["episode_id"], "result": {"outcome": "accepted_correction", "accepted": True}}
        return SimpleNamespace(ok=True, read=AsyncMock(return_value=json.dumps(row).encode()))

    @asynccontextmanager
    async def serve(config):
        yield SimpleNamespace(post=post)

    def export(output, *, max_audio_files):
        nonlocal active_exports
        active_exports += 1
        try:
            assert active_exports == 1
            assert max_audio_files == 2
            rows = (output / "episode_results.jsonl").read_text().splitlines()
            exports.append(len(rows))
            if len(rows) == 1:
                assert threading.get_ident() != loop_thread
                if exports.count(1) == timed_count:
                    loop.call_soon_threadsafe(timer_started.set)
                    assert finish_export.wait(2)
            return {"accepted_examples": len(rows)}
        finally:
            active_exports -= 1

    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setattr("environment_servers.audiomc_sim.collect.serve_graph", serve)
    monkeypatch.setattr("environment_servers.audiomc_sim.collect.export_collection", export)
    task = asyncio.create_task(
        collect(
            Namespace(
                output=tmp_path,
                env_file=None,
                max_rollouts=2,
                target_accepted=1,
                concurrency=1,
                fixed_rollouts=True,
                export_every=export_every,
                export_every_min=0.001,
                max_audio_files=2,
            )
        )
    )
    try:
        await asyncio.wait_for(timer_started.wait(), 2)
        assert second_started.is_set() and not task.done()
        finish_second.set()
        await asyncio.wait_for(second_replied.wait(), 2)
        # The event loop runs during export, but ledger publication shares the export lock.
        assert len((tmp_path / "episode_results.jsonl").read_text().splitlines()) == 1
    finally:
        finish_export.set()
        finish_second.set()
        await asyncio.wait_for(task, 2)
    assert exports == [1] * timed_count + ([2, 2] if export_every else [2])


@pytest.mark.asyncio
async def test_call_context_isolated_between_sessions():
    events = []

    async def call(role):
        with trace_call(events, role) as timing:
            await asyncio.sleep(0)
            assert CURRENT_CALL.get() is timing
            timing["usage"] = {"output_tokens": int(role)}

    await asyncio.gather(call("1"), call("2"))
    assert CURRENT_CALL.get() is None
    assert {e["role"]: e["usage"]["output_tokens"] for e in events} == {"1": 1, "2": 2}
    with pytest.raises(ValueError):
        with trace_call(events, "error"):
            raise ValueError("Invalid response")
    assert events[-1]["success"] is False and events[-1]["error_type"] == "ValueError"
    assert CURRENT_CALL.get() is None


def test_prometheus_aggregate_counters_and_skip_buckets():
    values = prometheus_values("""vllm:num_requests_running{engine="0"} 2
vllm:request_success_total{finished_reason="stop"} 3
vllm:request_success_total{finished_reason="length"} 1
vllm:request_queue_time_seconds_sum{engine="0"} 0.5
vllm:request_queue_time_seconds_count{engine="0"} 4
vllm:request_queue_time_seconds_bucket{le="1"} 4
vllm:request_success_created{engine="0"} 123
""")
    assert values == {
        "vllm:num_requests_running": 2,
        "vllm:request_success_total": 4,
        "vllm:request_queue_time_seconds_sum": 0.5,
        "vllm:request_queue_time_seconds_count": 4,
    }


def test_graph_admission_matches_collector_concurrency(tmp_path):
    graph = graph_config(
        output=tmp_path,
        base_port=18240,
        concurrency=16,
    )
    assert graph["environment"]["environment_servers"]["audiomc_sim"]["max_concurrent_episodes"] == 16


def test_benchmark_report_uses_counter_deltas_and_separates_outcomes(tmp_path):
    from environment_servers.audiomc_sim.benchmark_report import summarize_run

    episodes = tmp_path / "artifacts/episodes"
    episodes.mkdir(parents=True)
    (tmp_path / "runtime.json").write_text(json.dumps({"collection_seconds": 100, "completed_in_process": 2}))
    (tmp_path / "episode_results.jsonl").write_text(
        json.dumps({"result": {"outcome": "accepted_correction", "accepted": True}})
        + "\n"
        + json.dumps({"failure": {"failure_reason": "Timeout"}})
        + "\n"
    )
    timing = {
        "kind": "timing",
        "role": "assistant",
        "elapsed_seconds": 10,
        "success": True,
        "usage": {"input_tokens": 50, "output_tokens": 100},
    }
    (episodes / "a.json").write_text(
        json.dumps(
            {
                "invocations": [timing],
                "resource_invocations": [],
                "turn_evaluations": [{"turn_index": 1, "judgment": {"passed": False}}],
            }
        )
    )
    samples = []
    for start, queue_sum, count in [(True, 20, 10), (False, 22, 11)]:
        samples.append(
            {
                "timestamp": 1 if start else 101,
                "event_loop_lag_seconds": 0.01,
                "collector_cpu_percent": 2,
                "gpu": [[50, 80000, 100, 20]],
                "vllm": {
                    "vllm:request_queue_time_seconds_sum": queue_sum,
                    "vllm:request_queue_time_seconds_count": count,
                    "vllm:num_requests_running": 1,
                    "vllm:num_requests_waiting": 0,
                },
            }
        )
    (tmp_path / "system_metrics.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    report = summarize_run(tmp_path)
    assert report["completed_rollouts"] == 2
    assert report["outcomes"] == {"accepted_correction": 1, "infrastructure_failure": 1}
    assert report["vllm_request_means"]["request_queue_time_seconds"] == 2
    assert report["first_failure_exchanges"] == {"2": 1}
    assert report["roles"]["assistant"]["total_output_tokens"] == 100
    assert report["roles"]["assistant"]["latency_seconds"]["mean"] == 10
    assert report["gpu_utilization_percent"]["mean"] == 50
