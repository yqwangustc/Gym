# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable submission, endpoint preflight, service lifecycle and durable stopping."""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from environment_servers.audiomc_sim.backends import TEXT_ROLES, LocalServing, load_backends
from environment_servers.audiomc_sim.slurm import incontainer_run, preflight, run, submit
from environment_servers.audiomc_sim.slurm.config import load_settings


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("AUDIO_TEST_BASE", raising=False)
    monkeypatch.delenv("AUDIO_TEST_KEY", raising=False)
    repo = Path(__file__).resolve().parents[3]
    (tmp_path / "credentials.env").write_text("AUDIO_TEST_BASE=https://api.invalid/v1\nAUDIO_TEST_KEY=test-secret\n")
    (tmp_path / "personas.jsonl").write_text('{"uuid":"one","persona":"A curious person."}\n')
    (tmp_path / "assistant.md").write_text("Answer conversationally.\n")
    backends = {
        role: {"api_base": "${oc.env:AUDIO_TEST_BASE}", "api_key": "${oc.env:AUDIO_TEST_KEY}", "model_name": "shared"}
        for role in (*TEXT_ROLES, "tts")
    }
    backends["tts"].update(model_name="speech", voices=["alloy"])
    backends["assistant"] = {
        "api_base": "http://127.0.0.1:8000/v1",
        "model_name": "assistant",
        "serving": {"checkpoint": "/models/assistant", "startup_timeout_seconds": 1},
    }
    OmegaConf.save(OmegaConf.create(backends), tmp_path / "backend.yaml")
    task = {
        "repos": {"nemo_gym": str(repo)},
        "container": "./image.sqsh",
        "environment": "credentials.env",
        "backend": "backend.yaml",
        "output": "output",
        "mc_resources": {
            "persona": "personas.jsonl",
            "eval_axis": str(repo / "resources_servers/audiomc_sim/data/eval_axis.yaml"),
            "topics": str(repo / "resources_servers/audiomc_sim/data/topics.yaml"),
            "assistant_system_prompt": "assistant.md",
        },
        "simulation_tasks": {"target_trajectory": 2, "concurrency": 4, "max_turns": 2, "max_teacher_trials": 2},
        "slurm": {"gpus": 1, "partition": "test", "account": "test-account"},
    }
    OmegaConf.save(OmegaConf.create(task), tmp_path / "tasks.settings.yaml")
    settings = load_settings(tmp_path / "tasks.settings.yaml")
    return settings, load_backends(settings.backend), tmp_path / "tasks.settings.yaml"


def test_environment_loaded_before_interpolation_and_snapshots_keep_key_reference(deployment):
    settings, backends, path = deployment
    assert settings.output == path.parent / "output"
    assert settings.container == str(path.parent / "image.sqsh")
    assert backends["user"]["api_base"] == "https://api.invalid/v1"
    assert backends["user"]["api_key"] == "${oc.env:AUDIO_TEST_KEY}"
    assert backends["assistant"]["checkpoint"] == "/models/assistant"
    assert "test-secret" not in json.dumps(backends)
    assert settings.simulation_tasks.max_rollouts == 4


def test_submission_and_container_commands_use_settings_without_cluster_defaults(deployment):
    settings, _, path = deployment
    command = submit.sbatch_command(settings, path)
    assert "--partition=test" in command and "--account=test-account" in command
    assert "--gpus-per-node=1" in command and str(path) in command
    command = submit.srun_command(settings, path)
    assert f"--container-image={settings.container}" in command
    assert "environment_servers.audiomc_sim.slurm.incontainer_run" in command
    assert str(path) == command[-1]
    assert not any("ORD_" in value or "/lustre/" in value for value in command)


def test_nemo_and_standard_serving_flags_and_ep_validation(deployment):
    _, backends, _ = deployment
    backend = backends["assistant"]
    backend["serving"].update(
        tp=1,
        dp=8,
        ep=8,
        interface="nemo_audio",
        entrypoint="/repo/serve.py",
        audio_limit=50,
        engine_args=["--mm-processor-cache-gb", "0"],
    )
    command = incontainer_run.server_command(backend)
    assert command[:2] == ["python3", "/repo/serve.py"]
    assert command[command.index("--audio-limit") + 1] == "50"
    assert command.index("--") < command.index("--data-parallel-size")
    assert "--enable-expert-parallel" in command and "--enable-prefix-caching" in command
    backend["serving"].update(interface="vllm", prefix_cache=False)
    command = incontainer_run.server_command(backend)
    assert "--" not in command and "--limit-mm-per-prompt" in command
    assert "--no-enable-prefix-caching" in command
    with pytest.raises(ValidationError, match="EP size"):
        LocalServing(checkpoint="model", tp=2, dp=4, ep=4)


@pytest.mark.asyncio
async def test_preflight_checks_external_assistant_deduplicates_and_releases(deployment, monkeypatch):
    settings, backends, _ = deployment
    backends["assistant"].update(serving=None, api_base="https://assistant.invalid/v1")
    calls, released = [], []

    class Response:
        def __init__(self, url):
            self.url = url
            self.status = 410 if "assistant.invalid" in url else 200

        async def json(self):
            return {"choices": [{"message": {"content": "hello"}}]}

        async def read(self):
            return b"RIFF0000WAVEpayload"

        def release(self):
            released.append(self.url)

    async def request(method, url, **kwargs):
        calls.append(url)
        return Response(url)

    monkeypatch.setattr(preflight, "load_backends", lambda _: backends)
    monkeypatch.setattr(preflight, "request", request)
    results = await preflight.check(settings)
    assert len(calls) == 3 and calls == released
    assert results[1]["role"] == "assistant" and not results[1]["ok"]
    with pytest.raises(RuntimeError, match="preflight failed"):
        await preflight.require_preflight(settings)


def test_shared_local_model_starts_once_and_conflicts_rejected(deployment):
    _, backends, _ = deployment
    backends["user"] = dict(backends["assistant"])
    backends["teacher"] = {**backends["assistant"], "serving": None}
    assert len(preflight.managed_endpoints(backends)) == 1
    backends["user"] = {**backends["user"], "model_name": "different"}
    with pytest.raises(ValueError, match="Conflicting"):
        preflight.managed_endpoints(backends)


def test_prepare_materializes_requested_prompt_and_resumes_frozen_tasks(deployment):
    settings, backends, _ = deployment
    run.prepare_collection(settings, backends)
    rows = [json.loads(line) for line in (settings.output / "tasks.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(row["task_input"]["assistant_instructions"] == "Answer conversationally.\n" for row in rows)
    original = (settings.output / "tasks.jsonl").read_bytes()
    run.prepare_collection(settings, backends)
    assert (settings.output / "tasks.jsonl").read_bytes() == original
    settings.simulation_tasks.concurrency += 1
    with pytest.raises(ValueError, match="settings differ"):
        run.prepare_collection(settings, backends)


@pytest.mark.asyncio
async def test_deadline_cancels_collection_before_final_export(deployment, monkeypatch):
    settings, backends, _ = deployment
    settings.simulation_tasks.export_reserve_min = 0.001
    cancelled = asyncio.Event()

    async def collect(args):
        assert not args.fixed_rollouts and args.target_accepted == 2 and args.concurrency == 4
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    exports = []

    def export(path, **kwargs):
        assert cancelled.is_set()
        exports.append(kwargs["max_audio_files"])
        return {"rollout_counter": 1, "accepted_examples": 1}

    monkeypatch.setattr(run, "collect", collect)
    monkeypatch.setattr(run, "export_collection", export)
    # Preparation precedes the bounded collection; give it enough time even on slow CI.
    run.prepare_collection(settings, backends)
    await run.run_collection(settings, backends, job_end=run.time() + 0.12)
    runtime = json.loads((settings.output / "slurm_runtime.json").read_text())
    assert runtime["deadline_interrupted"] and runtime["accepted_examples"] == 1
    assert exports == [50_000]


@pytest.mark.asyncio
async def test_controller_waits_for_shared_service_then_cleans_up(deployment, monkeypatch):
    settings, backends, path = deployment
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(run.time() + 3600))
    events = []

    class Process:
        pid = 99999
        returncode = None

    async def start(*args, **kwargs):
        events.append("start")
        return Process()

    async def healthy(backend):
        events.append("health")
        return "start" in events

    async def collect(*args, **kwargs):
        events.append("collect")
        return {}

    async def stop(process):
        events.append("stop")

    async def preflight_ok(settings):
        events.append("preflight")

    async def free_port(_):
        events.append("port_free")

    monkeypatch.setattr(incontainer_run, "require_free_port", free_port)
    monkeypatch.setattr(incontainer_run, "load_settings", lambda _: settings)
    monkeypatch.setattr(incontainer_run, "load_backends", lambda _: backends)
    monkeypatch.setattr(incontainer_run, "require_preflight", preflight_ok)
    monkeypatch.setattr(incontainer_run, "healthy", healthy)
    monkeypatch.setattr(incontainer_run.asyncio, "create_subprocess_exec", start)
    monkeypatch.setattr(incontainer_run, "run_collection", collect)
    monkeypatch.setattr(incontainer_run, "stop_process", stop)
    await incontainer_run.run(path)
    assert events == ["preflight", "port_free", "start", "health", "collect", "stop"]


@pytest.mark.asyncio
async def test_startup_failure_never_collects_and_stops_owned_process(deployment, monkeypatch):
    settings, backends, path = deployment
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(run.time() + 3600))
    events = []

    class Process:
        returncode = 1

    async def start(*args, **kwargs):
        return Process()

    async def healthy(_):
        return False

    async def stop(_):
        events.append("stopped")

    async def preflight_ok(_):
        pass

    monkeypatch.setattr(incontainer_run, "require_free_port", preflight_ok)

    async def never_collect(*args, **kwargs):
        pytest.fail("collection started before readiness")

    monkeypatch.setattr(incontainer_run, "load_settings", lambda _: settings)
    monkeypatch.setattr(incontainer_run, "load_backends", lambda _: backends)
    monkeypatch.setattr(incontainer_run, "require_preflight", preflight_ok)
    monkeypatch.setattr(incontainer_run, "healthy", healthy)
    monkeypatch.setattr(incontainer_run.asyncio, "create_subprocess_exec", start)
    monkeypatch.setattr(incontainer_run, "stop_process", stop)
    monkeypatch.setattr(incontainer_run, "run_collection", never_collect)
    with pytest.raises(RuntimeError, match="exited"):
        await incontainer_run.run(path)
    assert events == ["stopped"]


@pytest.mark.asyncio
async def test_external_preflight_failure_prevents_submission(deployment, monkeypatch):
    _, _, path = deployment
    submitted = []

    async def fail(_):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(submit, "require_preflight", fail)
    monkeypatch.setattr(submit.subprocess, "run", lambda *args, **kwargs: submitted.append(args))
    monkeypatch.setattr("sys.argv", ["submit", "--settings", str(path)])
    # main() owns asyncio.run; execute it outside this test's running loop.
    with pytest.raises(RuntimeError, match="API unavailable"):
        await asyncio.to_thread(submit.main)
    assert submitted == []


@pytest.mark.asyncio
async def test_actual_health_request_requires_expected_model(deployment):
    from aiohttp import web

    settings, backends, _ = deployment
    app = web.Application()

    async def health(request):
        return web.Response(status=200)

    async def models(request):
        return web.json_response({"data": [{"id": "expected"}]})

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        backend = {**backends["assistant"], "api_base": f"http://127.0.0.1:{port}/v1", "model_name": "wrong"}
        assert not await incontainer_run.healthy(backend)
        backend["model_name"] = "expected"
        assert await incontainer_run.healthy(backend)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cancellation_exports_before_returning(deployment, monkeypatch):
    settings, backends, _ = deployment
    run.prepare_collection(settings, backends)
    ready, cancelled = asyncio.Event(), asyncio.Event()

    async def collect(args):
        ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    exported = []

    def export(*args, **kwargs):
        assert cancelled.is_set()
        exported.append(True)
        return {"rollout_counter": 1, "accepted_examples": 1}

    monkeypatch.setattr(run, "collect", collect)
    monkeypatch.setattr(run, "export_collection", export)
    task = asyncio.create_task(run.run_collection(settings, backends, job_end=run.time() + 3600))
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert exported == [True]
    assert (settings.output / "slurm_runtime.json").exists()


def test_submission_freezes_host_paths_and_batch_runs_srun_directly(deployment, monkeypatch, tmp_path):
    settings, _, path = deployment
    # Shell metacharacters must survive as literal arguments in the generated script.
    settings.output = tmp_path / "output with spaces $(touch INJECTED)"
    settings.slurm.python = "/container/python with spaces"
    monkeypatch.setattr(submit, "load_settings", lambda _: settings)
    monkeypatch.setattr("sys.argv", ["submit", "--settings", str(path)])
    commands = []

    async def preflight_ok(_):
        assert not (settings.output / "run.sbatch").exists()

    monkeypatch.setattr(submit, "require_preflight", preflight_ok)
    with monkeypatch.context() as patch:
        patch.setattr(submit.subprocess, "run", lambda command, **kwargs: commands.append(command))
        submit.main()
    launch = settings.output / "launch_settings.yaml"
    snapshot = OmegaConf.to_container(OmegaConf.load(launch), resolve=False)
    assert snapshot["environment"] == str(settings.environment)
    assert snapshot["repos"]["nemo_gym"] == str(settings.repos.nemo_gym)
    script = settings.output / "run.sbatch"
    assert commands == [submit.sbatch_command(settings, script)]
    assert "test-secret" not in launch.read_text()
    assert "test-secret" not in script.read_text()
    # Execute the real bash script with a stand-in for srun; no host Python is invoked.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    srun = bin_dir / "srun"
    srun.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$CAPTURE_ARGS"\n')
    srun.chmod(0o755)
    capture = tmp_path / "args"
    subprocess.run(
        ["bash", str(script)],
        check=True,
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "CAPTURE_ARGS": str(capture)},
    )
    assert capture.read_bytes().decode().split("\0")[:-1] == submit.srun_command(settings, launch)[1:]
    assert not (tmp_path / "INJECTED").exists()


@pytest.mark.asyncio
async def test_managed_port_rejects_any_listener_and_leaves_it_running(deployment):
    _, backends, _ = deployment

    async def close(reader, writer):
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(close, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    backend = {**backends["assistant"], "api_base": f"http://127.0.0.1:{port}/v1"}
    try:
        # This listener has no HTTP health endpoint; it must still block startup.
        with pytest.raises(RuntimeError, match="already occupied"):
            await incontainer_run.require_free_port(backend)
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()
    await incontainer_run.require_free_port(backend)
    await incontainer_run.require_free_port({**backend, "api_base": f"http://localhost:{port}/v1"})


@pytest.mark.asyncio
async def test_invalid_gpu_plan_aborts_preflight_before_api_requests(deployment, monkeypatch):
    settings, _, _ = deployment
    settings.slurm.gpus = 2  # The managed model has TP1/DP1 and no explicit subset.

    async def never_request(*args, **kwargs):
        pytest.fail("invalid GPU settings should fail before calling any API")

    monkeypatch.setattr(preflight, "request", never_request)
    with pytest.raises(ValueError, match="Selected GPU count 2 must equal TP"):
        await preflight.check(settings)


@pytest.mark.asyncio
async def test_controller_translates_slurm_devices_for_child(deployment, monkeypatch):
    settings, backends, path = deployment
    settings.slurm.gpus = 2
    backends["assistant"]["serving"]["cuda_visible_devices"] = [0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(run.time() + 3600))
    captured = []

    class Process:
        pid = 99999
        returncode = None

    async def noop(*args, **kwargs):
        return {}

    async def healthy(_):
        return True

    async def start(*args, **kwargs):
        captured.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
        return Process()

    monkeypatch.setattr(incontainer_run, "load_settings", lambda _: settings)
    monkeypatch.setattr(incontainer_run, "load_backends", lambda _: backends)
    monkeypatch.setattr(incontainer_run, "require_preflight", noop)
    monkeypatch.setattr(incontainer_run, "require_free_port", noop)
    monkeypatch.setattr(incontainer_run, "healthy", healthy)
    monkeypatch.setattr(incontainer_run.asyncio, "create_subprocess_exec", start)
    monkeypatch.setattr(incontainer_run, "run_collection", noop)
    monkeypatch.setattr(incontainer_run, "stop_process", noop)
    await incontainer_run.run(path)
    assert captured == ["4"]


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["deadline", "cancel"])
async def test_final_export_waits_for_cancelled_periodic_export(deployment, monkeypatch, interrupt):
    import threading
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from environment_servers.audiomc_sim import collect as collector

    settings, backends, _ = deployment
    settings.simulation_tasks.export_every_min = 0.0001
    settings.simulation_tasks.export_reserve_min = 0.0001
    run.prepare_collection(settings, backends)
    export_started, release_export = threading.Event(), threading.Event()
    order = []

    @asynccontextmanager
    async def graph(_):
        async def post(**kwargs):
            await asyncio.Event().wait()

        yield SimpleNamespace(post=post)

    def export(*args, **kwargs):
        if not export_started.is_set():
            order.append("periodic_start")
            export_started.set()
            if not release_export.wait(3):
                raise AssertionError("export was not released")
            order.append("periodic_finish")
        else:
            order.append("final_start")
        return {"rollout_counter": 0, "accepted_examples": 0}

    original_collect = collector.collect

    async def unmonitored(args):
        args.monitor = False
        await original_collect(args)

    monkeypatch.setattr(collector, "serve_graph", graph)
    monkeypatch.setattr(collector, "export_collection", export)
    monkeypatch.setattr(run, "export_collection", export)
    monkeypatch.setattr(run, "collect", unmonitored)
    budget = 0.10 if interrupt == "deadline" else 3600
    task = asyncio.create_task(run.run_collection(settings, backends, job_end=run.time() + budget))
    try:
        assert await asyncio.to_thread(export_started.wait, 2)
        if interrupt == "cancel":
            task.cancel()
        await asyncio.sleep(0.15)
        assert not task.done() and order == ["periodic_start"]
    finally:
        release_export.set()
        if interrupt == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        else:
            await asyncio.wait_for(task, 2)
    assert order == ["periodic_start", "periodic_finish", "final_start"]
