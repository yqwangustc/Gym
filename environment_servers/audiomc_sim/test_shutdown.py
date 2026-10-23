# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real HTTP cancellation drains finalization before shutting down Resources."""

import asyncio
import socket
from contextlib import asynccontextmanager

import pytest
import uvicorn
from aiohttp import ClientTimeout
from fastapi import FastAPI

from environment_servers.audiomc_sim.collect import shutdown_graph
from nemo_gym import server_utils


@pytest.mark.asyncio
@pytest.mark.parametrize("resources_available", [True, False])
async def test_shutdown_drains_canceled_http_handlers(monkeypatch, resources_available):
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "{}")
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    client = server_utils.get_global_aiohttp_client()
    entered = asyncio.Event()
    events = []
    participants = []
    sockets = []

    @asynccontextmanager
    async def resources_lifespan(app):
        yield
        events.append("resources_stopped")

    resources = FastAPI(lifespan=resources_lifespan)

    @resources.post("/verify")
    async def verify():
        await asyncio.sleep(0.1)
        events.append("verified")
        return {"saved": True}

    environment = FastAPI()
    resource_url = None

    @environment.post("/run")
    async def run():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            try:
                async with asyncio.timeout(1):
                    response = await server_utils.request(
                        "POST",
                        resource_url + "/verify",
                        _internal=True,
                        _max_num_tries=1,
                        timeout=ClientTimeout(total=1),
                    )
                    await response.read()
            except Exception:
                events.append("finalization_failed")
            events.append("handler_drained")
            raise

    try:
        # Deliberately list Resources first: shutdown order must follow dependencies.
        for kind, app in [("resources_servers", resources), ("environment_servers", environment)]:
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=sock.getsockname()[1],
                    log_level="critical",
                    timeout_graceful_shutdown=0.05,
                )
            )
            task = asyncio.create_task(server.serve(sockets=[sock]))
            participants.append((kind, server, task))
            if kind == "resources_servers":
                resource_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        async with asyncio.timeout(5):
            while not all(server.started for _, server, _ in participants):
                await asyncio.sleep(0.01)
        if not resources_available:
            participants[0][1].should_exit = True
            await participants[0][2]
        environment_url = f"http://127.0.0.1:{sockets[-1].getsockname()[1]}"
        call = asyncio.create_task(
            server_utils.request(
                "POST",
                environment_url + "/run",
                _max_num_tries=1,
                timeout=ClientTimeout(total=5),
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(shutdown_graph(participants), 5)
        await asyncio.gather(call, return_exceptions=True)
        assert "handler_drained" in events
        if resources_available:
            assert events.index("verified") < events.index("handler_drained") < events.index("resources_stopped")
        else:
            assert "finalization_failed" in events and "verified" not in events
        assert all(task.done() for _, _, task in participants)
        assert all(not server.server_state.tasks for _, server, _ in participants)
    finally:
        for _, server, task in participants:
            server.should_exit = True
            if not task.done():
                task.cancel()
        await asyncio.gather(*(task for _, _, task in participants), return_exceptions=True)
        await client.close()
        for sock in sockets:
            sock.close()
