# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make real API requests before Slurm submission; skip services managed by the job."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from aiohttp import ClientError, ClientTimeout
from omegaconf import OmegaConf

from environment_servers.audiomc_sim.backends import LocalServing, load_backends
from environment_servers.audiomc_sim.slurm.config import TaskSettings, load_settings
from environment_servers.audiomc_sim.slurm.gpus import validate_selection
from nemo_gym.server_utils import request


def managed_endpoints(backends: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Deduplicate shared local services and reject conflicting definitions."""
    managed = {}
    for role, backend in backends.items():
        if not backend.get("serving"):
            continue
        base = backend["api_base"]
        if base in managed:
            prior = managed[base]
            if prior["serving"] != backend["serving"] or prior["model_name"] != backend["model_name"]:
                raise ValueError(f"Conflicting local serving settings for {base} ({role})")
        managed[base] = backend
    for role, backend in backends.items():
        prior = managed.get(backend["api_base"])
        if prior and prior["model_name"] != backend["model_name"]:
            raise ValueError(f"Shared local endpoint has a different model_name for {role}")
    return managed


async def check(settings: TaskSettings) -> list[dict[str, object]]:
    """Probe every externally managed text/speech backend, including API assistants."""
    os.environ.setdefault("NEMO_GYM_CONFIG_DICT", "{}")
    backends = load_backends(settings.backend)
    managed = managed_endpoints(backends)
    for backend in managed.values():
        validate_selection(LocalServing.model_validate(backend["serving"]), available=settings.slurm.gpus)
    results, seen = [], set()
    for role, backend in backends.items():
        if backend["api_base"] in managed:
            continue
        key = OmegaConf.to_container(OmegaConf.create({"key": backend["api_key"]}), resolve=True)["key"]
        speech = role == "tts"
        identity = (backend["api_base"], backend["model_name"], key, speech)
        if identity in seen:
            continue
        seen.add(identity)
        payload = {"model": backend["model_name"]}
        if speech:
            payload.update(input="Hello.", voice=backend["voices"][0], response_format="wav")
        else:
            payload.update(messages=[{"role": "user", "content": "Say hello."}], max_tokens=32)
        result = {"role": role, "model": backend["model_name"]}
        try:
            response = await request(
                "POST",
                backend["api_base"] + ("/audio/speech" if speech else "/chat/completions"),
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=ClientTimeout(total=90),
                _max_num_tries=1,
            )
            try:
                result["status"] = response.status
                result["ok"] = 200 <= response.status < 300
                if speech and result["ok"]:
                    body = await response.read()
                    result["ok"] = body.startswith(b"RIFF") and body[8:12] == b"WAVE"
                elif result["ok"]:
                    result["ok"] = bool((await response.json()).get("choices"))
            finally:
                response.release()
        except (ClientError, TimeoutError, ValueError) as error:
            result.update(ok=False, error_type=type(error).__name__)
        results.append(result)
    return results


async def require_preflight(settings: TaskSettings) -> None:
    """Abort startup if any configured external model fails an actual request."""
    results = await check(settings)
    print(json.dumps(results, indent=2), flush=True)
    if not all(result["ok"] for result in results):
        raise RuntimeError("API preflight failed; aborting startup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        type=Path,
        required=True,
        help="Task YAML containing resources, backend path and collection settings.",
    )
    args = parser.parse_args()
    asyncio.run(require_preflight(load_settings(args.settings)))


if __name__ == "__main__":
    main()
