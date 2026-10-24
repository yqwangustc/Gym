# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make real API requests before Slurm submission; skip services managed by the job."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from time import perf_counter

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


async def check(settings: TaskSettings, *, managed_teacher: bool = False) -> list[dict[str, object]]:
    """Probe every externally managed text/speech backend, including API assistants."""
    os.environ.setdefault("NEMO_GYM_CONFIG_DICT", "{}")
    print(f"[preflight] Loading backend settings: {settings.backend}", flush=True)
    backends = load_backends(settings.backend)
    managed = managed_endpoints(backends)
    print(
        f"[preflight] PASS backend configuration: {len(backends)} roles, {len(managed)} managed services", flush=True
    )
    for backend in managed.values():
        serving = LocalServing.model_validate(backend["serving"])
        validate_selection(serving, available=settings.slurm.gpus)
        print(
            f"[preflight] PASS planned GPU selection for {backend['model_name']}: "
            f"allocation={settings.slurm.gpus}, TP={serving.tp}, DP={serving.dp}, EP={serving.ep} "
            "(actual allocation checked inside the job)",
            flush=True,
        )
    phase = "managed Teacher reasoning" if managed_teacher else "external API"
    print(f"[preflight] Starting {phase} probes (90-second timeout per request)", flush=True)
    results, seen = [], {}
    for role, backend in backends.items():
        if managed_teacher:
            if role != "teacher" or backend["api_base"] not in managed:
                continue
        elif backend["api_base"] in managed:
            print(f"[preflight] DEFER {role}: locally managed; health checked after service startup", flush=True)
            continue
        key = OmegaConf.to_container(OmegaConf.create({"key": backend["api_key"]}), resolve=True)["key"]
        speech = role == "tts"
        needs_reasoning = role == "teacher"
        identity = (
            backend["api_base"],
            backend["model_name"],
            key,
            speech,
            backend.get("reasoning_effort"),
            needs_reasoning,
        )
        if identity in seen:
            print(f"[preflight] SKIP {role}: shares the {seen[identity]} probe", flush=True)
            continue
        seen[identity] = role
        payload = {"model": backend["model_name"]}
        if speech:
            payload.update(input="Hello.", voice=backend["voices"][0], response_format="wav")
        else:
            payload.update(
                messages=[
                    {
                        "role": "user",
                        "content": "What is 17 times 23? Give only the answer." if needs_reasoning else "Say hello.",
                    }
                ],
                max_tokens=1024 if needs_reasoning else 32,
            )
            if backend.get("reasoning_effort") is not None:
                payload["reasoning_effort"] = backend["reasoning_effort"]
        result = {"role": role, "model": backend["model_name"]}
        if backend.get("reasoning_effort") is not None:
            result["reasoning_effort"] = backend["reasoning_effort"]
        if speech:
            detail = f"WAV synthesis, voice={backend['voices'][0]}"
        elif needs_reasoning:
            detail = f"final answer + native reasoning, effort={backend.get('reasoning_effort') or 'provider default'}"
        else:
            detail = "chat completion"
        print(f"[preflight] CHECK {role}: {backend['model_name']} ({detail})", flush=True)
        started = perf_counter()
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
                if not result["ok"]:
                    result["error"] = f"HTTP {response.status}"
                if speech and result["ok"]:
                    body = await response.read()
                    result["ok"] = body.startswith(b"RIFF") and body[8:12] == b"WAVE"
                    if not result["ok"]:
                        result["error"] = "TTS endpoint did not return WAV audio"
                elif result["ok"]:
                    choices = (await response.json()).get("choices") or []
                    result["ok"] = bool(choices)
                    if not choices:
                        result["error"] = "Chat endpoint returned no choices"
                    if needs_reasoning:
                        choice = choices[0] if choices else {}
                        message = choice.get("message") or {}
                        reasoning = message.get("reasoning_content") or message.get("reasoning")
                        answer = message.get("content")
                        result["native_reasoning_available"] = isinstance(reasoning, str) and bool(reasoning.strip())
                        if not backend["uses_reasoning_parser"]:
                            result.update(ok=False, error="Teacher requires uses_reasoning_parser=true")
                        elif not result["native_reasoning_available"]:
                            result.update(ok=False, error="Teacher endpoint returned no native reasoning text")
                        elif not isinstance(answer, str) or not answer.strip():
                            result.update(ok=False, error="Teacher endpoint returned no final answer")
                        elif choice.get("finish_reason") == "length":
                            result.update(ok=False, error="Teacher reasoning probe did not complete")
            finally:
                response.release()
        except (ClientError, TimeoutError, ValueError) as error:
            result.update(ok=False, error_type=type(error).__name__)
        verdict = "PASS" if result["ok"] else "FAIL"
        if not result["ok"]:
            evidence = result.get("error", result.get("error_type", "response validation failed"))
        elif speech:
            evidence = "WAV audio received"
        elif needs_reasoning:
            evidence = "final answer and native reasoning received"
        else:
            evidence = "chat response received"
        status = f"HTTP {result['status']}; " if "status" in result else ""
        print(f"[preflight] {verdict} {role}: {status}{evidence} ({perf_counter() - started:.1f}s)", flush=True)
        results.append(result)
    return results


async def require_preflight(settings: TaskSettings, *, managed_teacher: bool = False) -> None:
    """Abort startup if any configured external model fails an actual request."""
    started = perf_counter()
    try:
        results = await check(settings, managed_teacher=managed_teacher)
    except Exception as error:
        print(f"[preflight] FAIL configuration/probe setup: {type(error).__name__}", flush=True)
        raise
    print(json.dumps(results, indent=2), flush=True)
    passed = sum(bool(result["ok"]) for result in results)
    verdict = "PASS" if passed == len(results) else "FAIL"
    print(
        f"[preflight] {verdict}: {passed}/{len(results)} API probes passed in {perf_counter() - started:.1f}s",
        flush=True,
    )
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
    print(f"[preflight] Loading task settings and environment: {args.settings}", flush=True)
    asyncio.run(require_preflight(load_settings(args.settings)))


if __name__ == "__main__":
    main()
