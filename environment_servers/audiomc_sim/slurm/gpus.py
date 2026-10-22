# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate parallelism and map allocation-relative GPU selections safely."""

import asyncio
import json
import os

from environment_servers.audiomc_sim.backends import LocalServing


def validate_selection(serving: LocalServing, *, available: int) -> None:
    """Validate GPU count and selection bounds against planned or actual capacity."""
    selected = serving.cuda_visible_devices
    count = available if selected is None else len(selected)
    if selected is not None and any(index >= available for index in selected):
        raise ValueError(f"cuda_visible_devices must index the {available} allocated GPUs, starting at zero")
    expected = serving.tp * serving.dp
    if count != expected:
        raise ValueError(f"Selected GPU count {count} must equal TP * DP ({serving.tp} * {serving.dp} = {expected})")
    # EP, duplicate indices and negative indices are checked by LocalServing.


def mapped_devices(serving: LocalServing, devices: list[str]) -> str | None:
    """Return child CUDA visibility, or None to preserve the inherited environment."""
    validate_selection(serving, available=len(devices))
    if serving.cuda_visible_devices is None:
        return None
    return ",".join(devices[index] for index in serving.cuda_visible_devices)


async def allocated_devices(*, python: str) -> list[str]:
    """Preserve Slurm IDs/UUIDs; discover CUDA ordinals in the serving runtime if absent."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        value = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        devices = [] if value in {"", "-1"} else [device.strip() for device in value.split(",")]
        if any(not device or device.startswith("-") for device in devices) or len(set(devices)) != len(devices):
            raise ValueError("CUDA_VISIBLE_DEVICES must contain distinct GPU identifiers")
        return devices
    # Query CUDA directly instead of NVML, which can enumerate devices outside the
    # container's usable allocation. Keep driver initialization out of the controller.
    discovery = """import ctypes, json
cuda = ctypes.CDLL("libcuda.so.1")
status = cuda.cuInit(0)
if status == 100:  # CUDA_ERROR_NO_DEVICE
    print(0)
else:
    if status:
        raise RuntimeError(f"cuInit failed: {status}")
    count = ctypes.c_int()
    status = cuda.cuDeviceGetCount(ctypes.byref(count))
    if status:
        raise RuntimeError(f"cuDeviceGetCount failed: {status}")
    print(json.dumps(count.value))
"""
    process = await asyncio.create_subprocess_exec(
        python,
        "-c",
        discovery,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(60):
            stdout, _ = await process.communicate()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode:
        raise RuntimeError(f"CUDA GPU discovery failed in {python}; set a CUDA-capable serving.python")
    try:
        count = json.loads(stdout.decode(errors="replace").strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise ValueError("CUDA discovery did not return a GPU count") from error
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("CUDA discovery did not return a nonnegative GPU count")
    return [str(index) for index in range(count)]
