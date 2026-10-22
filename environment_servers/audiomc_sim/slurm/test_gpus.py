# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU selections retain Slurm device identities and match serving parallelism."""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from environment_servers.audiomc_sim.backends import LocalServing
from environment_servers.audiomc_sim.slurm import gpus


@pytest.mark.parametrize(
    "devices",
    [
        ["4", "5", "6", "7"],
        ["GPU-aaaa", "GPU-bbbb", "GPU-cccc", "GPU-dddd"],
        ["MIG-GPU-a/1/0", "MIG-GPU-a/2/0", "MIG-GPU-b/1/0", "MIG-GPU-b/2/0"],
    ],
)
def test_selection_indexes_allocation_in_order(devices):
    serving = LocalServing(checkpoint="model", tp=1, dp=2, ep=2, cuda_visible_devices=[2, 0])
    assert gpus.mapped_devices(serving, devices) == f"{devices[2]},{devices[0]}"


def test_omitted_selection_preserves_environment_but_requires_exact_count():
    serving = LocalServing(checkpoint="model", tp=1, dp=2, ep=1)
    assert gpus.mapped_devices(serving, ["4", "5"]) is None
    with pytest.raises(ValueError, match="GPU count 4 must equal TP"):
        gpus.mapped_devices(serving, ["4", "5", "6", "7"])


def test_physical_ids_are_not_accepted_as_out_of_range_allocation_indices():
    serving = LocalServing(checkpoint="model", dp=4, ep=4, cuda_visible_devices=[4, 5, 6, 7])
    with pytest.raises(ValueError, match="index the 4 allocated GPUs"):
        gpus.mapped_devices(serving, ["4", "5", "6", "7"])


@pytest.mark.parametrize("selection", [[0, 0], [-1], [0, 1]])
def test_invalid_selections_fail_before_startup(selection):
    with pytest.raises(ValidationError):
        LocalServing(checkpoint="model", cuda_visible_devices=selection)


def test_ep_is_disabled_or_matches_tp_times_dp():
    assert LocalServing(checkpoint="model", tp=2, dp=2, ep=1).ep == 1
    assert LocalServing(checkpoint="model", tp=2, dp=2, ep=4).ep == 4
    with pytest.raises(ValidationError, match="EP size"):
        LocalServing(checkpoint="model", tp=2, dp=2, ep=2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value,expected", [("4,5", ["4", "5"]), ("GPU-a,MIG-b", ["GPU-a", "MIG-b"]), ("", []), ("-1", [])]
)
async def test_inherited_visibility_never_queries_or_reorders_cuda(monkeypatch, value, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    discover = AsyncMock(side_effect=AssertionError("unexpected discovery"))
    monkeypatch.setattr(gpus.asyncio, "create_subprocess_exec", discover)
    assert await gpus.allocated_devices(python="serving-python") == expected
    discover.assert_not_called()


@pytest.mark.asyncio
async def test_absent_visibility_discovers_cuda_in_serving_interpreter(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    class Process:
        returncode = 0
        communicate = AsyncMock(return_value=(b"2\n", b""))

    create = AsyncMock(return_value=Process())
    monkeypatch.setattr(gpus.asyncio, "create_subprocess_exec", create)
    assert await gpus.allocated_devices(python="/image/bin/python") == ["0", "1"]
    assert create.call_args.args[0] == "/image/bin/python"
    assert "cuDeviceGetCount" in create.call_args.args[-1]


@pytest.mark.asyncio
async def test_duplicate_inherited_identifiers_are_rejected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,4")
    with pytest.raises(ValueError, match="distinct GPU"):
        await gpus.allocated_devices(python="python3")
