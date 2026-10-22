#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run from the Gym checkout with its Python environment activated.
set -euo pipefail
exec python -m environment_servers.audiomc_sim.slurm.submit "$@"
