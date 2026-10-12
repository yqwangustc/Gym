# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Labels for the byte-preserving OpenASR datasets prepared in text-evals."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Subset = Literal[
    "ami",
    "earnings22",
    "gigaspeech",
    "librispeech_test_clean",
    "librispeech_test_other",
    "spgispeech",
    "tedlium",
    "voxpopuli",
]


class OpenASRMetadata(BaseModel):
    """Reference text and source identity stay outside candidate-visible input."""

    model_config = ConfigDict(extra="allow")

    subset: Subset
    id: str = Field(min_length=1)
    text: str | None
