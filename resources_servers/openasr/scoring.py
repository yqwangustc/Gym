# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned English transcript normalization and inspectable word edit counts."""

from dataclasses import dataclass

from rapidfuzz.distance.Levenshtein import editops

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.openasr.vendor.english_text_normalizer import EnglishTextNormalizer


@dataclass(frozen=True)
class EditCounts:
    """Sufficient statistics for corpus WER; errors are never clipped."""

    reference_words: int
    hypothesis_words: int
    hits: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float | None:
        return self.edits / self.reference_words if self.reference_words else None


def word_error_counts(*, reference: str, hypothesis: str) -> EditCounts:
    """Align words with RapidFuzz's compiled unit-cost Levenshtein operations.

    Pass token lists, rather than transcript strings, to measure word errors.
    Equal-cost alignments use the pinned library's tie rule; diagnostic S/D/I
    counts can differ from the former Python alignment without changing WER.
    """
    ref, hyp = reference.split(), hypothesis.split()
    operations = editops(ref, hyp)
    substitutions = deletions = insertions = 0
    for operation, _, _ in operations:
        if operation == "replace":
            substitutions += 1
        elif operation == "delete":
            deletions += 1
        else:  # RapidFuzz editops emits only replace, delete, and insert.
            insertions += 1
    return EditCounts(
        reference_words=len(ref),
        hypothesis_words=len(hyp),
        hits=len(ref) - substitutions - deletions,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
    )


def final_transcript(response: NeMoGymResponse) -> str:
    """Extract final text after the last closing think tag.

    The audio model reassembles parsed reasoning and final content before the
    reference evaluator extracts the text after the final ``</think>``.
    The ASR profile disables thinking and does not synthesize VoiceBench's
    empty-think fallback. Reasoning-only output therefore has no transcript.
    Raw response items remain intact on the verify response.
    """
    answer = ""
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            answer = "\n".join(part.text for part in item.content if part.type == "output_text")
            break
    reasoning = ""
    for item in reversed(response.output):
        if item.type == "reasoning":
            reasoning = "".join(part.text for part in (item.summary or []))
            break
    prediction = f"<think>\n{reasoning}\n</think>\n{answer}" if reasoning else answer
    prediction = prediction.strip()
    if "</think>" in prediction:
        return prediction.rsplit("</think>", 1)[-1].lstrip("\n")
    return prediction


def normalize_transcript(text: str) -> str:
    """Use the vendored baseline normalizer, retaining its exact string output."""
    return EnglishTextNormalizer()(text)
