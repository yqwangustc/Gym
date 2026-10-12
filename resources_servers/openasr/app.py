# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic English ASR verification and corpus word error rate."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.reward_profile import compute_aggregate_metrics
from resources_servers.openasr.scoring import final_transcript, normalize_transcript, word_error_counts
from resources_servers.openasr.task_data import OpenASRMetadata, Subset


MetricValue = int | float | bool | None


class OpenASRConfig(BaseResourcesServerConfig):
    """A named instance evaluates one prepared split with a fixed scoring profile."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    subset: Subset
    scoring_version: Literal["asr_v1"] = "asr_v1"
    expected_num_rows: int = Field(gt=0)
    expected_sample_ids_file: str | None = None


class OpenASRVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: OpenASRMetadata


class OpenASRVerifyResponse(OpenASRVerifyRequest, BaseVerifyResponse):
    """Retain raw input/output, normalized transcripts, and corpus statistics."""

    sample_id: str
    subset: Subset
    scoring_version: Literal["asr_v1"]
    score_status: Literal["scored", "missing_reference"]
    score_valid: bool
    candidate_valid: bool
    candidate_status: str | None
    candidate_incomplete_details: dict[str, str] | None
    candidate_thinking: list[str]
    candidate_answer: str
    normalized_reference: str | None
    normalized_hypothesis: str
    reference_words: int
    hypothesis_words: int
    hits: int
    substitutions: int
    deletions: int
    insertions: int
    edits: int
    wer: float | None
    empty_hypothesis: bool
    zero_reference_words: bool


def _expected_ids(path: str, *, subset: str) -> set[str]:
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        values = json.loads(text)
    else:
        values = []
        # JSONL has physical newline record boundaries; Unicode separators can
        # legitimately occur inside JSON strings and are not new records.
        for line in text.split("\n"):
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                row = json.loads(line)
                metadata = row.get("verifier_metadata", row)
                if metadata.get("subset", subset) != subset:
                    raise ValueError("expected_sample_ids_file contains a different OpenASR subset")
                values.append(metadata["id"])
            else:
                values.append(line.strip())
    if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError("expected_sample_ids_file requires a nonempty string list or Gym JSONL source IDs")
    if len(set(values)) != len(values):
        raise ValueError("expected_sample_ids_file contains duplicate IDs")
    return set(values)


class OpenASRServer(SimpleResourcesServer):
    """A stateless verifier using the pinned normalization and corpus WER definition."""

    config: OpenASRConfig
    _expected_sample_ids: set[str] | None = PrivateAttr(default=None)

    def model_post_init(self, context: object) -> None:
        if self.config.expected_sample_ids_file:
            self._expected_sample_ids = _expected_ids(self.config.expected_sample_ids_file, subset=self.config.subset)
            if len(self._expected_sample_ids) != self.config.expected_num_rows:
                raise ValueError("expected_sample_ids_file count differs from expected_num_rows")
        super().model_post_init(context)

    async def verify(self, body: OpenASRVerifyRequest) -> OpenASRVerifyResponse:
        if body.verifier_metadata.subset != self.config.subset:
            raise ValueError(f"resource {self.config.subset} cannot score subset {body.verifier_metadata.subset}")
        answer = final_transcript(body.response)
        hypothesis = normalize_transcript(answer)
        reference = body.verifier_metadata.text
        ref_norm = normalize_transcript(reference) if reference is not None else None
        counts = word_error_counts(reference=ref_norm or "", hypothesis=hypothesis)
        candidate_valid = body.response.status == "completed" and body.response.incomplete_details is None
        score_valid = reference is not None
        return OpenASRVerifyResponse.model_validate(
            body.model_dump()
            | {
                "reward": float(score_valid and counts.edits == 0 and candidate_valid),
                "sample_id": body.verifier_metadata.id,
                "subset": self.config.subset,
                "scoring_version": self.config.scoring_version,
                "score_status": "scored" if score_valid else "missing_reference",
                "score_valid": score_valid,
                "candidate_valid": candidate_valid,
                "candidate_status": body.response.status,
                "candidate_incomplete_details": body.response.incomplete_details.model_dump()
                if body.response.incomplete_details is not None
                else None,
                "candidate_thinking": [
                    part.text
                    for item in body.response.output
                    if item.type == "reasoning"
                    for part in (item.summary or [])
                ],
                "candidate_answer": answer,
                "normalized_reference": ref_norm,
                "normalized_hypothesis": hypothesis,
                "reference_words": counts.reference_words if score_valid else 0,
                "hypothesis_words": counts.hypothesis_words,
                "hits": counts.hits if score_valid else 0,
                "substitutions": counts.substitutions if score_valid else 0,
                "deletions": counts.deletions if score_valid else 0,
                "insertions": counts.insertions if score_valid else 0,
                "edits": counts.edits if score_valid else 0,
                "wer": counts.wer if score_valid else None,
                "empty_hypothesis": not bool(hypothesis.split()),
                "zero_reference_words": score_valid and not counts.reference_words,
            }
        )

    def compute_metrics(self, tasks: Sequence[Sequence[Mapping[str, object]]]) -> dict[str, MetricValue]:
        """Sum word errors over every received row and report evaluation coverage.

        This profile expects one rollout per source sample. Duplicate or missing
        sample IDs, masked verifier failures, and incomplete candidates prevent
        ``complete``. The published WER describes the received scoreable rows;
        it is never the average per-row WER or a transformed average reward.
        """
        rows = [row for task in tasks for row in task]
        identities = []
        for row in rows:
            metadata = row.get("verifier_metadata")
            sample_id = row.get("sample_id")
            if sample_id is None and isinstance(metadata, Mapping):
                sample_id = metadata.get("id")
            identities.append(sample_id if isinstance(sample_id, str) else None)
        observed = {identity for identity in identities if identity is not None}
        scored = [row for row in rows if row.get("score_valid") is True and not row.get("mask_sample")]
        skipped = [
            row
            for row in rows
            if row.get("score_status") == "missing_reference"
            and row.get("score_valid") is False
            and not row.get("mask_sample")
        ]
        invalid = len(rows) - len(scored) - len(skipped)
        totals: dict[str, int] = {}
        for field in (
            "reference_words",
            "hypothesis_words",
            "hits",
            "substitutions",
            "deletions",
            "insertions",
            "edits",
        ):
            values = [row.get(field) for row in scored]
            if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values):
                raise ValueError(f"OpenASR scored rows require nonnegative integer {field}")
            totals[field] = sum(values)
        if totals["edits"] != totals["substitutions"] + totals["deletions"] + totals["insertions"]:
            raise ValueError("OpenASR aggregate edit count differs from substitutions + deletions + insertions")
        reference_words = totals["reference_words"]
        # The baseline evaluator returns zero when no reference words exist. Keep that
        # result together with an explicit denominator-validity flag.
        wer = totals["edits"] / reference_words if reference_words else 0.0
        duplicates = len(identities) - len(observed) - identities.count(None)
        unexpected = len(observed - self._expected_sample_ids) if self._expected_sample_ids is not None else 0
        missing = (
            len(self._expected_sample_ids - observed)
            if self._expected_sample_ids is not None
            else max(0, self.config.expected_num_rows - len(observed))
        )
        candidate_invalid = sum(row.get("candidate_valid") is not True for row in rows)
        sample_complete = (
            len(rows) == len(observed) == self.config.expected_num_rows
            and not duplicates
            and not missing
            and not unexpected
            and not invalid
        )
        metrics: dict[str, MetricValue] = {
            "openasr/count": len(rows),
            "openasr/expected_rows": self.config.expected_num_rows,
            "openasr/scored_rows": len(scored),
            "openasr/skipped_missing_reference_rows": len(skipped),
            "openasr/invalid_rows": invalid,
            "openasr/candidate_invalid_rows": candidate_invalid,
            "openasr/empty_hypothesis_rows": sum(row.get("empty_hypothesis") is True for row in scored),
            "openasr/zero_reference_word_rows": sum(row.get("zero_reference_words") is True for row in scored),
            "openasr/distinct_sample_ids": len(observed),
            "openasr/duplicate_rows": duplicates,
            "openasr/missing_rows": missing,
            "openasr/unexpected_sample_ids": unexpected,
            "openasr/missing_sample_id_rows": identities.count(None),
            "openasr/corpus_wer": wer,
            "openasr/corpus_wer_percent": 100 * wer,
            "openasr/wer_denominator_valid": reference_words > 0,
            "openasr/sample_complete": sample_complete,
            "openasr/complete": sample_complete and not candidate_invalid and reference_words > 0,
        }
        metrics.update({f"openasr/{field}": value for field, value in totals.items()})
        return metrics

    def get_key_metrics(self, agent_metrics: Mapping[str, object]) -> dict[str, object]:
        return {key: value for key, value in agent_metrics.items() if key.startswith("openasr/")}

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        # The base aggregator filters masked rows before the custom hook. Include
        # them in our coverage checks so failed verification cannot appear complete.
        aggregate = compute_aggregate_metrics(body.verify_responses)
        metrics = self.compute_metrics([body.verify_responses])
        aggregate.agent_metrics.update(metrics)
        aggregate.key_metrics.update(metrics)
        return aggregate


if __name__ == "__main__":
    OpenASRServer.run_webserver()
