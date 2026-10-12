# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoring, native Gym aggregation, and source identity/coverage checks."""

import asyncio
import itertools
import json
from unittest.mock import MagicMock

import editdistance
import pytest
from pydantic import ValidationError

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.openasr.app import OpenASRConfig, OpenASRServer, OpenASRVerifyRequest
from resources_servers.openasr.scoring import final_transcript, normalize_transcript, word_error_counts


def candidate(text="red green", *, status="completed", output=None):
    return NeMoGymResponse.model_validate(
        {
            "id": "candidate",
            "created_at": 0,
            "model": "audio-model",
            "object": "response",
            "status": status,
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "output": output
            if output is not None
            else [
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }
    )


def request(sample_id="a", *, reference="red green", response=None, subset="librispeech_test_clean"):
    return OpenASRVerifyRequest.model_validate(
        {
            "id": f"openasr/{subset}/{sample_id}",
            "responses_create_params": {
                "input": [{"role": "user", "content": "Transcribe the recording."}],
                "metadata": {"audio_ref": '{"archive":"clean.tar","member":"audio/a.flac"}'},
            },
            "verifier_metadata": {"subset": subset, "id": sample_id, "text": reference, "source": "kept"},
            "response": (response or candidate()).model_dump(),
        }
    )


def server(tmp_path, *, expected=1, ids=None, **overrides):
    values = {
        "subset": "librispeech_test_clean",
        "name": "openasr",
        "host": "127.0.0.1",
        "port": 8154,
        "entrypoint": "app.py",
        "expected_num_rows": expected,
    }
    if ids is not None:
        path = tmp_path / "ids.json"
        path.write_text(json.dumps(ids))
        values["expected_sample_ids_file"] = str(path)
    return OpenASRServer(config=OpenASRConfig(**(values | overrides)), server_client=MagicMock(spec=ServerClient))


def score(svc, sample_id="a", *, reference="red green", text="red green", response=None):
    return asyncio.run(svc.verify(request(sample_id, reference=reference, response=response or candidate(text))))


def test_exact_normalization_reward_and_raw_trace_preservation(tmp_path):
    svc = server(tmp_path)
    response = candidate(
        output=[
            {"id": "thinking", "type": "reasoning", "summary": [{"type": "summary_text", "text": "private trace"}]},
            candidate("Mr. Smith cannot pay twenty-one dollars.").output[0].model_dump(),
        ]
    )
    original = response.model_dump(mode="json")
    result = score(svc, reference="mister smith cannot pay $21", response=response)
    assert svc.config.scoring_version == result.scoring_version == "asr_v1"
    assert result.reward == 1 and result.wer == 0 and result.score_valid and result.candidate_valid
    assert result.normalized_reference == result.normalized_hypothesis == "mister smith cannot pay $21"
    assert result.candidate_thinking == ["private trace"]
    assert result.response.model_dump(mode="json") == original
    assert result.verifier_metadata.source == "kept"
    assert result.responses_create_params.metadata["audio_ref"].endswith('"audio/a.flac"}')
    assert result.reference_words == result.hits == 5
    svc.server_client.post.assert_not_called()


def test_substitution_deletion_insertion_and_unbounded_wer(tmp_path):
    svc = server(tmp_path)
    changed = score(svc, reference="red green blue", text="red yellow blue extra")
    assert changed.reward == 0 and changed.substitutions == changed.insertions == 1 and changed.deletions == 0
    assert changed.edits == 2 and changed.hits == 2 and changed.wer == pytest.approx(2 / 3)
    long = score(svc, reference="red", text="red green blue yellow")
    assert long.wer == 3 and long.edits == long.insertions == 3


def test_empty_hypothesis_is_scored_as_all_deletions(tmp_path):
    svc = server(tmp_path)
    result = score(svc, text="")
    assert result.score_valid and result.candidate_valid and result.reward == 0
    assert result.empty_hypothesis and result.deletions == result.reference_words == 2 and result.wer == 1
    metrics = svc.compute_metrics([[result.model_dump()]])
    assert metrics["openasr/complete"] and metrics["openasr/corpus_wer_percent"] == 100
    assert metrics["openasr/empty_hypothesis_rows"] == 1


def test_refusal_text_is_not_used_as_a_transcript(tmp_path):
    response = candidate(
        output=[
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": "Cannot help"}],
            }
        ]
    )
    result = score(server(tmp_path), response=response)
    assert result.candidate_answer == "" and result.wer == 1 and result.deletions == 2


def test_missing_reference_skips_legacy_score_and_zero_denominator_is_explicit(tmp_path):
    svc = server(tmp_path)
    missing = score(svc, reference=None)
    assert not missing.score_valid and missing.score_status == "missing_reference" and missing.wer is None
    assert missing.reference_words == missing.edits == missing.insertions == 0
    metrics = svc.compute_metrics([[missing.model_dump()]])
    assert metrics["openasr/skipped_missing_reference_rows"] == 1
    assert metrics["openasr/corpus_wer_percent"] == 0 and not metrics["openasr/wer_denominator_valid"]
    assert not metrics["openasr/complete"]


def test_normalized_empty_reference_retains_insertions_in_corpus(tmp_path):
    svc = server(tmp_path, expected=2)
    empty_ref = score(svc, reference="[noise]", text="red green")
    real_ref = score(svc, "b", reference="red green", text="red green")
    assert empty_ref.score_valid and empty_ref.zero_reference_words and empty_ref.wer is None
    assert empty_ref.reference_words == 0 and empty_ref.insertions == empty_ref.edits == 2
    metrics = svc.compute_metrics([[empty_ref.model_dump()], [real_ref.model_dump()]])
    assert metrics["openasr/corpus_wer_percent"] == 100 and metrics["openasr/complete"]


def test_native_aggregation_uses_corpus_counts_not_mean_wer_or_reward(tmp_path):
    svc = server(tmp_path, expected=2, ids=["a", "b"])
    short = score(svc, reference="red", text="green").model_dump(mode="json") | {"_ng_task_index": 0}
    long = score(
        svc,
        "b",
        reference="red green blue yellow black white orange purple brown",
        text=("red green blue yellow black white orange purple brown"),
    ).model_dump(mode="json") | {"_ng_task_index": 1}
    aggregate = asyncio.run(svc.aggregate_metrics(AggregateMetricsRequest(verify_responses=[short, long])))
    metrics = aggregate.key_metrics
    assert metrics["openasr/reference_words"] == 10 and metrics["openasr/edits"] == 1
    assert metrics["openasr/corpus_wer"] == 0.1 and metrics["openasr/corpus_wer_percent"] == 10
    assert metrics["openasr/complete"]
    assert aggregate.agent_metrics["mean/reward"] == 0.5


def test_incomplete_candidate_retains_legacy_score_and_prevents_completion(tmp_path):
    svc = server(tmp_path)
    result = score(svc, response=candidate(status="incomplete"))
    assert result.score_valid and result.wer == 0 and not result.candidate_valid and result.reward == 0
    assert result.candidate_incomplete_details == {"reason": "max_output_tokens"}
    metrics = svc.compute_metrics([[result.model_dump()]])
    assert metrics["openasr/corpus_wer_percent"] == 0 and metrics["openasr/sample_complete"]
    assert metrics["openasr/candidate_invalid_rows"] == 1 and not metrics["openasr/complete"]


def test_masked_verifier_failure_is_counted_in_native_aggregate(tmp_path):
    svc = server(tmp_path, expected=2, ids=["a", "b"])
    valid = score(svc).model_dump(mode="json") | {"_ng_task_index": 0}
    failed = request("b").model_dump(mode="json") | {
        "reward": 0.0,
        "mask_sample": True,
        "_ng_task_index": 1,
        "_ng_rollout_index": 0,
    }
    aggregate = asyncio.run(svc.aggregate_metrics(AggregateMetricsRequest(verify_responses=[valid, failed])))
    metrics = aggregate.key_metrics
    assert metrics["openasr/count"] == metrics["openasr/distinct_sample_ids"] == 2
    assert metrics["openasr/scored_rows"] == metrics["openasr/invalid_rows"] == 1
    assert metrics["openasr/corpus_wer_percent"] == 0 and not metrics["openasr/complete"]
    all_failed = asyncio.run(svc.aggregate_metrics(AggregateMetricsRequest(verify_responses=[failed])))
    assert all_failed.key_metrics["openasr/invalid_rows"] == 1
    assert not all_failed.key_metrics["openasr/wer_denominator_valid"]


def test_expected_ids_reject_missing_duplicate_and_unexpected_outputs(tmp_path):
    svc = server(tmp_path, expected=2, ids=["a", "b"])
    a = score(svc).model_dump()
    duplicate = svc.compute_metrics([[a], [a]])
    assert duplicate["openasr/duplicate_rows"] == duplicate["openasr/missing_rows"] == 1
    assert not duplicate["openasr/complete"]
    bad = score(svc, "c").model_dump()
    unexpected = svc.compute_metrics([[a], [bad]])
    assert unexpected["openasr/missing_rows"] == unexpected["openasr/unexpected_sample_ids"] == 1
    assert not unexpected["openasr/complete"]


def test_schema_and_runtime_reject_wrong_subset_and_profile(tmp_path):
    with pytest.raises(ValidationError):
        server(tmp_path, scoring_version="unreviewed")
    with pytest.raises(ValidationError):
        request(sample_id="")
    with pytest.raises(ValueError, match="cannot score subset"):
        asyncio.run(server(tmp_path).verify(request(subset="librispeech_test_other")))
    with pytest.raises(ValueError, match="duplicate IDs"):
        server(tmp_path, expected=2, ids=["a", "a"])
    with pytest.raises(ValueError, match="count differs"):
        server(tmp_path, expected=2, ids=["a"])


def test_expected_ids_accept_prepared_gym_jsonl_and_plain_id_lines(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(request().model_dump(mode="json")) + "\n")
    svc = server(tmp_path, expected_sample_ids_file=str(path))
    assert svc._expected_sample_ids == {"a"}
    path.write_text("a\n")
    assert server(tmp_path, expected_sample_ids_file=str(path))._expected_sample_ids == {"a"}
    path.write_text(json.dumps(request(subset="tedlium").model_dump(mode="json")) + "\n")
    with pytest.raises(ValueError, match="different OpenASR subset"):
        server(tmp_path, expected_sample_ids_file=str(path))


def test_metrics_empty_input_and_corrupt_counts_are_explicit(tmp_path):
    svc = server(tmp_path)
    empty = svc.compute_metrics([])
    assert empty["openasr/count"] == empty["openasr/corpus_wer_percent"] == 0
    assert empty["openasr/missing_rows"] == 1 and not empty["openasr/complete"]
    corrupt = score(svc).model_dump() | {"edits": "0"}
    with pytest.raises(ValueError, match="nonnegative integer edits"):
        svc.compute_metrics([[corrupt]])
    corrupt["edits"] = 1
    with pytest.raises(ValueError, match="aggregate edit count differs"):
        svc.compute_metrics([[corrupt]])


@pytest.mark.parametrize(
    "answer,thinking,expected",
    [
        (" A \n", None, "A"),
        ("\nA\n", "thought", "A"),
        ("<think>reason</think>\nA\n", None, "A"),
        ("<think>first</think>A</think>\nB", "thought", "B"),
        ("", "reasoning transcript", ""),
        ("", None, ""),
        ("The audio is red green.", None, "The audio is red green."),
    ],
)
def test_final_transcript_matches_legacy_audio_prediction(answer, thinking, expected):
    output = [candidate(answer).output[0].model_dump()]
    if thinking is not None:
        output.insert(
            0, {"id": "reasoning", "type": "reasoning", "summary": [{"type": "summary_text", "text": thinking}]}
        )
    assert final_transcript(candidate(output=output)) == expected


def test_edit_counts_match_source_distance_for_ambiguous_and_empty_sequences():
    sequences = [list(items) for length in range(4) for items in itertools.product(["red", "blue"], repeat=length)]
    for ref, hyp in itertools.product(sequences, repeat=2):
        counts = word_error_counts(reference=" ".join(ref), hypothesis=" ".join(hyp))
        assert counts.edits == editdistance.eval(ref, hyp)
        assert counts.hits + counts.substitutions + counts.deletions == len(ref)
        assert counts.hits + counts.substitutions + counts.insertions == len(hyp)


@pytest.mark.parametrize(
    "reference,hypothesis,edits",
    [
        ("ab c", "a bc", 2),
        ("cat", "hippopotamus", 1),
        ("naïve café", "naïve tea", 1),
        ("red\tblue\n green", "red blue green", 0),
    ],
)
def test_distance_is_over_words_not_characters(reference, hypothesis, edits):
    assert word_error_counts(reference=reference, hypothesis=hypothesis).edits == edits


def test_long_transcript_with_substitution_deletion_and_insertions():
    reference = [f"word{i}" for i in range(5000)]
    hypothesis = reference.copy()
    hypothesis[1000] = "replacement"
    del hypothesis[2500]
    hypothesis.extend(["extra", "tokens"])
    counts = word_error_counts(reference=" ".join(reference), hypothesis=" ".join(hypothesis))
    assert counts.edits == editdistance.eval(reference, hypothesis) == 4
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 1, 2)
    assert counts.hits == 4998
    assert counts.reference_words == 5000 and counts.hypothesis_words == 5001


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Dr. Jones won't pay twenty dollars.", "doctor jones will not pay $20"),
        ("[noise] Um, colour (laugh) theatre.", "color theater"),
        ("I'M READY!", "i am ready"),
    ],
)
def test_source_specific_normalization_contract(text, expected):
    assert normalize_transcript(text) == expected
