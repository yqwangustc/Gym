# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable exports retain all rubrics and exact audio bytes without transcript leakage."""

import hashlib
import io
import json
import tarfile
import wave

import pytest

from resources_servers.audiomc_sim.export import export_collection
from responses_api_models.vllm_model.tar_audio import TarAudioReader


def test_export_filters_outcomes_and_reads_audio_with_gym_adapter(tmp_path):
    artifacts = tmp_path / "artifacts"
    episodes = artifacts / "episodes"
    episodes.mkdir(parents=True)
    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 240)
    audio = wav_bytes.getvalue()
    with tarfile.open(artifacts / "original.tar", "w") as archive:
        info = tarfile.TarInfo("episode/u0.wav")
        info.size = len(audio)
        archive.addfile(info, io.BytesIO(audio))
    ref = {"archive": "original.tar", "member": "episode/u0.wav", "sha256": hashlib.sha256(audio).hexdigest()}
    turns = [{"user_turn_id": "u0", "audio": ref, "rubrics": [{"rubric_id": "r0"}]}]
    accepted = {
        "outcome": "accepted_correction",
        "metadata": {},
        "turn_evaluations": turns,
        "training_example": {
            "metadata": {"task": {"assistant_instructions": "Be helpful.", "developer_instructions": None}},
            "schema_version": "1.7",
            "turn_evaluations": turns,
            "conversation": [
                {"turn_id": "u0", "role": "user", "text": "Private transcript"},
                {"turn_id": "a0", "role": "assistant", "text": "Teacher answer", "generated_by_teacher": True},
            ],
        },
    }
    excluded = {"outcome": "teacher_correction_exhausted", "metadata": {}, "turn_evaluations": turns}
    (episodes / "a.json").write_text(json.dumps(accepted))
    (episodes / "b.json").write_text(json.dumps(excluded))
    summary = export_collection(tmp_path)
    assert summary["accepted_examples"] == 1
    assert summary["rollout_counter"] == 2
    assert summary["audio_members"] == 1
    example = json.loads((tmp_path / "rollout.jsonl").read_text())
    assert example["turn_evaluations"][0]["rubrics"] == [{"rubric_id": "r0"}]
    assert example["conversation"][-1]["text"] == "Teacher answer"
    ref = example["conversation"][0]["audio"]
    assert ref["archive"] == "audio.tar" and ref["member"] == "episode/u0.wav"
    assert TarAudioReader().read_member(tmp_path / ref["archive"], member=ref["member"]) == audio
    assert json.loads((tmp_path / "inspection.jsonl").read_text())["outcome"] == "teacher_correction_exhausted"
    assert export_collection(tmp_path) == summary
    # A changed source cannot silently produce a seemingly valid training archive.
    accepted["turn_evaluations"][0]["audio"]["sha256"] = "invalid"
    (episodes / "a.json").write_text(json.dumps(accepted))
    with pytest.raises(ValueError):
        export_collection(tmp_path)


def _write_sharding_fixture(root, count):
    artifacts = root / "artifacts"
    episodes = artifacts / "episodes"
    episodes.mkdir(parents=True)
    refs, payloads = [], {}
    with tarfile.open(artifacts / "original.tar", "w") as archive:
        for index in range(count):
            member = f"episode/user_turn_{index:04d}.wav"
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(bytes([index, 0]) * 240)
            payload = buffer.getvalue()
            payloads[member] = payload
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            refs.append({"archive": "original.tar", "member": member, "sha256": hashlib.sha256(payload).hexdigest()})
    if count:
        accepted_refs = refs[:3]
        evaluations = [{"user_turn_id": f"u{i}", "audio": ref} for i, ref in enumerate(accepted_refs)]
        prefix = []
        for index in range(len(accepted_refs)):
            if index:
                prefix.append({"role": "assistant", "text": "Previous answer"})
            prefix.append({"turn_id": f"u{index}", "role": "user", "text": "Private transcript"})
        (episodes / "accepted.json").write_text(
            json.dumps(
                {
                    "outcome": "accepted_correction",
                    "metadata": {},
                    "turn_evaluations": evaluations,
                    "training_example": {
                        "metadata": {"task": {"assistant_instructions": "Be helpful", "developer_instructions": None}},
                        "schema_version": "1.7",
                        "turn_evaluations": evaluations,
                        "conversation": [
                            dict(t, generated_by_teacher=False) if t["role"] == "assistant" else t for t in prefix
                        ]
                        + [
                            {
                                "turn_id": "answer",
                                "role": "assistant",
                                "text": "Teacher answer",
                                "generated_by_teacher": True,
                            }
                        ],
                    },
                }
            )
        )
    if count > 3:
        (episodes / "excluded.json").write_text(
            json.dumps({"outcome": "teacher_correction_exhausted", "metadata": {}, "audio": refs[3:4]})
        )
    if count > 4:
        partial = artifacts / "partial"
        partial.mkdir()
        (partial / "failed.json").write_text(json.dumps({"audio": refs[4:]}))
        (root / "episode_results.jsonl").write_text(
            json.dumps({"episode_id": {"rollout_id": "failed", "attempt": 0}, "failure": {"terminal": True}}) + "\n"
        )
    return payloads


@pytest.mark.parametrize("count,limit", [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (10, 1)])
def test_shard_limits_and_all_exported_references(tmp_path, count, limit):
    payloads = _write_sharding_fixture(tmp_path, count)
    summary = export_collection(tmp_path, max_audio_files=limit)
    shard_count = max(1, (count + limit - 1) // limit)
    names = (
        ["audio.tar"] if shard_count == 1 else [f"audio-{i:05d}-of-{shard_count:05d}.tar" for i in range(shard_count)]
    )
    assert summary["audio_archives"] == names
    assert summary["audio_members"] == count
    assert summary["max_audio_files"] == limit
    assert sorted(p.name for p in tmp_path.glob("*.tar")) == names
    locations = {}
    for name in names:
        with tarfile.open(tmp_path / name) as archive:
            members = archive.getmembers()
            assert len(members) <= limit
            for member in members:
                assert member.name not in locations
                assert archive.extractfile(member).read() == payloads[member.name]
                locations[member.name] = name
    assert set(locations) == set(payloads)
    checked = set()
    reader = TarAudioReader()

    def check_refs(value):
        if isinstance(value, dict):
            if {"archive", "member"} <= value.keys():
                assert value["archive"] == locations[value["member"]]
                assert (
                    reader.read_member(tmp_path / value["archive"], member=value["member"])
                    == payloads[value["member"]]
                )
                checked.add(value["member"])
            for child in value.values():
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)

    for filename in ("rollout.jsonl", "episodes.jsonl", "inspection.jsonl"):
        for line in (tmp_path / filename).read_text().splitlines():
            row = json.loads(line)
            check_refs(row)
            if filename == "rollout.jsonl":
                assert row["schema_version"] == "1.7"
    assert checked == set(payloads)
    assert export_collection(tmp_path, max_audio_files=limit) == summary


def test_resharding_removes_only_obsolete_export_archives(tmp_path):
    _write_sharding_fixture(tmp_path, 5)
    unrelated = tmp_path / "audio-unrelated.tar"
    unrelated.write_bytes(b"unrelated")
    for limit in (5, 2, 1, 5):
        summary = export_collection(tmp_path, max_audio_files=limit)
        assert sorted(p.name for p in tmp_path.glob("*.tar") if p != unrelated) == summary["audio_archives"]
        assert unrelated.read_bytes() == b"unrelated"
    assert (tmp_path / "artifacts/original.tar").exists()


def test_failed_sharding_preserves_published_export(tmp_path):
    payloads = _write_sharding_fixture(tmp_path, 5)
    export_collection(tmp_path, max_audio_files=2)
    published = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    # Corrupt a member in the second shard, after the first has been staged.
    with tarfile.open(tmp_path / "artifacts/original.tar", "a") as archive:
        member = sorted(payloads)[2]
        info = tarfile.TarInfo(member)
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="Audio hash mismatch"):
        export_collection(tmp_path, max_audio_files=2)
    assert all(p.read_bytes() == content for p, content in published.items())
    assert not list(tmp_path.glob(".audio-export-*"))


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_audio_limit_rejected_before_writing(tmp_path, limit):
    with pytest.raises(ValueError, match="max_audio_files must be >= 1"):
        export_collection(tmp_path, max_audio_files=limit)
    assert not list(tmp_path.iterdir())
