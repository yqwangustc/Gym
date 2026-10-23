# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compaction preserves complete datasets, causal training inputs and resumability."""

import hashlib
import io
import json
import tarfile
import wave
from pathlib import Path

import pytest

from resources_servers.audiomc_sim.export import export_collection
from resources_servers.audiomc_sim.storage import started_attempt_count


def trajectory(root, rollout, *, turns=2, max_turns=2, attempt=0):
    key = rollout + (f"-a{attempt}" if attempt else "")
    identity = {"rollout_id": rollout, "attempt": attempt}
    evaluations, conversation = [], []
    for i in range(turns):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x00\x00" * 240)
        data = buffer.getvalue()
        archive = root / "artifacts/audio" / key / f"turn_{i:04d}.tar"
        archive.parent.mkdir(parents=True, exist_ok=True)
        member = f"{key}/user_turn_{i:04d}.wav"
        with tarfile.open(archive, "w") as tar:
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        audio = {
            "archive": f"audio/{key}/turn_{i:04d}.tar",
            "member": member,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        evaluations.append({"user_turn_id": f"u{i}", "audio": audio})
        user = {"turn_id": f"u{i}", "role": "user", "text": f"Request {i}"}
        assistant = {
            "turn_id": f"a{i}",
            "role": "assistant",
            "text": f"Reply {i}",
            "generated_by_teacher": i == turns - 1,
        }
        conversation.extend([user, assistant])
    record = {
        "outcome": "accepted_correction",
        "metadata": {
            "episode_id": identity,
            "task_id": {"taskset": "test", "task_id": rollout},
            "task": {"max_turns": max_turns, "axis_id": "voice_editing"},
        },
        "turn_evaluations": evaluations,
        "teacher_attempts": [{"turn_index": turns - 1}],
        "training_example": {
            "metadata": {"task": {"assistant_instructions": "Voice assistant", "developer_instructions": None}},
            "schema_version": "1.7",
            "turn_evaluations": evaluations,
            "conversation": conversation,
        },
    }
    target = root / "artifacts/episodes" / f"{key}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record))
    request = root / "requests" / f"{rollout}-attempt-{attempt}.json"
    request.parent.mkdir(exist_ok=True)
    request.write_text(json.dumps({"episode_id": identity}))
    return record, target, request


def verify_portable_audio(root):
    for line in (root / "episodes.jsonl").read_text().splitlines():
        record = json.loads(line)
        for turn in record["turn_evaluations"]:
            ref = turn["audio"]
            with tarfile.open(root / ref["archive"]) as tar:
                assert hashlib.sha256(tar.extractfile(ref["member"]).read()).hexdigest() == ref["sha256"]


def test_reexport_preserves_conversation_and_instructions_after_cleanup(tmp_path):
    _, raw, _ = trajectory(tmp_path, "full")
    export_collection(tmp_path)
    assert not raw.exists()
    original = json.loads((tmp_path / "rollout.jsonl").read_text())
    for _ in range(2):
        export_collection(tmp_path)
        exported = json.loads((tmp_path / "episodes.jsonl").read_text())
        example = json.loads((tmp_path / "rollout.jsonl").read_text())
        assert example == original == exported["training_example"]
        assert example["metadata"]["task"]["assistant_instructions"] == "Voice assistant"
        assert example["conversation"][-1]["text"] == "Reply 1"
        assert example["conversation"][-1]["generated_by_teacher"] is True
        assert [t["audio"] for t in example["conversation"] if t["role"] == "user"] == [
            t["audio"] for t in example["turn_evaluations"]
        ]
    verify_portable_audio(tmp_path)


def test_cleanup_preserves_short_and_incomplete_traces_and_incremental_exports(tmp_path):
    full, full_path, full_request = trajectory(tmp_path, "full", attempt=1)
    _, short_path, short_request = trajectory(tmp_path, "short", turns=1, max_turns=3)
    pending, pending_path, pending_request = trajectory(tmp_path, "pending", turns=1, max_turns=3)
    pending_path.unlink()
    partial = tmp_path / "artifacts/partial/pending.json"
    partial.parent.mkdir()
    partial.write_text(json.dumps({"audio": [t["audio"] for t in pending["turn_evaluations"]]}))
    old_request = tmp_path / "requests/full-attempt-0.json"
    old_request.write_text("{}")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts/system.md").write_text("frozen prompt")
    with (tmp_path / "episode_results.jsonl").open("w") as stream:
        for record in [full, json.loads(short_path.read_text())]:
            stream.write(
                json.dumps({"episode_id": record["metadata"]["episode_id"], "result": {"outcome": record["outcome"]}})
                + "\n"
            )
    before = started_attempt_count(tmp_path)
    first = export_collection(tmp_path, max_audio_files=2)
    assert first["accepted_examples"] == 2 and first["audio_members"] == 4
    assert not full_path.exists() and not full_request.exists()
    assert not (tmp_path / "artifacts/audio/full-a1").exists()
    assert short_path.exists() and short_request.exists()
    assert partial.exists() and pending_request.exists() and (tmp_path / "artifacts/audio/pending").exists()
    assert old_request.exists() and (tmp_path / "prompts/system.md").exists()
    assert started_attempt_count(tmp_path) == before == 4
    verify_portable_audio(tmp_path)
    second = export_collection(tmp_path, max_audio_files=1)
    assert second["accepted_examples"] == 2 and second["audio_members"] == 4
    assert len(second["audio_archives"]) == 4
    verify_portable_audio(tmp_path)
    trajectory(tmp_path, "later")
    third = export_collection(tmp_path, max_audio_files=3)
    assert third["accepted_examples"] == 3 and third["audio_members"] == 6
    assert len(third["audio_archives"]) == 2
    verify_portable_audio(tmp_path)
    assert len((tmp_path / "rollout.jsonl").read_text().splitlines()) == 3


def test_keep_raw_option_and_repeated_export_after_cleanup(tmp_path):
    _, path, request = trajectory(tmp_path, "full")
    first = export_collection(tmp_path, cleanup_finished_trajectories=False)
    assert path.exists() and request.exists()
    assert export_collection(tmp_path) == first
    assert not path.exists() and not request.exists()
    assert export_collection(tmp_path) == first
    verify_portable_audio(tmp_path)


def test_hash_failure_never_removes_sources(tmp_path):
    record, path, request = trajectory(tmp_path, "full")
    record["turn_evaluations"][0]["audio"]["sha256"] = "invalid"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash mismatch"):
        export_collection(tmp_path)
    assert path.exists() and request.exists() and (tmp_path / "artifacts/audio/full").exists()


def test_publication_failure_never_cleans_raw_records(tmp_path, monkeypatch):
    _, path, request = trajectory(tmp_path, "full")
    original = Path.replace

    def replace(source, target):
        if source.name == "episodes.jsonl.tmp":
            raise OSError("simulated publication failure")
        return original(source, target)

    with monkeypatch.context() as context:
        context.setattr(Path, "replace", replace)
        with pytest.raises(OSError, match="publication failure"):
            export_collection(tmp_path)
    assert path.exists() and request.exists() and (tmp_path / "artifacts/audio/full").exists()
    assert export_collection(tmp_path)["accepted_examples"] == 1
    verify_portable_audio(tmp_path)


@pytest.mark.asyncio
async def test_resume_recovers_compacted_episode_without_ledger_or_regeneration(tmp_path, monkeypatch):
    from argparse import Namespace
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from environment_servers.audiomc_sim.collect import collect

    _, path, request = trajectory(tmp_path, "audiomc-test-0000")
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "test", "graph": {}}))
    (tmp_path / "tasks.jsonl").write_text(json.dumps({"task_input": {"axis_id": "voice_editing"}}) + "\n")
    export_collection(tmp_path)
    assert not path.exists() and not request.exists()
    assert not (tmp_path / "episode_results.jsonl").exists()
    client = SimpleNamespace(post=AsyncMock())

    @asynccontextmanager
    async def graph(_):
        yield client

    monkeypatch.setattr("environment_servers.audiomc_sim.collect.serve_graph", graph)
    await collect(Namespace(output=tmp_path, env_file=None, max_rollouts=1, target_accepted=1))
    client.post.assert_not_awaited()
    row = json.loads((tmp_path / "episode_results.jsonl").read_text())
    assert row["result"]["outcome"] == "accepted_correction"
    assert row["result"]["artifact_path"] == str(tmp_path / "episodes.jsonl")
    assert started_attempt_count(tmp_path) == 1
    assert json.loads((tmp_path / "summary.json").read_text())["accepted_examples"] == 1
    verify_portable_audio(tmp_path)


def test_failed_resharding_restores_only_remaining_data_copy(tmp_path, monkeypatch):
    trajectory(tmp_path, "full")
    previous = export_collection(tmp_path, max_audio_files=1)
    assert not list((tmp_path / "artifacts/episodes").glob("*.json"))
    summary_before = (tmp_path / "summary.json").read_bytes()
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.parent == tmp_path and path.name == previous["audio_archives"][0]:
            raise OSError("simulated archive publication failure")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "unlink", unlink)
        with pytest.raises(OSError, match="archive publication failure"):
            export_collection(tmp_path, max_audio_files=2)
    assert (tmp_path / "summary.json").read_bytes() == summary_before
    assert not (tmp_path / "audio.tar").exists()
    assert not (tmp_path / ".export-backup").exists()
    verify_portable_audio(tmp_path)
    assert export_collection(tmp_path, max_audio_files=2)["accepted_examples"] == 1
    verify_portable_audio(tmp_path)


def test_restart_rolls_back_a_killed_export_before_reading_compacted_records(tmp_path):
    trajectory(tmp_path, "full")
    previous = export_collection(tmp_path)
    backup = tmp_path / ".export-backup"
    backup.mkdir()
    names = [*previous["audio_archives"], "rollout.jsonl", "episodes.jsonl", "inspection.jsonl", "summary.json"]
    for name in names:
        (backup / name).hardlink_to(tmp_path / name)
    (backup / "files.json").write_text(json.dumps(names))
    # Model a kill after some output files have been replaced, before commit.
    for name, data in [("audio.tar", b"corrupt replacement"), ("episodes.jsonl", b""), ("summary.json", b"{}")]:
        temporary = tmp_path / (name + ".interrupted")
        temporary.write_bytes(data)
        temporary.replace(tmp_path / name)
    assert export_collection(tmp_path) == previous
    assert not backup.exists()
    verify_portable_audio(tmp_path)


def test_interrupted_cleanup_uses_exported_index_on_retry(tmp_path, monkeypatch):
    record, path, request = trajectory(tmp_path, "full")
    partial = tmp_path / "artifacts/partial/full.json"
    partial.parent.mkdir()
    partial.write_text(json.dumps({"audio": [t["audio"] for t in record["turn_evaluations"]]}))
    original = Path.unlink

    def unlink(target, *args, **kwargs):
        if target == partial:
            raise OSError("simulated cleanup interruption")
        return original(target, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "unlink", unlink)
        with pytest.raises(OSError, match="cleanup interruption"):
            export_collection(tmp_path)
    assert not path.exists() and partial.exists() and request.exists()
    verify_portable_audio(tmp_path)
    assert export_collection(tmp_path)["accepted_examples"] == 1
    assert not partial.exists() and not request.exists()
    assert not (tmp_path / "artifacts/audio/full").exists()
    assert started_attempt_count(tmp_path) == 1  # No request or ledger is needed for this durable completion.
