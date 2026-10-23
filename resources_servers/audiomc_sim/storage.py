# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read compacted episodes and safely reclaim exported per-trajectory source files."""

import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def episode_key(record: dict[str, object]) -> str | None:
    """Identify a physical rollout attempt using the same capture key as Resources."""
    identity = record.get("metadata", {}).get("episode_id", {})
    rollout = identity.get("rollout_id")
    attempt = identity.get("attempt", 0)
    if not isinstance(rollout, str) or not rollout or not isinstance(attempt, int) or attempt < 0:
        return None
    key = rollout + (f"-a{attempt}" if attempt else "")
    if Path(key).name != key or key in {".", ".."}:
        raise ValueError("Invalid episode capture key")
    return key


def read_episode_records(root: Path) -> tuple[dict[str, dict[str, object]], dict[str, Path]]:
    """Merge durable exports with newer raw records; raw records win for the same attempt."""
    records, sources = {}, {}
    published = root / "episodes.jsonl"
    if published.exists():
        with published.open() as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if (key := episode_key(record)) is not None:
                        records[key] = record
    for path in sorted((root / "artifacts/episodes").glob("*.json")):
        record = json.loads(path.read_text())
        key = episode_key(record) or path.stem
        records[key], sources[key] = record, path
    return records, sources


def cleanup_exported_trajectories(root: Path, records: dict[str, dict[str, object]], sources: dict[str, Path]) -> None:
    """Remove max-turn raw files only after records and verified audio have been exported."""
    for record in records.values():
        key = episode_key(record)
        metadata = record.get("metadata", {})
        limit = metadata.get("task", {}).get("max_turns")
        if key is None or not isinstance(limit, int) or limit < 1 or len(record.get("turn_evaluations", [])) != limit:
            continue
        # Delete the raw index before its data: after any interruption, the durable
        # exported record must win instead of a raw record pointing at removed audio.
        if key in sources:
            sources[key].unlink(missing_ok=True)
        partial = root / "artifacts/partial" / f"{key}.json"
        partial.unlink(missing_ok=True)
        audio = root / "artifacts/audio" / key
        if audio.is_symlink():
            audio.unlink()
        elif audio.exists():
            shutil.rmtree(audio)
        identity = metadata["episode_id"]
        request = root / "requests" / f"{identity['rollout_id']}-attempt-{identity.get('attempt', 0)}.json"
        request.unlink(missing_ok=True)


def started_attempt_count(root: Path) -> int:
    """Count both completed and pending attempts even after request files are compacted."""
    identities = set()
    published = root / "episodes.jsonl"
    if published.exists():
        with published.open() as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if episode_key(record) is not None:
                        identity = record["metadata"]["episode_id"]
                        identities.add((identity["rollout_id"], identity.get("attempt", 0)))
    ledger = root / "episode_results.jsonl"
    if ledger.exists():
        with ledger.open() as stream:
            for line in stream:
                if line.strip():
                    identity = json.loads(line)["episode_id"]
                    identities.add((identity["rollout_id"], identity.get("attempt", 0)))
    for path in (root / "requests").glob("*.json"):
        rollout, separator, attempt = path.stem.rpartition("-attempt-")
        if separator and attempt.isdigit():
            identities.add((rollout, int(attempt)))
    return len(identities)


_EXPORT_JSON = ("rollout.jsonl", "episodes.jsonl", "inspection.jsonl", "summary.json")


def _published_name(name: str) -> bool:
    return bool(name in _EXPORT_JSON or name == "audio.tar" or re.fullmatch(r"audio-\d{5,}-of-\d{5,}\.tar", name))


def _published_files(root: Path) -> list[Path]:
    return [p for p in root.iterdir() if p.is_file() and _published_name(p.name)]


def recover_export(root: Path) -> None:
    """Roll back interrupted publication before reading compacted episode data."""
    backup = root / ".export-backup"
    if not backup.exists():
        return
    marker = backup / "files.json"
    if marker.exists():
        names = json.loads(marker.read_text())
        if not isinstance(names, list) or any(
            not isinstance(name, str) or not _published_name(name) for name in names
        ):
            raise ValueError("Invalid export recovery file list")
        for path in _published_files(root):
            if path.name not in names:
                path.unlink()
        for name in names:
            original = backup / name
            if original.exists():
                original.replace(root / name)
            elif not (root / name).exists():
                raise FileNotFoundError(f"Missing export recovery file: {name}")
        marker.unlink()
    shutil.rmtree(backup)


@contextmanager
def publish_export(root: Path) -> Iterator[None]:
    """Keep prior output inodes until all JSONL and audio files commit together."""
    recover_export(root)
    backup = root / ".export-backup"
    backup.mkdir()
    try:
        names = []
        for path in _published_files(root):
            (backup / path.name).hardlink_to(path)
            names.append(path.name)
        marker = backup / "files.json"
        marker_tmp = backup / "files.json.tmp"
        marker_tmp.write_text(json.dumps(names))
        marker_tmp.replace(marker)
        yield
    except BaseException:
        recover_export(root)
        raise
    else:
        # Removing the marker is the commit point. Leftover backup links after a
        # crash are harmless and the next startup can remove them without rollback.
        marker.unlink()
        shutil.rmtree(backup)
