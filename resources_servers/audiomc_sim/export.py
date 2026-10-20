# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export completed episodes with portable, hash-checked audio archives."""

import argparse
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


DEFAULT_MAX_AUDIO_FILES = 50_000


def export_collection(root: Path, *, max_audio_files: int = DEFAULT_MAX_AUDIO_FILES) -> dict[str, object]:
    """Export examples and count-bounded audio archives, replacing each file atomically.

    Audio from excluded episodes is retained for inspection too. Input prefixes
    use transcript text only as annotation; responses_create_params replays the
    exact audio-only assistant input with the accepted teacher response as target.
    """
    if max_audio_files < 1:
        raise ValueError("max_audio_files must be >= 1")
    artifact_root = root / "artifacts"
    records = [json.loads(p.read_text()) for p in sorted((artifact_root / "episodes").glob("*.json"))]
    partials = {p.stem: json.loads(p.read_text()) for p in (artifact_root / "partial").glob("*.json")}
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    model_manifest = {
        name: {"endpoint": c["base_url"], "model": c["model"]}
        for name, entry in manifest.get("graph", {}).items()
        if "responses_api_models" in entry
        for c in entry["responses_api_models"].values()
    }
    model_manifest["assistant_checkpoint"] = manifest.get("assistant_checkpoint")
    audio_refs = {}

    def discover(value: object) -> None:
        if isinstance(value, dict):
            if {"archive", "member", "sha256"} <= value.keys():
                key = value["member"]
                if key in audio_refs and audio_refs[key] != value:
                    raise ValueError(f"Conflicting audio identity: {key}")
                audio_refs[key] = dict(value)
            for item in value.values():
                discover(item)
        elif isinstance(value, list):
            for item in value:
                discover(item)

    for record in records + list(partials.values()):
        discover(record)
    members = sorted(audio_refs)
    shard_count = max(1, (len(members) + max_audio_files - 1) // max_audio_files)
    archives = [
        "audio.tar" if shard_count == 1 else f"audio-{index:05d}-of-{shard_count:05d}.tar"
        for index in range(shard_count)
    ]
    member_archives = {}
    # Validate/build every shard before replacing any existing exported archive.
    with TemporaryDirectory(prefix=".audio-export-", dir=root) as staging:
        for index, archive in enumerate(archives):
            with tarfile.open(Path(staging) / archive, "w") as target:
                for member in members[index * max_audio_files : (index + 1) * max_audio_files]:
                    ref = audio_refs[member]
                    with tarfile.open(artifact_root / ref["archive"], "r:") as source:
                        stream = source.extractfile(ref["member"])
                        if stream is None:
                            raise ValueError(f"Missing audio: {member}")
                        data = stream.read()
                    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
                        raise ValueError(f"Audio hash mismatch: {member}")
                    info = tarfile.TarInfo(member)
                    info.size = len(data)
                    target.addfile(info, io.BytesIO(data))
                    member_archives[member] = archive
        for archive in archives:
            (Path(staging) / archive).replace(root / archive)

    def rewrite(value: object) -> None:
        if isinstance(value, dict):
            if {"archive", "member", "sha256"} <= value.keys():
                value["archive"] = member_archives[value["member"]]
            for item in value.values():
                rewrite(item)
        elif isinstance(value, list):
            for item in value:
                rewrite(item)

    accepted, inspection = [], []
    counts = Counter(
        {
            "accepted_correction": 0,
            "teacher_correction_exhausted": 0,
            "max_turn_reached_without_failing": 0,
            "ungradable": 0,
            "infrastructure_failure": 0,
        }
    )
    for record in records:
        rewrite(record)
        record["metadata"]["models"] = model_manifest
        counts[record["outcome"]] += 1
        if record["outcome"] == "accepted_correction":
            example = record["training_example"]
            example["metadata"]["models"] = model_manifest
            evaluations = {t["user_turn_id"]: t for t in example["turn_evaluations"]}
            prefix = example["input_prefix"]
            inputs = [{"role": "system", "content": prefix["system_prompt"]}]
            if prefix["developer_prompt"]:
                inputs.append({"role": "developer", "content": prefix["developer_prompt"]})
            refs = []
            for turn in prefix["turns"]:
                if turn["role"] == "user":
                    audio = evaluations[turn["turn_id"]]["audio"]
                    turn["audio"] = audio
                    refs.append([{"archive": audio["archive"], "member": audio["member"]}])
                    inputs.append({"role": "user", "content": ""})
                else:
                    inputs.append({"role": "assistant", "content": turn["text"]})
            example["responses_create_params"] = {"input": inputs, "metadata": {"audio_refs": json.dumps(refs)}}
            example["supervised_target"] = example["corrected_response"]["text"]
            accepted.append(example)
        else:
            inspection.append(record)
    ledger = root / "episode_results.jsonl"
    if ledger.exists():
        failures = [json.loads(line) for line in ledger.read_text().splitlines() if json.loads(line).get("failure")]
        for failure in failures:
            identity = failure["episode_id"]
            attempt = identity.get("attempt", 0)
            key = identity["rollout_id"] + (f"-a{attempt}" if attempt else "")
            trace = partials.get(key)
            if trace:
                rewrite(trace)
                failure["partial_trace"] = trace
        inspection.extend(failures)
        counts["infrastructure_failure"] = len(failures)
    for name, rows in [("rollout.jsonl", accepted), ("inspection.jsonl", inspection), ("episodes.jsonl", records)]:
        temp = root / f"{name}.tmp"
        temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        temp.replace(root / name)
    summary = {
        "rollout_counter": sum(counts.values()),
        "accepted_examples": len(accepted),
        "audio_members": len(audio_refs),
        "audio_archives": archives,
        "max_audio_files": max_audio_files,
        "counters": dict(counts),
        "by_axis": dict(Counter(r.get("metadata", {}).get("task", {}).get("axis_id", "unknown") for r in records)),
        "accepted_turn_depths": dict(Counter(str(len(e["turn_evaluations"])) for e in accepted)),
        "accepted_with_prior_history": sum(len(e["turn_evaluations"]) > 1 for e in accepted),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    # Remove only obsolete exports, after JSONL references point to the new set.
    for path in root.iterdir():
        if path.name not in archives and (
            path.name == "audio.tar" or re.fullmatch(r"audio-\d{5,}-of-\d{5,}\.tar", path.name)
        ):
            path.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--max-audio-files",
        type=int,
        default=DEFAULT_MAX_AUDIO_FILES,
        help="Maximum WAV members per archive (default: %(default)s). Larger exports use "
        "audio-00000-of-NNNNN.tar shards; a single archive keeps the name audio.tar.",
    )
    args = parser.parse_args()
    if args.max_audio_files < 1:
        parser.error("Require max-audio-files >= 1")
    print(json.dumps(export_collection(args.directory, max_audio_files=args.max_audio_files), indent=2))


if __name__ == "__main__":
    main()
