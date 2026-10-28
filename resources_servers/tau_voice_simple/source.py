# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate or fetch the Tau ``data/`` tree that matches the pinned tau2 package.

The tau2 wheel does not ship the repository's ``data`` directory (domains,
tasks, policies, user-simulator guidelines), so the server either uses a
prepared copy or clones the pinned revision once into a shared directory.
"""

import hashlib
import shutil
import time
import uuid
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory


TAU2_BENCH_REPO_URL = "https://github.com/sierra-research/tau2-bench"
# Tau v1.0.0, the revision the VLMEvalKitMcore reference evaluated (also pinned in requirements.txt).
TAU2_BENCH_REF = "8ebb7499622fc2be9b9d510d6f7a7653461f4f29"
DOMAINS = ("airline", "retail", "telecom")
_LOCK_STALE_SECONDS = 3600.0


def data_identity(data_dir: Path) -> dict[str, str]:
    """Portable identity for the policies, fixtures and scoring data used by Tau."""
    return {
        str(p.relative_to(data_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(data_dir.rglob("*"))
        if p.is_file() and p.suffix in {".json", ".jsonl", ".md", ".txt"}
    }


def has_tau2_data(data_dir: Path) -> bool:
    root = data_dir / "tau2"
    return (root / "user_simulator" / "simulation_guidelines_voice.md").is_file() and all(
        (root / "domains" / domain).is_dir() for domain in DOMAINS
    )


def ensure_tau2_data_dir(data_dir: str | Path, repo_url: str = TAU2_BENCH_REPO_URL, ref: str = TAU2_BENCH_REF) -> Path:
    """Return ``data_dir`` after making sure it holds the pinned Tau data tree."""
    data_dir = Path(data_dir)
    if has_tau2_data(data_dir):
        return data_dir
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = data_dir.parent / f".{data_dir.name}.lockdir"
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > _LOCK_STALE_SECONDS:
                    shutil.rmtree(lock, ignore_errors=True)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(5.0)
    try:
        if not has_tau2_data(data_dir):
            with TemporaryDirectory(prefix=f"tau2-{uuid.uuid4().hex[:8]}-", dir=data_dir.parent) as tmp:
                checkout = Path(tmp) / "tau2-bench"
                run(
                    ["git", "clone", "--quiet", "--filter=blob:none", repo_url, str(checkout)],
                    check=True,
                )
                run(["git", "-C", str(checkout), "checkout", "--quiet", ref], check=True)
                if data_dir.exists():
                    shutil.rmtree(data_dir)
                shutil.copytree(checkout / "data", data_dir, ignore=shutil.ignore_patterns("results"))
            if not has_tau2_data(data_dir):
                raise RuntimeError(f"Tau data from {repo_url}@{ref} is missing domains or voice guidelines")
    finally:
        shutil.rmtree(lock, ignore_errors=True)
    return data_dir
