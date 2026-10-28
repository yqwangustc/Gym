# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Task contracts and environments for the shared Tau-Voice-Simple loop.

Upstream runtimes execute in isolated subprocesses. They never run a user,
candidate, TTS engine, or conversation orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PACK_DOMAINS = {
    "tau": ("telecom", "retail", "airline"),
    "eva": ("airline", "itsm", "medical_hr"),
    "mivas": ("healthcare", "legal", "customer-support"),
}


def selected_pack():
    name = os.environ.get("TAU_VOICE_SIMPLE_TASK_PACK", "tau").strip().lower()
    if name not in PACK_DOMAINS:
        raise ValueError(f"Unknown task pack {name!r}; choose {tuple(PACK_DOMAINS)}")
    return name


def selected_domains():
    pack = selected_pack()
    raw = os.environ.get("TAU_VOICE_SIMPLE_DOMAINS", "")
    domains = tuple(x.strip() for x in raw.split(",")) if raw else PACK_DOMAINS[pack]
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("Task-pack domains must be nonempty and unique")
    if any(x not in PACK_DOMAINS[pack] for x in domains):
        raise ValueError(f"Invalid {pack} domains: {domains}")
    return domains


class PackWorker:
    """One episode's source runtime; JSON lines are the only protocol."""

    def __init__(self, pack, root, data_root, domain, artifact_dir):
        self.log = (Path(artifact_dir) / "task_pack_worker.log").open("a")
        worker_env = dict(os.environ)
        # Tool workers do not load the candidate serving stack. In particular,
        # client PYTHONPATH can run vLLM sitecustomize before main() and write
        # startup banners into the worker's protocol stream.
        worker_env.pop("PYTHONPATH", None)
        worker_env.pop("VLMEVAL_PATCH_NEMO_TRANSFORMER_VLLM", None)
        self.process = subprocess.Popen(
            [
                os.environ.get(f"TAU_VOICE_SIMPLE_{pack.upper()}_PYTHON", sys.executable),
                "-u",
                str(Path(__file__).with_name("task_pack_worker.py")),
                pack,
                str(root),
                str(data_root),
                domain,
                str(artifact_dir),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            cwd=root,
            env=worker_env,
        )

    def request(self, op, **payload):
        self.process.stdin.write(json.dumps({"op": op, **payload}) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                self.log.flush()
                tail = Path(self.log.name).read_text()[-4000:]
                raise RuntimeError(f"Task-pack worker exited: {tail}")
            if line.startswith("TAU_TASK_PACK_RESPONSE "):
                result = json.loads(line.removeprefix("TAU_TASK_PACK_RESPONSE "))
                break
            self.log.write(line)
            self.log.flush()
        if "error" in result:
            raise RuntimeError(f"Task-pack {op} failed: {result['error']}")
        return result["result"]

    def close(self):
        if self.log.closed:
            return
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()
        self.log.close()


@dataclass
class SchemaTool:
    openai_schema: dict

    @property
    def name(self):
        return self.openai_schema["function"]["name"]


class TaskPack:
    def __init__(self, name, domain, task_split):
        if domain not in PACK_DOMAINS[name]:
            raise ValueError(f"Unknown {name} domain: {domain}")
        if task_split not in {"base", "test"}:
            raise ValueError(f"{name} only supports base/test (the public test set)")
        self.name, self.domain = name, domain
        key = f"TAU_VOICE_SIMPLE_{name.upper()}_ROOT"
        if not os.environ.get(key):
            raise ValueError(f"Set {key} to the upstream source checkout")
        self.root = Path(os.environ[key]).resolve(strict=True)
        self.data_root = (
            Path(os.environ.get("TAU_VOICE_SIMPLE_MIVAS_DATA_ROOT", str(self.root / "industries"))).resolve(
                strict=True
            )
            if name == "mivas"
            else self.root
        )
        with tempfile.TemporaryDirectory(prefix="tau-task-inventory-") as tmp:
            worker = PackWorker(name, self.root, self.data_root, domain, tmp)
            try:
                inventory = worker.request("inventory")
            finally:
                worker.close()
        self.contracts = {x["id"]: x for x in inventory}
        if len(self.contracts) != len(inventory) or not inventory:
            raise ValueError("Empty or duplicate task IDs in upstream inventory")

    def load_tasks(self):
        from tau2.data_model.tasks import Task, UserScenario

        # Ground truth stays outside Tau's Task: only the caller contract goes
        # to the user, and only the policy/tools go to the candidate.
        return [
            Task(
                id=k,
                user_scenario=UserScenario(instructions=v["caller"]),
                user_tools=[],
            )
            for k, v in self.contracts.items()
        ]

    def build_environment(self, task, artifact_dir):
        from tau2.environment.environment import Environment

        pack = self

        class PackEnvironment(Environment):
            def __init__(self):
                self.worker = PackWorker(pack.name, pack.root, pack.data_root, pack.domain, artifact_dir)
                try:
                    self.context = self.worker.request("init", task_id=str(task.id))
                    expected = pack.contracts[str(task.id)]["contract_sha256"]
                    if self.context["contract_sha256"] != expected:
                        raise RuntimeError("Task contract changed since dataset construction")
                    super().__init__(domain_name=pack.domain, policy=self.context["policy"])
                except BaseException:
                    self.worker.close()
                    raise

            @property
            def session_ended(self):
                return self.context.get("session_ended", False)

            def get_policy(self):
                return self.context["policy"]

            def get_tools(self):
                return [SchemaTool(x) for x in self.context["tools"]]

            def _has_tool(self, name):
                return name in {t.name for t in self.get_tools()}

            def _is_mutating_tool(self, name):
                return True

            def use_tool(self, tool_name, **kwargs):
                result = self.worker.request("tool", name=tool_name, arguments=kwargs)
                self.context = result["context"]
                return result["output"]

            def set_state(
                self,
                initialization_data,
                initialization_actions,
                message_history,
                strict=True,
            ):
                if initialization_data or initialization_actions or message_history:
                    raise ValueError("External packs initialize from their own task fixtures")

            def save_evidence(self):
                return self.worker.request("save")

            def close(self):
                self.worker.close()

        return PackEnvironment()

    def score(self, row):
        artifact_dir = Path(row["simulation_path"]).parent
        evidence_path = artifact_dir / "task_pack_evidence.json"
        evidence = json.loads(evidence_path.read_text())
        contract = self.contracts[str(row["task_id"])]
        if (
            evidence["pack"],
            evidence["domain"],
            evidence["task_id"],
            evidence["contract_sha256"],
        ) != (self.name, self.domain, str(row["task_id"]), contract["contract_sha256"]):
            raise ValueError("Task-pack evidence does not match this task contract")
        worker = PackWorker(self.name, self.root, self.data_root, self.domain, artifact_dir)
        try:
            verdict = worker.request("score", evidence_path=str(evidence_path))
        finally:
            worker.close()
        return verdict


def contract_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
