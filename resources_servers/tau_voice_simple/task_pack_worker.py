# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Isolated upstream task/tool/verifier process; no model or dialogue runtime."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from task_packs import contract_hash


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Runtime:
    def __init__(self, pack, root, data_root, domain, artifact_dir):
        self.pack, self.root, self.data_root = pack, root, data_root
        self.domain, self.artifact_dir = domain, artifact_dir
        self.calls, self.counts = [], Counter()
        self.session_ended = False
        self.task_id = None
        if pack == "eva":
            sys.path.insert(0, str(root / "src"))
            from eva.models.agents import AgentConfig
            from eva.utils import culture

            self.culture = culture
            self.config = AgentConfig.from_yaml(root / f"configs/agents/{domain}_agent.yaml")
            self.records = {str(r["id"]): r for r in json.loads((root / f"data/{domain}_dataset.json").read_text())}
            self.schemas = self.config.build_tools_for_agent()
            self.schema_map = {s["function"]["name"]: s for s in self.schemas}
            paths = [
                *root.joinpath("src/eva").rglob("*.py"),
                root / f"configs/agents/{domain}_agent.yaml",
                root / f"data/{domain}_dataset.json",
                *root.joinpath(f"data/{domain}_scenarios").glob("*.json"),
                *root.joinpath("data/itsm_aliases").rglob("*.json"),
            ]
        else:
            self.industry_dir = data_root / domain
            tasks_file = self.industry_dir / "tasks.jsonl"
            if tasks_file.is_file():
                records = [json.loads(x) for x in tasks_file.read_text().splitlines() if x.strip()]
                self.records = {}
                for r in records:
                    key = r.get("case_key") or r.get("task_id") or r["task_name"].split(":")[0]
                    if str(key) in self.records:
                        raise ValueError(f"Duplicate MIVAS case {key}")
                    self.records[str(key)] = r
            else:
                self.records = {}
                for p in sorted((self.industry_dir / "tasks").glob("*/task.json")):
                    r = json.loads(p.read_text())
                    if "exp_db_state" not in r:
                        r["exp_db_state"] = json.loads(p.with_name("exp_db_state.json").read_text())
                    self.records[p.parent.name] = r
            self.blueprint = json.loads((self.industry_dir / "agent_blueprint.json").read_text())
            self.agents = {a["name"]: a for a in self.blueprint["agents"]}
            self.active = self.blueprint["agents"][0]["name"]
            self.schema_map = {
                s["name"]: {
                    "type": "function",
                    "function": {
                        "name": s["name"],
                        "description": s.get("description", ""),
                        "parameters": s["inputSchema"],
                    },
                }
                for s in json.loads((self.industry_dir / "tools.json").read_text())["tools"]
            }
            for a in self.agents.values():
                (self.industry_dir / a["system_prompt"]).resolve().relative_to(self.industry_dir)
                for t in a["tools"]:
                    if t["name"] not in self.schema_map:
                        raise ValueError(f"Missing industry tool schema: {t['name']}")
                    if t.get("handoff") and t["handoff_to"] not in self.agents:
                        raise ValueError(f"Unknown handoff target: {t}")
            paths = [
                *root.joinpath("runtime").glob("*.py"),
                *root.joinpath("verifiers").glob("*.py"),
                root / f"industries/{domain}/tool_server.py",
                root / f"industries/{domain}/db/schema.sql",
                root / f"industries/{domain}/db/seed.sql",
                *self.industry_dir.rglob("*.json"),
                *self.industry_dir.rglob("*.jsonl"),
                *self.industry_dir.rglob("*.md"),
            ]
        # Content seal covers dirty checkouts as well as pinned commits.
        self.source_files = {
            (
                f"source/{p.relative_to(root)}" if p.is_relative_to(root) else f"data/{p.relative_to(data_root)}"
            ): contract_hash(p.read_text())
            for p in sorted(set(paths))
        }
        self.source_sha256 = contract_hash(self.source_files)

    def resolve_eva(self, value, record):
        return self.culture.resolve_scenario_db(
            value,
            record.get("culture_overrides"),
            "en",
            record.get("romanized_culture_overrides"),
            aliases_dir=self.root / "data/itsm_aliases" if self.domain == "itsm" else None,
        )

    def caller(self, record):
        if self.pack == "eva":
            goal = self.culture.resolve_user_goal(
                record["user_goal"],
                record.get("culture_overrides"),
                "en",
                record.get("romanized_culture_overrides"),
                record.get("starting_utterances"),
                aliases_dir=self.root / "data/itsm_aliases" if self.domain == "itsm" else None,
            )
            persona = self.culture.resolve_user_config(
                record["user_config"],
                record.get("culture_overrides"),
                "en",
                record.get("romanized_culture_overrides"),
            )
            return json.dumps(
                {
                    "current_date_time": record["current_date_time"],
                    "user_goal": goal,
                    "user_config": persona,
                },
                ensure_ascii=False,
            )
        # Explicit caller fields only: no verifier/gold-state/expected-tool leakage.
        return json.dumps(
            {
                k: record[k]
                for k in (
                    "intent",
                    "traits",
                    "behaviors",
                    "scripted_responses",
                    "customer_name",
                )
                if k in record
            },
            ensure_ascii=False,
        )

    def task_hash(self, key):
        return contract_hash({"record": self.records[key], "sources": self.source_sha256})

    def inventory(self):
        for key, record in self.records.items():
            if self.pack == "eva":
                expected = record["ground_truth"]["expected_scenario_db"]
                if not (self.root / f"data/{self.domain}_scenarios/{key}.json").is_file():
                    raise ValueError(f"Missing EVA initial fixture for {key}")
            else:
                expected = record["exp_db_state"]
                if record.get("customer_available_tools"):
                    raise ValueError("This MIVAS adaptation does not implement caller-side tools")
                if not isinstance(record["exp_tool_calls"], list) or not isinstance(record["exp_handoff_path"], list):
                    raise ValueError(f"Invalid MIVAS verifier contract for {key}")
            if not isinstance(expected, dict):
                raise ValueError(f"Missing expected database state for {key}")  # noqa: TRY004
        return [
            {
                "id": key,
                "caller": self.caller(r),
                "contract_sha256": self.task_hash(key),
            }
            for key, r in self.records.items()
        ]

    def init(self, task_id):
        self.task_id = task_id
        self.record = self.records[task_id]
        if self.pack == "eva":
            from eva.assistant.tools.tool_executor import ToolExecutor

            db = json.loads((self.root / f"data/{self.domain}_scenarios/{task_id}.json").read_text())
            self.initial_state = self.resolve_eva(db, self.record)
            db_path = self.artifact_dir / "initial_scenario_db.json"
            db_path.write_text(json.dumps(self.initial_state, ensure_ascii=False, indent=2))
            self.executor = ToolExecutor(
                str(self.root / f"configs/agents/{self.domain}_agent.yaml"),
                str(db_path),
                self.config.tool_module_path,
                self.record["current_date_time"],
            )
            if any(self.executor._get_tool_function(name) is None for name in self.schema_map):
                raise ValueError("EVA tool schema has no executable implementation")
        else:
            os.environ["MIVAS_DB_PATH"] = str(self.artifact_dir / "mivas_db/runtime.db")
            os.environ.pop("MIVAS_DB_SHARED", None)
            self.server = load_file(
                "tau_mivas_tool_server",
                self.root / f"industries/{self.domain}/tool_server.py",
            )
            for agent in self.agents.values():
                for tool in agent["tools"]:
                    if (
                        not tool.get("handoff")
                        and tool["name"] != "end_call"
                        and tool["name"] not in self.server.DISPATCH
                    ):
                        raise ValueError(f"No MIVAS handler for {tool['name']}")
            with self.server.db.scope("episode", fresh=True):
                self.initial_state = self.server.state()
        return self.context()

    def context(self):
        if self.pack == "eva":
            policy = (
                f"{self.config.role}\n{self.config.description}\n"
                f"Current date and time: {self.record['current_date_time']}\n"
                f"{self.config.instructions}"
            )
            tools = self.schemas
        else:
            agent = self.agents[self.active]
            policy = (self.industry_dir / agent["system_prompt"]).read_text()
            tools = [self.schema_map[t["name"]] for t in agent["tools"]]
        return {
            "policy": policy,
            "tools": tools,
            "session_ended": self.session_ended,
            "contract_sha256": self.task_hash(self.task_id),
        }

    def tool(self, name, arguments):
        available = {x["function"]["name"] for x in self.context()["tools"]}
        if self.session_ended or name not in available:
            output = {"ok": False, "error_code": "TOOL_NOT_AVAILABLE"}
        elif self.pack == "eva":
            output = asyncio.run(self.executor.execute(name, arguments))
        else:
            from jsonschema import ValidationError, validate

            spec = next(t for t in self.agents[self.active]["tools"] if t["name"] == name)
            try:
                validate(arguments, self.schema_map[name]["function"]["parameters"])
            except ValidationError as exc:
                output = {
                    "ok": False,
                    "error_code": "INVALID_ARGUMENTS",
                    "message": exc.message,
                }
            else:
                if spec.get("handoff"):
                    self.active = spec["handoff_to"]
                    output = {"ok": True, "data": {"active_agent": self.active}}
                elif name == "end_call":
                    output = {"ok": True, "data": {"ended": True}}
                    self.session_ended = True
                else:
                    with self.server.db.scope("episode"):
                        output = self.server.dispatch_tool(name, self.server.ToolCall(arguments=arguments))
                    if spec.get("session"):
                        self.session_ended = True
        self.calls.append(
            {
                "name": name,
                "parameters": arguments,
                "output": output,
                "start_offset_ms": len(self.calls),
            }
        )
        return {"output": output, "context": self.context()}

    def save(self):
        if self.pack == "eva":
            state = self.executor.db
        else:
            with self.server.db.scope("episode"):
                state = self.server.state()
        evidence = {
            "pack": self.pack,
            "domain": self.domain,
            "task_id": self.task_id,
            "contract_sha256": self.task_hash(self.task_id),
            "source_sha256": self.source_sha256,
            "source_files": self.source_files,
            "final_state": state,
            "calls": self.calls,
            "session_ended": self.session_ended,
        }
        path = self.artifact_dir / "task_pack_evidence.json"
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
        return str(path)

    def score(self, evidence_path):
        evidence = json.loads(Path(evidence_path).read_text())
        key = evidence["task_id"]
        if evidence["contract_sha256"] != self.task_hash(key):
            raise ValueError("Upstream sources changed since inference; restore the sealed sources")
        if evidence["final_state"] is None:
            raise ValueError("Missing final state; cannot score this episode")
        if self.pack == "eva":
            from eva.metrics.accuracy.task_completion import TaskCompletion
            from eva.utils.hash_utils import get_dict_hash

            expected = self.resolve_eva(
                self.records[key]["ground_truth"]["expected_scenario_db"],
                self.records[key],
            )
            result = asyncio.run(
                TaskCompletion().compute(
                    SimpleNamespace(
                        expected_scenario_db=expected,
                        final_scenario_db=evidence["final_state"],
                        final_scenario_db_hash=get_dict_hash(evidence["final_state"]),
                    )
                )
            )
            return {
                "reward": float(result.score),
                "components": result.model_dump(mode="json"),
            }
        verifier = load_file("tau_mivas_verifier", self.root / "verifiers/verify_task_run.py")
        result = {"tool_calls": [{"name": c["name"], "actual": [c]} for c in evidence["calls"]]}
        schemas = {t["name"]: t for t in json.loads((self.industry_dir / "tools.json").read_text())["tools"]}
        task = self.records[key]
        if task.get("exp_db_state") is None:
            raise ValueError("Missing expected MIVAS state; refusing a vacuous pass")
        verdict = verifier.verify_result(result, task, evidence["final_state"], industry=self.domain, schemas=schemas)
        if verdict["state"]["skipped"] or verdict["state"]["passed"] is None:
            raise ValueError("MIVAS state verification was skipped")
        return {"reward": float(verdict["passed"]), "components": verdict}


def main():
    protocol_out = sys.stdout
    # Dependencies may print startup banners; stdout is reserved for the protocol.
    with contextlib.redirect_stdout(sys.stderr):
        pack, root, data, domain, artifact = sys.argv[1:]
        runtime = Runtime(pack, Path(root), Path(data), domain, Path(artifact))
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request.pop("op")
            if op not in {"inventory", "init", "tool", "save", "score"}:
                raise ValueError(f"Unknown task-pack operation {op}")
            with contextlib.redirect_stdout(sys.stderr):
                result = getattr(runtime, op)(**request)
            response = {"result": result}
        except Exception as exc:  # noqa: BLE001 - preserve the parent/worker protocol on tool errors
            traceback.print_exc(file=sys.stderr)
            response = {"error": f"{type(exc).__name__}: {exc}"}
        protocol_out.write("TAU_TASK_PACK_RESPONSE " + json.dumps(response, ensure_ascii=False) + "\n")
        protocol_out.flush()


if __name__ == "__main__":
    main()
