# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One Tau-Voice Simple episode driven by Tau's own half-duplex orchestrator.

The candidate lives outside this process (the Gym agent calls the policy
model), so the orchestrator's agent is a relay: each time Tau asks the agent
for its next message, the relay returns the message the Gym agent submitted.
Everything else (environment, tools, user simulator, step and error limits,
timeout, communication checks, scoring) is Tau's unmodified code, which keeps
termination and reward identical to the VLMEvalKitMcore reference.

Importing this module imports tau2, so ``TAU2_DATA_DIR`` must be set first.
"""

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import Role
from tau2.registry import registry
from tau2.runner.build import build_environment
from tau2.utils.utils import get_now

from resources_servers.tau_voice_simple.airline_assertions import assertions_for_task
from resources_servers.tau_voice_simple.orchestrator import CALLER_PROTOCOL, VoiceOrchestrator
from resources_servers.tau_voice_simple.tts import TTSClient
from resources_servers.tau_voice_simple.user import VoiceUserSimulator


CONTEXT_EXHAUSTED_KEY = "tau_voice_simple_context_exhausted"


def load_tasks(domain: str, task_split: str) -> dict[str, Task]:
    """Tau tasks for one domain split, with the curated airline NL assertions adopted."""
    tasks = {}
    for task in registry.get_tasks_loader(domain)(task_split_name=task_split):
        assertions = assertions_for_task(domain, task_split, str(task.id))
        if assertions is not None:
            if task.evaluation_criteria is None:
                raise ValueError(f"Tau {domain} task {task.id} has no evaluation criteria")
            task = task.model_copy(deep=True)
            task.evaluation_criteria.nl_assertions = list(assertions)
            if RewardType.NL_ASSERTION not in task.evaluation_criteria.reward_basis:
                task.evaluation_criteria.reward_basis.append(RewardType.NL_ASSERTION)
        tasks[str(task.id)] = task
    return tasks


def configure_nl_assertions_judge(model: str, llm_args: dict[str, Any]) -> None:
    """Point Tau's NL-assertion judge at the configured model.

    The evaluator module binds ``DEFAULT_LLM_NL_ASSERTIONS`` at import time, so
    both bindings must change (as in the reference's ``scoring.py``).
    """
    import tau2.config as tau_config
    from tau2.evaluator import evaluator_nl_assertions

    for module in (tau_config, evaluator_nl_assertions):
        module.DEFAULT_LLM_NL_ASSERTIONS = model
        module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(llm_args)


@dataclass
class _RelayState:
    messages: list = field(default_factory=list)


class RelayAgent(HalfDuplexAgent):
    """Tau agent whose next message is supplied by the Gym agent."""

    def __init__(self, tools, domain_policy: str):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.pending: AssistantMessage | None = None

    def get_init_state(self, message_history=None) -> _RelayState:
        return _RelayState()

    def generate_next_message(self, message, state):
        if self.pending is None:
            raise RuntimeError("Tau requested an agent message before the Gym agent submitted one")
        reply, self.pending = self.pending, None
        return reply, state

    def set_seed(self, seed: int) -> None:
        # The candidate's sampling seed is set by the Gym agent's requests.
        pass

    @classmethod
    def is_stop(cls, message) -> bool:
        """End only episodes whose history no longer fits the candidate's context."""
        return bool((getattr(message, "raw_data", None) or {}).get(CONTEXT_EXHAUSTED_KEY))


@dataclass(frozen=True)
class EpisodeSettings:
    interaction_mode: str
    max_steps: int
    max_errors: int
    timeout_seconds: float
    user_llm: str
    user_llm_args: dict[str, Any]


def reference_agent_write_names(task: Task, environment) -> tuple[str, ...]:
    """Mutating tools the reference solution calls as the assistant."""
    actions = task.evaluation_criteria.actions if task.evaluation_criteria is not None else None
    return tuple(
        sorted(
            {
                str(action.name)
                for action in actions or ()
                if action.requestor == "assistant" and environment._is_mutating_tool(action.name)
            }
        )
    )


def _user_tools(environment, task: Task):
    try:
        return environment.get_user_tools(include=task.user_tools) or None
    except (AttributeError, TypeError, ValueError):
        # Most Tau domains have no user-side tools; Tau's runner treats that as none.
        return None


class Episode:
    """A seeded Tau episode that advances until the candidate must speak or act."""

    def __init__(
        self,
        *,
        domain: str,
        task: Task,
        seed: int,
        simulation_id: str,
        settings: EpisodeSettings,
        tts_client: TTSClient | None,
        environment=None,
        pack_backend=None,
        artifact_dir=None,
    ):
        self.domain = domain
        self.task = task
        self.settings = settings
        environment = environment or build_environment(domain, solo_mode=False)
        self.environment = environment
        self.pack_backend = pack_backend
        self.artifact_dir = artifact_dir
        self.tools = environment.get_tools()
        self.policy = environment.get_policy()
        self.agent = RelayAgent(tools=self.tools, domain_policy=self.policy)
        self.user = VoiceUserSimulator(
            llm=settings.user_llm,
            instructions=str(task.user_scenario),
            tools=_user_tools(environment, task),
            llm_args=dict(settings.user_llm_args),
            tts_client=tts_client if settings.interaction_mode == "s2t" else None,
            required_agent_write_names=reference_agent_write_names(task, environment),
        )
        self.orchestrator = VoiceOrchestrator(
            domain=domain,
            agent=self.agent,
            user=self.user,
            environment=environment,
            task=task,
            max_steps=settings.max_steps,
            max_errors=settings.max_errors,
            seed=seed,
            solo_mode=False,
            simulation_id=simulation_id,
            validate_communication=True,
            timeout=settings.timeout_seconds,
        )
        self.simulation: SimulationRun | None = None

    @property
    def done(self) -> bool:
        return self.orchestrator.done

    def start(self) -> dict[str, Any]:
        """Initialize Tau (as ``Orchestrator.run`` does) and run until the candidate's first turn."""
        orchestrator = self.orchestrator
        orchestrator._run_start_time = get_now()
        orchestrator._run_start_perf = time.perf_counter()
        orchestrator.initialize()
        return self._advance()

    def submit(self, reply: AssistantMessage) -> dict[str, Any]:
        """Deliver the candidate's reply and run until its next turn or the end."""
        if self.done:
            raise RuntimeError("the episode has already ended")
        if self.orchestrator.to_role != Role.AGENT:
            raise RuntimeError("the episode is not waiting for the agent")
        self.agent.pending = reply
        self._step()
        return self._advance()

    def _step(self) -> None:
        self.orchestrator.step()
        if self.pack_backend and self.environment.session_ended:
            self.orchestrator.done = True
            self.orchestrator.termination_reason = TerminationReason.AGENT_STOP
        self.orchestrator._check_termination()

    def _advance(self) -> dict[str, Any]:
        orchestrator = self.orchestrator
        while not orchestrator.done and orchestrator.to_role != Role.AGENT:
            self._step()
        if orchestrator.done:
            return {"done": True, "termination_reason": orchestrator.termination_reason.value}
        self.tools = self.environment.get_tools()
        self.policy = self.environment.get_policy()
        return {
            "done": False,
            "observation": self._observation(orchestrator.message),
            "domain_policy": self.policy,
            "tools": [t.openai_schema for t in self.tools],
        }

    def _observation(self, message) -> dict[str, Any]:
        """What the candidate receives: user speech (or text in t2t) or tool results."""
        if isinstance(message, UserMessage):
            if self.settings.interaction_mode == "t2t":
                return {"type": "user_text", "text": message.content}
            if not message.is_audio or not self.user.audio_turns:
                if (message.raw_data or {}).get("tau_voice_simple_empty_user_turn"):
                    return {"type": "user_text", "text": message.content or ""}
                raise RuntimeError("Tau-Voice Simple s2t candidate input must be audio")
            turn_index, wav = self.user.audio_turns[-1]
            return {
                "type": "user_audio",
                "turn_index": turn_index,
                "audio_wav_base64": base64.b64encode(wav).decode("ascii"),
            }
        tool_messages = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        if not all(isinstance(item, ToolMessage) for item in tool_messages):
            raise TypeError(f"Unsupported message for the agent: {type(message).__name__}")
        return {
            "type": "tool_results",
            "results": [{"tool_call_id": item.id, "content": item.content or ""} for item in tool_messages],
        }

    def finalize(self, info: dict[str, Any]) -> SimulationRun:
        """Build Tau's SimulationRun once the episode has ended."""
        if self.simulation is None:
            if not self.done:
                raise RuntimeError("the episode has not ended")
            simulation = self.orchestrator._finalize()
            info.update(
                caller_protocol=CALLER_PROTOCOL,
                user_tts_wait_seconds=self.user.tts_wait_seconds,
                episode_timeout_excludes_user_tts=True,
            )
            if self.pack_backend:
                self.environment.save_evidence()
                info.update(
                    task_pack=self.pack_backend.name,
                    task_contract_sha256=self.pack_backend.contracts[str(self.task.id)]["contract_sha256"],
                )
            simulation.info = dict(simulation.info or {}) | {"tau_voice_simple": info}
            self.simulation = simulation
        return self.simulation

    def evaluate(self) -> dict[str, Any]:
        if self.pack_backend:
            verdict = self.pack_backend.score(
                {
                    "task_id": str(self.task.id),
                    "simulation_path": str(self.artifact_dir / "simulation.json"),
                }
            )
            return {"reward": verdict["reward"], "components": verdict["components"]}
        reward_info = evaluate_simulation(
            simulation=self.simulation,
            task=self.task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=self.domain,
            mode=CommunicationMode.HALF_DUPLEX,
        )
        return reward_info.model_dump(mode="json")

    def close(self):
        if self.pack_backend:
            self.environment.close()


def assistant_message(content: str | None, tool_call: dict[str, Any] | None, raw_data: dict | None):
    """Tau message for one candidate turn: spoken text or exactly one tool call."""
    if tool_call is not None:
        arguments = tool_call["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        call = ToolCall(id=tool_call["id"], name=tool_call["name"], arguments=arguments, requestor="assistant")
        return AssistantMessage.text(content=None, tool_calls=[call], raw_data=raw_data or None)
    return AssistantMessage.text(content=content, raw_data=raw_data or None)


def trajectory_summary(simulation: SimulationRun) -> dict[str, Any]:
    """Per-episode counters used by the aggregate metrics."""
    assistant = [message for message in simulation.messages if isinstance(message, AssistantMessage)]
    users = [message for message in simulation.messages if isinstance(message, UserMessage)]
    failures = [
        failure
        for message in assistant
        for failure in ((message.raw_data or {}).get("protocol_failures") or [])
        if isinstance(failure, dict)
    ]
    categories: dict[str, int] = {}
    for failure in failures:
        category = str(failure.get("category"))
        categories[category] = categories.get(category, 0) + 1
    return {
        "num_messages": len(simulation.messages),
        "num_assistant_turns": len(assistant),
        "num_tool_calls": sum(len(message.tool_calls or []) for message in assistant),
        "num_user_audio_turns": sum(bool(message.is_audio) for message in users),
        "protocol_fallbacks": sum(bool((message.raw_data or {}).get("protocol_fallback")) for message in assistant),
        "protocol_failure_categories": categories,
        "context_exhausted": any(RelayAgent.is_stop(message) for message in assistant),
        "stop_guard_fired": sum(
            bool((message.raw_data or {}).get("tau_voice_simple_stop_guard")) for message in users
        ),
    }


def simulation_json(simulation: SimulationRun) -> bytes:
    """The trajectory as JSON, without the inline audio already archived as WAVs."""
    payload = simulation.model_dump(mode="json")
    for message in payload.get("messages") or []:
        if message.get("audio_content"):
            message["audio_content"] = None
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
