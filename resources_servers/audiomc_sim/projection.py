# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project verified conversation prefixes into training trajectories."""

from pydantic import JsonValue

from resources_servers.audiomc_sim.contracts import Candidate, SavedTrainingExample, TurnEvaluation


def project_training_example(
    *,
    example_id: str,
    rollout_id: str,
    turns: list[TurnEvaluation],
    teacher_attempts: list[Candidate],
    metadata: dict[str, JsonValue],
    termination_outcome: str,
) -> SavedTrainingExample | None:
    """Keep the accepted prefix if it contains a verified Teacher replacement.

    The unresolved exchange and its candidates stay in the episode inspection
    record, not in the training conversation or its turn evaluations.
    """
    prefix = []
    for turn in turns:
        if turn.accepted_response is None or turn.accepted_response.judgment.passed is not True:
            break
        prefix.append(turn)
    rewritten = [turn for turn in prefix if turn.generated_by_teacher]
    if not rewritten:
        return None
    attempts = [candidate for candidate in teacher_attempts if candidate.turn_index < len(prefix)]
    last_rewrite = rewritten[-1].turn_index
    conversation = []
    for turn in prefix:
        accepted = turn.accepted_response
        conversation.extend(
            [
                {"turn_id": turn.user_turn_id, "role": "user", "text": turn.user_text},
                {
                    "turn_id": f"a{turn.turn_index}",
                    "role": "assistant",
                    "text": accepted.text,
                    "generated_by_teacher": turn.generated_by_teacher,
                    "thinking": accepted.thinking,
                },
            ]
        )
    return SavedTrainingExample(
        example_id=example_id,
        rollout_id=rollout_id,
        turn_evaluations=prefix,
        rejected_teacher_attempts=[c for c in attempts if c.turn_index == last_rewrite and c.judgment.passed is False],
        metadata=metadata | {"termination_outcome": termination_outcome},
        conversation=conversation,
        teacher_attempts=attempts,
    )
