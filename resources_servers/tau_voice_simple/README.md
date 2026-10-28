# Tau-Voice Simple

Tau-Voice Simple runs [Tau2](https://github.com/sierra-research/tau2-bench)
customer-service episodes in which the candidate hears the customer. Tau's LLM
user simulator writes each customer turn, GPT-4o mini TTS speaks it, and the
candidate answers in text or with exactly one tool call per turn
(speech in, text out, half duplex). It is a port of VLMEvalKitMcore's
`TauVoiceSimple` (s2t mode) and keeps its prompts, limits, and scoring.

The evaluation set is Tau v1.0.0's `base` split: 278 episodes (telecom 114,
retail 114, airline 50), one trial each with seed 42. Each session uses one of
11 voices (alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer,
verse), assigned round-robin within each domain.

## How an episode runs

The work is split between this resources server and the
`tau_voice_simple_agent` agent:

| Component | Owns |
| --- | --- |
| `tau_voice_simple` server | Tau's environment and tools, the user simulator, TTS, Tau's step/error/time limits, scoring, and the audio archive |
| `tau_voice_simple_agent` | The candidate's chat history and system prompt, policy calls, the turn protocol and its retries |

1. `/seed_session` builds the episode and runs Tau until the candidate must
   act. It returns the domain policy, the tool schemas, and the first user turn
   as a WAV clip.
2. The agent sends the whole history (every user turn as an inline
   `input_audio` clip) to the policy model and submits the reply to
   `/agent_turn`. Tau executes tool calls, and the user simulator answers
   spoken replies. The server runs Tau until the candidate's next turn and
   returns either tool results or the next user clip.
3. When Tau ends the episode, `/verify` scores it with Tau's evaluator
   (`EvaluationType.ALL`, half duplex) and returns the reward.

Tau's own `Orchestrator` drives the episode. Its agent is a relay that returns
the message the Gym agent submitted, so termination (`user_stop`,
`agent_stop`, `max_steps`, `too_many_errors`, `timeout`) and communication
checks use Tau's code. A small orchestrator subclass excludes time spent waiting
for user TTS from the episode timeout. Andrew's user-message normalizer keeps
user tool calls and records any accompanying dropped text; no second mixed-turn
communication policy is introduced.

The candidate side follows the reference:

- The system prompt is the voice response contract plus Tau's domain policy.
- After each user clip comes the instruction "Listen to this user turn and
  respond using the audio only. Do not assume access to a transcript."
- A reply that breaks the turn protocol is retried twice, then replaced by a
  fixed fallback reply. The protocol rejects multiple tool calls, text mixed
  with a tool call, malformed or non-object tool arguments, unknown tools,
  textual tool calls, empty visible text, and stray `<think>` fragments.
- vLLM context-length rejections get the reference's single retry. If the
  history no longer fits, the agent says the call has reached its maximum
  length, which ends the episode.

The user simulator adds the reference's stop rule and STOP guard for tasks
with a pending agent write. On airline, 24 read-only tasks are scored with
curated NL assertions (`airline_assertions.py`), judged by an LLM.

## Configuration

`configs/tau_voice_simple.yaml` holds the reference settings:

| Setting | Value |
| --- | --- |
| User simulator | `openai/openai/openai/gpt-5.6-luna`, temperature 1 |
| NL-assertion judge | `openai/openai/openai/gpt-5.4-mini`, temperature 0 |
| TTS | `openai/openai/gpt-4o-mini-tts`, WAV output |
| Episode limits | 100 steps, 10 tool errors, 1200 s |

All three models use one OpenAI-compatible endpoint, read from
`TAU_VOICE_SIMPLE_LLM_API_BASE` and `TAU_VOICE_SIMPLE_LLM_API_KEY`.

The candidate's vLLM server must parse tool calls (`--enable-auto-tool-choice
--tool-call-parser qwen3_coder` for Nemotron) and allow one audio clip per user
turn (`--limit-mm-per-prompt '{"audio": 100}'`). The policy model server should
set `propagate_context_overflow_errors: true` so the agent sees vLLM's
context-length errors.

The tau2 wheel does not include Tau's `data/` tree. Set `tau2_data_dir` to a
prepared copy at revision `8ebb7499622fc2be9b9d510d6f7a7653461f4f29`. If the
directory is missing, the server clones that revision once and copies `data/`
into it.

Set `audio_archive_path` to collect every episode's synthesized user turns
(`<episode>/user_turn_NNNN.wav`, PCM16 mono, as returned by TTS) and its Tau
trajectory (`<episode>/simulation.json`, with inline audio removed) into one
uncompressed tar per run.

`artifact_root` (or `episodes/` beside the audio archive) also preserves each
simulation, episode identity and individual WAVs before judging. A judge outage
therefore leaves inference evidence available for CPU rescoring. Saved rollouts
include `simulation_path`; masked failures remain zero in metric denominators.

Transient TTS transport errors, empty responses,429/5xx and the gateway's known
database-overload401 retry the identical request. `tts_max_retries: -1` means
unlimited retries; ordinary authentication/budget failures still surface.
`tts_timeout_seconds: 0` disables the per-request timeout. TTS wait time does
not consume the episode timeout. Audio remains unaugmented.

Candidate `instructions` may replace the voice wrapper while preserving the
mandatory domain policy. Context recovery distinguishes exact token counts from
lower bounds; the latter defer output allowance to vLLM. A zero remaining output
allowance ends only that episode, with a recorded context failure.

## Metrics

| Metric | Meaning |
| --- | --- |
| `tau_voice_simple/<domain>/success` | Mean Tau reward x 100; the reference's "Tau-Voice Simple Success" |
| `tau_voice_simple/<domain>/premature_episodes` | Episodes that ended other than by `user_stop` or `agent_stop` |
| `tau_voice_simple/<domain>/termination/<reason>` | Episodes per termination reason |
| `tau_voice_simple/macro_success` | Mean of the domain success rates |

# Licensing information
Code: Apache 2.0
Data: Tau2 tasks and policies, MIT

Dependencies
- nemo_gym: Apache 2.0
- tau2: MIT
