# AudioMC simulation resources

This server supports synthetic collection implemented by the
[AudioMC Environment Server](../../environment_servers/audiomc_sim/README.md).
It is separate from the existing `audiomc` benchmark.

## Internal routes

| Route | Purpose |
| --- | --- |
| `/seed_session` | Bind episode/task identity and supplied persona/axis/topic configuration. |
| `/build_rubrics` | Freeze requirements before seeing the current assistant response. |
| `/synthesize` | Synthesize the frozen user utterance, cache it, and store immutable WAV tar data. |
| `/grade` | Grade a candidate against the frozen prefix and rubric set. |
| `/checkpoint` | Retain environment diagnostics for partial-failure inspection during cleanup. |
| `/verify` | Validate outcomes against stored evidence and save the completed episode. |
| `/close_session` | Release state and save available unfinished diagnostics. |
| `/aggregate_metrics` | Count collection outcomes without assigning RL reward. |

These are harness routes, not tools exposed to the evaluated assistant. Run one Resources
worker per instance; all state is process-local, session-locked, and checked against
signed session cookies. Seed/input digests detect conflicting retries. Closed session
IDs remain tombstoned for the configured TTL. Durable records are local to the configured
`artifact_root`; colocate Resources and the assistant model adapter or mount shared storage.

Rubric Builder returns `{rubrics: [...]}` with contiguous zero-based `index` fields.
The harness creates IDs from episode/turn/index. Critiquer returns `{grades: [...], passed: ...}`.
Every rubric must be graded exactly once, in order. Aggregate verdicts must agree with
item verdicts. Unsupported rubric evidence, malformed JSON, and contradictory grades
cannot become accepted corrections. Empty rubric lists or null judgments are ungradable.

By default, the Environment Server stops after repairing the first failed original answer.
Each Teacher candidate is graded with the same frozen rubrics; the first passing replacement
is accepted. Runs without any verified correction remain inspection data.

Teacher returns a plain-text spoken answer. Its API's separate native reasoning
is preserved by the Gym model wrapper and saved as `Candidate.thinking` for every
Teacher trial, including rejected or ungradable candidates. It is not a generated
explanation of an already accepted answer and is not requested in JSON.

Teacher receives only assistant/developer instructions and visible conversation
history through the latest user utterance. Private evaluation configuration,
rubrics, persona, current failed response, prior trials, and critique feedback are
excluded. Every retry starts fresh with the same context. Resources grades only
the answer against the frozen rubrics. Only an accepted answer enters later
helper/student dialogue; reasoning is never spoken or sent for grading. Student
replies have no stored reasoning. An empty/incomplete answer or missing native
reasoning is a generation failure. Preflight checks native-reasoning support for
Teacher using its configured `reasoning_effort` (default examples: `minimal`).

Set `simulation_tasks.keep_generation_until_max_turn_reached: true` in Slurm task
settings, or pass `--keep-generation-until-max-turn-reached` to the collector, to
continue with each accepted teacher reply through `max_turns`. Subsequent planning,
user generation, rubric building and assistant requests see the accepted reply;
the rejected original response is retained only for inspection. Teacher retry
budgets apply independently to each turn (five attempts by default). A trajectory
with any verified Teacher rewrite is retained even if a later repair is exhausted,
ungradable, or interrupted by a generation error or cancellation. Training keeps
only the contiguous accepted conversation prefix; the unresolved exchange stays
in the inspection record. Stop reasons and training eligibility are independent.
Hard process termination can still prevent finalization.

## Training projection and audio

Accepted examples use schema `1.7` (`SavedTrainingExample` in `contracts.py`).
`conversation` holds the accepted dialogue prefix, user audio references, and each assistant
reply's `generated_by_teacher` flag and optional `thinking` annotation.
`turn_evaluations` retains all rubrics, original `policy_response` and grade, and
verified `accepted_response`. `teacher_attempts` retains all trials with their
`turn_index`; `rejected_teacher_attempts` contains rejected candidates for the last
teacher correction's turn. Instructions live in
`metadata.task.assistant_instructions` and `metadata.task.developer_instructions`.

Training consumers reconstruct one example per Teacher-written assistant reply.
Use only preceding conversation messages, prepend the saved instructions, and use
that reply's text as the target. For audio-only requests, user content is empty;
build `metadata.audio_refs` as a JSON string with one archive/member group per
preceding user turn, matching Gym's tar-audio adapter. User transcripts remain
inspection annotations and must not enter audio-only model inputs. Keep any
`thinking` annotation separate from spoken answer text. Rejected candidates and
critique feedback are not training targets or conversation context.

A conversation may provide several training examples while counting once toward
the collection quota.

Task preparation samples one voice per task from `tts.voices` and saves it as
`task_input.tts_voice`. Resources uses that saved voice across all turns and retries.
Old tasks without a voice use a deterministic task-level fallback from `tts_voices`.
The chosen voice is retained in episode metadata and `AudioRef.voice`. Legacy
`tts_voice` still defines a fixed voice when no pool is supplied.

Per-turn audio archives are immutable while collection is running, so the adapter's
archive identity/offset cache remains valid. Export verifies all member hashes and
stores at most 50,000 unique WAV members in each archive by default. A single archive
is named `audio.tar`; larger exports use zero-based shard names such as
`audio-00000-of-00010.tar` through `audio-00009-of-00010.tar` for ten shards.
Excluded episodes' audio is included for human inspection and counts toward the limit.

Member paths remain `<conversation-id>/user_turn_XXXX.wav`, sorted by conversation
and turn. A conversation may span shards when it crosses the file-count limit.
All audio references in exported training examples, episode traces, inspection data
and `metadata.audio_refs` point to the appropriate shard. `summary.json` records
the archive names and configured limit. Change the limit during collection with
`--max-audio-files`, or re-export existing artifacts without making model calls:

```bash
.venv/bin/python -m resources_servers.audiomc_sim.export /path/to/collection \
  --max-audio-files 50000
```

Export is repeatable. It stages and hash-checks all shards before replacing any
published archive, replaces each output file atomically, and removes obsolete
export archives after publishing the new references. Publication of the complete
file set is not a single atomic operation; read exports after the export finishes.
Before publication, existing output files are hard-linked into `.export-backup/`.
A failed export restores them; startup also restores an interrupted publication.
Do not remove that directory during collection or recovery.

After a successful periodic, count-triggered, or final export, completed trajectories
with exactly `max_turns` evaluated exchanges have their raw episode/partial records,
per-turn audio archives, and matching request file removed. Shorter trajectories,
unfinished work, and shared prompt copies remain. Pass `--keep-raw-artifacts` to
collection or export to retain all raw files instead.

Compacted trajectories remain in `episodes.jsonl` and the exported audio archives;
subsequent exports merge them with new raw records, including when changing shard
size. Keep these files for resume/re-export. Training consumers only need
`rollout.jsonl` and its referenced audio archives.

TTS retries use capped exponential backoff with additive jitter and honor provider
`Retry-After` cooldowns. `tts_retry_backoff` configures the delay policy;
`tts_request_attempts` and `tts_request_timeout_seconds` bound individual calls.
The transport does not add hidden retries; canceled requests stop immediately.
