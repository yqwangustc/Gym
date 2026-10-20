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

The Environment Server stops at the first failed original answer. The Teacher sees
that answer and critique separately from the conversation prefix. Each teacher candidate
is graded with the same rubrics; the first passing replacement is accepted. Exhausted
teacher attempts and all-pass turn-limit runs are inspection data, not training examples.

## Training projection and audio

Accepted examples use `SavedTrainingExample` in `contracts.py`, retaining all turn-level
rubrics, original assistant responses, and grades. `input_prefix.turns` contains annotated
transcript history; user turns acquire structured `audio` references at export.
`responses_create_params` is the audio-only model-visible prefix: user content is empty,
and `metadata.audio_refs` is a JSON string containing one archive/member group per user
turn, matching the existing Gym tar-audio adapter. Do not train by concatenating the
annotated transcript history. `supervised_target` is the accepted teacher replacement;
prior assistant text is context, and rejected answers/critique metadata are not SFT targets.

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
Original per-turn artifacts remain available under `artifacts/` for live replay
and recovery.
