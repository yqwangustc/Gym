# AudioMC simulation

Collect multi-turn voice conversations that expose a failure of an evaluated assistant,
then save a teacher replacement that passes the same frozen rubrics. This is synthetic
training-data collection, not an RL environment with a success reward.

The Environment Server orchestrates Planner, User, Assistant, and Teacher. It reuses
Gym's Simple Agent and Model Servers. The Resources Server owns session state,
Rubric Builder, Critiquer, TTS, and final consistency checks. No new Agent Server is needed.
See [Resources](../../resources_servers/audiomc_sim/README.md) for routes and saved data.

## Default models

| Roles | Model | Connection |
| --- | --- | --- |
| Planner, Rubric Builder, Critiquer | `openai/openai/gpt-6-sol` | NVIDIA inference API |
| User, Teacher | `gcp/google/gemini-4-argon` | NVIDIA inference API |
| Assistant | `vllm_local` | Local audio vLLM, port 8000 |
| TTS | `openai/openai/gpt-4o-mini-tts` | NVIDIA inference API |

All three text/audio assistant connections use Gym's OpenAI-compatible chat adapter.
Reasoning parsing is enabled; only final assistant text enters the spoken dialogue.
The default assistant generation budget is 8192 tokens to accommodate its reasoning.
The speech service must produce mono PCM16 WAV. Streaming WAV headers are supported.
TTS is assumed faithful; acoustic cues are outside this version's evaluation scope.

## Configure each role

Edit or copy the complete example [backend.yaml](backend.yaml) and pass it with
`--backend-config environment_servers/audiomc_sim/backend.yaml`. The default file is
[configs/backends.yaml](configs/backends.yaml). Planner, Rubric Builder, Critiquer,
User, Teacher, Assistant and TTS each have their own settings dictionary.
For example, these entries put User on local vLLM and Teacher on the inference API:

```yaml
user:
  api_base: http://127.0.0.1:8001/v1
  model_name: vllm_local
teacher:
  api_base: ${oc.env:OPENAI_API_BASE,https://inference-api.nvidia.com/v1}
  model_name: gcp/google/gemini-4-argon
  api_key: ${oc.env:OPENAI_API_KEY}
```

Keep the other role entries from the supplied file. Moving one role requires only
editing its entry. All endpoints must expose the supported OpenAI-compatible API;
the Assistant endpoint must accept audio. TTS uses `api_base + /audio/speech`.
The collector connects to model servers that are already running.

| Setting | Meaning |
| --- | --- |
| `api_base` | Local or remote API base URL; supports `${oc.env:VARIABLE}` and `${oc.env:VARIABLE,default}`. |
| `model_name` | Model identifier sent to that endpoint. |
| `api_key` | Optional authentication value, default `unused`. Use environment references for secrets; each role can use a different credential variable. |
| `uses_reasoning_parser` | Optional text-model adapter setting, default `true`; configure for the served model's reasoning/final response format. |
| `checkpoint` | Optional Assistant provenance identifier; does not load or serve a model. |
| `voices` | Nonempty TTS voice list; one entry is sampled per conversation. Examples include all 13 voices verified through the NVIDIA endpoint. |

TTS selects one voice uniformly from the configured entries and keeps it for all
user turns in that conversation:

```yaml
tts:
  api_base: ${oc.env:OPENAI_API_BASE,https://inference-api.nvidia.com/v1}
  model_name: openai/openai/gpt-4o-mini-tts
  api_key: ${oc.env:OPENAI_API_KEY}
  voices: [alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse, marin, cedar]
```

Voice sampling happens when preparing tasks, using a separate seeded random generator.
The result is saved in each task as `task_input.tts_voice` in `tasks.jsonl`. Every turn,
retry and replay of that task reuses its saved voice, independently of rollout IDs
and concurrent execution. Changing the pool does not change persona/topic sampling.
The selected voice is stored in episode metadata and each audio reference. Voice IDs must be supported
by your configured TTS endpoint. Legacy `voice: alloy` remains supported as a one-entry
pool; specify either `voices` or `voice`. Omitting both keeps the default `alloy`.
The provided YAML examples enable all 13 verified voices; existing frozen runs retain
their original configuration when resumed.

`--env-file` loads literal environment assignments before resolving YAML. Existing
process environment variables take precedence. The default remote-role entries use
`OPENAI_API_BASE` if set, otherwise the NVIDIA endpoint. An entirely local setup with
omitted/`unused` keys does not require `OPENAI_API_KEY`.

The collector validates all seven role entries before starting a new graph. Identical
helper settings share one Gym model wrapper. Different role settings get independent
wrappers; each wrapper consumes a localhost port, not another copy of model weights.

A new collection saves effective settings in `backend_config.yaml`, alongside
`manifest.json` and `gym_config.yaml`. Credential environment references remain
unresolved in saved files. Use environment references rather than literal API keys.
Resuming uses the saved graph; use a new output directory to change model routing.
All model endpoints, identifiers, credentials and the optional Assistant checkpoint
are configured in the YAML. For provenance, set `assistant.checkpoint` there;
the collector does not accept separate CLI backend overrides.

## Run on one host

From the repository root, install the development environment with `uv sync --extra dev`.
Start the user's audio-capable vLLM server separately. The collector connects to an
already running server; it does not install or modify the checkpoint or serving code.

```bash
environment_servers/audiomc_sim/start-assistant.sh
```

This launcher uses the existing patched vLLM installation and enables 50 audio clips
per prompt. The original launcher defaults to one clip and cannot replay multi-turn
audio history. Set `AUDIO_LIMIT` at least as high as your collection `--max-turns`.

In another terminal, collect five accepted corrections:

```bash
.venv/bin/python -m environment_servers.audiomc_sim.collect \
  --personas /home/yongqiangw/work/codex_tasks/audiomc-simulation/resources/persona/nemotron-usa.jsonl \
  --output /home/yongqiangw/work/codex_tasks/audiomc-simulation/initial_rollouts/v0 \
  --env-file /home/yongqiangw/.env \
  --backend-config environment_servers/audiomc_sim/backend.yaml
```

The default is at most 20 rollouts, six exchanges per rollout, three teacher attempts.
Change endpoints/models per role in `--backend-config`. `--env-file` accepts literal
exported assignments; it never executes shell code. Only locally hosted model servers
consume GPU memory. The collector runs native Gym HTTP wrappers on consecutive localhost
ports starting at 18240; the default shared configuration uses six ports (18240–18245),
and six distinct text backends need nine ports. Change `--base-port` if necessary.
HTTP calls use Gym's pooled aiohttp transport and ServerClient.

Repeating the same command resumes terminal episodes from `episode_results.jsonl` and
uses the frozen `tasks.jsonl`/`manifest.json`. Existing manifests take precedence over
model/sampling CLI flags: use a new output directory to change those settings. Interrupted
nonterminal attempts resume with a new physical attempt ID, preserving earlier audio.
Completed artifacts whose HTTP replies were lost are recovered without rerunning inference.
Role prompts are snapshotted into the output directory when starting a new collection. Infrastructure failures are retained and counted,
and are never substituted for accepted corrections. Transport replay of `/run` is disabled.

The output includes:

- `rollout.jsonl`: accepted training examples only, with every turn's rubrics and grades.
- `audio.tar` or `audio-00000-of-NNNNN.tar` shards: uncompressed WAV archives,
  referenced by relative archive/member paths. `--max-audio-files` limits each
  archive to 50,000 WAV files by default, including inspection audio. All JSONL
  references are updated to the correct shard; a conversation may cross a shard boundary.
- `episodes.jsonl`: complete finalized traces, including planner/user/teacher outputs.
- `inspection.jsonl`: excluded traces and infrastructure failures, with available partial state.
- `summary.json`: rollout/outcome counters, accepted count, audio count.
- `tasks.jsonl`, `manifest.json`, `gym_config.yaml`: sampled inputs and reproducible wiring.
- `artifacts/`: original per-episode records and immutable per-turn audio archives.

The collector exits unsuccessfully if its rollout budget does not yield the requested
accepted count. Inspect the saved exclusions; never loosen acceptance checks to meet a quota.

## Standard Gym wiring

`configs/audiomc-sim.yaml` is a low-level Gym graph example for Hydra, distinct from
the per-role standalone `backends.yaml`. Set `OPENAI_API_KEY` and
`AUDIOMC_OUTPUT_DIR`, compose this config with your Gym launch settings, and dispatch
materialized tasks to the Environment Server `/run`. Its taskset route is
`audiomc_sim:generated`. The standalone collector exports `gym_config.yaml` with explicit
localhost addresses; it is also useful for inspecting the exact resolved graph.

Prepare tasks without inference:

```bash
.venv/bin/python -m resources_servers.audiomc_sim.prepare \
  --personas /path/to/nemotron-usa.jsonl --output /path/to/tasks.jsonl --count 20
```

Preparation scans the complete local persona corpus once, keeps compact byte offsets,
and samples UUID/general persona without loading the corpus into memory. Sampling is
seeded. LLM sampling remains stochastic; the seed is not a promise of identical inference.
Evaluation axes and topics are snapshotted under Resources `data/`. Hashes, persona UUID,
and source revision are recorded with each task. Prompts load once from Markdown files
with strict Jinja placeholders. Environment owns Planner/User/Teacher/Assistant prompts;
Resources owns Rubric Builder/Critiquer prompts. Restart servers after changing prompts.

## Checks

```bash
.venv/bin/pytest --import-mode=importlib environment_servers/audiomc_sim resources_servers/audiomc_sim
.venv/bin/ruff check environment_servers/audiomc_sim resources_servers/audiomc_sim
```

Real audio/model rollouts are also required for integration validation. Tests do not
establish that generated rubrics are semantically valid; inspect collection examples
and calibrate rubric/critique quality before scaling collection.

## Concurrent collection and throughput experiments

Use `--concurrency N` to run up to N independent conversations. The collector configures
the Environment Server with the same admission limit. Each conversation retains its own
Resources and Assistant sessions; turns within a conversation stay sequential.

`--fixed-rollouts --max-rollouts 100` runs exactly 100 task attempts regardless of how
many corrections are accepted. Without `--fixed-rollouts`, accepted-quota collection
stops scheduling new work once its quota is met and drains already running conversations;
this can exceed the accepted target by at most the in-flight worker count minus one.

Ledger and rollout-timing writes are serialized. Archive export runs once after collection
by default. Set `--export-every N` for checkpoints after every N completed attempts, or
`--export-every-min N` for time-based checkpoints independent of conversation completions.
Both triggers can be enabled together and share the same export lock. Checkpoint export
runs in a background thread; result publication waits for it, while in-flight model calls
can continue. Repeatedly exporting large archives adds work and can lower throughput.
Original per-turn archives and episode files remain durable throughout collection.

For a cluster job with a four-hour wall-time limit, add the following to your collection
command to start exporting completed trajectories after 230 minutes:

```bash
--export-every-min 230
```

The first interval starts when collection begins, after task preparation and Gym startup.
Subsequent intervals start after the previous timed export finishes; resuming the process
restarts the timer. Each checkpoint rebuilds the full export accumulated so far, including
the configured audio shards and inspection data. A final export still runs on normal
completion, and a short run does not wait for its next timed checkpoint before exiting.

This option checkpoints data; it does not impose a wall-time stop. The ten-minute margin
assumes preparation/startup and export fit within the remaining job time. Trajectories
finished after the last checkpoint remain in raw artifacts until a later export. Resume
with the same output directory to recover durable completed episodes and retry interrupted
work. Keep the output directory on storage that survives job termination.

`--monitor` samples local vLLM Prometheus metrics, GPU utilization/memory/power, collector
CPU/RSS, active conversations, and event-loop lag every five seconds. Role timing and
available token usage are retained in each episode's invocation records, including
available partial-failure traces. `rollout_timings.jsonl` records completed-attempt latency;
`runtime.json` separates collection duration from final export duration.

Run the four sequential 100-task experiments with one shared frozen task/prompt snapshot:

```bash
.venv/bin/python -m environment_servers.audiomc_sim.benchmark \
  --output /path/to/concurrency-experiments \
  --personas /path/to/nemotron-usa.jsonl \
  --axes /path/to/eval_axis.yaml \
  --topics /path/to/expanded_topics.yaml \
  --env-file /path/to/credentials.env \
  --backend-config /path/to/backends.yaml
```

The suite uses concurrency 4, 8, 16, and 32, seed 44, six exchanges, and three teacher
trials. It freezes the selected backend YAML for all four levels (defaulting to the NVIDIA
API and local Assistant configuration shown above). It skips completed
runs on resume; an interrupted run's runtime statistics describe only the resumed process,
so flag resumed measurements before comparing whole-run throughput.

```bash
.venv/bin/python -m environment_servers.audiomc_sim.benchmark_report /path/to/concurrency-experiments
```

Use accepted examples per hour alongside failure counts, request latency, generated tokens,
GPU utilization, and queue/cache measurements. Concurrent conversations spend time in API
stages; their count differs from the assistant's active vLLM sequence count. Current vLLM
scheduling capacity is 32 sequences. Provider rate limits must be diagnosed from observed
responses; increased API latency alone does not establish a quota.
