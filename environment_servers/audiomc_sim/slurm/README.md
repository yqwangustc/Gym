# Slurm collection

This single-node launcher uses a task YAML and per-role backend YAML. There is no
concurrency calibration: set `simulation_tasks.concurrency` before submission.
`tasks.ord.yaml` and `backend.ord.yaml` are deployment examples; cluster-specific
paths, partition and account appear only in those example files.

From the Gym checkout on a login node, using its installed Python environment:

```bash
.venv/bin/python -m environment_servers.audiomc_sim.slurm.submit \
  --settings environment_servers/audiomc_sim/slurm/tasks.ord.yaml
```

Submission runs preflight automatically and aborts before `sbatch` if it fails.
No separate preflight command is needed for normal submission.

Optionally, to check configuration and API accessibility without submitting a job:

```bash
.venv/bin/python -m environment_servers.audiomc_sim.slurm.preflight \
  --settings environment_servers/audiomc_sim/slurm/tasks.ord.yaml
```

Copy the example task/backend files for another cluster. Any filename works,
including `tasks.settings.yaml`. No `ORD_*` variables are used.

## Task settings

Paths are relative to the task file, with `~` and OmegaConf environment references
such as `${oc.env:OPENAI_API_BASE}` supported. `environment` is loaded before other
settings resolve. It accepts literal `KEY=value` or `export KEY=value` assignments;
existing process variables take precedence. Shell expressions are not executed.
Use environment references for credentials in backend YAML rather than literal keys.

- `repos.nemo_gym`: checkout mounted in the container and used as working directory.
- `container`: Slurm/Pyxis `.sqsh` image path or registry image reference.
- `mc_resources`: `persona`, `eval_axis`, `topics`, and `assistant_system_prompt` paths.
- `environment`: credentials/environment file; `null` uses only inherited variables.
- `backend`: per-role backend YAML path.
- `output`: directory for raw records, logs, frozen configuration and portable exports.
- `simulation_tasks`: `target_trajectory` counts accepted teacher-corrected SFT
  examples. `max_rollouts` bounds all attempts, defaulting to twice the target.
  `seed`, `max_turns`, `max_teacher_trials`, and `concurrency` control generation.
  `keep_generation_until_max_turn_reached` defaults to `false`; set it to `true`
  to continue after a failed assistant reply using a rubric-passing teacher
  replacement. Teacher trials reset for each turn (default: five). If a later
  repair fails, keep the accepted prefix when it contains a successful Teacher
  rewrite; exclude the unresolved exchange from training. Stop-reason counters
  remain independent of accepted trajectory counts.
  `export_every_min` defaults to 230; `export_reserve_min` defaults to 10;
  `max_audio_files` defaults to 50,000. `base_port` sets the internal Gym port range.
  `cleanup_finished_trajectories` defaults to `true`: after each successful export,
  completed trajectories that reached `max_turns` have their raw artifacts and
  matching request removed. Set it to `false` to retain these files.
- `slurm`: partition/account (optional), allocation `time`, `gpus`,
  `cpus_per_task`, `exclusive`, `job_name`, `container_mounts`, and container `python`.
  The launcher requests one node and one task.

The container interpreter must have Gym and its dependencies installed, alongside
its serving dependencies. The launcher does not replace the image's patched vLLM.
Prepare a compatible image or a compatible mounted environment and set
`slurm.python` to its interpreter. The ORD example uses the checkout's mounted `.venv` for Gym and the image's
`python3` for vLLM. The `.venv` interpreter and libraries must be compatible with
the image and its symlink targets must be mounted. Host Python must also have Gym dependencies for
configuration and preflight. Pyxis/Enroot container flags must be available.

The launcher automatically mounts the checkout, configuration directories, resource
parents and output at their absolute paths. Add checkpoints, server source trees
and other dependencies to `slurm.container_mounts`; the ORD example mounts `/lustre`
and `/home`. Keep outputs and checkpoints on storage visible to the allocated node.

## Backend settings

Each role retains `api_base`, `model_name`, `api_key`, and its existing options.
Add `serving` to a role to have the job manage a local vLLM process. With no
`serving`, the endpoint must already be available and is checked during preflight,
including when the Assistant is API-hosted. Moving a role requires only YAML edits.
The default and ORD examples use `nvidia/zai-org/glm-5.2` for User and Teacher.
Their backend settings can be configured independently; identical settings share
a model wrapper.

```yaml
user:
  api_base: http://127.0.0.1:8001/v1
  model_name: user_local
  serving:
    entrypoint: vllm.entrypoints.openai.api_server
    checkpoint: /models/user-model
    cuda_visible_devices: [0, 1, 2, 3]
    tp: 4
    dp: 1
    ep: 1
    max_num_seqs_per_dp_rank: 32
    prefix_cache: true
teacher:
  api_base: http://127.0.0.1:8001/v1
  model_name: user_local
```

Roles sharing a local endpoint launch only one service. Duplicate `serving`
definitions must agree. GPU selections are indices in the allocation's visible GPU
list, not physical IDs. For example, inherited `CUDA_VISIBLE_DEVICES=4,5,6,7`
and a selection of `[0, 2]` give the child `CUDA_VISIBLE_DEVICES=4,6`. GPU UUIDs
and MIG identifiers retain their original form. Allocate disjoint devices when
running multiple models. Omit the list to preserve all GPUs exposed by Slurm.

Preflight checks selection bounds and requires the selected GPU count to equal
`tp * dp` against the planned `slurm.gpus`. Inside the container, startup validates
the actual allocation again and translates explicit selections through the inherited
`CUDA_VISIBLE_DEVICES`. If that variable is absent, CUDA device discovery runs in
`serving.python`. An allocation/count mismatch aborts before launching vLLM.

`serving.entrypoint` is a Python script or module; `python` selects its interpreter.
Explicit relative checkpoint/entrypoint paths (`./...`, `../...`) resolve against
the backend file. Hugging Face model IDs remain unchanged. `tp`, `dp`, and `ep`
control parallelism; vLLM derives EP group size from TP times DP, so `ep` is either
1 (disabled) or `tp * dp`. `max_num_seqs_per_dp_rank`, `prefix_cache`, `audio_limit`,
`max_model_len`, `gpu_memory_utilization`, and `reasoning_parser` map to serving flags.
The Assistant checkpoint recorded in exported provenance comes from `serving.checkpoint`.

`interface: nemo_audio` selects the existing audio wrapper's `--audio-limit` and
`--` argument separator. Standard `interface: vllm` uses
`--limit-mm-per-prompt`. `extra_args` precede the separator and `engine_args` follow
it. `startup_timeout_seconds` defaults to 1200. `backend.ord.yaml` reproduces the
previous TP1/DP8/EP8, 32 sequences per rank, prefix-cache and audio-limit-50 setup.

## Startup, stopping and output

1. Login-node preflight loads task settings and probes each distinct external text
   or speech model with an actual request. Any failure prevents `sbatch`.
2. Before calling `sbatch`, `submit.py` writes `launch_settings.yaml` with absolute
   host paths (including the environment file, so container HOME does not change
   its location) and generates `run.sbatch` in the output directory. Slurm executes
   this script, which directly invokes `srun` with the configured container; no
   second host Python invocation is needed. Inside the container,
   `incontainer_run.py` repeats API preflight and starts all configured local services.
3. Every managed port must be unoccupied, even if an existing service is healthy
   and advertises the same model name. Occupied ports abort startup with instructions
   to stop the service or change the port; existing processes are never stopped.
   Use `serving: null` to intentionally use an external service, which is checked
   during API preflight. New local services must pass `/health` and expose their
   configured names in `/v1/models`. The job shuts down only processes it started.
   Startup failure prevents collection.
4. The normal collector starts at the configured concurrency. It stops scheduling
   once the accepted target is reached, allowing in-flight conversations to finish.
   The target can therefore be exceeded. Exhausting `max_rollouts` below the target
   is an error. No warmup, probing, or automatic concurrency selection is performed.
   By default, a conversation ends at its first assistant failure and accepted
   teacher correction. With `keep_generation_until_max_turn_reached: true`, the
   corrected reply enters the conversation history and generation continues.
   Trajectories with no assistant failures remain excluded as before.
5. Collection is interrupted at the actual Slurm end time minus the export reserve.
   Completed records are exported; unfinished conversations are counted separately.
   Cancellation finalizes earlier verified rewrites when graceful shutdown permits;
   the unresolved exchange never becomes a training target.
   Startup and task preparation consume allocation time. SIGTERM/SIGINT also trigger
   final export. Cancellation waits for any in-progress periodic export to finish
   before starting the final export, including its worker thread. A hard Slurm kill
   cannot guarantee saving, so retain the reserve.

Output uses the ordinary collector layout: `rollout.jsonl`, `audio.tar` or bounded
shards, `episodes.jsonl`, `inspection.jsonl`, `summary.json`, raw `artifacts/`,
`episode_results.jsonl`, invocation and system metrics, frozen `tasks.jsonl`,
`manifest.json`, prompt copies, `task_settings.json`, and `backend_config.yaml`.
Additional files are `services/vllm-N.log`, `slurm-JOBID.log`, and
`slurm_runtime.json` with allocation deadline and interruption counts. API-key
interpolations stay unresolved in backend snapshots.

Reuse the same output with the same task/backend settings to resume; completed
attempts are recovered from durable records, including compacted `episodes.jsonl`
and exported audio. Keep these files for resume or re-export after cleanup.
Cleanup runs at the next periodic, count-triggered, or final export; shared prompts,
shorter completed trajectories, and unfinished traces are retained. Older task
snapshots missing newly defaulted settings can resume using those defaults.
Use a new output directory when changing
models, prompts or generation settings. The earlier ORD calibration artifacts are
historical experiment outputs and are not part of this launcher.

## Continuing after teacher corrections

```yaml
simulation_tasks:
  target_trajectory: 12000
  max_turns: 6
  max_teacher_trials: 5
  keep_generation_until_max_turn_reached: true
  concurrency: 128
```

The target counts accepted conversations, not individual teacher replacements.
A conversation is saved only after all allowed turns have accepted replies and
at least one required a teacher correction. A failed or ungradable repair excludes
all its earlier corrections from training output and retains the trace for inspection.

Saved `conversation` includes every accepted assistant reply with a
`generated_by_teacher` boolean. Each `turn_evaluations` entry retains the original
`policy_response` and `judgment`, plus the `accepted_response` actually used in
history and its `generated_by_teacher` flag. `teacher_attempts` retains all trials,
with a `turn_index` for each. `rejected_teacher_attempts` contains only rejected
candidates for the last teacher correction's turn.

Saved example schema version is `1.7`. Training consumers reconstruct one example
for each Teacher-written reply from preceding `conversation` messages and
`metadata.task.assistant_instructions` / `developer_instructions`. User messages
supply audio references; their transcripts are inspection annotations, not audio-only
model inputs. `turn_evaluations` supplies rubrics and grades.

## Teacher correction and native reasoning

For each failed student turn, Teacher receives only assistant/developer
instructions and the visible conversation through the latest user utterance.
It returns the spoken answer as plain text; its native API reasoning is captured
separately in `thinking`. No JSON wrapper or later rationale-generation step is
used. Rubrics, evaluation configuration, persona, current failed response, prior
Teacher trials, and critique feedback are excluded from Teacher input.
Critiquer grades only the answer against the frozen rubrics. Every retry uses
identical visible context in a fresh request until the first pass or the trial limit.

Set `teacher.reasoning_effort: minimal` in backend YAML (as in the default and ORD
examples). The setting is forwarded to the endpoint and included in model-wrapper
sharing decisions; it does not alter other roles sharing the model's weights.
API preflight separately probes Teacher with its configured effort and requires a
nonempty answer plus exposed native reasoning. HTTP 200 alone is insufficient.
A managed local Teacher is probed after startup and before collection. Models
that return no reasoning, or configurations disabling its parser, fail startup.

Every Teacher trial retains answer, native reasoning, and grade. Accepted answer
text enters later dialogue; reasoning never enters spoken text or grading requests.
Empty/incomplete answers or missing reasoning are generation failures. Use a new
collection output directory to adopt these templates and backend settings.
