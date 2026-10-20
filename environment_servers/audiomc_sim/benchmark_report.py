# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize completed concurrency experiments and the measured pipeline stages."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def distribution(values: list[float]) -> dict[str, float | int]:
    """Return descriptive sample statistics without assuming a normal distribution."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": max(values),
    }


def read_rows(path: Path) -> list[dict[str, object]]:
    """Read an optional newline-delimited metrics file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize_run(path: Path) -> dict[str, object]:
    """Combine finalized/partial traces, worker timings and observed backend metrics."""
    ledger = read_rows(path / "episode_results.jsonl")
    runtime_path = path / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    records = [json.loads(p.read_text()) for p in sorted((path / "artifacts/episodes").glob("*.json"))]
    partials = [json.loads(p.read_text()) for p in sorted((path / "artifacts/partial").glob("*.json"))]
    events = []
    for record in records:
        events.extend(record.get("invocations", []))
        events.extend(record.get("resource_invocations", []))
    for partial in partials:
        events.extend(partial.get("environment_trace", {}).get("invocations", []))
        events.extend(partial.get("resource_invocations", []))
    timings = [event for event in events if event.get("kind") == "timing"]
    roles = defaultdict(list)
    for event in timings:
        roles[event["role"]].append(event)
    role_stats = {}
    for role, calls in roles.items():
        elapsed = [c["elapsed_seconds"] for c in calls]
        outputs = [c["usage"]["output_tokens"] for c in calls if c.get("usage")]
        inputs = [c["usage"]["input_tokens"] for c in calls if c.get("usage")]
        cached = [((c.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens", 0) for c in calls]
        role_stats[role] = {
            "latency_seconds": distribution(elapsed),
            "total_wall_seconds": sum(elapsed),
            "failed_calls": sum(c.get("success") is False for c in calls),
            "output_tokens": distribution(outputs),
            "total_output_tokens": sum(outputs),
            "total_input_tokens": sum(inputs),
            "total_cached_input_tokens": sum(v or 0 for v in cached),
        }
    samples = read_rows(path / "system_metrics.jsonl")
    backend_samples = [s["vllm"] for s in samples if "vllm" in s]
    delta = {}
    if len(backend_samples) > 1:
        first, last = backend_samples[0], backend_samples[-1]
        for key in first.keys() & last.keys():
            if key.endswith(("_sum", "_count", "_total")):
                delta[key] = last[key] - first[key]
    native_means = {}
    for stem in [
        "request_queue_time_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "e2e_request_latency_seconds",
        "time_to_first_token_seconds",
        "request_generation_tokens",
    ]:
        count = delta.get(f"vllm:{stem}_count", 0)
        if count:
            native_means[stem] = delta[f"vllm:{stem}_sum"] / count
    gpu_samples = [s["gpu"][0] for s in samples if s.get("gpu")]
    outcomes = Counter((r.get("result") or {}).get("outcome", "infrastructure_failure") for r in ledger)
    failures = [
        t["turn_index"] + 1 for r in records for t in r["turn_evaluations"] if t["judgment"]["passed"] is False
    ]
    log = (path / "collection.log").read_text(errors="replace") if (path / "collection.log").exists() else ""
    rollout_rows = read_rows(path / "rollout_timings.jsonl")
    host_samples = read_rows(path.parent / "host_metrics.jsonl")
    if rollout_rows:
        begin = min(r["finished_at"] - r["elapsed_seconds"] for r in rollout_rows)
        finish = max(r["finished_at"] for r in rollout_rows)
        host_samples = [r for r in host_samples if begin <= r["timestamp"] <= finish]
    serving_cpu = [sum(p["cpu_percent"] for p in r["serving_processes"]) for r in host_samples]
    middle = {}
    ordered_completions = sorted(rollout_rows, key=lambda r: r["finished_at"])
    if len(ordered_completions) >= 60:
        window_seconds = ordered_completions[59]["finished_at"] - ordered_completions[19]["finished_at"]
        middle = {
            "completion_window": [20, 60],
            "seconds": window_seconds,
            "rollouts_per_hour": 40 * 3600 / max(window_seconds, 1e-9),
            "accepted_per_hour": sum(r["outcome"] == "accepted_correction" for r in ordered_completions[20:60])
            * 3600
            / max(window_seconds, 1e-9),
        }
    if middle:
        window_start = ordered_completions[19]["finished_at"]
        window_end = ordered_completions[59]["finished_at"]
        selected_samples = [sample for sample in samples if window_start <= sample["timestamp"] <= window_end]
        middle["gpu_utilization_percent"] = distribution(
            [sample["gpu"][0][0] for sample in selected_samples if sample.get("gpu")]
        )
        middle["vllm_running_requests"] = distribution(
            [sample.get("vllm", {}).get("vllm:num_requests_running", 0) for sample in selected_samples]
        )
        middle["vllm_waiting_requests"] = distribution(
            [sample.get("vllm", {}).get("vllm:num_requests_waiting", 0) for sample in selected_samples]
        )
        overlap = defaultdict(float)
        for event in timings:
            if "started_at" in event:
                begin = event["started_at"]
                end = begin + event["elapsed_seconds"]
                overlap[event["role"]] += max(0, min(end, window_end) - max(begin, window_start))
        middle["mean_inflight_by_role"] = {role: seconds / middle["seconds"] for role, seconds in overlap.items()}
    exclusion_causes = Counter()
    for row in ledger:
        if row.get("failure"):
            reason = row["failure"]["failure_reason"].lower()
            cause = (
                "assistant_incomplete_or_empty"
                if "assistant returned incomplete or empty" in reason
                else "timeout"
                if "timeout" in reason
                else "other_transport_or_protocol_error"
            )
            exclusion_causes[cause] += 1
    result = {
        "middle_window": middle,
        "exclusion_causes": dict(exclusion_causes),
        "serving_cpu_percent": distribution(serving_cpu),
        "directory": str(path),
        "complete": runtime_path.exists(),
        "completed_rollouts": len(ledger),
        "runtime": runtime,
        "outcomes": dict(outcomes),
        "roles": role_stats,
        "rollout_latency_seconds": distribution(
            [r["elapsed_seconds"] for r in read_rows(path / "rollout_timings.jsonl")]
        ),
        "first_failure_exchanges": dict(Counter(map(str, failures))),
        "assistant_requests": len(roles.get("assistant", [])),
        "total_call_wall_seconds": sum(e["elapsed_seconds"] for e in timings),
        "gpu_utilization_percent": distribution([s[0] for s in gpu_samples]),
        "gpu_memory_mib": distribution([s[1] for s in gpu_samples]),
        "gpu_power_watts": distribution([s[2] for s in gpu_samples]),
        "event_loop_lag_seconds": distribution([s["event_loop_lag_seconds"] for s in samples]),
        "collector_cpu_percent": distribution([s["collector_cpu_percent"] for s in samples]),
        "vllm_running_requests": distribution([s.get("vllm:num_requests_running", 0) for s in backend_samples]),
        "vllm_waiting_requests": distribution([s.get("vllm:num_requests_waiting", 0) for s in backend_samples]),
        "vllm_kv_cache_fraction": distribution([s.get("vllm:kv_cache_usage_perc", 0) for s in backend_samples]),
        "vllm_counter_deltas": delta,
        "vllm_request_means": native_means,
        "observed_http_errors": dict(
            Counter(str(e["http_status"]) for e in timings if e.get("success") is False and e.get("http_status"))
        ),
        "rate_limit_log_mentions": sum("429" in line or "rate limit" in line.lower() for line in log.splitlines()),
        "metric_scrape_errors": sum("vllm_scrape_error" in row for row in samples),
    }
    return result


def report(root: Path) -> dict[str, object]:
    """Publish a comparison table and detailed machine-readable measurements."""
    runs = {str(c): summarize_run(root / f"c{c}") for c in (4, 8, 16, 32) if (root / f"c{c}/manifest.json").exists()}
    result = {"runs": runs}
    (root / "comparison.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# AudioMC concurrency experiments",
        "",
        "Each level runs the same 100 task configurations with frozen prompts, seed 44, a six-exchange cap, and three teacher trials. Levels run sequentially against one local assistant with max-num-seqs 32 and audio-limit 50. Inference is stochastic; realized dialogue lengths and token counts can differ.",
        "",
        "| Concurrency | Completed | Accepted | Collection minutes | Rollouts/hour | Accepted/hour | Mean GPU % | Peak waiting requests | Infrastructure failures |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c, r in runs.items():
        rt = r["runtime"]
        lines.append(
            f"| {c} | {r['completed_rollouts']} | {r['outcomes'].get('accepted_correction', 0)} | {rt.get('collection_seconds', 0) / 60:.2f} | {rt.get('rollouts_per_hour', 0):.1f} | {rt.get('accepted_per_hour', 0):.1f} | {r['gpu_utilization_percent'].get('mean', 0):.1f} | {r['vllm_waiting_requests'].get('max', 0):g} | {r['outcomes'].get('infrastructure_failure', 0)} |"
        )
    for c, r in runs.items():
        lines += [
            "",
            f"## Concurrency {c}",
            "",
            "End-to-end role latency includes transport, backend waiting, generation, and visible retry time. Summed role wall time overlaps across conversations; it is not GPU compute time.",
            "",
            "| Role | Calls | Mean seconds | P95 seconds | Mean output tokens | Total wall seconds | Failed calls |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for role, stats in sorted(r["roles"].items()):
            latency = stats["latency_seconds"]
            lines.append(
                f"| {role} | {latency['count']} | {latency['mean']:.2f} | {latency['p95']:.2f} | {stats['output_tokens'].get('mean', 0):.1f} | {stats['total_wall_seconds']:.1f} | {stats['failed_calls']} |"
            )
        lines += [
            "",
            f"Native vLLM means: `{json.dumps(r['vllm_request_means'])}`.",
            "",
            f"Outcomes: `{json.dumps(r['outcomes'])}`. First-failure exchange counts: `{json.dumps(r['first_failure_exchanges'])}`.",
        ]
    lines += [
        "",
        "## Measurement notes",
        "",
        "A common middle-window diagnostic uses the interval between the 20th and 60th completions (40 completions), excluding initial fill and the final worker drain at every tested concurrency. Full-batch throughput remains reported separately. Collection throughput excludes graph startup and final archive export; export time is recorded separately in runtime.json. The collector writes one durable ledger entry per attempted rollout. Accepted/hour excludes ungradable, correction-exhausted, all-pass-at-cap and infrastructure outcomes. Acceptance is the existing automated grading outcome; these performance experiments do not independently adjudicate rubric validity.",
        "",
        "GPU utilization, queue depth and cache gauges are five-second samples. Request latency/queue/prefill/decode means use differences in vLLM's cumulative histogram sums/counts at run boundaries. Role timings include hidden provider latency and retries; they cannot by themselves prove a provider quota. Rate-limit log mentions are diagnostic text counts rather than a complete count of upstream HTTP retries.",
        "",
        "The exported rollout.jsonl in each run contains accepted corrections; episodes.jsonl and inspection.jsonl retain outcomes and available excluded traces. Audio is packed once at the end to keep repeated export from becoming the collection bottleneck.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = report(args.directory)
    print(
        json.dumps(
            {
                c: {"completed": r["completed_rollouts"], "complete": r["complete"], "outcomes": r["outcomes"]}
                for c, r in result["runs"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
