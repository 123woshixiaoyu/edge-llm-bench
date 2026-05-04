#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.vision_router import VisionRouter


LOCAL_TASKS = {"detect", "classify"}
REMOTE_TASKS = {"vqa", "scene_description"}


@dataclass(frozen=True)
class Scenario:
    name: str
    task_type: str
    privacy: str
    quality: str
    expected_behavior: str
    latency_budget_ms: int = 3000
    local_available: bool = True
    remote_available: bool = True
    queue_depth_override: int | None = None
    simulated_remote_latency_ms: float = 1.0


class ReliabilityHarness:
    def __init__(
        self,
        *,
        image_path: Path,
        engine_path: Path,
        model_dir: Path,
        local_queue_limit: int,
        timeout_s: float,
    ):
        self.image_path = image_path
        self.local_queue_limit = local_queue_limit
        self.timeout_ms = timeout_s * 1000
        self.router = VisionRouter(
            model_dir=model_dir,
            local_cv_backend="yolo_tensorrt_fp16",
            yolo_engine_path=engine_path,
            fallback_to_mobilenet=False,
            remote_backend=None,
        )
        self._counter_lock = threading.Lock()
        self._local_lock = threading.Lock()
        self._local_inflight = 0
        self._remote_inflight = 0

    def snapshot(self) -> tuple[int, int]:
        with self._counter_lock:
            return self._local_inflight, self._remote_inflight

    def decide(self, scenario: Scenario) -> tuple[str, bool, str, list[str], int, int, int]:
        local_inflight, remote_inflight = self.snapshot()
        queue_depth = scenario.queue_depth_override
        if queue_depth is None:
            queue_depth = local_inflight
        reasons: list[str] = []
        fallback_used = False

        if scenario.latency_budget_ms < 100:
            return (
                "reject",
                False,
                "latency budget is too low for reliable routing",
                [f"latency_budget_ms={scenario.latency_budget_ms} is below the v0.7 minimum"],
                queue_depth,
                local_inflight,
                remote_inflight,
            )

        if scenario.privacy == "local_only" and scenario.task_type in REMOTE_TASKS:
            return (
                "reject",
                False,
                "privacy local_only blocks remote semantic vision",
                [
                    "privacy=local_only prevents sending image to remote VLM",
                    "local CV cannot satisfy semantic VQA/scene_description",
                ],
                queue_depth,
                local_inflight,
                remote_inflight,
            )

        if scenario.task_type in LOCAL_TASKS and scenario.quality != "high":
            if not scenario.local_available:
                if scenario.privacy == "local_only":
                    return (
                        "reject",
                        False,
                        "local backend unavailable for local_only task",
                        ["privacy=local_only but local CV backend is unavailable"],
                        queue_depth,
                        local_inflight,
                        remote_inflight,
                    )
                if scenario.remote_available:
                    return (
                        "remote",
                        True,
                        "",
                        ["local CV backend unavailable; privacy allows remote fallback"],
                        queue_depth,
                        local_inflight,
                        remote_inflight,
                    )
                return (
                    "reject",
                    False,
                    "both local CV and remote VLM are unavailable",
                    ["local CV unavailable and remote VLM unavailable"],
                    queue_depth,
                    local_inflight,
                    remote_inflight,
                )
            if queue_depth >= self.local_queue_limit:
                if scenario.privacy == "local_only":
                    return (
                        "reject",
                        False,
                        "local queue overloaded and privacy prevents remote fallback",
                        [f"local_queue_depth={queue_depth} reached limit={self.local_queue_limit}"],
                        queue_depth,
                        local_inflight,
                        remote_inflight,
                    )
                if scenario.remote_available:
                    return (
                        "remote",
                        True,
                        "",
                        [
                            f"local_queue_depth={queue_depth} reached limit={self.local_queue_limit}",
                            "privacy allows remote fallback",
                        ],
                        queue_depth,
                        local_inflight,
                        remote_inflight,
                    )
                return (
                    "reject",
                    False,
                    "local queue overloaded and remote unavailable",
                    [
                        f"local_queue_depth={queue_depth} reached limit={self.local_queue_limit}",
                        "remote VLM backend unavailable",
                    ],
                    queue_depth,
                    local_inflight,
                    remote_inflight,
                )
            return (
                "local",
                False,
                "",
                [
                    "task is covered by YOLO TensorRT local CV",
                    f"local_queue_depth={queue_depth} is below limit={self.local_queue_limit}",
                ],
                queue_depth,
                local_inflight,
                remote_inflight,
            )

        if scenario.privacy != "local_only" and scenario.remote_available:
            return (
                "remote",
                False,
                "",
                ["task requires semantic or high-quality visual handling; remote VLM is available"],
                queue_depth,
                local_inflight,
                remote_inflight,
            )
        return (
            "reject",
            False,
            "remote VLM backend unavailable",
            ["remote VLM backend unavailable"],
            queue_depth,
            local_inflight,
            remote_inflight,
        )

    def run(self, scenario: Scenario, *, request_id: str, concurrency: int) -> dict[str, object]:
        start = time.perf_counter()
        route, fallback_used, reject_reason, reasons, queue_depth, local_at_decision, remote_at_decision = self.decide(
            scenario
        )
        status = "success"
        backend_error = False
        timeout = False
        backend_latency_ms: float | str = ""
        error = ""

        if route == "reject":
            status = "rejected"
        elif route == "local":
            with self._counter_lock:
                self._local_inflight += 1
            try:
                with self._local_lock:
                    local_result = self.router.run_local_cv(self.image_path)
                backend_latency_ms = local_result.inference_latency_ms or local_result.total_latency_ms or ""
                if not local_result.ok:
                    status = "backend_error"
                    backend_error = True
                    error = local_result.error
                    reasons.append(local_result.error)
            finally:
                with self._counter_lock:
                    self._local_inflight -= 1
        elif route == "remote":
            with self._counter_lock:
                self._remote_inflight += 1
            try:
                if scenario.simulated_remote_latency_ms > self.timeout_ms:
                    time.sleep(min(self.timeout_ms / 1000, 0.05))
                    status = "timeout"
                    timeout = True
                    backend_error = True
                    error = "simulated remote timeout"
                    backend_latency_ms = round(self.timeout_ms, 2)
                    reasons.append("remote mock exceeded timeout budget")
                else:
                    time.sleep(max(0.0, scenario.simulated_remote_latency_ms) / 1000)
                    backend_latency_ms = round(scenario.simulated_remote_latency_ms, 2)
            finally:
                with self._counter_lock:
                    self._remote_inflight -= 1

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        match_expected = matches_expected(scenario.expected_behavior, route, status, fallback_used, backend_error, timeout)
        return {
            "request_id": request_id,
            "scenario": scenario.name,
            "concurrency": concurrency,
            "task_type": scenario.task_type,
            "privacy": scenario.privacy,
            "quality": scenario.quality,
            "route": route,
            "expected_behavior": scenario.expected_behavior,
            "status": status,
            "fallback_used": fallback_used,
            "reject_reason": reject_reason,
            "backend_error": backend_error,
            "timeout": timeout,
            "queue_depth_at_decision": queue_depth,
            "local_inflight": local_at_decision,
            "remote_inflight": remote_at_decision,
            "latency_ms": latency_ms,
            "backend_latency_ms": backend_latency_ms,
            "match_expected": match_expected,
            "error": error,
            "reasons": json.dumps(reasons, ensure_ascii=False),
        }


def matches_expected(
    expected_behavior: str,
    route: str,
    status: str,
    fallback_used: bool,
    backend_error: bool,
    timeout: bool,
) -> bool:
    if expected_behavior == "local_success":
        return route == "local" and status == "success" and not fallback_used
    if expected_behavior == "remote_success":
        return route == "remote" and status == "success" and not fallback_used
    if expected_behavior == "reject":
        return route == "reject" and status == "rejected"
    if expected_behavior == "remote_fallback":
        return route == "remote" and status == "success" and fallback_used
    if expected_behavior == "local_or_remote_success":
        return status == "success" and route in {"local", "remote"}
    if expected_behavior == "backend_error":
        return backend_error and status in {"backend_error", "timeout"}
    if expected_behavior == "timeout":
        return timeout and status == "timeout"
    return False


def load_scenarios() -> list[Scenario]:
    return [
        Scenario("normal_local_detect", "detect", "allow_remote", "low", "local_or_remote_success"),
        Scenario("normal_local_classify", "classify", "allow_remote", "low", "local_or_remote_success"),
        Scenario("normal_remote_scene", "scene_description", "allow_remote", "medium", "remote_success"),
        Scenario("normal_remote_vqa", "vqa", "allow_remote", "high", "remote_success"),
        Scenario("privacy_vqa_local_only", "vqa", "local_only", "medium", "reject"),
        Scenario("impossible_latency_budget", "scene_description", "allow_remote", "medium", "reject", latency_budget_ms=50),
    ]


def failure_scenarios(timeout_s: float) -> list[Scenario]:
    return [
        Scenario("normal_local_detect", "detect", "allow_remote", "low", "local_success"),
        Scenario("normal_remote_scene", "scene_description", "allow_remote", "medium", "remote_success"),
        Scenario("privacy_local_only_vqa", "vqa", "local_only", "medium", "reject"),
        Scenario(
            "local_backend_unavailable_private",
            "detect",
            "local_only",
            "low",
            "reject",
            local_available=False,
        ),
        Scenario(
            "remote_backend_unavailable_scene",
            "scene_description",
            "allow_remote",
            "medium",
            "reject",
            remote_available=False,
        ),
        Scenario(
            "local_queue_overloaded_allow_remote",
            "detect",
            "allow_remote",
            "low",
            "remote_fallback",
            queue_depth_override=2,
        ),
        Scenario(
            "local_queue_overloaded_private",
            "detect",
            "local_only",
            "low",
            "reject",
            queue_depth_override=2,
        ),
        Scenario("impossible_latency_budget", "vqa", "allow_remote", "medium", "reject", latency_budget_ms=50),
        Scenario(
            "remote_timeout",
            "scene_description",
            "allow_remote",
            "medium",
            "timeout",
            simulated_remote_latency_ms=timeout_s * 1000 + 100,
        ),
    ]


def parse_concurrency(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    fraction = idx - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_concurrency: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_concurrency.setdefault(int(row["concurrency"]), []).append(row)
    summaries: list[dict[str, object]] = []
    for concurrency, group in sorted(by_concurrency.items()):
        latencies = [float(row["latency_ms"]) for row in group]
        total = len(group)
        summaries.append(
            {
                "concurrency": concurrency,
                "total_requests": total,
                "success_count": sum(1 for row in group if row["status"] == "success"),
                "reject_count": sum(1 for row in group if row["status"] == "rejected"),
                "backend_error_count": sum(1 for row in group if str(row["backend_error"]).lower() == "true"),
                "timeout_count": sum(1 for row in group if str(row["timeout"]).lower() == "true"),
                "fallback_count": sum(1 for row in group if str(row["fallback_used"]).lower() == "true"),
                "route_local_count": sum(1 for row in group if row["route"] == "local"),
                "route_remote_count": sum(1 for row in group if row["route"] == "remote"),
                "route_reject_count": sum(1 for row in group if row["route"] == "reject"),
                "p50_latency_ms": round(percentile(latencies, 0.50), 2),
                "p95_latency_ms": round(percentile(latencies, 0.95), 2),
                "p99_latency_ms": round(percentile(latencies, 0.99), 2),
                "pass_rate": round(sum(1 for row in group if row["match_expected"] is True) / total, 4) if total else 0.0,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal v0.7 reliability benchmark for the vision router.")
    parser.add_argument("--mode", choices=["vision"], default="vision")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("serving/results/raw/reliability_benchmark.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("serving/results/raw/reliability_summary.csv"))
    parser.add_argument("--failure-out", type=Path, default=Path("serving/results/raw/reliability_failure_modes.csv"))
    parser.add_argument("--timeout-s", type=float, default=0.1)
    parser.add_argument("--sample-image", type=Path, default=REPO_ROOT / "results/figures/camera_v05_positive_detection.jpg")
    parser.add_argument("--engine-path", type=Path, default=Path("/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/rainbow/models/vision/mobilenet_ssd"))
    parser.add_argument("--local-queue-limit", type=int, default=2)
    args = parser.parse_args()

    harness = ReliabilityHarness(
        image_path=args.sample_image,
        engine_path=args.engine_path,
        model_dir=args.model_dir,
        local_queue_limit=args.local_queue_limit,
        timeout_s=args.timeout_s,
    )
    templates = load_scenarios()
    rows: list[dict[str, object]] = []
    for concurrency in parse_concurrency(args.concurrency):
        jobs = [templates[idx % len(templates)] for idx in range(args.requests)]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    harness.run,
                    scenario,
                    request_id=f"rel-c{concurrency}-{idx:03d}",
                    concurrency=concurrency,
                )
                for idx, scenario in enumerate(jobs)
            ]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: str(row["request_id"]))
    write_csv(args.out, rows)
    summary_rows = summarize(rows)
    write_csv(args.summary_out, summary_rows)

    failure_rows = [
        harness.run(scenario, request_id=f"failure-{idx:03d}", concurrency=0)
        for idx, scenario in enumerate(failure_scenarios(args.timeout_s))
    ]
    write_csv(args.failure_out, failure_rows)

    all_rows = rows + failure_rows
    ok = all(row["match_expected"] is True for row in all_rows)
    print(
        json.dumps(
            {
                "benchmark_out": str(args.out),
                "summary_out": str(args.summary_out),
                "failure_out": str(args.failure_out),
                "benchmark_rows": len(rows),
                "failure_rows": len(failure_rows),
                "benchmark_pass_rate": round(
                    sum(1 for row in rows if row["match_expected"] is True) / len(rows), 4
                )
                if rows
                else 0.0,
                "failure_pass_rate": round(
                    sum(1 for row in failure_rows if row["match_expected"] is True) / len(failure_rows), 4
                )
                if failure_rows
                else 0.0,
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
