from __future__ import annotations

import argparse
import base64
import csv
import json
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPO_ROOT / "demo"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

import event_store  # noqa: E402
from edgelog_events import EdgeLogConfig, EdgeLogEventEngine, EdgeLogROI  # noqa: E402
from vlm_verifier import SmolVLM2FastVerifier, verification_to_dict  # noqa: E402


OUT_PATH = REPO_ROOT / "serving" / "results" / "raw" / "edgelog_real_mode_validation.csv"
SAMPLE_IMAGE = REPO_ROOT / "results" / "figures" / "camera_v05_positive_detection.jpg"

FIELDS = [
    "case",
    "expected",
    "actual",
    "status",
    "source",
    "route",
    "backend",
    "latency_ms",
    "error",
    "notes",
]


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def get_json(url: str, timeout_s: float = 8.0) -> tuple[bool, dict[str, Any], str, float]:
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return True, json.loads(response.read().decode("utf-8")), "", round((time.perf_counter() - start) * 1000, 2)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, {}, str(exc), round((time.perf_counter() - start) * 1000, 2)


def post_json(url: str, payload: dict[str, Any], timeout_s: float = 60.0) -> tuple[int, dict[str, Any], str, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, json.loads(response.read().decode("utf-8")), "", round((time.perf_counter() - start) * 1000, 2)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"error": raw[:500]}
        return exc.code, data, "", round((time.perf_counter() - start) * 1000, 2)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return 0, {}, str(exc), round((time.perf_counter() - start) * 1000, 2)


def add_row(
    rows: list[dict[str, Any]],
    *,
    case: str,
    expected: str,
    actual: str,
    passed: bool,
    source: str,
    route: str = "",
    backend: str = "",
    latency_ms: Any = "",
    error: str = "",
    notes: str = "",
) -> None:
    rows.append(
        {
            "case": case,
            "expected": expected,
            "actual": actual,
            "status": "PASS" if passed else "FAIL",
            "source": source,
            "route": route,
            "backend": backend,
            "latency_ms": latency_ms,
            "error": error,
            "notes": notes,
        }
    )


def text_payload(prompt: str, task_type: str = "summary", quality: str = "high") -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "task_type": task_type,
        "privacy": "allow_remote",
        "quality": quality,
        "latency_budget_ms": 20000,
        "stream": False,
        "max_tokens": 192,
    }


def reset_temp_store(tmp_dir: Path) -> None:
    event_store.EVENT_ROOT = tmp_dir / "events"
    event_store.EVENT_IMAGE_DIR = event_store.EVENT_ROOT / "images"
    event_store.EVENT_LOG_PATH = event_store.EVENT_ROOT / "events.jsonl"
    event_store.LATEST_SNAPSHOT_PATH = event_store.EVENT_ROOT / "latest_snapshot.jpg"


def normalize_chat(data: dict[str, Any]) -> tuple[str, str, str]:
    decision = data.get("route_decision") or data.get("decision") or {}
    text = ""
    if data.get("choices"):
        text = data["choices"][0].get("message", {}).get("content", "")
    return decision.get("route", ""), decision.get("selected_model") or data.get("model", ""), text


def run_validation(gateway_url: str, verifier_url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gateway_url = gateway_url.rstrip("/")
    verifier_url = verifier_url.rstrip("/")
    sample_b64 = encode_image(SAMPLE_IMAGE)

    ok_gateway, gateway_health, gateway_error, _ = get_json(f"{gateway_url}/health")
    ok_llm, llm_health, llm_error, _ = get_json("http://127.0.0.1:8081/health")
    ok_vlm, vlm_health, vlm_error, _ = get_json("http://127.0.0.1:8091/health")
    ok_verifier, verifier_health, verifier_error, _ = get_json(f"{verifier_url}/health")
    full_stack_ready = (
        ok_gateway
        and gateway_health.get("local_backend_available") is True
        and gateway_health.get("remote_backend_available") is True
        and gateway_health.get("vision_remote_vlm_available") is True
        and gateway_health.get("vision_remote_is_mock") is False
        and ok_llm
        and ok_vlm
        and ok_verifier
        and verifier_health.get("ready") is True
    )
    add_row(
        rows,
        case="health_full_stack",
        expected="gateway/local/remote/vlm/verifier all ready",
        actual=json.dumps(
            {
                "gateway": gateway_health,
                "remote_llm": llm_health,
                "remote_vlm": vlm_health,
                "smolvlm2": verifier_health,
            },
            sort_keys=True,
        ),
        passed=full_stack_ready,
        source="real_stack_health",
        backend=str(verifier_health.get("backend", "")),
        error="; ".join(item for item in [gateway_error, llm_error, vlm_error, verifier_error] if item),
    )

    camera_payload = {
        "task_type": "detect",
        "privacy": "allow_remote",
        "quality": "low",
        "latency_budget_ms": 5000,
        "image_source": "camera",
        "use_yolo_trt": True,
    }
    status, vision_local, vision_local_error, elapsed = post_json(f"{gateway_url}/v1/vision/analyze", camera_payload, timeout_s=60)
    detections = vision_local.get("detections") or []
    labels = vision_local.get("detected_labels") or []
    proposal_created = bool(detections or labels)
    add_row(
        rows,
        case="cheap_trigger_real_camera",
        expected="real camera local YOLO route with proposal evidence",
        actual=f"status={status} route={vision_local.get('route')} labels={labels} detections={len(detections)}",
        passed=status == 200 and vision_local.get("route") == "local" and vision_local.get("local_cv_inference_latency_ms") is not None,
        source="real_camera",
        route=str(vision_local.get("route", "")),
        backend=str(vision_local.get("local_cv_backend") or vision_local.get("selected_backend") or ""),
        latency_ms=vision_local.get("local_cv_inference_latency_ms") or elapsed,
        error=vision_local_error or vision_local.get("error", ""),
        notes="proposal_created_from_labels_or_boxes=" + str(proposal_created),
    )

    verifier = SmolVLM2FastVerifier(verifier_url, timeout_s=10)
    verification = verification_to_dict(
        verifier.verify(
            {"natural_language_rule": "Alert me when there is a chair or indoor furniture visible."},
            {
                "proposal_reason": "chair detected by cheap local trigger",
                "objects": ["chair"],
                "keyframe_path": str(SAMPLE_IMAGE),
                "roi_name": "desk_roi",
            },
        )
    )
    promoted_status = "verified" if verification.get("final_answer") == "YES" else verification.get("semantic_status")
    add_row(
        rows,
        case="smolvlm2_fast_verifier_real",
        expected="parse_success true and event promoted/rejected/unknown",
        actual=f"answer={verification.get('final_answer')} status={promoted_status}",
        passed=verification.get("final_answer") in {"YES", "NO", "UNKNOWN"} and not verification.get("error"),
        source="sample_keyframe_real_smolvlm2",
        backend=str(verification.get("verifier_backend", "")),
        latency_ms=verification.get("latency_ms", ""),
        error=str(verification.get("error", "")),
    )

    remote_payload = {
        "task_type": "scene_description",
        "privacy": "allow_remote",
        "quality": "high",
        "latency_budget_ms": 30000,
        "prompt": "Describe this EdgeLog event keyframe in one concise sentence. Do not explain reasoning.",
        "image_source": "upload",
        "image_base64": sample_b64,
        "use_yolo_trt": True,
        "max_tokens": 64,
    }
    status, remote_vlm, remote_error, elapsed = post_json(f"{gateway_url}/v1/vision/analyze", remote_payload, timeout_s=260)
    remote_text = remote_vlm.get("remote_response_text") or remote_vlm.get("final_answer_text") or ""
    add_row(
        rows,
        case="slow_vlm_describer_real",
        expected="route=remote remote_is_mock=false nonempty description",
        actual=f"status={status} route={remote_vlm.get('route')} mock={remote_vlm.get('remote_is_mock')} chars={len(remote_text)}",
        passed=status == 200 and remote_vlm.get("route") == "remote" and remote_vlm.get("remote_is_mock") is False and bool(remote_text.strip()),
        source="sample_keyframe_real_gemma_vlm",
        route=str(remote_vlm.get("route", "")),
        backend=str(remote_vlm.get("remote_model") or remote_vlm.get("selected_backend") or ""),
        latency_ms=remote_vlm.get("remote_latency_ms") or elapsed,
        error=remote_error or remote_vlm.get("error", ""),
    )

    text_status, text_data, text_error, text_elapsed = post_json(
        f"{gateway_url}/v1/chat/completions",
        text_payload(
            "Summarize these EdgeLog events in one sentence: verified chair event, unknown hidden sensor event, rejected vehicle event."
        ),
        timeout_s=120,
    )
    text_route, text_backend, text = normalize_chat(text_data)
    add_row(
        rows,
        case="text_llm_summary_real",
        expected="real text LLM returns nonempty summary with route/backend",
        actual=f"status={text_status} route={text_route} backend={text_backend} chars={len(text)}",
        passed=text_status == 200 and bool(text.strip()) and bool(text_route),
        source="real_text_llm",
        route=text_route,
        backend=text_backend,
        latency_ms=text_data.get("total_latency_ms") or text_elapsed,
        error=text_error or text_data.get("error", ""),
    )

    with tempfile.TemporaryDirectory() as tmp:
        reset_temp_store(Path(tmp))
        verified = event_store.append_event(
            {
                "event_type": "semantic_event",
                "status": "verified",
                "event_rule": "chair visible",
                "route": "remote",
                "objects": ["chair"],
                "risk_level": "medium",
                "semantic_status": "yes",
                "verification": {"final_answer": "YES", "semantic_status": "yes", "reason": "chair visible"},
                "semantic_description": "A chair is visible near the desk ROI.",
                "trigger_matched": True,
            },
            image_source=str(SAMPLE_IMAGE),
        )
        event_store.append_event(
            {
                "event_type": "semantic_event",
                "status": "rejected",
                "event_rule": "red vehicle visible",
                "route": "remote",
                "objects": ["chair"],
                "risk_level": "low",
                "semantic_status": "no",
                "verification": {"final_answer": "NO", "semantic_status": "no", "reason": "no vehicle"},
                "trigger_matched": True,
            },
            image_source=str(SAMPLE_IMAGE),
        )
        hits = event_store.search_events("chair", event_type="semantic_event", semantic_status="yes")
        summary = event_store.build_daily_summary()
        add_row(
            rows,
            case="event_search_real_store",
            expected="verified searchable and rejected excluded from high-value summary",
            actual=f"hits={len(hits)} summary={summary['counts_by_type']}",
            passed=len(hits) == 1 and hits[0].get("event_id") == verified.get("event_id") and summary["counts_by_type"].get("semantic_event") == 1,
            source="temp_store",
        )
        add_row(
            rows,
            case="daily_summary_real_or_deterministic",
            expected="deterministic summary includes correct counts",
            actual=json.dumps(summary["counts_by_type"], sort_keys=True),
            passed=summary["counts_by_type"].get("semantic_event") == 1,
            source="temp_store",
        )
        ordinary = event_store.append_event(
            {"source": "live_event_stream", "route": "local", "objects": ["chair"], "semantic_status": "not_required"},
            image_source=str(SAMPLE_IMAGE),
        )
        add_row(
            rows,
            case="storage_retention_guard",
            expected="ordinary frame latest snapshot only",
            actual=f"stored_event={ordinary.get('stored_event')} stored_image={ordinary.get('stored_image')}",
            passed=ordinary.get("stored_event") is False,
            source="temp_store",
        )

    reject_payload = {
        "task_type": "vqa",
        "privacy": "local_only",
        "quality": "high",
        "latency_budget_ms": 3000,
        "prompt": "What is happening in this image?",
        "image_source": "upload",
        "image_base64": sample_b64,
        "use_yolo_trt": True,
    }
    reject_status, reject_data, reject_error, reject_elapsed = post_json(f"{gateway_url}/v1/vision/analyze", reject_payload, timeout_s=60)
    reject_route = reject_data.get("route") or ("reject" if reject_status >= 400 else "")
    add_row(
        rows,
        case="privacy_local_only_reject",
        expected="semantic vision local_only rejected, no remote send",
        actual=f"status={reject_status} route={reject_route} reasons={reject_data.get('reasons')}",
        passed=reject_route == "reject",
        source="sample_keyframe_policy",
        route=reject_route,
        latency_ms=reject_data.get("total_latency_ms") or reject_elapsed,
        error=reject_error or reject_data.get("error", ""),
    )

    failure = verification_to_dict(
        SmolVLM2FastVerifier("http://127.0.0.1:65500", timeout_s=0.2).verify(
            {"natural_language_rule": "Alert me when a chair is visible."},
            {"proposal_reason": "chair detected", "objects": ["chair"], "keyframe_path": str(SAMPLE_IMAGE)},
        )
    )
    add_row(
        rows,
        case="graceful_backend_failure",
        expected="failed verifier recorded without crash",
        actual=f"semantic_status={failure.get('semantic_status')} error={bool(failure.get('error'))}",
        passed=failure.get("semantic_status") == "failed" and bool(failure.get("error")),
        source="unused_port_failure_simulation",
        backend=str(failure.get("verifier_backend", "")),
        latency_ms=failure.get("latency_ms", ""),
        error="simulated unavailable verifier port",
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EdgeLog real-mode semantic event stack.")
    parser.add_argument("--gateway-url", default="http://192.168.1.102:8000")
    parser.add_argument("--verifier-url", default="http://127.0.0.1:8092")
    parser.add_argument("--out", default=str(OUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = run_validation(args.gateway_url, args.verifier_url)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    failures = [row for row in rows if row["status"] == "FAIL"]
    if failures:
        print(json.dumps(failures, indent=2, ensure_ascii=False))
        raise SystemExit(f"{len(failures)} EdgeLog real-mode validation checks failed")


if __name__ == "__main__":
    main()
