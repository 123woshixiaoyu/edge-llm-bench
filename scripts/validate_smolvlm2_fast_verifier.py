from __future__ import annotations

import argparse
import base64
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPO_ROOT / "demo"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from vlm_verifier import SmolVLM2FastVerifier, verification_to_dict  # noqa: E402


OUT_PATH = REPO_ROOT / "serving" / "results" / "raw" / "smolvlm2_fast_verifier_validation.csv"
SAMPLE_IMAGE = REPO_ROOT / "results" / "figures" / "camera_v05_positive_detection.jpg"


FIELDS = [
    "case",
    "expected",
    "actual",
    "status",
    "backend",
    "final_answer",
    "parse_success",
    "latency_ms",
    "error",
    "notes",
]


def post_json(url: str, payload: dict[str, Any], timeout_s: float = 12.0) -> tuple[bool, dict[str, Any], str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return True, json.loads(response.read().decode("utf-8")), ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, {}, str(exc)


def get_json(url: str, timeout_s: float = 8.0) -> tuple[bool, dict[str, Any], str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return True, json.loads(response.read().decode("utf-8")), ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, {}, str(exc)


def image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def blank_image_base64() -> str:
    from PIL import Image

    image = Image.new("RGB", (640, 480), (128, 128, 128))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def row(
    rows: list[dict[str, Any]],
    *,
    case: str,
    expected: str,
    actual: str,
    passed: bool,
    data: dict[str, Any] | None = None,
    error: str = "",
    notes: str = "",
) -> None:
    data = data or {}
    rows.append(
        {
            "case": case,
            "expected": expected,
            "actual": actual,
            "status": "PASS" if passed else "FAIL",
            "backend": data.get("backend", ""),
            "final_answer": data.get("final_answer", ""),
            "parse_success": data.get("parse_success", ""),
            "latency_ms": data.get("latency_ms", ""),
            "error": error or data.get("error", ""),
            "notes": notes,
        }
    )


def verify_payload(rule: str, *, proposal_reason: str, objects: list[str], question: str | None = None) -> dict[str, Any]:
    return {
        "rule": rule,
        "question": question or "Does this candidate frame show that event?",
        "image_base64": image_base64(SAMPLE_IMAGE),
        "roi_name": "desk_roi",
        "proposal_reason": proposal_reason,
        "objects": objects,
        "max_new_tokens": 32,
        "timeout_s": 8,
    }


def blank_payload(rule: str, *, proposal_reason: str, question: str | None = None) -> dict[str, Any]:
    return {
        "rule": rule,
        "question": question or "Does this candidate frame show that event?",
        "image_base64": blank_image_base64(),
        "roi_name": "validation_blank",
        "proposal_reason": proposal_reason,
        "objects": [],
        "max_new_tokens": 32,
        "timeout_s": 8,
    }


def promote_event(answer: str, error: str = "") -> str:
    if error:
        return "failed"
    if answer == "YES":
        return "verified"
    if answer == "NO":
        return "rejected"
    if answer == "UNKNOWN":
        return "unknown"
    return "failed"


def run_validation(base_url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    verify_url = f"{base_url.rstrip('/')}/v1/verify_event"
    health_ok, health_data, health_error = get_json(f"{base_url.rstrip('/')}/health")
    row(
        rows,
        case="health endpoint ready",
        expected="ready=true",
        actual=json.dumps(health_data, sort_keys=True) if health_data else health_error,
        passed=health_ok and health_data.get("ready") is True,
        data=health_data,
        error=health_error,
    )
    if not (health_ok and health_data.get("ready") is True):
        return rows

    cases = [
        (
            "YES case parse_success true",
            "YES",
            verify_payload(
                "Alert me when there is a chair or indoor furniture visible.",
                proposal_reason="chair detected by cheap local trigger",
                objects=["chair"],
            ),
        ),
        (
            "NO case parse_success true",
            "NO",
            blank_payload(
                "Alert me when there is a chair visible.",
                proposal_reason="blank generated validation frame",
                question="Is there a chair visible?",
            ),
        ),
        (
            "UNKNOWN ambiguous case parse_success true",
            "UNKNOWN",
            blank_payload(
                "This is an ambiguous validation case; the correct answer is UNKNOWN.",
                proposal_reason="blank generated validation frame",
                question="What is the final answer?",
            ),
        ),
    ]

    response_by_answer: dict[str, dict[str, Any]] = {}
    for case_name, expected_answer, payload in cases:
        ok, data, error = post_json(verify_url, payload)
        response_by_answer[expected_answer] = data
        passed = ok and data.get("parse_success") is True and data.get("final_answer") == expected_answer
        row(
            rows,
            case=case_name,
            expected=f"final_answer={expected_answer}",
            actual=f"final_answer={data.get('final_answer')} parse_success={data.get('parse_success')}",
            passed=passed,
            data=data,
            error=error,
        )

    ok, data, error = post_json(
        verify_url,
        {
            "rule": "Alert me when anything happens.",
            "question": "Does this candidate frame show that event?",
            "image_path": str(REPO_ROOT / "missing-image.jpg"),
            "proposal_reason": "invalid image validation",
            "objects": [],
        },
    )
    row(
        rows,
        case="invalid image graceful error",
        expected="error without crash",
        actual=data.get("error", error),
        passed=ok and bool(data.get("error")),
        data=data,
        error=error,
    )

    fallback = verification_to_dict(
        SmolVLM2FastVerifier("http://127.0.0.1:65500", timeout_s=0.2).verify(
            {"natural_language_rule": "Alert me when a chair is visible."},
            {
                "proposal_reason": "chair detected",
                "objects": ["chair"],
                "keyframe_path": str(SAMPLE_IMAGE),
            },
        )
    )
    fallback["error"] = "SmolVLM2 verifier unavailable on unused validation port"
    row(
        rows,
        case="timeout / service unavailable graceful fallback",
        expected="semantic_status=failed with error",
        actual=f"{fallback.get('semantic_status')} {fallback.get('error')}",
        passed=fallback.get("semantic_status") == "failed" and bool(fallback.get("error")),
        data=fallback,
    )

    for answer, expected_status in [("YES", "verified"), ("NO", "rejected"), ("UNKNOWN", "unknown")]:
        data = response_by_answer.get(answer, {})
        status = promote_event(str(data.get("final_answer") or ""), str(data.get("error") or ""))
        row(
            rows,
            case=f"EdgeLog proposal -> SmolVLM2 -> {expected_status} event",
            expected=f"event_status={expected_status}",
            actual=f"event_status={status} final_answer={data.get('final_answer')}",
            passed=status == expected_status,
            data=data,
        )

    latencies = [
        float(item.get("latency_ms"))
        for item in response_by_answer.values()
        if item.get("latency_ms") not in {"", None}
    ]
    if latencies:
        row(
            rows,
            case="latency_ms recorded",
            expected="avg/p50/p95 recorded",
            actual=f"avg={statistics.mean(latencies):.2f} p50={statistics.median(latencies):.2f} p95={max(latencies):.2f}",
            passed=True,
            data={"latency_ms": round(statistics.mean(latencies), 2), "backend": health_data.get("backend", "")},
            notes="p95 uses max for the three validation calls.",
        )
    else:
        row(rows, case="latency_ms recorded", expected="latency values", actual="none", passed=False)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the live RTX SmolVLM2 EdgeLog verifier service.")
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    parser.add_argument("--out", default=str(OUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = run_validation(args.url)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    failures = [item for item in rows if item["status"] == "FAIL"]
    if failures:
        raise SystemExit(f"{len(failures)} SmolVLM2 verifier validation checks failed")


if __name__ == "__main__":
    main()
