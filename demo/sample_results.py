from __future__ import annotations

import csv
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = REPO_ROOT / "results" / "figures" / "camera_v05_positive_detection.jpg"
TEXT_SMOKE_CSV = REPO_ROOT / "serving" / "results" / "raw" / "dual_real_backend_smoke.csv"
VISION_SMOKE_CSV = REPO_ROOT / "serving" / "results" / "raw" / "vision_router_yolo_trt_smoke.csv"
RELIABILITY_SUMMARY_CSV = REPO_ROOT / "serving" / "results" / "raw" / "reliability_summary.csv"
LOCAL_CV_SUMMARY_CSV = REPO_ROOT / "serving" / "results" / "raw" / "local_cv_runtime_summary.csv"
INTERACTIVE_SMOKE_CSV = REPO_ROOT / "serving" / "results" / "raw" / "interactive_gateway_smoke.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_jsonish(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _first(rows: list[dict[str, str]], **filters: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            return row
    return None


def _sample_preview_notice() -> str:
    return "Sample mode only stores a preview. Use real backend mode to see the full response."


def _labels_summary(labels: list[Any], detections: list[dict[str, Any]]) -> str:
    label_text = ", ".join(str(label) for label in labels) if labels else "no labels"
    if not detections:
        return f"Detected labels: {label_text}."
    top = detections[0]
    return (
        f"Detected labels: {label_text}. "
        f"Top detection: {top.get('label', 'object')} "
        f"confidence={top.get('confidence', 'n/a')} box={top.get('box', [])}."
    )


def _vision_final_answer(
    *,
    route: str,
    labels: list[Any],
    detections: list[dict[str, Any]],
    reasons: list[Any],
    remote_text: str = "",
) -> tuple[str, str]:
    if route == "local":
        return "Jetson local CV", _labels_summary(labels, detections)
    if route == "remote":
        return "RTX remote VLM", remote_text or "Remote route sample stores routing evidence but no full VLM text."
    if route == "reject":
        return "policy reject", "; ".join(str(reason) for reason in reasons if reason)
    return "unknown", ""


def select_text_sample(
    prompt: str,
    task_type: str,
    privacy: str,
    quality: str,
    latency_budget_ms: int,
) -> dict[str, Any]:
    rows = read_csv(TEXT_SMOKE_CSV)
    row: dict[str, str] | None = None

    if latency_budget_ms < 800 and task_type in {"code", "reasoning"}:
        row = _first(rows, route="reject") or (rows[-1] if rows else None)
    elif privacy == "local_only":
        row = _first(rows, task_type=task_type, privacy="local_only") or _first(rows, route="local")
    elif quality == "high" or task_type in {"code", "reasoning"}:
        row = _first(rows, task_type=task_type, route="remote") or _first(rows, route="remote")
    else:
        row = _first(rows, task_type=task_type, route="local") or _first(rows, route="local")

    if not row:
        return {
            "mode": "sample",
            "route": "reject",
            "selected_backend": "",
            "backend_latency_ms": None,
            "total_latency_ms": 0,
            "response_preview": "No sample row is available for this request.",
            "full_response": "No sample row is available for this request.",
            "sample_truncated": True,
            "reasons": ["sample data unavailable"],
            "input_preview": prompt[:120],
        }

    preview = row.get("short_response_preview", "")
    return {
        "mode": "sample",
        "route": row.get("route", ""),
        "selected_backend": row.get("selected_model", ""),
        "backend_latency_ms": row.get("backend_latency_ms") or None,
        "total_latency_ms": row.get("total_latency_ms") or None,
        "response_preview": preview,
        "full_response": preview,
        "sample_truncated": True,
        "sample_truncation_note": _sample_preview_notice(),
        "reasons": [reason.strip() for reason in row.get("reasons", "").split(";") if reason.strip()],
        "input_preview": prompt[:120],
        "sample_request_id": row.get("request_id", ""),
        "sample_source": str(TEXT_SMOKE_CSV.relative_to(REPO_ROOT)),
    }


def call_text_backend(
    api_base_url: str,
    prompt: str,
    task_type: str,
    privacy: str,
    quality: str,
    latency_budget_ms: int,
) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "task_type": task_type,
        "privacy": privacy,
        "quality": quality,
        "latency_budget_ms": latency_budget_ms,
        "stream": False,
        "max_tokens": 128,
    }
    start = time.perf_counter()
    try:
        request = urllib.request.Request(
            f"{api_base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read().decode("utf-8")
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        data = json.loads(raw)
        message = ""
        if data.get("choices"):
            message = data["choices"][0].get("message", {}).get("content", "")
        decision = data.get("route_decision") or {}
        return {
            "mode": "real",
            "route": decision.get("route", "unknown"),
            "selected_backend": decision.get("selected_model") or data.get("model", ""),
            "backend_latency_ms": data.get("backend_latency_ms"),
            "total_latency_ms": data.get("total_latency_ms", elapsed_ms),
            "response_preview": message[:500] or raw[:500],
            "full_response": message or raw,
            "sample_truncated": False,
            "reasons": decision.get("reasons", ["real backend response did not include route reasons"]),
            "input_preview": prompt[:120],
        }
    except urllib.error.HTTPError as exc:
        try:
            data = json.loads(exc.read().decode("utf-8"))
            decision = data.get("decision") or {}
            return {
                "mode": "real",
                "route": decision.get("route", "reject"),
                "selected_backend": decision.get("selected_model") or "",
                "backend_latency_ms": data.get("backend_latency_ms"),
                "total_latency_ms": data.get("total_latency_ms"),
                "response_preview": data.get("error", str(exc)),
                "full_response": data.get("error", str(exc)),
                "sample_truncated": False,
                "reasons": decision.get("reasons", [data.get("error", str(exc))]),
                "input_preview": prompt[:120],
            }
        except Exception:
            sample = select_text_sample(prompt, task_type, privacy, quality, latency_budget_ms)
            sample["mode"] = "sample fallback"
            sample["reasons"] = [f"real backend returned HTTP {exc.code}", *sample["reasons"]]
            return sample
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        sample = select_text_sample(prompt, task_type, privacy, quality, latency_budget_ms)
        sample["mode"] = "sample fallback"
        sample["reasons"] = [f"real backend unavailable: {exc}", *sample["reasons"]]
        return sample


def select_vision_sample(task_type: str, privacy: str, quality: str) -> dict[str, Any]:
    rows = read_csv(VISION_SMOKE_CSV)
    interactive_rows = read_csv(INTERACTIVE_SMOKE_CSV)
    row = _first(rows, task_type=task_type, privacy=privacy, quality=quality)
    if not row and quality == "high":
        row = _first(rows, task_type=task_type, quality="high")
    if not row:
        row = _first(rows, task_type=task_type) or (rows[0] if rows else None)

    if not row:
        return {
            "mode": "sample",
            "route": "reject",
            "selected_backend": "",
            "detected_labels": [],
            "detections": [],
            "local_cv_precheck_labels": [],
            "local_cv_precheck_detections": [],
            "final_answer_source": "policy reject",
            "final_answer_text": "No sample row is available for this request.",
            "full_response": "No sample row is available for this request.",
            "sample_truncated": True,
            "reasons": ["sample data unavailable"],
        }

    route = row.get("route", "")
    labels = parse_jsonish(row.get("detected_labels"), [])
    detections = parse_jsonish(row.get("detections"), [])
    reasons = parse_jsonish(row.get("reasons"), [row.get("reasons", "")])
    interactive = _first(interactive_rows, task_type=task_type, privacy=privacy, quality=quality, route=route)
    remote_text = (interactive or {}).get("response_preview", "")
    final_source, final_text = _vision_final_answer(
        route=route,
        labels=labels,
        detections=detections,
        reasons=reasons,
        remote_text=remote_text,
    )
    return {
        "mode": "sample",
        "route": route,
        "selected_backend": row.get("selected_model", ""),
        "remote_is_mock": row.get("remote_is_mock", ""),
        "local_cv_backend": row.get("local_cv_backend", ""),
        "local_cv_model": row.get("local_cv_model", ""),
        "detected_labels": labels,
        "detections": detections,
        "local_cv_precheck_labels": labels,
        "local_cv_precheck_detections": detections,
        "capture_latency_ms": row.get("capture_latency_ms") or None,
        "local_cv_inference_latency_ms": row.get("local_cv_inference_latency_ms") or None,
        "local_cv_total_latency_ms": row.get("local_cv_total_latency_ms") or None,
        "remote_latency_ms": row.get("remote_latency_ms") or None,
        "total_latency_ms": row.get("total_latency_ms") or None,
        "remote_response_preview": remote_text,
        "remote_response_text": remote_text,
        "final_answer_source": final_source,
        "final_answer_text": final_text,
        "full_response": final_text,
        "sample_truncated": True,
        "sample_truncation_note": _sample_preview_notice(),
        "reasons": reasons,
        "sample_request_id": row.get("request_id", ""),
        "sample_source": str(VISION_SMOKE_CSV.relative_to(REPO_ROOT)),
    }


def _encode_image_file(image_path: Any) -> str | None:
    try:
        if hasattr(image_path, "getvalue"):
            return base64.b64encode(image_path.getvalue()).decode("ascii")
        path = Path(image_path)
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return None
    return None


def call_vision_backend(
    api_base_url: str,
    *,
    image_source: str,
    image_path: Any,
    task_type: str,
    privacy: str,
    quality: str,
    latency_budget_ms: int,
    prompt: str,
    timeout_s: float = 260.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_type": task_type,
        "privacy": privacy,
        "quality": quality,
        "latency_budget_ms": latency_budget_ms,
        "prompt": prompt or None,
        "image_source": "camera" if image_source == "camera" else "upload",
        "use_yolo_trt": True,
        "max_tokens": 96,
    }
    if image_source != "camera":
        image_b64 = _encode_image_file(image_path)
        if image_b64:
            payload["image_base64"] = image_b64
        else:
            sample = select_vision_sample(task_type, privacy, quality)
            sample["mode"] = "sample fallback"
            sample["reasons"] = ["failed to encode selected image", *sample.get("reasons", [])]
            return sample

    start = time.perf_counter()
    try:
        request = urllib.request.Request(
            f"{api_base_url.rstrip('/')}/v1/vision/analyze",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        data["mode"] = "real"
        data["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
        data["selected_backend"] = data.get("selected_backend") or ""
        labels = data.get("detected_labels") or []
        detections = data.get("detections") or []
        route = data.get("route", "")
        remote_text = data.get("remote_response_text") or ""
        final_source, final_text = _vision_final_answer(
            route=route,
            labels=labels,
            detections=detections,
            reasons=data.get("reasons") or [],
            remote_text=remote_text,
        )
        data["local_cv_precheck_labels"] = labels
        data["local_cv_precheck_detections"] = detections
        data["remote_response_preview"] = remote_text[:500]
        data["final_answer_source"] = final_source
        data["final_answer_text"] = final_text
        data["full_response"] = remote_text or final_text
        data["sample_truncated"] = False
        return data
    except urllib.error.HTTPError as exc:
        try:
            data = json.loads(exc.read().decode("utf-8"))
            data["mode"] = "real"
            data["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
            data["selected_backend"] = data.get("selected_backend") or ""
            labels = data.get("detected_labels") or []
            detections = data.get("detections") or []
            final_source, final_text = _vision_final_answer(
                route=data.get("route", "reject"),
                labels=labels,
                detections=detections,
                reasons=data.get("reasons") or [data.get("error", str(exc))],
                remote_text=data.get("remote_response_text") or "",
            )
            data["local_cv_precheck_labels"] = labels
            data["local_cv_precheck_detections"] = detections
            data["final_answer_source"] = final_source
            data["final_answer_text"] = final_text
            data["full_response"] = data.get("remote_response_text") or final_text
            data["sample_truncated"] = False
            return data
        except Exception:
            sample = select_vision_sample(task_type, privacy, quality)
            sample["mode"] = "sample fallback"
            sample["reasons"] = [f"real vision backend returned HTTP {exc.code}", *sample.get("reasons", [])]
            return sample
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        sample = select_vision_sample(task_type, privacy, quality)
        sample["mode"] = "sample fallback"
        sample["reasons"] = [f"real vision backend unavailable: {exc}", *sample.get("reasons", [])]
        return sample


def draw_detections(image_path: Any, detections: list[dict[str, Any]]) -> Image.Image | None:
    if hasattr(image_path, "read"):
        image = Image.open(image_path).convert("RGB")
    else:
        path = Path(image_path) if image_path else SAMPLE_IMAGE
        if not path.exists():
            return None
        image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = detection.get("box") or []
        if len(box) != 4:
            continue
        label = detection.get("label", "object")
        confidence = detection.get("confidence", 0)
        x1, y1, x2, y2 = [int(float(value)) for value in box]
        draw.rectangle((x1, y1, x2, y2), outline=(21, 120, 255), width=4)
        draw.text((x1 + 4, max(0, y1 - 18)), f"{label} {confidence:.2f}", fill=(21, 120, 255))
    return image


def backend_status() -> dict[str, Any]:
    reliability = read_csv(RELIABILITY_SUMMARY_CSV)
    return {
        "mode": "sample",
        "local_llm": "Qwen3.5 0.8B Q4_K_M on Jetson",
        "remote_llm": "Qwen3.5 4B Q4_K_M on RTX",
        "local_cv": "YOLOv8n TensorRT FP16 on Jetson; MobileNet-SSD OpenCV DNN baseline retained",
        "remote_vlm": "Gemma 4 E2B-it Q4_K_M + mmproj-F16 on RTX",
        "queue_state": "v0.7 reliability benchmark uses prototype in-process inflight tracking",
        "latest_reliability_pass_rate": reliability[-1].get("pass_rate") if reliability else "unknown",
    }


def key_result_tables() -> dict[str, list[dict[str, Any]]]:
    local_cv_rows = read_csv(LOCAL_CV_SUMMARY_CSV)
    reliability_rows = read_csv(RELIABILITY_SUMMARY_CSV)
    return {
        "local_cv_runtime_summary": local_cv_rows,
        "reliability_summary": reliability_rows,
        "backend_recommendations": [
            {
                "profile": "text_local_default",
                "backend": "Qwen3.5 0.8B Q4_K_M on Jetson",
                "why": "Highest Jetson useful throughput per watt while staying inside memory limits.",
            },
            {
                "profile": "text_quality_fallback",
                "backend": "Qwen3.5 4B Q4_K_M on RTX",
                "why": "Larger text model for code, reasoning, and high-quality text tasks.",
            },
            {
                "profile": "vision_local_fast_path",
                "backend": "YOLOv8n TensorRT FP16 on Jetson",
                "why": "Optimized local CV path, 14.54 ms average inference in Project 3.",
            },
            {
                "profile": "vision_remote_semantic_backend",
                "backend": "Gemma 4 E2B-it Q4_K_M + mmproj-F16 on RTX",
                "why": "Real remote VLM for scene descriptions and VQA.",
            },
        ],
    }


def real_backend_health(api_base_url: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{api_base_url.rstrip('/')}/health", timeout=3) as response:
            raw = response.read().decode("utf-8")
        return {
            "mode": "real",
            "healthy": True,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "response": json.loads(raw),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "mode": "sample fallback",
            "healthy": False,
            "error": str(exc),
            "sample_status": backend_status(),
        }
