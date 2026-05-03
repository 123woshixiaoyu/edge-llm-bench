from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .camera import CameraFrame, capture_frame
from .local_cv import LocalCVResult, run_mobilenet_ssd
from .vision_analyzer import ImageSource, VisionPrivacy, VisionQuality, VisionTaskType, analyze_vision_task
from .vision_policy import VisionDecision, VisionPolicyEngine


@dataclass
class VisionRequest:
    task_type: VisionTaskType
    privacy: VisionPrivacy
    quality: VisionQuality
    image_source: ImageSource = "camera"
    latency_budget_ms: int = 3000
    request_id: str | None = None
    expected_route: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VisionResult:
    request_id: str
    request: VisionRequest
    decision: VisionDecision
    capture: CameraFrame
    local_cv: LocalCVResult
    remote_latency_ms: float | None
    total_latency_ms: float
    match_expected: bool | None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "request": self.request.to_dict(),
            "decision": self.decision.to_dict(),
            "capture": self.capture.to_dict(),
            "local_cv": self.local_cv.to_dict(),
            "remote_latency_ms": self.remote_latency_ms,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "match_expected": self.match_expected,
            "error": self.error,
        }


def mock_remote_vlm_result(task_type: str, labels: list[str]) -> tuple[str, float]:
    start = time.perf_counter()
    label_text = ", ".join(labels) if labels else "unknown objects"
    text = f"[mock_remote_vlm] task={task_type}; local_cv_labels={label_text}; real remote VLM is deferred to v0.5b."
    latency_ms = (time.perf_counter() - start) * 1000
    return text, round(latency_ms, 2)


class VisionRouter:
    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        policy: VisionPolicyEngine | None = None,
    ):
        self.model_dir = model_dir
        self.policy = policy or VisionPolicyEngine()

    def route_existing_frame(
        self,
        request: VisionRequest,
        *,
        capture: CameraFrame,
        local_cv: LocalCVResult,
    ) -> VisionResult:
        start = time.perf_counter()
        request_id = request.request_id or str(uuid.uuid4())
        analysis = analyze_vision_task(request.task_type, request.image_source)
        local_cv_available = local_cv.ok
        decision = self.policy.decide(
            analysis=analysis,
            privacy=request.privacy,
            quality=request.quality,
            latency_budget_ms=request.latency_budget_ms,
            local_cv_available=local_cv_available,
        )
        remote_latency_ms = None
        error = local_cv.error if decision.route == "local" and not local_cv.ok else ""
        if decision.route == "remote":
            _, remote_latency_ms = mock_remote_vlm_result(request.task_type, local_cv.detected_labels)
        match_expected = None
        if request.expected_route is not None:
            match_expected = decision.route == request.expected_route
        total_latency_ms = (time.perf_counter() - start) * 1000
        capture_latency = capture.capture_latency_ms or 0.0
        cv_latency = local_cv.total_latency_ms or local_cv.inference_latency_ms or 0.0
        total_latency_ms += capture_latency + cv_latency + (remote_latency_ms or 0.0)
        return VisionResult(
            request_id=request_id,
            request=request,
            decision=decision,
            capture=capture,
            local_cv=local_cv,
            remote_latency_ms=remote_latency_ms,
            total_latency_ms=total_latency_ms,
            match_expected=match_expected,
            error=error,
        )

    def capture_and_analyze(
        self,
        *,
        image_path: Path,
        device: str | None = None,
        sensor_id: int = 0,
    ) -> tuple[CameraFrame, LocalCVResult]:
        capture = capture_frame(image_path, device=device, sensor_id=sensor_id, prefer_argus=True)
        if not capture.ok:
            local_cv = LocalCVResult(
                ok=False,
                model="mobilenet_ssd_voc_opencv_dnn",
                model_dir=str(self.model_dir or ""),
                detected_labels=[],
                detections=[],
                inference_latency_ms=None,
                total_latency_ms=None,
                error=capture.error,
            )
            return capture, local_cv
        local_cv = run_mobilenet_ssd(Path(capture.image_path), model_dir=self.model_dir)
        return capture, local_cv


def detections_json(result: LocalCVResult) -> str:
    return json.dumps([detection.to_dict() for detection in result.detections], ensure_ascii=False)
