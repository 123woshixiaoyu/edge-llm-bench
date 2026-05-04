from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends.vlm import HttpVLMBackend
from .camera import CameraFrame, capture_frame
from .local_cv import LocalCVResult, run_mobilenet_ssd
from .local_cv_yolo import YoloTensorRTDetector
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
    prompt: str | None = None
    max_tokens: int = 128

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
    remote_model: str | None
    remote_response_text: str
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
            "remote_model": self.remote_model,
            "remote_response_text": self.remote_response_text,
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
        local_cv_backend: str = "mobilenet_ssd_opencv_dnn",
        yolo_engine_path: Path | None = None,
        fallback_to_mobilenet: bool = True,
        policy: VisionPolicyEngine | None = None,
        remote_backend: HttpVLMBackend | None = None,
    ):
        self.model_dir = model_dir
        self.local_cv_backend = local_cv_backend
        self.yolo_engine_path = yolo_engine_path
        self.fallback_to_mobilenet = fallback_to_mobilenet
        self._yolo_detector: YoloTensorRTDetector | None = None
        local_cv_model = "yolov8n_tensorrt_fp16" if local_cv_backend == "yolo_tensorrt_fp16" else "mobilenet_ssd_voc_opencv_dnn"
        self.policy = policy or VisionPolicyEngine(local_cv_model=local_cv_model)
        self.remote_backend = remote_backend

    def _annotate_local_cv(
        self,
        result: LocalCVResult,
        *,
        backend: str,
        engine_path: str = "",
        fallback_used: bool = False,
    ) -> LocalCVResult:
        result.local_cv_backend = backend
        result.engine_path = engine_path
        result.fallback_used = fallback_used
        return result

    def default_remote_prompt(self, request: VisionRequest, local_cv: LocalCVResult) -> str:
        labels = ", ".join(local_cv.detected_labels) if local_cv.detected_labels else "none"
        if request.task_type == "vqa":
            return f"What is visible in this image? Local CV detected: {labels}."
        if request.task_type == "scene_description":
            return f"Describe the scene in one or two sentences. Local CV detected: {labels}."
        return f"Describe the image and mention any important objects. Local CV detected: {labels}."

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
        if getattr(local_cv, "fallback_used", False):
            decision.reasons.append(f"local CV fallback used: {local_cv.error or 'fallback reason unavailable'}")
        remote_latency_ms = None
        remote_model = None
        remote_response_text = ""
        error = local_cv.error if decision.route == "local" and (not local_cv.ok or getattr(local_cv, "fallback_used", False)) else ""
        if decision.route == "remote":
            if self.remote_backend is None:
                remote_response_text, remote_latency_ms = mock_remote_vlm_result(
                    request.task_type, local_cv.detected_labels
                )
            else:
                prompt = request.prompt or self.default_remote_prompt(request, local_cv)
                remote_result = self.remote_backend.complete_image(
                    image_path=capture.image_path,
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                )
                remote_latency_ms = remote_result.latency_ms
                remote_model = remote_result.model
                remote_response_text = remote_result.text
                decision.remote_is_mock = False
                if remote_result.model:
                    decision.selected_model = remote_result.model
                if not remote_result.ok:
                    error = remote_result.error or "remote VLM backend error"
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
            remote_model=remote_model,
            remote_response_text=remote_response_text,
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
                model=self.local_cv_backend,
                model_dir=str(self.model_dir or ""),
                detected_labels=[],
                detections=[],
                inference_latency_ms=None,
                total_latency_ms=None,
                error=capture.error,
            )
            engine_path = str(self.yolo_engine_path or "") if self.local_cv_backend == "yolo_tensorrt_fp16" else ""
            return capture, self._annotate_local_cv(local_cv, backend=self.local_cv_backend, engine_path=engine_path)
        local_cv = self.run_local_cv(Path(capture.image_path))
        return capture, local_cv

    def run_local_cv(self, image_path: Path) -> LocalCVResult:
        if self.local_cv_backend == "mobilenet_ssd_opencv_dnn":
            result = run_mobilenet_ssd(image_path, model_dir=self.model_dir)
            return self._annotate_local_cv(result, backend="mobilenet_ssd_opencv_dnn")
        if self.local_cv_backend == "yolo_tensorrt_fp16":
            engine_path = str(self.yolo_engine_path or "")
            try:
                if self._yolo_detector is None:
                    self._yolo_detector = YoloTensorRTDetector(engine_path=self.yolo_engine_path, warmup=3)
                result = self._yolo_detector.detect(image_path)
                if result.ok:
                    return self._annotate_local_cv(result, backend="yolo_tensorrt_fp16", engine_path=engine_path)
                if not self.fallback_to_mobilenet:
                    return self._annotate_local_cv(result, backend="yolo_tensorrt_fp16", engine_path=engine_path)
                fallback = run_mobilenet_ssd(image_path, model_dir=self.model_dir)
                fallback.error = f"fallback_to_mobilenet_after_yolo_error: {result.error}"
                return self._annotate_local_cv(
                    fallback,
                    backend="yolo_tensorrt_fp16",
                    engine_path=engine_path,
                    fallback_used=True,
                )
            except Exception as exc:
                if not self.fallback_to_mobilenet:
                    result = LocalCVResult(
                        ok=False,
                        model="yolov8n_tensorrt_fp16",
                        model_dir=str(Path(self.yolo_engine_path).parent if self.yolo_engine_path else ""),
                        detected_labels=[],
                        detections=[],
                        inference_latency_ms=None,
                        total_latency_ms=None,
                        error=f"YOLO TensorRT backend unavailable: {exc}",
                    )
                    return self._annotate_local_cv(result, backend="yolo_tensorrt_fp16", engine_path=engine_path)
                fallback = run_mobilenet_ssd(image_path, model_dir=self.model_dir)
                fallback.error = f"fallback_to_mobilenet_after_yolo_unavailable: {exc}"
                return self._annotate_local_cv(
                    fallback,
                    backend="yolo_tensorrt_fp16",
                    engine_path=engine_path,
                    fallback_used=True,
                )
        result = LocalCVResult(
            ok=False,
            model=self.local_cv_backend,
            model_dir=str(self.model_dir or ""),
            detected_labels=[],
            detections=[],
            inference_latency_ms=None,
            total_latency_ms=None,
            error=f"unknown local_cv_backend: {self.local_cv_backend}",
        )
        return self._annotate_local_cv(result, backend=self.local_cv_backend)


def detections_json(result: LocalCVResult) -> str:
    return json.dumps([detection.to_dict() for detection in result.detections], ensure_ascii=False)
