from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from .backends.vlm import HttpVLMBackend
from .camera import CameraFrame
from .local_cv import LocalCVResult
from .schemas import VisionAnalyzeRequest, VisionAnalyzeResponse
from .vision_policy import VisionPolicyEngine
from .vision_router import VisionRequest, VisionRouter


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_vision_router(config_dir: Path) -> VisionRouter:
    policy = _read_yaml(config_dir / "policy.yaml")
    models = _read_yaml(config_dir / "models.yaml")
    vision_policy = policy.get("vision_policy", {})
    local_cv = models.get("local_cv", {})
    remote_vlm = models.get("remote_vlm", {})

    backend = str(vision_policy.get("local_cv_backend") or local_cv.get("default_backend") or "mobilenet_ssd_opencv_dnn")
    yolo_cfg = local_cv.get("yolo_tensorrt_fp16", {})
    fallback_cfg = local_cv.get("mobilenet_ssd_opencv_dnn", {})
    engine_path = Path(
        str(
            vision_policy.get("local_cv_engine_path")
            or yolo_cfg.get("engine_path")
            or "/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine"
        )
    )
    fallback_model_dir = Path(
        str(
            vision_policy.get("fallback_model_dir")
            or fallback_cfg.get("model_dir")
            or "/home/rainbow/models/vision/mobilenet_ssd"
        )
    )
    fallback_to_mobilenet = bool(vision_policy.get("fallback_to_mobilenet", False))

    remote_cfg = remote_vlm.get(str(remote_vlm.get("default_backend", "")), {})
    remote_url = str(vision_policy.get("remote_vlm_base_url") or remote_cfg.get("remote_url") or "").strip()
    remote_timeout_s = float(vision_policy.get("remote_vlm_timeout_s", 180.0))
    remote_backend = HttpVLMBackend(remote_url, timeout_s=remote_timeout_s) if remote_url else None
    remote_model = str(remote_vlm.get("default_backend") or "mock_remote_vlm")
    policy_engine = VisionPolicyEngine(
        local_cv_model="yolov8n_tensorrt_fp16" if backend == "yolo_tensorrt_fp16" else "mobilenet_ssd_voc_opencv_dnn",
        remote_vlm_model=remote_model,
        remote_available=True,
        max_local_latency_budget_ms=int(vision_policy.get("max_local_latency_budget_ms", 3000)),
    )
    return VisionRouter(
        model_dir=fallback_model_dir,
        local_cv_backend=backend,
        yolo_engine_path=engine_path,
        fallback_to_mobilenet=fallback_to_mobilenet,
        policy=policy_engine,
        remote_backend=remote_backend,
    )


def _strip_data_uri(value: str) -> str:
    if "," in value and value.split(",", 1)[0].startswith("data:"):
        return value.split(",", 1)[1]
    return value


def _frame_from_file(path: Path, *, request_id: str, source: str, latency_ms: float = 0.0) -> CameraFrame:
    resolution = None
    try:
        from PIL import Image

        with Image.open(path) as image:
            resolution = f"{image.width}x{image.height}"
    except Exception:
        resolution = None
    return CameraFrame(
        ok=path.exists() and path.stat().st_size > 0,
        image_path=str(path),
        device_path=None,
        camera_type=source,
        backend="uploaded image" if source == "upload" else "sample image",
        resolution=resolution,
        capture_latency_ms=round(latency_ms, 2),
        stable=path.exists(),
        permission_issue=False,
        video_devices=[],
        media_devices=[],
        gstreamer_available=False,
        v4l2src_available=False,
        nvarguscamerasrc_available=False,
        error="" if path.exists() else f"{source} image not found for request {request_id}",
    )


def _image_base64(path: str | Path) -> str | None:
    try:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except Exception:
        return None


def _prepare_image(
    request: VisionAnalyzeRequest,
    *,
    request_id: str,
    serving_root: Path,
) -> tuple[CameraFrame, Path | None]:
    runtime_dir = serving_root / "results/raw/interactive_gateway_frames"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if request.image_source == "upload":
        if not request.image_base64:
            path = runtime_dir / f"{request_id}_missing_upload.jpg"
            return _frame_from_file(path, request_id=request_id, source="upload"), None
        start = time.perf_counter()
        image_bytes = base64.b64decode(_strip_data_uri(request.image_base64), validate=False)
        path = runtime_dir / f"{request_id}_upload.jpg"
        path.write_bytes(image_bytes)
        latency_ms = (time.perf_counter() - start) * 1000
        return _frame_from_file(path, request_id=request_id, source="upload", latency_ms=latency_ms), path

    if request.image_source == "sample":
        sample_path = serving_root.parent / "results/figures/camera_v05_positive_detection.jpg"
        return _frame_from_file(sample_path, request_id=request_id, source="sample"), sample_path

    return CameraFrame(
        ok=False,
        image_path=str(runtime_dir / f"{request_id}_camera.jpg"),
        device_path=None,
        camera_type="camera",
        backend="pending camera capture",
        resolution=None,
        capture_latency_ms=None,
        stable=False,
        permission_issue=False,
        video_devices=[],
        media_devices=[],
        gstreamer_available=False,
        v4l2src_available=False,
        nvarguscamerasrc_available=False,
        error="",
    ), runtime_dir / f"{request_id}_camera.jpg"


def analyze_vision_request(
    *,
    request: VisionAnalyzeRequest,
    router: VisionRouter,
    serving_root: Path,
) -> tuple[VisionAnalyzeResponse, int]:
    request_id = request.request_id or str(uuid.uuid4())
    if request.use_yolo_trt and router.local_cv_backend != "yolo_tensorrt_fp16":
        router.local_cv_backend = "yolo_tensorrt_fp16"
        router.policy.local_cv_model = "yolov8n_tensorrt_fp16"

    if router.remote_backend is not None and (request.task_type in {"vqa", "scene_description"} or request.quality == "high"):
        router.policy.remote_available = router.remote_backend.available()

    if request.image_source == "camera":
        capture, local_cv = router.capture_and_analyze(image_path=serving_root / f"results/raw/interactive_gateway_frames/{request_id}_camera.jpg")
    else:
        capture, image_path = _prepare_image(request, request_id=request_id, serving_root=serving_root)
        local_cv = router.run_local_cv(image_path) if image_path and capture.ok else LocalCVResult(
            ok=False,
            model=router.local_cv_backend,
            model_dir=str(router.model_dir or ""),
            detected_labels=[],
            detections=[],
            inference_latency_ms=None,
            total_latency_ms=None,
            error=capture.error or "image preparation failed",
        )
        local_cv = router._annotate_local_cv(
            local_cv,
            backend=router.local_cv_backend,
            engine_path=str(router.yolo_engine_path or "") if router.local_cv_backend == "yolo_tensorrt_fp16" else "",
        )

    vision_request = VisionRequest(
        task_type=request.task_type,
        privacy=request.privacy,
        quality=request.quality,
        image_source="camera" if request.image_source == "camera" else "file",
        latency_budget_ms=request.latency_budget_ms,
        request_id=request_id,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
    )
    result = router.route_existing_frame(vision_request, capture=capture, local_cv=local_cv)
    status = "success"
    status_code = 200
    if result.decision.route == "reject":
        status = "rejected"
        status_code = 400
    elif result.error:
        status = "backend_error"
        status_code = 502

    response = VisionAnalyzeResponse(
        request_id=result.request_id,
        route=result.decision.route,
        selected_backend=result.decision.selected_model,
        local_cv_backend=getattr(local_cv, "local_cv_backend", router.local_cv_backend),
        local_cv_model=local_cv.model,
        remote_model=result.remote_model,
        remote_is_mock=result.decision.remote_is_mock,
        capture_metadata=result.capture.to_dict(),
        detected_labels=list(local_cv.detected_labels),
        detections=[detection.to_dict() for detection in local_cv.detections],
        remote_response_text=result.remote_response_text,
        image_base64=_image_base64(result.capture.image_path),
        capture_latency_ms=result.capture.capture_latency_ms,
        local_cv_inference_latency_ms=local_cv.inference_latency_ms,
        local_cv_total_latency_ms=local_cv.total_latency_ms,
        remote_latency_ms=result.remote_latency_ms,
        total_latency_ms=round(result.total_latency_ms, 2),
        reasons=list(result.decision.reasons),
        error=result.error or result.capture.error or local_cv.error,
        status=status,
    )
    return response, status_code
