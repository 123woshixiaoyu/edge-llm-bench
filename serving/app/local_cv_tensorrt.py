from __future__ import annotations

import os
import shutil
from pathlib import Path

from .local_cv import LocalCVResult


def default_engine_path() -> Path:
    return Path(
        os.environ.get(
            "EDGE_VISION_TRT_ENGINE",
            "/home/rainbow/models/vision/ssd_mobilenet_onnx/ssd_mobilenet_v1_10_fp16.engine",
        )
    )


def find_trtexec() -> str | None:
    return shutil.which("trtexec") or (
        "/usr/src/tensorrt/bin/trtexec" if Path("/usr/src/tensorrt/bin/trtexec").exists() else None
    )


def run_tensorrt_fp16(
    image_path: Path,
    *,
    engine_path: Path | None = None,
) -> LocalCVResult:
    model_name = "ssd_mobilenet_v1_tensorrt_fp16"
    engine_path = Path(engine_path) if engine_path else default_engine_path()
    model_dir = str(engine_path.parent)

    if not Path(image_path).exists():
        return LocalCVResult(False, model_name, model_dir, [], [], None, None, f"image not found: {image_path}")
    if find_trtexec() is None:
        return LocalCVResult(
            False,
            model_name,
            model_dir,
            [],
            [],
            None,
            None,
            "TensorRT trtexec is not installed; install TensorRT packages and rebuild the FP16 engine",
        )
    try:
        import tensorrt  # noqa: F401
    except Exception as exc:
        return LocalCVResult(False, model_name, model_dir, [], [], None, None, f"TensorRT Python import failed: {exc}")
    if not engine_path.exists():
        return LocalCVResult(False, model_name, model_dir, [], [], None, None, f"TensorRT engine missing: {engine_path}")

    return LocalCVResult(
        False,
        model_name,
        model_dir,
        [],
        [],
        None,
        None,
        "TensorRT runtime adapter is staged, but engine inference is not implemented in Phase 1",
    )
