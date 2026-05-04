from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .local_cv import Detection, LocalCVResult


def default_worker_path() -> Path:
    return Path(os.environ.get("EDGE_YOLO_TRT_CPP_WORKER", "/home/rainbow/edge-llm-bench/cpp/yolo_trt/build/yolo_trt_worker"))


def default_engine_path() -> Path:
    return Path(os.environ.get("EDGE_YOLO_TRT_ENGINE", "/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine"))


class YoloTrtCppDetector:
    """Persistent Python wrapper around the C++ TensorRT worker process."""

    def __init__(
        self,
        *,
        worker_path: Path | None = None,
        engine_path: Path | None = None,
        startup_timeout_s: float = 20.0,
    ):
        self.worker_path = Path(worker_path) if worker_path else default_worker_path()
        self.engine_path = Path(engine_path) if engine_path else default_engine_path()
        self.model_name = "yolov8n_tensorrt_cpp_fp16"
        self.model_dir = str(self.engine_path.parent)
        self.startup_timeout_s = startup_timeout_s
        self.engine_init_latency_ms = None
        self._closed = False

        start = time.perf_counter()
        env = os.environ.copy()
        ld_paths = [
            "/usr/lib/aarch64-linux-gnu/nvidia",
            "/usr/local/cuda/targets/aarch64-linux/lib",
            env.get("LD_LIBRARY_PATH", ""),
        ]
        env["LD_LIBRARY_PATH"] = ":".join(path for path in ld_paths if path)
        self.process = subprocess.Popen(
            [str(self.worker_path), str(self.engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        ready_line = self._read_stderr_line(start)
        self.engine_init_latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if "ready" not in ready_line:
            raise RuntimeError(f"C++ TensorRT worker did not become ready: {ready_line}")

    def _read_stderr_line(self, start: float) -> str:
        assert self.process.stderr is not None
        while True:
            if time.perf_counter() - start > self.startup_timeout_s:
                raise TimeoutError("timed out waiting for C++ TensorRT worker startup")
            line = self.process.stderr.readline()
            if line:
                return line.strip()
            if self.process.poll() is not None:
                raise RuntimeError(f"C++ TensorRT worker exited with code {self.process.returncode}")

    def detect(self, image_path: Path) -> LocalCVResult:
        start_total = time.perf_counter()
        if self.process.poll() is not None:
            return LocalCVResult(
                False,
                self.model_name,
                self.model_dir,
                [],
                [],
                None,
                round((time.perf_counter() - start_total) * 1000, 2),
                f"C++ TensorRT worker is not running: code {self.process.returncode}",
            )
        try:
            assert self.process.stdin is not None
            assert self.process.stdout is not None
            self.process.stdin.write(str(image_path) + "\n")
            self.process.stdin.flush()
            raw = self.process.stdout.readline()
            if not raw:
                raise RuntimeError("C++ TensorRT worker returned no output")
            data: dict[str, Any] = json.loads(raw)
            detections = [
                Detection(
                    label=str(item.get("label", "")),
                    confidence=round(float(item.get("confidence", 0.0)), 4),
                    box=[int(value) for value in item.get("box", [])],
                )
                for item in data.get("detections", [])
            ]
            labels = [str(label) for label in data.get("detected_labels", [])] or ["no_detection"]
            return LocalCVResult(
                ok=bool(data.get("ok", False)),
                model=str(data.get("model", self.model_name)),
                model_dir=self.model_dir,
                detected_labels=labels,
                detections=detections,
                inference_latency_ms=round(float(data.get("inference_latency_ms", 0.0)), 2),
                total_latency_ms=round(float(data.get("total_latency_ms", 0.0)), 2),
                error=str(data.get("error", "")),
            )
        except Exception as exc:
            return LocalCVResult(
                False,
                self.model_name,
                self.model_dir,
                [],
                [],
                None,
                round((time.perf_counter() - start_total) * 1000, 2),
                f"C++ TensorRT wrapper failed: {exc}",
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
        except Exception:
            self.process.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
