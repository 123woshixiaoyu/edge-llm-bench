#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.local_cv_yolo import _prepare_yolo_input


CUDA_MEMCPY_HOST_TO_DEVICE = 1


def preload_tensorrt_libraries() -> None:
    for path in (
        "/usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so",
        "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so.12",
    ):
        if Path(path).exists():
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


class CudaRuntime:
    def __init__(self) -> None:
        candidates = [
            "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so.12",
            "/usr/local/cuda/lib64/libcudart.so.12",
            "libcudart.so.12",
        ]
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                self.lib = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                last_error = exc
        else:
            raise RuntimeError(f"failed to load CUDA runtime: {last_error}")

        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, code: int, op: str) -> None:
        if code != 0:
            message = self.lib.cudaGetErrorString(code)
            detail = message.decode("utf-8", errors="replace") if message else f"CUDA error {code}"
            raise RuntimeError(f"{op} failed: {detail}")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr

    def free(self, ptr: ctypes.c_void_p) -> None:
        if ptr:
            self.lib.cudaFree(ptr)

    def memcpy_htod(self, dst: ctypes.c_void_p, src: np.ndarray) -> None:
        self.check(
            self.lib.cudaMemcpy(
                dst,
                ctypes.c_void_p(src.ctypes.data),
                src.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy H2D",
        )


class YoloEntropyCalibrator:
    def __init__(self, trt_module, image_paths: list[Path], cache_path: Path, batch_size: int = 1):
        self.trt = trt_module
        self.image_paths = image_paths
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.index = 0
        self.cuda = CudaRuntime()
        self.batch = np.empty((batch_size, 3, 640, 640), dtype=np.float32)
        self.device_input = self.cuda.malloc(self.batch.nbytes)
        self._base = self.trt.IInt8EntropyCalibrator2.__init__
        self._base(self)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names) -> list[int] | None:
        if self.index >= len(self.image_paths):
            return None

        import cv2

        batch_paths = self.image_paths[self.index : self.index + self.batch_size]
        if len(batch_paths) < self.batch_size:
            return None

        tensors = []
        for path in batch_paths:
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"failed to read calibration image: {path}")
            tensor, _ = _prepare_yolo_input(image)
            tensors.append(tensor[0])

        self.batch[...] = np.stack(tensors, axis=0)
        self.cuda.memcpy_htod(self.device_input, np.ascontiguousarray(self.batch))
        self.index += self.batch_size
        return [int(self.device_input.value)]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)

    def __del__(self):
        try:
            self.cuda.free(self.device_input)
        except Exception:
            pass


def make_calibrator_class(trt_module):
    class Calibrator(YoloEntropyCalibrator, trt_module.IInt8EntropyCalibrator2):
        pass

    return Calibrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a YOLOv8n TensorRT INT8 engine with image calibration.")
    parser.add_argument("--onnx", type=Path, default=Path("/home/rainbow/models/vision/yolo_nano/yolov8n.onnx"))
    parser.add_argument("--engine", type=Path, default=Path("/home/rainbow/models/vision/yolo_nano/yolov8n_int8.engine"))
    parser.add_argument("--calib-dir", type=Path, default=Path("/home/rainbow/models/vision/yolo_nano/calibration_images"))
    parser.add_argument("--calib-cache", type=Path, default=Path("/home/rainbow/models/vision/yolo_nano/yolov8n_int8_calibration.cache"))
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--fp16-fallback", action="store_true", help="Allow FP16 kernels where INT8 is unavailable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preload_tensorrt_libraries()
    import tensorrt as trt

    image_paths = sorted(
        path for path in args.calib_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise SystemExit(f"no calibration images found in {args.calib_dir}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(args.onnx.read_bytes()):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise SystemExit("failed to parse ONNX model")

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.INT8)
    if args.fp16_fallback:
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
    config.add_optimization_profile(profile)

    Calibrator = make_calibrator_class(trt)
    config.int8_calibrator = Calibrator(trt, image_paths, args.calib_cache, batch_size=1)

    print(f"TensorRT version: {trt.__version__}")
    print(f"ONNX: {args.onnx}")
    print(f"INT8 engine: {args.engine}")
    print(f"Calibration images: {len(image_paths)}")
    print(f"Calibration cache: {args.calib_cache}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT returned no serialized INT8 engine")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    print(f"Wrote engine bytes: {args.engine.stat().st_size}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        "/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib",
    )
    raise SystemExit(main())
