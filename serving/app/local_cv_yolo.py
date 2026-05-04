from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .local_cv import Detection, LocalCVResult


COCO80_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def default_yolo_onnx_path() -> Path:
    return Path(os.environ.get("EDGE_YOLO_ONNX_MODEL", "/home/rainbow/models/vision/yolo_nano/yolov8n.onnx"))


def default_yolo_engine_path() -> Path:
    return Path(
        os.environ.get("EDGE_YOLO_TRT_ENGINE", "/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine")
    )


def _letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, float, float]:
    import cv2

    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    pad_x = (size - new_width) / 2
    pad_y = (size - new_height) / 2

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    x0 = int(round(pad_x - 0.1))
    y0 = int(round(pad_y - 0.1))
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized
    return canvas, ratio, pad_x, pad_y


def _prepare_yolo_input(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, dict[str, float]]:
    import cv2

    letterboxed, ratio, pad_x, pad_y = _letterbox(image, size)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
    return np.ascontiguousarray(tensor), {"ratio": ratio, "pad_x": pad_x, "pad_y": pad_y, "size": float(size)}


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_box + area_boxes - inter, 1e-6)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_detections: int) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        overlaps = _iou(boxes[current], boxes[order[1:]])
        order = order[1:][overlaps <= iou_threshold]
    return keep


def _reshape_yolo_output(raw_outputs: list[np.ndarray]) -> np.ndarray:
    if not raw_outputs:
        raise ValueError("YOLO produced no output tensors")
    output = np.asarray(raw_outputs[0])
    output = np.squeeze(output, axis=0) if output.ndim == 3 and output.shape[0] == 1 else np.squeeze(output)
    if output.ndim != 2:
        raise ValueError(f"unsupported YOLO output shape: {output.shape}")
    if output.shape[0] in (84, 85):
        output = output.T
    if output.shape[-1] not in (6, 84, 85):
        raise ValueError(f"unsupported YOLO detection width: {output.shape}")
    return output.astype(np.float32, copy=False)


def _postprocess_yolov8(
    raw_outputs: list[np.ndarray],
    *,
    original_width: int,
    original_height: int,
    meta: dict[str, float],
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    output = _reshape_yolo_output(raw_outputs)
    if output.shape[-1] == 6:
        boxes = output[:, :4]
        scores = output[:, 4]
        class_ids = output[:, 5].astype(np.int32)
    else:
        boxes = output[:, :4]
        if output.shape[-1] == 85:
            scores_by_class = output[:, 5:] * output[:, 4:5]
        else:
            scores_by_class = output[:, 4:]
        class_ids = np.argmax(scores_by_class, axis=1)
        scores = scores_by_class[np.arange(scores_by_class.shape[0]), class_ids]

    mask = scores >= confidence_threshold
    if not np.any(mask):
        return []
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if output.shape[-1] != 6:
        cx, cy, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes = np.column_stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))

    ratio = meta["ratio"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)

    keep = _nms(boxes, scores, iou_threshold, max_detections)
    detections: list[Detection] = []
    for index in keep:
        class_id = int(class_ids[index])
        label = COCO80_CLASSES[class_id] if 0 <= class_id < len(COCO80_CLASSES) else f"class_{class_id}"
        x1, y1, x2, y2 = boxes[index]
        detections.append(
            Detection(
                label=label,
                confidence=round(float(scores[index]), 4),
                box=[int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            )
        )
    detections.sort(key=lambda detection: detection.confidence, reverse=True)
    return detections


def _read_image(image_path: Path):
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    return image


class YoloOnnxDetector:
    def __init__(
        self,
        *,
        model_path: Path | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        providers: list[str] | None = None,
        warmup: int = 0,
    ):
        self.model_path = Path(model_path) if model_path else default_yolo_onnx_path()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.providers = providers or ["CPUExecutionProvider"]
        self.model_name = "yolov8n_onnxruntime_cpu_reuse"

        start_init = time.perf_counter()
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        self.input_meta = self.session.get_inputs()[0]
        self.output_meta = self.session.get_outputs()
        self.session_init_latency_ms = round((time.perf_counter() - start_init) * 1000, 2)
        if warmup > 0:
            self.warmup(warmup)

    def warmup(self, iterations: int = 3) -> None:
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        for _ in range(max(0, iterations)):
            self.session.run(None, {self.input_meta.name: dummy})

    def detect(self, image_path: Path) -> LocalCVResult:
        start_total = time.perf_counter()
        if not Path(image_path).exists():
            return LocalCVResult(False, self.model_name, str(self.model_path.parent), [], [], None, None, f"image not found: {image_path}")
        try:
            image = _read_image(image_path)
            original_height, original_width = image.shape[:2]
            input_tensor, meta = _prepare_yolo_input(image)
            start_infer = time.perf_counter()
            raw_outputs = self.session.run(None, {self.input_meta.name: input_tensor})
            inference_latency_ms = (time.perf_counter() - start_infer) * 1000
            detections = _postprocess_yolov8(
                raw_outputs,
                original_width=original_width,
                original_height=original_height,
                meta=meta,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                max_detections=100,
            )
        except Exception as exc:
            return LocalCVResult(False, self.model_name, str(self.model_path.parent), [], [], None, None, f"YOLO ONNX failed: {exc}")

        labels = sorted({detection.label for detection in detections}) or ["no_detection"]
        return LocalCVResult(
            ok=True,
            model=self.model_name,
            model_dir=str(self.model_path.parent),
            detected_labels=labels,
            detections=detections,
            inference_latency_ms=round(inference_latency_ms, 2),
            total_latency_ms=round((time.perf_counter() - start_total) * 1000, 2),
            error="",
        )


class _CudaRuntime:
    def __init__(self):
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
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
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

    def stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def memcpy_htod_async(self, dst: ctypes.c_void_p, src: np.ndarray, stream: ctypes.c_void_p) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                dst,
                ctypes.c_void_p(src.ctypes.data),
                src.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            ),
            "cudaMemcpyAsync H2D",
        )

    def memcpy_dtoh_async(self, dst: np.ndarray, src: ctypes.c_void_p, stream: ctypes.c_void_p) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst.ctypes.data),
                src,
                dst.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            ),
            "cudaMemcpyAsync D2H",
        )

    def synchronize(self, stream: ctypes.c_void_p) -> None:
        self.check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        if stream:
            self.lib.cudaStreamDestroy(stream)


def _preload_tensorrt_libraries() -> None:
    for path in (
        "/usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so",
        "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so.12",
    ):
        if Path(path).exists():
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


class YoloTensorRTDetector:
    def __init__(
        self,
        *,
        engine_path: Path | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        warmup: int = 0,
    ):
        self.engine_path = Path(engine_path) if engine_path else default_yolo_engine_path()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_name = "yolov8n_tensorrt_fp16"
        self.device_buffers: dict[str, ctypes.c_void_p] = {}
        self.host_outputs: dict[str, np.ndarray] = {}
        self.stream = ctypes.c_void_p()

        start_init = time.perf_counter()
        _preload_tensorrt_libraries()
        import tensorrt as trt

        self.trt = trt
        self.cuda = _CudaRuntime()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with self.engine_path.open("rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_name = ""
        self.output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        if not self.input_name:
            raise RuntimeError("TensorRT engine has no input tensor")
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(self.input_name, (1, 3, 640, 640))

        self.stream = self.cuda.stream()
        self._allocate_buffers()
        self.engine_init_latency_ms = round((time.perf_counter() - start_init) * 1000, 2)
        if warmup > 0:
            self.warmup(warmup)

    def _tensor_shape(self, name: str) -> tuple[int, ...]:
        shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
        if any(dim <= 0 for dim in shape):
            shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
        if any(dim <= 0 for dim in shape):
            raise RuntimeError(f"dynamic tensor shape is unresolved for {name}: {shape}")
        return shape

    def _allocate_buffers(self) -> None:
        input_shape = self._tensor_shape(self.input_name)
        input_nbytes = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize
        self.device_buffers[self.input_name] = self.cuda.malloc(input_nbytes)
        self.context.set_tensor_address(self.input_name, int(self.device_buffers[self.input_name].value))

        for name in self.output_names:
            shape = self._tensor_shape(name)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, dtype=dtype)
            device = self.cuda.malloc(host.nbytes)
            self.host_outputs[name] = host
            self.device_buffers[name] = device
            self.context.set_tensor_address(name, int(device.value))

    def _infer_tensor(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        self.cuda.memcpy_htod_async(self.device_buffers[self.input_name], input_tensor, self.stream)
        if not self.context.execute_async_v3(stream_handle=int(self.stream.value)):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        for name in self.output_names:
            self.cuda.memcpy_dtoh_async(self.host_outputs[name], self.device_buffers[name], self.stream)
        self.cuda.synchronize(self.stream)
        return [self.host_outputs[name].copy() for name in self.output_names]

    def warmup(self, iterations: int = 3) -> None:
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        for _ in range(max(0, iterations)):
            self._infer_tensor(dummy)

    def detect(self, image_path: Path) -> LocalCVResult:
        start_total = time.perf_counter()
        if not Path(image_path).exists():
            return LocalCVResult(False, self.model_name, str(self.engine_path.parent), [], [], None, None, f"image not found: {image_path}")
        try:
            image = _read_image(image_path)
            original_height, original_width = image.shape[:2]
            input_tensor, meta = _prepare_yolo_input(image)
            start_infer = time.perf_counter()
            raw_outputs = self._infer_tensor(input_tensor)
            inference_latency_ms = (time.perf_counter() - start_infer) * 1000
            detections = _postprocess_yolov8(
                raw_outputs,
                original_width=original_width,
                original_height=original_height,
                meta=meta,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                max_detections=100,
            )
        except Exception as exc:
            return LocalCVResult(False, self.model_name, str(self.engine_path.parent), [], [], None, None, f"YOLO TensorRT failed: {exc}")

        labels = sorted({detection.label for detection in detections}) or ["no_detection"]
        return LocalCVResult(
            ok=True,
            model=self.model_name,
            model_dir=str(self.engine_path.parent),
            detected_labels=labels,
            detections=detections,
            inference_latency_ms=round(inference_latency_ms, 2),
            total_latency_ms=round((time.perf_counter() - start_total) * 1000, 2),
            error="",
        )

    def close(self) -> None:
        if hasattr(self, "cuda"):
            for ptr in self.device_buffers.values():
                self.cuda.free(ptr)
            self.device_buffers.clear()
            self.cuda.destroy_stream(self.stream)
            self.stream = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
