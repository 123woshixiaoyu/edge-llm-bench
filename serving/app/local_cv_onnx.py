from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .local_cv import Detection, LocalCVResult


COCO_91_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def default_onnx_model_path() -> Path:
    return Path(
        os.environ.get(
            "EDGE_VISION_ONNX_MODEL",
            "/home/rainbow/models/vision/ssd_mobilenet_onnx/ssd_mobilenet_v1_10.onnx",
        )
    )


def _shape_dim(value: Any, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default


def _prepare_input(image, input_meta):
    import cv2
    import numpy as np

    shape = list(input_meta.shape)
    input_type = input_meta.type.lower()
    if len(shape) != 4:
        raise ValueError(f"expected 4D image input, got shape={shape}")

    nchw = bool(shape[1] == 3 or shape[1] == "3")
    if nchw:
        height = _shape_dim(shape[2], 300)
        width = _shape_dim(shape[3], 300)
    else:
        height = _shape_dim(shape[1], 300)
        width = _shape_dim(shape[2], 300)

    resized = cv2.resize(image, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if "uint8" in input_type:
        tensor = rgb.astype(np.uint8)
    else:
        tensor = rgb.astype(np.float32)
    if nchw:
        tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, axis=0)


def _find_output(outputs: dict[str, Any], *needles: str):
    for name, value in outputs.items():
        lowered = name.lower()
        if all(needle in lowered for needle in needles):
            return value
    return None


def _parse_tf_style_outputs(outputs: dict[str, Any], width: int, height: int, threshold: float) -> list[Detection]:
    import numpy as np

    boxes = _find_output(outputs, "detection_boxes")
    if boxes is None:
        boxes = _find_output(outputs, "box")
    scores = _find_output(outputs, "detection_scores")
    if scores is None:
        scores = _find_output(outputs, "score")
    classes = _find_output(outputs, "detection_classes")
    if classes is None:
        classes = _find_output(outputs, "class")
    if boxes is None or scores is None or classes is None:
        return []

    boxes = np.asarray(boxes).reshape(-1, 4)
    scores = np.asarray(scores).reshape(-1)
    classes = np.asarray(classes).reshape(-1)

    detections: list[Detection] = []
    for box, score, class_id in zip(boxes, scores, classes):
        confidence = float(score)
        if confidence < threshold:
            continue
        cid = int(class_id)
        label = COCO_91_CLASSES.get(cid, f"class_{cid}")
        y1, x1, y2, x2 = box.astype(float).tolist()
        pixel_box = [
            max(0, int(x1 * width)),
            max(0, int(y1 * height)),
            min(width, int(x2 * width)),
            min(height, int(y2 * height)),
        ]
        detections.append(Detection(label, round(confidence, 4), pixel_box))
    return detections


def _parse_detection_matrix(outputs: dict[str, Any], width: int, height: int, threshold: float) -> list[Detection]:
    import numpy as np

    for value in outputs.values():
        array = np.asarray(value)
        if array.ndim >= 2 and array.shape[-1] == 7:
            detections: list[Detection] = []
            for row in array.reshape(-1, 7):
                confidence = float(row[2])
                if confidence < threshold:
                    continue
                cid = int(row[1])
                label = COCO_91_CLASSES.get(cid, f"class_{cid}")
                x1, y1, x2, y2 = row[3:7].astype(float).tolist()
                if max(x1, y1, x2, y2) <= 1.5:
                    x1, x2 = x1 * width, x2 * width
                    y1, y2 = y1 * height, y2 * height
                detections.append(
                    Detection(
                        label,
                        round(confidence, 4),
                        [max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))],
                    )
                )
            return detections
    return []


def run_ssd_mobilenet_onnx(
    image_path: Path,
    *,
    model_path: Path | None = None,
    confidence_threshold: float = 0.2,
) -> LocalCVResult:
    start_total = time.perf_counter()
    model_path = Path(model_path) if model_path else default_onnx_model_path()
    model_name = "ssd_mobilenet_v1_onnxruntime_cpu"

    if not Path(image_path).exists():
        return LocalCVResult(False, model_name, str(model_path.parent), [], [], None, None, f"image not found: {image_path}")
    if not model_path.exists():
        return LocalCVResult(False, model_name, str(model_path.parent), [], [], None, None, f"ONNX model missing: {model_path}")

    try:
        import cv2
        import onnxruntime as ort
    except Exception as exc:
        return LocalCVResult(False, model_name, str(model_path.parent), [], [], None, None, f"ONNXRuntime import failed: {exc}")

    image = cv2.imread(str(image_path))
    if image is None:
        return LocalCVResult(False, model_name, str(model_path.parent), [], [], None, None, f"failed to read image: {image_path}")
    height, width = image.shape[:2]

    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        input_tensor = _prepare_input(image, input_meta)
        start_infer = time.perf_counter()
        raw_outputs = session.run(None, {input_meta.name: input_tensor})
        inference_latency_ms = (time.perf_counter() - start_infer) * 1000
        outputs = {meta.name: value for meta, value in zip(session.get_outputs(), raw_outputs)}
        detections = _parse_tf_style_outputs(outputs, width, height, confidence_threshold)
        if not detections:
            detections = _parse_detection_matrix(outputs, width, height, confidence_threshold)
    except Exception as exc:
        return LocalCVResult(False, model_name, str(model_path.parent), [], [], None, None, f"ONNXRuntime inference failed: {exc}")

    detections.sort(key=lambda detection: detection.confidence, reverse=True)
    labels = sorted({detection.label for detection in detections}) or ["no_detection"]
    total_latency_ms = (time.perf_counter() - start_total) * 1000
    return LocalCVResult(
        ok=True,
        model=model_name,
        model_dir=str(model_path.parent),
        detected_labels=labels,
        detections=detections,
        inference_latency_ms=round(inference_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        error="",
    )


def detections_json(detections: list[Detection]) -> str:
    return json.dumps([detection.to_dict() for detection in detections], ensure_ascii=False)
