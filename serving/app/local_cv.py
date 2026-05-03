from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass
class Detection:
    label: str
    confidence: float
    box: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocalCVResult:
    ok: bool
    model: str
    model_dir: str
    detected_labels: list[str]
    detections: list[Detection]
    inference_latency_ms: float | None
    total_latency_ms: float | None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["detections"] = [detection.to_dict() for detection in self.detections]
        return data

    def detected_labels_json(self) -> str:
        return json.dumps(self.detected_labels, ensure_ascii=False)


def default_model_dir() -> Path:
    return Path(os.environ.get("EDGE_VISION_MODEL_DIR", "/home/rainbow/models/vision/mobilenet_ssd"))


def mobilenet_ssd_files(model_dir: Path) -> tuple[Path, Path]:
    return model_dir / "deploy.prototxt", model_dir / "mobilenet_iter_73000.caffemodel"


def run_mobilenet_ssd(
    image_path: Path,
    *,
    model_dir: Path | None = None,
    confidence_threshold: float = 0.2,
) -> LocalCVResult:
    start_total = time.perf_counter()
    model_dir = Path(model_dir) if model_dir else default_model_dir()
    proto_path, weights_path = mobilenet_ssd_files(model_dir)
    model_name = "mobilenet_ssd_voc_opencv_dnn"

    if not Path(image_path).exists():
        return LocalCVResult(False, model_name, str(model_dir), [], [], None, None, f"image not found: {image_path}")
    if not proto_path.exists() or not weights_path.exists():
        return LocalCVResult(
            False,
            model_name,
            str(model_dir),
            [],
            [],
            None,
            None,
            f"missing MobileNet-SSD files under {model_dir}",
        )

    try:
        import cv2
        import numpy as np
    except Exception as exc:
        return LocalCVResult(False, model_name, str(model_dir), [], [], None, None, f"opencv import failed: {exc}")

    image = cv2.imread(str(image_path))
    if image is None:
        return LocalCVResult(False, model_name, str(model_dir), [], [], None, None, f"failed to read image: {image_path}")

    try:
        net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        start_infer = time.perf_counter()
        net.setInput(blob)
        output = net.forward()
        inference_latency_ms = (time.perf_counter() - start_infer) * 1000
    except Exception as exc:
        return LocalCVResult(False, model_name, str(model_dir), [], [], None, None, f"OpenCV DNN failed: {exc}")

    height, width = image.shape[:2]
    detections: list[Detection] = []
    for row in output.reshape(-1, 7):
        confidence = float(row[2])
        class_id = int(row[1])
        if confidence < confidence_threshold or class_id <= 0 or class_id >= len(VOC_CLASSES):
            continue
        x1, y1, x2, y2 = (row[3:7] * np.array([width, height, width, height])).astype("int")
        box = [
            max(0, int(x1)),
            max(0, int(y1)),
            min(width, int(x2)),
            min(height, int(y2)),
        ]
        detections.append(Detection(VOC_CLASSES[class_id], round(confidence, 4), box))

    detections.sort(key=lambda detection: detection.confidence, reverse=True)
    labels = sorted({detection.label for detection in detections})
    if not labels:
        labels = ["no_detection"]
    total_latency_ms = (time.perf_counter() - start_total) * 1000
    return LocalCVResult(
        ok=True,
        model=model_name,
        model_dir=str(model_dir),
        detected_labels=labels,
        detections=detections,
        inference_latency_ms=round(inference_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        error="",
    )
