from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


EDGELOG_EVENT_TYPES = {
    "person_enter_exit",
    "roi_intrusion",
    "object_change",
    "loitering",
}


@dataclass
class EdgeLogROI:
    name: str = "watch_zone"
    x1: float = 0.25
    y1: float = 0.25
    x2: float = 0.75
    y2: float = 0.75
    normalized: bool = True


@dataclass
class EdgeLogConfig:
    roi: EdgeLogROI
    loitering_threshold_s: float = 10.0
    cooldown_seconds: float = 10.0
    object_move_threshold_px: float = 80.0
    confidence_threshold: float = 0.3


def _now_iso(now_s: float | None = None) -> str:
    now_s = time.time() if now_s is None else now_s
    return datetime.fromtimestamp(now_s, tz=timezone.utc).isoformat()


def _label(detection: dict[str, Any]) -> str:
    return str(detection.get("label", "")).strip().lower()


def _confidence(detection: dict[str, Any]) -> float:
    try:
        return float(detection.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _box(detection: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = detection.get("box") or []
    if len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def _center(detection: dict[str, Any]) -> tuple[float, float] | None:
    box = _box(detection)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _roi_bounds(roi: EdgeLogROI, image_size: tuple[int, int] | None) -> tuple[float, float, float, float]:
    if not roi.normalized:
        return roi.x1, roi.y1, roi.x2, roi.y2
    width, height = image_size or (1280, 720)
    return roi.x1 * width, roi.y1 * height, roi.x2 * width, roi.y2 * height


def _inside_roi(
    detection: dict[str, Any],
    roi: EdgeLogROI,
    image_size: tuple[int, int] | None,
) -> bool:
    center = _center(detection)
    if center is None:
        return False
    cx, cy = center
    x1, y1, x2, y2 = _roi_bounds(roi, image_size)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _top_confidence(detections: list[dict[str, Any]]) -> float | None:
    if not detections:
        return None
    return round(max(_confidence(detection) for detection in detections), 4)


def _object_signature(detections: list[dict[str, Any]], roi: EdgeLogROI, image_size: tuple[int, int] | None) -> str:
    parts: list[str] = []
    for detection in detections:
        label = _label(detection)
        if label == "person" or _confidence(detection) <= 0:
            continue
        if not _inside_roi(detection, roi, image_size):
            continue
        center = _center(detection)
        if center is None:
            continue
        cx, cy = center
        parts.append(f"{label}:{round(cx / 40) * 40}:{round(cy / 40) * 40}")
    return "|".join(sorted(parts))


def _centroid_shift(previous_signature: str, current_signature: str) -> float:
    if previous_signature == current_signature:
        return 0.0
    if not previous_signature or not current_signature:
        return math.inf
    return math.inf


class EdgeLogEventEngine:
    """Small in-memory event state machine for EdgeLog v1."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {
            "person_present": False,
            "person_present_since": None,
            "person_in_roi": False,
            "person_in_roi_since": None,
            "loitering_alerted": False,
            "object_signature": "",
            "last_fire": {},
            "active_events": {},
        }

    def reset(self) -> None:
        self.__init__()

    def _cooldown_ready(self, key: str, now_s: float, cooldown_seconds: float) -> bool:
        last_fire = float(self.state["last_fire"].get(key, 0.0))
        if now_s - last_fire < cooldown_seconds:
            return False
        self.state["last_fire"][key] = now_s
        return True

    def _event(
        self,
        event_type: str,
        *,
        now_s: float,
        status: str,
        objects: list[str],
        confidence: float | None,
        roi_name: str | None,
        risk_level: str,
        reason: str,
        start_time: str | None = None,
        duration_s: float | None = None,
    ) -> dict[str, Any]:
        timestamp = _now_iso(now_s)
        return {
            "event_type": event_type,
            "start_time": start_time or timestamp,
            "end_time": timestamp if status == "ended" else None,
            "duration_s": round(float(duration_s or 0.0), 2),
            "status": status,
            "objects": objects,
            "roi_name": roi_name,
            "confidence": confidence,
            "risk_level": risk_level,
            "semantic_status": "not_required",
            "semantic_description": "",
            "keyframe_path": None,
            "clip_path": None,
            "route": "local",
            "backend": "yolov8n_tensorrt_fp16",
            "latency_ms": None,
            "created_at": timestamp,
            "reasons": [reason],
            "trigger_matched": True,
            "alert": risk_level in {"medium", "high"},
        }

    def process_frame(
        self,
        detections: list[dict[str, Any]],
        *,
        config: EdgeLogConfig,
        now_s: float | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        now_s = time.time() if now_s is None else now_s
        events: list[dict[str, Any]] = []
        qualified = [
            detection
            for detection in detections
            if _confidence(detection) >= config.confidence_threshold
        ]
        persons = [detection for detection in qualified if _label(detection) == "person"]
        person_present = bool(persons)
        person_in_roi = any(_inside_roi(detection, config.roi, image_size) for detection in persons)

        if person_present and not self.state["person_present"]:
            self.state["person_present_since"] = now_s
            if self._cooldown_ready("person_enter", now_s, config.cooldown_seconds):
                events.append(
                    self._event(
                        "person_enter_exit",
                        now_s=now_s,
                        status="active",
                        objects=["person"],
                        confidence=_top_confidence(persons),
                        roi_name=None,
                        risk_level="low",
                        reason="person entered the camera view",
                    )
                )
        elif not person_present and self.state["person_present"]:
            start_s = float(self.state.get("person_present_since") or now_s)
            if self._cooldown_ready("person_exit", now_s, config.cooldown_seconds):
                events.append(
                    self._event(
                        "person_enter_exit",
                        now_s=now_s,
                        status="ended",
                        objects=["person"],
                        confidence=None,
                        roi_name=None,
                        risk_level="low",
                        reason="person left the camera view",
                        start_time=_now_iso(start_s),
                        duration_s=now_s - start_s,
                    )
                )
            self.state["person_present_since"] = None
        self.state["person_present"] = person_present

        if person_in_roi and not self.state["person_in_roi"]:
            self.state["person_in_roi_since"] = now_s
            self.state["loitering_alerted"] = False
            if self._cooldown_ready("roi_intrusion", now_s, config.cooldown_seconds):
                events.append(
                    self._event(
                        "roi_intrusion",
                        now_s=now_s,
                        status="active",
                        objects=["person"],
                        confidence=_top_confidence(persons),
                        roi_name=config.roi.name,
                        risk_level="medium",
                        reason=f"person entered ROI '{config.roi.name}'",
                    )
                )
        elif not person_in_roi and self.state["person_in_roi"]:
            self.state["person_in_roi_since"] = None
            self.state["loitering_alerted"] = False
        self.state["person_in_roi"] = person_in_roi

        if person_in_roi:
            since = float(self.state.get("person_in_roi_since") or now_s)
            duration = now_s - since
            if (
                duration >= config.loitering_threshold_s
                and not self.state.get("loitering_alerted")
                and self._cooldown_ready("loitering", now_s, config.cooldown_seconds)
            ):
                self.state["loitering_alerted"] = True
                events.append(
                    self._event(
                        "loitering",
                        now_s=now_s,
                        status="active",
                        objects=["person"],
                        confidence=_top_confidence(persons),
                        roi_name=config.roi.name,
                        risk_level="high",
                        reason=(
                            f"person remained in ROI '{config.roi.name}' "
                            f"for {duration:.1f}s"
                        ),
                        start_time=_now_iso(since),
                        duration_s=duration,
                    )
                )

        current_signature = _object_signature(qualified, config.roi, image_size)
        previous_signature = str(self.state.get("object_signature") or "")
        if previous_signature and current_signature != previous_signature:
            shift = _centroid_shift(previous_signature, current_signature)
            if shift >= config.object_move_threshold_px and self._cooldown_ready(
                "object_change", now_s, config.cooldown_seconds
            ):
                object_labels = sorted(
                    {
                        _label(detection)
                        for detection in qualified
                        if _label(detection) != "person"
                        and _inside_roi(detection, config.roi, image_size)
                    }
                )
                events.append(
                    self._event(
                        "object_change",
                        now_s=now_s,
                        status="ended",
                        objects=object_labels or ["object"],
                        confidence=_top_confidence(qualified),
                        roi_name=config.roi.name,
                        risk_level="medium",
                        reason=f"object signature changed inside ROI '{config.roi.name}'",
                    )
                )
        self.state["object_signature"] = current_signature
        return events
