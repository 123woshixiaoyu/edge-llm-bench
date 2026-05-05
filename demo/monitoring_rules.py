from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class MonitoringRule:
    rule_name: str
    enabled: bool
    target_labels: list[str]
    confidence_threshold: float
    persistence_frames: int
    cooldown_seconds: float
    require_vlm_confirmation: bool
    vlm_prompt: str
    privacy: str


def detection_signature(detections: list[dict[str, Any]]) -> str:
    parts = []
    for detection in detections:
        label = str(detection.get("label", "object")).lower()
        confidence = float(detection.get("confidence") or 0.0)
        parts.append(f"{label}:{confidence:.2f}")
    return "|".join(sorted(parts))


def evaluate_rule(
    detections: list[dict[str, Any]],
    rule: MonitoringRule,
    state: dict[str, Any],
) -> dict[str, Any]:
    if not rule.enabled:
        return {"trigger_matched": False, "reason": "rule_disabled", "matches": []}

    targets = {label.strip().lower() for label in rule.target_labels if label.strip()}
    if not targets:
        return {"trigger_matched": False, "reason": "no_target_labels", "matches": []}

    matches = []
    for detection in detections:
        label = str(detection.get("label", "")).lower()
        confidence = float(detection.get("confidence") or 0.0)
        if label in targets and confidence >= rule.confidence_threshold:
            matches.append(detection)

    key = f"rule:{rule.rule_name}"
    now = time.time()
    if not matches:
        state[f"{key}:persistence"] = 0
        return {"trigger_matched": False, "reason": "no_matching_detection", "matches": []}

    persistence = int(state.get(f"{key}:persistence", 0)) + 1
    state[f"{key}:persistence"] = persistence
    if persistence < max(1, rule.persistence_frames):
        return {
            "trigger_matched": False,
            "reason": f"waiting_for_persistence:{persistence}/{rule.persistence_frames}",
            "matches": matches,
        }

    last_trigger = float(state.get(f"{key}:last_trigger", 0.0))
    if now - last_trigger < rule.cooldown_seconds:
        return {"trigger_matched": False, "reason": "cooldown_active", "matches": matches}

    state[f"{key}:last_trigger"] = now
    return {"trigger_matched": True, "reason": "rule_matched", "matches": matches}
