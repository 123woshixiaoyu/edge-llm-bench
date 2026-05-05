from __future__ import annotations

import base64
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_ROOT = REPO_ROOT / "runtime_data" / "events"
EVENT_IMAGE_DIR = EVENT_ROOT / "images"
EVENT_LOG_PATH = EVENT_ROOT / "events.jsonl"


def _json_default(value: Any) -> str:
    return str(value)


def _image_suffix(source: Any) -> str:
    if isinstance(source, (str, Path)):
        suffix = Path(source).suffix.lower()
        return suffix if suffix in {".jpg", ".jpeg", ".png"} else ".jpg"
    name = getattr(source, "name", "")
    suffix = Path(name).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png"} else ".jpg"


def save_event_image(event_id: str, source: Any) -> str | None:
    if source is None:
        return None

    EVENT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _image_suffix(source)
    target = EVENT_IMAGE_DIR / f"{event_id}{suffix}"

    try:
        if isinstance(source, bytes):
            target.write_bytes(source)
            return str(target)

        if isinstance(source, str) and len(source) > 200:
            try:
                target.write_bytes(base64.b64decode(source))
                return str(target)
            except Exception:
                pass

        if hasattr(source, "getvalue"):
            target.write_bytes(source.getvalue())
            return str(target)

        path = Path(source)
        if path.exists() and path.is_file():
            shutil.copyfile(path, target)
            return str(target)
    except Exception:
        return None

    return None


def append_event(event: dict[str, Any], image_source: Any = None) -> dict[str, Any]:
    EVENT_ROOT.mkdir(parents=True, exist_ok=True)
    event_id = event.get("event_id") or uuid.uuid4().hex[:12]
    record = {
        "event_id": event_id,
        "timestamp": event.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        **event,
    }
    image_path = save_event_image(event_id, image_source)
    if image_path:
        record["image_path"] = image_path

    with EVENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return record


def load_recent_events(limit: int = 50, route_filter: str = "all") -> list[dict[str, Any]]:
    if not EVENT_LOG_PATH.exists():
        return []

    records: list[dict[str, Any]] = []
    with EVENT_LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if route_filter != "all" and record.get("route") != route_filter:
                continue
            records.append(record)

    return list(reversed(records[-limit:]))


def update_event(event_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    if not EVENT_LOG_PATH.exists():
        return None

    records: list[dict[str, Any]] = []
    updated: dict[str, Any] | None = None
    with EVENT_LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event_id") == event_id:
                record.update(updates)
                record["updated_at"] = datetime.now(timezone.utc).isoformat()
                updated = record
            records.append(record)

    with EVENT_LOG_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return updated


def summarize_recent_events(limit: int = 8) -> str:
    events = load_recent_events(limit=limit)
    if not events:
        return "No monitoring events have been recorded in this demo session yet."

    lines = []
    for event in events:
        labels = event.get("local_cv_labels") or []
        label_text = ", ".join(labels) if isinstance(labels, list) else str(labels)
        lines.append(
            "- "
            f"{event.get('timestamp', '')}: "
            f"{event.get('source', 'unknown')} "
            f"task={event.get('task_type', '')} "
            f"route={event.get('route', '')} "
            f"status={event.get('status', '')} "
            f"vlm_review={event.get('vlm_review_status', '')} "
            f"backend={event.get('selected_backend', '')} "
            f"labels={label_text} "
            f"error={event.get('error', '')}"
        )
    return "\n".join(lines)
