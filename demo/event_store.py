from __future__ import annotations

import base64
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_ROOT = REPO_ROOT / "runtime_data" / "events"
EVENT_IMAGE_DIR = EVENT_ROOT / "images"
EVENT_LOG_PATH = EVENT_ROOT / "events.jsonl"
LATEST_SNAPSHOT_PATH = EVENT_ROOT / "latest_snapshot.jpg"
STORAGE_POLICY_VERSION = "2026-05-retention-v1"
EDGELOG_SCHEMA_VERSION = "edgelog-v1"
EDGELOG_EVENT_TYPES = {
    "person_enter_exit",
    "roi_intrusion",
    "object_change",
    "loitering",
    "assistant_summary",
    "privacy_reject",
    "backend_error",
}


@dataclass(frozen=True)
class StoragePolicy:
    max_events: int = 500
    max_images_mb: int = 512
    retention_days: int = 7
    latest_snapshot_overwrite: bool = True


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def get_storage_policy() -> StoragePolicy:
    return StoragePolicy(
        max_events=_env_int("MONITORING_MAX_EVENTS", 500, minimum=1),
        max_images_mb=_env_int("MONITORING_MAX_IMAGES_MB", 512, minimum=0),
        retention_days=_env_int("MONITORING_RETENTION_DAYS", 7, minimum=1),
        latest_snapshot_overwrite=True,
    )


def _json_default(value: Any) -> str:
    return str(value)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return None


def _read_events() -> list[dict[str, Any]]:
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
            if isinstance(record, dict):
                records.append(record)
    return records


def _write_events(records: list[dict[str, Any]]) -> None:
    EVENT_ROOT.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _event_should_persist(event: dict[str, Any]) -> bool:
    if event.get("event_type") in EDGELOG_EVENT_TYPES:
        return True
    if event.get("source") in {"event_review", "monitoring_assistant"}:
        return True
    if event.get("user_save") is True:
        return True
    if event.get("trigger_matched") is True:
        return True
    if event.get("vlm_review_status") not in {None, "", "none"}:
        return True
    if event.get("route") == "reject":
        return True
    if event.get("alert") is True:
        return True
    if event.get("status") in {
        "backend_error",
        "sample fallback",
        "privacy_blocked",
        "local_alert",
        "incomplete_generation",
        "failed",
    }:
        return True
    return False


def _image_retention_reason(event: dict[str, Any]) -> str:
    if event.get("route") == "reject" and (
        event.get("privacy") == "local_only" or event.get("privacy_mode") == "local_only"
    ):
        return "privacy_reject_metadata_only"
    if event.get("user_save") is True:
        return "user_save"
    if event.get("trigger_matched") is True:
        return "triggered_event"
    if event.get("alert") is True or event.get("status") == "local_alert":
        return "local_alert"
    if event.get("vlm_review_status") not in {None, "", "none"}:
        return "vlm_review"
    if event.get("route") == "reject":
        return "policy_reject"
    if event.get("source") == "event_review":
        return "event_review"
    if event.get("source") == "monitoring_assistant":
        return "assistant_summary"
    if event.get("status") in {"backend_error", "sample fallback", "failed"}:
        return "error_or_fallback"
    return "event"


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


def save_latest_snapshot(source: Any) -> str | None:
    if source is None:
        return None
    EVENT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(source, bytes):
            LATEST_SNAPSHOT_PATH.write_bytes(source)
            return str(LATEST_SNAPSHOT_PATH)

        if isinstance(source, str) and len(source) > 200:
            try:
                LATEST_SNAPSHOT_PATH.write_bytes(base64.b64decode(source))
                return str(LATEST_SNAPSHOT_PATH)
            except Exception:
                pass

        if hasattr(source, "getvalue"):
            LATEST_SNAPSHOT_PATH.write_bytes(source.getvalue())
            return str(LATEST_SNAPSHOT_PATH)

        path = Path(source)
        if path.exists() and path.is_file():
            shutil.copyfile(path, LATEST_SNAPSHOT_PATH)
            return str(LATEST_SNAPSHOT_PATH)
    except Exception:
        return None
    return None


def _infer_sent_to_remote(event: dict[str, Any]) -> bool:
    if event.get("sent_to_remote") is not None:
        if isinstance(event.get("sent_to_remote"), str):
            return event.get("sent_to_remote", "").strip().lower() in {"true", "1", "yes"}
        return bool(event.get("sent_to_remote"))
    if event.get("privacy") == "local_only" or event.get("privacy_mode") == "local_only":
        return False
    if event.get("route") == "remote":
        return True
    if event.get("remote_latency_ms") not in {None, "", 0}:
        return True
    if event.get("vlm_review_status") in {"queued", "running", "done", "failed", "incomplete_generation"}:
        return True
    return False


def _retention_expires_at(timestamp: str, policy: StoragePolicy) -> str:
    base = _parse_time(timestamp) or datetime.now(timezone.utc)
    return (base + timedelta(days=policy.retention_days)).isoformat()


def _normalize_event_schema(
    event: dict[str, Any],
    *,
    policy: StoragePolicy,
    image_path: str | None = None,
    stored_image: bool | None = None,
    image_retention_reason: str | None = None,
) -> dict[str, Any]:
    timestamp = event.get("timestamp") or datetime.now(timezone.utc).isoformat()
    sent_to_remote = _infer_sent_to_remote(event)
    event_type = _infer_event_type(event)
    objects = event.get("objects")
    if objects is None:
        objects = event.get("local_cv_labels") or event.get("detected_labels") or []
    if isinstance(objects, str):
        objects = [part.strip() for part in objects.split(",") if part.strip()]
    semantic_status = event.get("semantic_status") or _infer_semantic_status(event)
    semantic_description = (
        event.get("semantic_description")
        or event.get("final_answer_text")
        or event.get("full_response")
        or ""
    )
    total_latency = event.get("latency_ms")
    if total_latency in {None, ""}:
        total_latency = event.get("total_latency_ms")
    record = {
        **event,
        "schema_version": EDGELOG_SCHEMA_VERSION,
        "event_type": event_type,
        "start_time": event.get("start_time") or timestamp,
        "end_time": event.get("end_time"),
        "duration_s": _safe_float(event.get("duration_s"), default=0.0),
        "status": event.get("status") or "ended",
        "objects": objects,
        "roi_name": event.get("roi_name"),
        "confidence": event.get("confidence"),
        "risk_level": event.get("risk_level") or _infer_risk_level(event),
        "semantic_status": semantic_status,
        "semantic_description": semantic_description,
        "keyframe_path": event.get("keyframe_path") or image_path or event.get("image_path"),
        "clip_path": event.get("clip_path"),
        "backend": event.get("backend") or event.get("selected_backend") or event.get("local_cv_backend") or "",
        "latency_ms": _safe_float(total_latency, default=None),
        "created_at": event.get("created_at") or timestamp,
        "timestamp": timestamp,
        "privacy_mode": event.get("privacy_mode") or event.get("privacy"),
        "sent_to_remote": sent_to_remote,
        "remote_backend": event.get("remote_backend")
        or (event.get("selected_backend") if sent_to_remote else ""),
        "retention_expires_at": event.get("retention_expires_at") or _retention_expires_at(timestamp, policy),
        "image_missing": bool(event.get("image_missing", False)),
        "storage_policy_version": STORAGE_POLICY_VERSION,
    }
    if stored_image is not None:
        record["stored_image"] = stored_image
    else:
        record["stored_image"] = bool(record.get("image_path"))
    if image_retention_reason is not None:
        record["image_retention_reason"] = image_retention_reason
    else:
        record["image_retention_reason"] = record.get("image_retention_reason") or (
            "event_image" if record.get("stored_image") else "metadata_only"
        )
    if image_path:
        record["image_path"] = image_path
        record["keyframe_path"] = record.get("keyframe_path") or image_path
        record["stored_image"] = True
    return record


def _safe_float(value: Any, *, default: float | None) -> float | None:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_event_type(event: dict[str, Any]) -> str:
    explicit = str(event.get("event_type") or "").strip()
    if explicit:
        return explicit
    if event.get("source") == "monitoring_assistant":
        return "assistant_summary"
    if event.get("route") == "reject":
        if event.get("privacy") == "local_only" or event.get("privacy_mode") == "local_only":
            return "privacy_reject"
        return "backend_error"
    if event.get("status") in {"backend_error", "sample fallback", "failed"}:
        return "backend_error"
    return "object_change"


def _infer_risk_level(event: dict[str, Any]) -> str:
    if event.get("risk_level"):
        return str(event["risk_level"])
    if event.get("route") == "reject" or event.get("status") in {"backend_error", "failed"}:
        return "medium"
    if event.get("alert") is True or event.get("trigger_matched") is True:
        return "medium"
    return "low"


def _infer_semantic_status(event: dict[str, Any]) -> str:
    if event.get("event_type") == "assistant_summary" or event.get("source") == "monitoring_assistant":
        return "completed"
    review_status = event.get("vlm_review_status")
    if review_status in {"queued", "running"}:
        return "pending"
    if review_status in {"done", "completed"}:
        return "completed"
    if review_status in {"failed", "incomplete_generation"}:
        return "failed"
    if event.get("route") == "remote" and event.get("final_answer_text"):
        return "completed"
    return "not_required"


def cleanup_storage(policy: StoragePolicy | None = None) -> None:
    policy = policy or get_storage_policy()
    try:
        records = _read_events()
        now = datetime.now(timezone.utc)

        kept: list[dict[str, Any]] = []
        for record in records:
            expires_at = _parse_time(record.get("retention_expires_at"))
            timestamp = _parse_time(record.get("timestamp")) or now
            if expires_at and expires_at < now:
                continue
            if not expires_at and timestamp + timedelta(days=policy.retention_days) < now:
                continue
            kept.append(record)

        if len(kept) > policy.max_events:
            kept = kept[-policy.max_events :]

        kept = [
            _normalize_event_schema(
                record,
                policy=policy,
                stored_image=bool(record.get("image_path")) and not record.get("image_missing", False),
                image_retention_reason=record.get("image_retention_reason"),
            )
            for record in kept
        ]

        referenced_images = {
            str(record.get("image_path"))
            for record in kept
            if record.get("image_path") and not record.get("image_missing")
        }

        EVENT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        for image in EVENT_IMAGE_DIR.iterdir():
            if not image.is_file():
                continue
            if image.resolve() == LATEST_SNAPSHOT_PATH.resolve():
                continue
            if str(image) not in referenced_images:
                try:
                    image.unlink()
                except OSError:
                    pass

        image_files = [path for path in EVENT_IMAGE_DIR.iterdir() if path.is_file()]
        image_files = [path for path in image_files if path.resolve() != LATEST_SNAPSHOT_PATH.resolve()]
        max_bytes = policy.max_images_mb * 1024 * 1024
        total_bytes = sum(path.stat().st_size for path in image_files if path.exists())

        if max_bytes >= 0 and total_bytes > max_bytes:
            for image in sorted(image_files, key=lambda path: path.stat().st_mtime if path.exists() else 0):
                if total_bytes <= max_bytes:
                    break
                try:
                    size = image.stat().st_size
                    image.unlink()
                    total_bytes -= size
                    for record in kept:
                        if record.get("image_path") == str(image):
                            record["image_missing"] = True
                            record["stored_image"] = False
                except OSError:
                    continue

        _write_events(kept)
    except Exception:
        return


def append_event(event: dict[str, Any], image_source: Any = None) -> dict[str, Any]:
    EVENT_ROOT.mkdir(parents=True, exist_ok=True)
    policy = get_storage_policy()
    event_id = event.get("event_id") or uuid.uuid4().hex[:12]
    record = {
        "event_id": event_id,
        "timestamp": event.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        **event,
    }

    if not _event_should_persist(record):
        latest_snapshot = save_latest_snapshot(image_source)
        record = _normalize_event_schema(
            record,
            policy=policy,
            stored_image=False,
            image_retention_reason="latest_snapshot_only",
        )
        record["stored_event"] = False
        if latest_snapshot:
            record["latest_snapshot_path"] = latest_snapshot
        cleanup_storage(policy)
        return record

    image_reason = _image_retention_reason(record)
    should_store_image = image_reason != "privacy_reject_metadata_only"
    image_path = save_event_image(event_id, image_source) if should_store_image else None
    record = _normalize_event_schema(
        record,
        policy=policy,
        image_path=image_path,
        stored_image=bool(image_path),
        image_retention_reason=image_reason if image_path else image_reason,
    )
    record["stored_event"] = True

    with EVENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    cleanup_storage(policy)
    return record


def load_recent_events(limit: int = 50, route_filter: str = "all") -> list[dict[str, Any]]:
    records = []
    for record in _read_events():
        if route_filter != "all" and record.get("route") != route_filter:
            continue
        records.append(record)

    return list(reversed(records[-limit:]))


def search_events(
    query: str = "",
    *,
    event_type: str = "all",
    risk_level: str = "all",
    semantic_status: str = "all",
    route_filter: str = "all",
    limit: int = 50,
) -> list[dict[str, Any]]:
    query_terms = [term.lower() for term in query.split() if term.strip()]
    matches: list[dict[str, Any]] = []
    for record in _read_events():
        if event_type != "all" and record.get("event_type") != event_type:
            continue
        if risk_level != "all" and record.get("risk_level") != risk_level:
            continue
        if semantic_status != "all" and record.get("semantic_status") != semantic_status:
            continue
        if route_filter != "all" and record.get("route") != route_filter:
            continue
        haystack = " ".join(
            [
                str(record.get("event_type", "")),
                " ".join(str(item) for item in record.get("objects", []) if item),
                str(record.get("semantic_description", "")),
                str(record.get("final_answer_text", "")),
                str(record.get("roi_name", "")),
                str(record.get("risk_level", "")),
                str(record.get("status", "")),
            ]
        ).lower()
        if query_terms and not all(term in haystack for term in query_terms):
            continue
        matches.append(record)
    return list(reversed(matches[-limit:]))


def build_daily_summary(date_prefix: str | None = None) -> dict[str, Any]:
    records = _read_events()
    if date_prefix:
        records = [
            record
            for record in records
            if str(record.get("start_time") or record.get("timestamp") or "").startswith(date_prefix)
        ]
    by_type: dict[str, int] = {}
    high_risk: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    completed_semantic: list[dict[str, Any]] = []
    timeline: list[str] = []
    for record in records:
        event_type = str(record.get("event_type") or "unknown")
        by_type[event_type] = by_type.get(event_type, 0) + 1
        if record.get("risk_level") == "high":
            high_risk.append(record)
        if record.get("semantic_status") == "pending":
            pending.append(record)
        if record.get("semantic_status") == "completed" and record.get("semantic_description"):
            completed_semantic.append(record)
        timeline.append(
            f"{record.get('start_time') or record.get('timestamp')}: "
            f"{event_type} route={record.get('route')} "
            f"risk={record.get('risk_level')} "
            f"objects={record.get('objects', [])}"
        )
    return {
        "date": date_prefix or "all retained events",
        "total_events": len(records),
        "counts_by_type": by_type,
        "high_risk_events": high_risk,
        "pending_semantic_reviews": pending,
        "completed_semantic_descriptions": completed_semantic,
        "timeline": timeline[-30:],
    }


def update_event(event_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    policy = get_storage_policy()
    records = _read_events()
    updated: dict[str, Any] | None = None
    for index, record in enumerate(records):
        if record.get("event_id") != event_id:
            continue
        record.update(updates)
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        record = _normalize_event_schema(
            record,
            policy=policy,
            stored_image=bool(record.get("image_path")) and not record.get("image_missing", False),
            image_retention_reason=record.get("image_retention_reason"),
        )
        records[index] = record
        updated = record
        break

    if updated is None:
        return None

    _write_events(records)
    cleanup_storage(policy)
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
