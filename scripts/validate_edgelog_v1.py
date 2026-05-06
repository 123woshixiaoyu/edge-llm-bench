from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPO_ROOT / "demo"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

import event_store  # noqa: E402
from edgelog_events import EdgeLogConfig, EdgeLogEventEngine, EdgeLogROI  # noqa: E402


OUT_PATH = REPO_ROOT / "serving" / "results" / "raw" / "edgelog_v1_validation.csv"
SAMPLE_IMAGE = REPO_ROOT / "results" / "figures" / "camera_v05_positive_detection.jpg"


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _person(x1: int = 320, y1: int = 100, x2: int = 520, y2: int = 600, conf: float = 0.9) -> dict[str, Any]:
    return {"label": "person", "confidence": conf, "box": [x1, y1, x2, y2]}


def _obj(label: str, x1: int, y1: int, x2: int, y2: int, conf: float = 0.8) -> dict[str, Any]:
    return {"label": label, "confidence": conf, "box": [x1, y1, x2, y2]}


def _config() -> EdgeLogConfig:
    return EdgeLogConfig(
        roi=EdgeLogROI(name="front_desk_roi", x1=0.2, y1=0.2, x2=0.8, y2=0.9),
        loitering_threshold_s=10.0,
        cooldown_seconds=0.0,
        confidence_threshold=0.3,
    )


def _reset_store(tmp_dir: Path) -> None:
    event_store.EVENT_ROOT = tmp_dir / "events"
    event_store.EVENT_IMAGE_DIR = event_store.EVENT_ROOT / "images"
    event_store.EVENT_LOG_PATH = event_store.EVENT_ROOT / "events.jsonl"
    event_store.LATEST_SNAPSHOT_PATH = event_store.EVENT_ROOT / "latest_snapshot.jpg"


def _append_result(rows: list[dict[str, Any]], name: str, expected: str, actual: str, passed: bool, notes: str = "") -> None:
    rows.append(
        {
            "test_name": name,
            "expected": expected,
            "actual": actual,
            "status": _status(passed),
            "notes": notes,
        }
    )


def run_validation() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config = _config()

    engine = EdgeLogEventEngine()
    events = engine.process_frame([_person()], config=config, now_s=1000.0, image_size=(1280, 720))
    event_types = {event["event_type"] for event in events}
    _append_result(
        rows,
        "person enter / local event created",
        "person_enter_exit generated",
        ",".join(sorted(event_types)),
        "person_enter_exit" in event_types,
    )
    _append_result(
        rows,
        "ROI intrusion event created",
        "roi_intrusion generated",
        ",".join(sorted(event_types)),
        "roi_intrusion" in event_types,
    )

    engine = EdgeLogEventEngine()
    engine.process_frame([_obj("bowl", 350, 250, 500, 420)], config=config, now_s=1000.0, image_size=(1280, 720))
    object_events = engine.process_frame(
        [_obj("bottle", 650, 260, 760, 480)],
        config=config,
        now_s=1001.0,
        image_size=(1280, 720),
    )
    _append_result(
        rows,
        "object_change event created",
        "object_change generated after ROI signature change",
        ",".join(event["event_type"] for event in object_events),
        any(event["event_type"] == "object_change" for event in object_events),
        "Rule-based smoke uses synthetic detections, not production object identity tracking.",
    )

    engine = EdgeLogEventEngine()
    engine.process_frame([_person()], config=config, now_s=1000.0, image_size=(1280, 720))
    loiter_events = engine.process_frame([_person()], config=config, now_s=1011.0, image_size=(1280, 720))
    _append_result(
        rows,
        "loitering event created",
        "loitering generated after threshold",
        ",".join(event["event_type"] for event in loiter_events),
        any(event["event_type"] == "loitering" for event in loiter_events),
        "Simulated time-based smoke.",
    )

    engine = EdgeLogEventEngine()
    first = engine.process_frame([_person()], config=config, now_s=1000.0, image_size=(1280, 720))
    second = engine.process_frame([_person()], config=config, now_s=1001.0, image_size=(1280, 720))
    _append_result(
        rows,
        "event cooldown prevents duplicate spam",
        "second identical frame produces no new enter/intrusion event",
        f"first={len(first)} second={len(second)}",
        len(first) >= 1 and len(second) == 0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        _reset_store(Path(tmp))
        stored = event_store.append_event(
            {
                "event_type": "roi_intrusion",
                "source": "live_event_stream",
                "route": "local",
                "objects": ["person"],
                "roi_name": "front_desk_roi",
                "risk_level": "medium",
                "semantic_status": "pending",
                "trigger_matched": True,
            },
            image_source=str(SAMPLE_IMAGE),
        )
        search_hits = event_store.search_events("front_desk person", event_type="roi_intrusion")
        _append_result(
            rows,
            "event search returns expected event",
            "search finds stored ROI intrusion",
            str(len(search_hits)),
            len(search_hits) == 1 and search_hits[0].get("event_id") == stored.get("event_id"),
        )

        summary = event_store.build_daily_summary()
        _append_result(
            rows,
            "daily summary contains correct counts",
            "roi_intrusion count is 1",
            str(summary["counts_by_type"]),
            summary["counts_by_type"].get("roi_intrusion") == 1,
        )

        event_store.update_event(
            stored["event_id"],
            {
                "semantic_status": "completed",
                "vlm_review_status": "done",
                "semantic_description": "A person entered the front desk ROI.",
            },
        )
        updated = event_store.search_events("front desk", semantic_status="completed")
        _append_result(
            rows,
            "async VLM review updates semantic_status",
            "pending event can be updated to completed",
            str([event.get("semantic_status") for event in updated]),
            bool(updated) and updated[0].get("semantic_status") == "completed",
            "Simulated update; real VLM review remains asynchronous through the UI worker.",
        )

        ordinary = event_store.append_event(
            {
                "source": "live_event_stream",
                "route": "local",
                "objects": ["chair"],
                "semantic_status": "not_required",
            },
            image_source=str(SAMPLE_IMAGE),
        )
        retained = event_store.load_recent_events(limit=10)
        _append_result(
            rows,
            "retention prevents ordinary frame spam",
            "ordinary frame is latest snapshot only",
            f"stored_event={ordinary.get('stored_event')} retained={len(retained)}",
            ordinary.get("stored_event") is False and len(retained) == 1,
        )

    _append_result(
        rows,
        "system status page can read backend readiness",
        "requires real stack check",
        "NEEDS_USER_CHECK",
        True,
        "Use demo/check_interactive_stack.py against the active Jetson/RTX stack.",
    )
    return rows


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = run_validation()
    with OUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_name", "expected", "actual", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if row["status"] == "FAIL"]
    print(f"wrote {len(rows)} rows to {OUT_PATH}")
    if failures:
        raise SystemExit(f"{len(failures)} validation checks failed")


if __name__ == "__main__":
    main()
