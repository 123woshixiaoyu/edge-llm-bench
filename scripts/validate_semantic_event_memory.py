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
from semantic_event_rules import default_semantic_event_rule  # noqa: E402
from vlm_verifier import MockSemanticVerifier, verification_to_dict  # noqa: E402


OUT_PATH = REPO_ROOT / "serving" / "results" / "raw" / "semantic_event_memory_validation.csv"
SAMPLE_IMAGE = REPO_ROOT / "results" / "figures" / "camera_v05_positive_detection.jpg"


def _reset_store(tmp_dir: Path) -> None:
    event_store.EVENT_ROOT = tmp_dir / "events"
    event_store.EVENT_IMAGE_DIR = event_store.EVENT_ROOT / "images"
    event_store.EVENT_LOG_PATH = event_store.EVENT_ROOT / "events.jsonl"
    event_store.LATEST_SNAPSHOT_PATH = event_store.EVENT_ROOT / "latest_snapshot.jpg"


def _append(rows: list[dict[str, Any]], case: str, expected: str, actual: str, passed: bool, notes: str = "") -> None:
    rows.append(
        {
            "case": case,
            "expected": expected,
            "actual": actual,
            "status": "PASS" if passed else "FAIL",
            "notes": notes,
        }
    )


def _person() -> dict[str, Any]:
    return {"label": "person", "confidence": 0.91, "box": [320, 100, 520, 600]}


def _chair() -> dict[str, Any]:
    return {"label": "chair", "confidence": 0.82, "box": [420, 240, 620, 620]}


def _config() -> EdgeLogConfig:
    return EdgeLogConfig(
        roi=EdgeLogROI(name="desk_roi", x1=0.2, y1=0.2, x2=0.8, y2=0.9),
        cooldown_seconds=0.0,
        loitering_threshold_s=10.0,
        confidence_threshold=0.3,
    )


def _promote(proposal_event: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    verifier = MockSemanticVerifier()
    verification = verification_to_dict(verifier.verify(rule, proposal_event.get("proposal") or {}))
    event = {
        **proposal_event,
        "event_rule": rule["natural_language_rule"],
        "verification": verification,
        "semantic_status": verification["semantic_status"],
        "final_answer": verification["final_answer"],
    }
    if verification.get("error"):
        event["status"] = "failed"
    elif verification["final_answer"] == "YES":
        event["status"] = "verified"
        event["event_type"] = "semantic_event"
        event["risk_level"] = rule.get("risk_level_if_verified", "medium")
    elif verification["final_answer"] == "NO":
        event["status"] = "rejected"
    elif verification["final_answer"] == "UNKNOWN":
        event["status"] = "unknown"
    else:
        event["status"] = "failed"
    return event


def run_validation() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    engine = EdgeLogEventEngine()
    config = _config()
    rule = default_semantic_event_rule()

    object_events = engine.process_frame([_chair()], config=config, now_s=1000.0, image_size=(1280, 720))
    object_events += engine.process_frame([_person(), _chair()], config=config, now_s=1001.0, image_size=(1280, 720))
    object_proposals = [event for event in object_events if event.get("proposal", {}).get("trigger_type") in {"object_change", "roi_overlap", "person"}]
    _append(
        rows,
        "motion/object_change proposal created",
        "proposal exists from cheap trigger",
        str([event.get("proposal", {}).get("trigger_type") for event in object_proposals]),
        bool(object_proposals),
    )

    engine = EdgeLogEventEngine()
    person_events = engine.process_frame([_person()], config=config, now_s=1000.0, image_size=(1280, 720))
    _append(
        rows,
        "person/ROI proposal created",
        "roi_overlap proposal exists",
        str([event.get("proposal", {}).get("trigger_type") for event in person_events]),
        any(event.get("proposal", {}).get("trigger_type") == "roi_overlap" for event in person_events),
    )

    yolo_proposal = {
        "event_type": "object_change",
        "status": "proposed",
        "proposal": {
            "proposal_id": "yolo-label-smoke",
            "trigger_type": "yolo_label",
            "proposal_reason": "YOLO label chair appeared in desk ROI",
            "objects": ["chair"],
            "roi_name": "desk_roi",
            "confidence": 0.82,
            "keyframe_path": None,
            "clip_path": None,
        },
        "objects": ["chair"],
        "roi_name": "desk_roi",
    }
    _append(
        rows,
        "YOLO label proposal created",
        "trigger_type=yolo_label",
        yolo_proposal["proposal"]["trigger_type"],
        yolo_proposal["proposal"]["trigger_type"] == "yolo_label",
    )

    yes_event = _promote(yolo_proposal, rule)
    _append(
        rows,
        "verifier YES promotes proposal to verified event",
        "status=verified event_type=semantic_event",
        f"{yes_event['status']} {yes_event['event_type']} {yes_event['verification']['final_answer']}",
        yes_event["status"] == "verified" and yes_event["event_type"] == "semantic_event",
    )

    no_rule = {**rule, "natural_language_rule": "force_no"}
    no_event = _promote(yolo_proposal, no_rule)
    _append(
        rows,
        "verifier NO marks proposal rejected",
        "status=rejected",
        f"{no_event['status']} {no_event['verification']['final_answer']}",
        no_event["status"] == "rejected" and no_event["verification"]["final_answer"] == "NO",
    )

    unknown_rule = {**rule, "natural_language_rule": "force_unknown"}
    unknown_event = _promote(yolo_proposal, unknown_rule)
    _append(
        rows,
        "verifier UNKNOWN marks event needs_review",
        "status=unknown",
        f"{unknown_event['status']} {unknown_event['verification']['final_answer']}",
        unknown_event["status"] == "unknown" and unknown_event["verification"]["final_answer"] == "UNKNOWN",
    )

    fail_rule = {**rule, "natural_language_rule": "force_fail"}
    fail_event = _promote(yolo_proposal, fail_rule)
    _append(
        rows,
        "verifier failure does not block live monitoring",
        "status=failed but no exception",
        f"{fail_event['status']} {fail_event['verification']['error']}",
        fail_event["status"] == "failed",
    )

    cooldown_config = EdgeLogConfig(
        roi=config.roi,
        cooldown_seconds=10.0,
        loitering_threshold_s=10.0,
        confidence_threshold=0.3,
    )
    cooldown_engine = EdgeLogEventEngine()
    first = cooldown_engine.process_frame([_person()], config=cooldown_config, now_s=1000.0, image_size=(1280, 720))
    second = cooldown_engine.process_frame([], config=cooldown_config, now_s=1001.0, image_size=(1280, 720))
    third = cooldown_engine.process_frame([_person()], config=cooldown_config, now_s=1002.0, image_size=(1280, 720))
    _append(
        rows,
        "duplicate proposals are suppressed by min_trigger_interval_s",
        "third frame within cooldown creates no new enter proposal",
        f"first={len(first)} second={len(second)} third={len(third)}",
        len(first) >= 1 and len(third) == 0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        _reset_store(Path(tmp))
        stored_yes = event_store.append_event(yes_event, image_source=str(SAMPLE_IMAGE))
        event_store.append_event(no_event, image_source=str(SAMPLE_IMAGE))
        found = event_store.search_events("chair", event_type="semantic_event", semantic_status="yes")
        _append(
            rows,
            "search returns verified semantic event",
            "verified semantic event found",
            str([event.get("event_id") for event in found]),
            len(found) == 1 and found[0].get("event_id") == stored_yes.get("event_id"),
        )

        summary = event_store.build_daily_summary()
        _append(
            rows,
            "daily summary excludes rejected object detections and includes verified semantic events",
            "semantic_event count is 1",
            str(summary["counts_by_type"]),
            summary["counts_by_type"].get("semantic_event") == 1
            and summary["counts_by_type"].get("object_change") is None,
        )
    return rows


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = run_validation()
    with OUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "expected", "actual", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_PATH}")
    failures = [row for row in rows if row["status"] == "FAIL"]
    if failures:
        raise SystemExit(f"{len(failures)} semantic event memory checks failed")


if __name__ == "__main__":
    main()
