from __future__ import annotations

import base64
import os
import queue
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st

from edgelog_events import EdgeLogConfig, EdgeLogEventEngine, EdgeLogROI
from event_store import (
    append_event,
    build_daily_summary,
    get_storage_policy,
    load_recent_events,
    search_events,
    summarize_recent_events,
    update_event,
)
from monitoring_rules import MonitoringRule, detection_signature, evaluate_rule
from output_validation import structured_vlm_prompt
from sample_results import (
    SAMPLE_IMAGE,
    backend_status,
    call_text_backend,
    call_vision_backend,
    draw_detections,
    real_backend_health,
    select_text_sample,
    select_vision_sample,
)


st.set_page_config(
    page_title="EdgeLog",
    layout="wide",
)


DEFAULT_GATEWAY_URL = "http://192.168.1.102:8000"
RULE_WIDGET_KEYS = {
    "enabled": "rule_enabled",
    "watch_label": "rule_watch_label",
    "confidence_threshold": "rule_confidence_threshold",
    "persistence_frames": "rule_persistence_frames",
    "cooldown_seconds": "rule_cooldown_seconds",
    "require_vlm_confirmation": "rule_require_vlm_confirmation",
    "privacy": "rule_privacy",
    "vlm_prompt": "rule_vlm_prompt",
}


def default_gateway_url() -> str:
    return os.environ.get("EDGE_GATEWAY_URL", DEFAULT_GATEWAY_URL)


def get_default_monitoring_rule_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "watch_label": "person",
        "confidence_threshold": 0.30,
        "persistence_frames": 1,
        "cooldown_seconds": 10,
        "require_vlm_confirmation": False,
        "privacy": "allow_remote",
        "vlm_prompt": "Does this image contain the watched object? Answer JSON only.",
    }


def ensure_monitoring_rule_config() -> dict[str, Any]:
    if "monitoring_rule_config" not in st.session_state:
        st.session_state["monitoring_rule_config"] = get_default_monitoring_rule_config()
    return st.session_state["monitoring_rule_config"]


def ensure_edgelog_engine() -> EdgeLogEventEngine:
    if "edgelog_event_engine" not in st.session_state:
        st.session_state["edgelog_event_engine"] = EdgeLogEventEngine()
    return st.session_state["edgelog_event_engine"]


def sync_rule_widgets_from_config(config: dict[str, Any]) -> None:
    for field, widget_key in RULE_WIDGET_KEYS.items():
        st.session_state[widget_key] = config[field]


def read_rule_widgets() -> dict[str, Any]:
    config = get_default_monitoring_rule_config()
    for field, widget_key in RULE_WIDGET_KEYS.items():
        if widget_key in st.session_state:
            config[field] = st.session_state[widget_key]
    config["confidence_threshold"] = float(config["confidence_threshold"])
    config["persistence_frames"] = int(config["persistence_frames"])
    config["cooldown_seconds"] = int(config["cooldown_seconds"])
    config["enabled"] = bool(config["enabled"])
    config["require_vlm_confirmation"] = bool(config["require_vlm_confirmation"])
    return config


def mode_label(use_sample_mode: bool) -> str:
    return "sample mode" if use_sample_mode else "real backend mode with sample fallback"


def _as_float(value: Any) -> float | None:
    if value in (None, "", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _image_for_result(result: dict[str, Any], fallback: Any) -> Any:
    if result.get("image_base64"):
        try:
            return BytesIO(base64.b64decode(result["image_base64"]))
        except Exception:
            return fallback
    return fallback


def _image_source_for_event(result: dict[str, Any], fallback: Any) -> Any:
    return result.get("image_base64") or fallback


def _labels(result: dict[str, Any]) -> list[Any]:
    return result.get("local_cv_precheck_labels") or result.get("detected_labels") or []


def _detections(result: dict[str, Any]) -> list[dict[str, Any]]:
    return result.get("local_cv_precheck_detections") or result.get("detections") or []


def _event_from_result(
    *,
    source: str,
    result: dict[str, Any],
    task_type: str,
    privacy: str,
    prompt: str = "",
) -> dict[str, Any]:
    return {
        "event_type": result.get("event_type"),
        "start_time": result.get("start_time"),
        "end_time": result.get("end_time"),
        "duration_s": result.get("duration_s"),
        "objects": result.get("objects"),
        "roi_name": result.get("roi_name"),
        "confidence": result.get("confidence"),
        "risk_level": result.get("risk_level"),
        "semantic_status": result.get("semantic_status"),
        "semantic_description": result.get("semantic_description"),
        "keyframe_path": result.get("keyframe_path"),
        "clip_path": result.get("clip_path"),
        "backend": result.get("backend"),
        "latency_ms": result.get("latency_ms"),
        "source": source,
        "prompt": prompt,
        "task_type": task_type,
        "privacy": privacy,
        "status": result.get("generation_status") or result.get("status") or "success",
        "frame_id": result.get("frame_id"),
        "detection_changed": result.get("detection_changed"),
        "trigger_matched": result.get("trigger_matched"),
        "vlm_review_status": result.get("vlm_review_status", "none"),
        "route": result.get("route"),
        "selected_backend": result.get("selected_backend") or result.get("local_cv_backend"),
        "local_cv_labels": _labels(result),
        "local_cv_latency_ms": _as_float(result.get("local_cv_inference_latency_ms")),
        "remote_latency_ms": _as_float(result.get("remote_latency_ms")),
        "total_latency_ms": _as_float(result.get("total_latency_ms")),
        "sent_to_remote": result.get("sent_to_remote"),
        "remote_backend": result.get("remote_backend")
        or (result.get("selected_backend") if result.get("route") == "remote" else ""),
        "alert": result.get("alert"),
        "user_save": result.get("user_save"),
        "final_answer_text": result.get("final_answer_text")
        or result.get("full_response")
        or result.get("response_preview")
        or "",
        "raw_output": result.get("raw_output", ""),
        "structured_answer": result.get("structured_answer"),
        "reasons": result.get("reasons") or [],
        "error": result.get("error") or "",
        "mode": result.get("mode"),
    }


def _record_event(
    *,
    source: str,
    result: dict[str, Any],
    task_type: str,
    privacy: str,
    prompt: str = "",
    image_source: Any = None,
    user_save: bool = False,
) -> dict[str, Any]:
    result["user_save"] = user_save
    event = append_event(
        _event_from_result(
            source=source,
            result=result,
            task_type=task_type,
            privacy=privacy,
            prompt=prompt,
        ),
        image_source=image_source,
    )
    if event.get("stored_event"):
        st.session_state["latest_event"] = event
    if event.get("image_path"):
        st.session_state["latest_image_path"] = event["image_path"]
    elif event.get("latest_snapshot_path"):
        st.session_state["latest_image_path"] = event["latest_snapshot_path"]
    return event


def _ensure_review_worker(api_base_url: str) -> queue.Queue:
    worker_key = "vlm_review_worker"
    queue_key = "vlm_review_queue"
    if (
        worker_key in st.session_state
        and st.session_state[worker_key].is_alive()
        and st.session_state.get("vlm_review_api_base_url") == api_base_url
    ):
        return st.session_state[queue_key]

    review_queue: queue.Queue = queue.Queue()

    def worker() -> None:
        while True:
            job = review_queue.get()
            if job is None:
                break
            event_id = job["event_id"]
            update_event(event_id, {"vlm_review_status": "running"})
            try:
                result = call_vision_backend(
                    api_base_url,
                    image_source="upload",
                    image_path=job["image_path"],
                    task_type="vqa",
                    privacy="allow_remote",
                    quality="high",
                    latency_budget_ms=15000,
                    prompt=job["prompt"],
                    max_tokens=job["max_tokens"],
                    timeout_s=260,
                )
                status = "done"
                if result.get("generation_status") == "incomplete_generation":
                    status = "incomplete_generation"
                elif result.get("route") != "remote" or result.get("error"):
                    status = "failed"
                update_event(
                    event_id,
                    {
                        "vlm_review_status": status,
                        "semantic_status": "completed" if status == "done" else "failed",
                        "semantic_description": result.get("final_answer_text") or result.get("full_response") or "",
                        "route": result.get("route"),
                        "remote_latency_ms": _as_float(result.get("remote_latency_ms")),
                        "total_latency_ms": _as_float(result.get("total_latency_ms")),
                        "final_answer_text": result.get("final_answer_text") or result.get("full_response") or "",
                        "raw_output": result.get("raw_output", ""),
                        "structured_answer": result.get("structured_answer"),
                        "reasons": result.get("reasons", []),
                        "error": result.get("error", ""),
                        "selected_backend": result.get("selected_backend"),
                        "sent_to_remote": result.get("route") == "remote",
                        "remote_backend": result.get("selected_backend") if result.get("route") == "remote" else "",
                        "status": status,
                        "mode": result.get("mode"),
                    },
                )
            except Exception as exc:
                update_event(
                    event_id,
                    {
                        "vlm_review_status": "failed",
                        "semantic_status": "failed",
                        "status": "failed",
                        "error": f"VLM review worker failed: {exc}",
                    },
                )
            finally:
                review_queue.task_done()

    thread = threading.Thread(target=worker, name="vlm-review-worker", daemon=True)
    thread.start()
    st.session_state[queue_key] = review_queue
    st.session_state[worker_key] = thread
    st.session_state["vlm_review_api_base_url"] = api_base_url
    return review_queue


def _select_image_source(label: str, *, default_camera: bool, key_prefix: str) -> tuple[str, Any]:
    options = ["sample image", "upload image", "Jetson camera"]
    index = 2 if default_camera else 0
    image_source = st.selectbox(label, options, index=index, key=f"{key_prefix}_image_source")
    image_file = None
    if image_source == "upload image":
        image_file = st.file_uploader("Upload a snapshot", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_upload")
    if image_source == "Jetson camera":
        return image_source, None
    return image_source, image_file if image_file is not None else str(SAMPLE_IMAGE)


def _display_local_precheck(result: dict[str, Any], image_payload: Any, caption: str) -> None:
    detections = _detections(result)
    labels = _labels(result)
    annotated = draw_detections(image_payload, detections)

    st.markdown("#### Local CV Precheck (Jetson)")
    st.caption(
        "The local YOLO TensorRT detector is the fast monitoring path. "
        "When the final route is remote, these boxes remain edge-side pre-analysis."
    )
    left, right = st.columns([1, 1])
    with left:
        if annotated is not None:
            st.image(annotated, caption=caption, use_container_width=True)
        else:
            st.warning("No snapshot image is available for display.")
    with right:
        metric_cols = st.columns(2)
        metric_cols[0].metric("Capture latency ms", result.get("capture_latency_ms") or "n/a")
        metric_cols[1].metric("YOLO inference ms", result.get("local_cv_inference_latency_ms") or "n/a")
        st.caption(
            "Capture latency is camera/frame acquisition plus the gateway capture pipeline. "
            "YOLO inference latency is the TensorRT model execution time. In snapshot mode, "
            "capture can be around 1s while YOLO inference is usually around 10-30ms."
        )
        st.write(f"Detected labels: {', '.join(str(label) for label in labels) if labels else 'none'}")
        st.json(
            {
                "local_cv_backend": result.get("local_cv_backend"),
                "local_cv_model": result.get("local_cv_model"),
                "detections": detections,
                "capture_latency_ms": result.get("capture_latency_ms"),
                "local_cv_total_latency_ms": result.get("local_cv_total_latency_ms"),
            }
        )


def _display_routing_decision(result: dict[str, Any]) -> None:
    st.markdown("#### Routing Decision")
    route_col, backend_col, latency_col = st.columns(3)
    route_col.metric("Route", result.get("route", "unknown"))
    backend_col.metric("Selected backend", result.get("selected_backend") or result.get("local_cv_backend") or "none")
    latency_col.metric("Total latency ms", result.get("total_latency_ms") or "n/a")
    st.json(
        {
            "mode": result.get("mode"),
            "remote_is_mock": result.get("remote_is_mock"),
            "remote_latency_ms": result.get("remote_latency_ms"),
            "status": result.get("status"),
            "error": result.get("error"),
            "reasons": result.get("reasons"),
        }
    )


def _display_final_answer(result: dict[str, Any]) -> None:
    route = result.get("route")
    st.markdown("#### Final Routed Answer")
    if result.get("generation_status") == "incomplete_generation" or result.get("status") == "incomplete_generation":
        st.error("Model did not produce a final answer before the output limit.")
        with st.expander("Debug raw model output"):
            st.text_area("Raw output", value=result.get("raw_output", ""), height=220)
        return
    if route == "local":
        st.success("Final answer source: Jetson local CV")
        st.write("Detection/classification was completed locally on Jetson.")
        st.json({"detected_labels": _labels(result), "detections": _detections(result)})
    elif route == "remote":
        st.info("Final answer source: RTX remote VLM")
        st.write("The semantic answer below comes from the remote VLM. YOLO boxes above are local precheck evidence.")
        text = result.get("full_response") or result.get("final_answer_text") or result.get("remote_response_text", "")
        st.text_area("Remote semantic answer", value=text, height=220)
    elif route == "reject":
        st.warning("Final answer source: policy reject")
        st.write("No model answer was produced.")
        st.json({"reject_reasons": result.get("reasons", [])})
    else:
        st.write(result.get("final_answer_text") or result.get("response_preview") or "")


def render_header() -> tuple[bool, str]:
    ensure_monitoring_rule_config()
    st.title("EdgeLog")
    st.caption(
        "Local-first video event memory box for semantic search and daily summaries."
    )
    st.write(
        "Jetson turns a fixed camera stream into structured events and keyframes. "
        "RTX backends asynchronously add semantic descriptions and daily summaries only when privacy allows."
    )

    left, right = st.columns([1, 2])
    with left:
        use_sample_mode = st.toggle("Use sample results", value=True)
    with right:
        api_base_url = st.text_input("Router API base URL", value=default_gateway_url())
        st.caption("Streamlit itself runs locally, but real backend mode should point to the Jetson Gateway.")
    st.info(f"Current mode: {mode_label(use_sample_mode)}")
    return use_sample_mode, api_base_url


def _run_live_monitor_capture(
    *,
    use_sample_mode: bool,
    api_base_url: str,
    image_source: str,
    image_payload: Any,
    task_type: str,
    privacy: str,
    quality: str,
    latency_budget_ms: int,
    max_tokens: int,
    rule_enabled: bool,
    watch_label: str,
    confidence_threshold: float,
    persistence_frames: int,
    cooldown_seconds: float,
    require_vlm_confirmation: bool,
    rule_privacy: str,
    vlm_prompt: str,
    roi_name: str,
    roi_x1: float,
    roi_y1: float,
    roi_x2: float,
    roi_y2: float,
    loitering_threshold_s: float,
    user_save: bool,
) -> None:
    if use_sample_mode:
        result = select_vision_sample(task_type, privacy, quality)
        image_for_display = str(SAMPLE_IMAGE)
    else:
        backend_source = "camera" if image_source == "Jetson camera" else "upload"
        with st.spinner("Calling Jetson Gateway for local monitoring detection."):
            result = call_vision_backend(
                api_base_url,
                image_source=backend_source,
                image_path=image_payload or str(SAMPLE_IMAGE),
                task_type=task_type,
                privacy=privacy,
                quality=quality,
                latency_budget_ms=int(latency_budget_ms),
                prompt="",
                max_tokens=int(max_tokens),
            )
        image_for_display = _image_for_result(result, image_payload or str(SAMPLE_IMAGE))

    frame_id = int(st.session_state.get("live_frame_id", 0)) + 1
    st.session_state["live_frame_id"] = frame_id
    signature = detection_signature(_detections(result))
    previous_signature = st.session_state.get("last_detection_signature")
    detection_changed = signature != previous_signature
    st.session_state["last_detection_signature"] = signature

    rule = MonitoringRule(
        rule_name="live_monitor_watch_label",
        enabled=rule_enabled,
        target_labels=[label.strip() for label in watch_label.split(",")],
        confidence_threshold=float(confidence_threshold),
        persistence_frames=int(persistence_frames),
        cooldown_seconds=float(cooldown_seconds),
        require_vlm_confirmation=bool(require_vlm_confirmation),
        vlm_prompt=structured_vlm_prompt(vlm_prompt),
        privacy=rule_privacy,
    )
    rule_state = st.session_state.setdefault("monitoring_rule_state", {})
    rule_result = evaluate_rule(_detections(result), rule, rule_state)
    engine = ensure_edgelog_engine()
    edge_events = engine.process_frame(
        _detections(result),
        config=EdgeLogConfig(
            roi=EdgeLogROI(
                name=roi_name,
                x1=float(roi_x1),
                y1=float(roi_y1),
                x2=float(roi_x2),
                y2=float(roi_y2),
                normalized=True,
            ),
            loitering_threshold_s=float(loitering_threshold_s),
            cooldown_seconds=float(cooldown_seconds),
            confidence_threshold=float(confidence_threshold),
        ),
    )
    primary_edge_event = edge_events[0] if edge_events else None
    result["frame_id"] = frame_id
    result["detection_changed"] = detection_changed
    result["trigger_matched"] = bool(rule_result["trigger_matched"] or primary_edge_event)
    result["vlm_review_status"] = "none"
    result["edge_events"] = edge_events

    if result.get("mode") == "sample fallback":
        st.warning("Real backend unavailable; showing committed sample fallback.")

    if primary_edge_event:
        result.update(primary_edge_event)
        st.warning(f"EdgeLog event: {primary_edge_event['event_type']} in {primary_edge_event.get('roi_name') or 'scene'}.")
    elif rule_result["trigger_matched"]:
        result.update(
            {
                "event_type": "object_change",
                "start_time": None,
                "end_time": None,
                "duration_s": 0.0,
                "status": "ended",
                "objects": [label.strip() for label in watch_label.split(",") if label.strip()],
                "roi_name": roi_name,
                "confidence": max(
                    [float(match.get("confidence") or 0.0) for match in rule_result.get("matches", [])] or [0.0]
                ),
                "risk_level": "medium",
                "semantic_status": "not_required",
                "semantic_description": "",
                "clip_path": None,
                "backend": "yolov8n_tensorrt_fp16",
                "latency_ms": _as_float(result.get("total_latency_ms")),
            }
        )
        st.warning("Candidate Event: trigger matched local detection rule.")
    if result["trigger_matched"]:
        if require_vlm_confirmation and rule_privacy == "allow_remote":
            result["vlm_review_status"] = "queued"
            result["semantic_status"] = "pending"
            result["sent_to_remote"] = True
            result["final_answer_text"] = "Candidate event queued for workstation VLM review."
        elif require_vlm_confirmation and rule_privacy == "local_only":
            result["vlm_review_status"] = "failed"
            result["event_type"] = "privacy_reject"
            result["semantic_status"] = "failed"
            result["route"] = "reject"
            result["status"] = "ended"
            result["sent_to_remote"] = False
            result["final_answer_text"] = (
                "Privacy Blocked: semantic review would require sending the image to the remote workstation."
            )
            result["reasons"] = [
                "privacy=local_only blocks remote semantic review for candidate event",
                *result.get("reasons", []),
            ]
        else:
            result["alert"] = True
            result["semantic_status"] = result.get("semantic_status") or "not_required"
            result["sent_to_remote"] = False
            result["final_answer_text"] = "Local alert generated from YOLO TensorRT detection rule."
    else:
        result["sent_to_remote"] = False
        st.caption(f"No candidate event triggered: {rule_result['reason']}")

    _display_local_precheck(result, image_for_display, "Current snapshot with local YOLO monitoring boxes")
    _display_routing_decision(result)
    st.success("Route/location: Processed on Jetson" if result.get("route") == "local" else "Route/location recorded by policy")

    event = _record_event(
        source="live_monitor",
        result=result,
        task_type=task_type,
        privacy=privacy,
        image_source=_image_source_for_event(result, image_payload or str(SAMPLE_IMAGE)),
        user_save=user_save,
    )
    st.session_state["last_live_monitor_result"] = result
    st.session_state["last_monitor_refresh_at"] = time.monotonic()

    if rule_result["trigger_matched"] and require_vlm_confirmation and rule_privacy == "allow_remote":
        if event.get("image_path"):
            review_queue = _ensure_review_worker(api_base_url)
            review_queue.put(
                {
                    "event_id": event["event_id"],
                    "image_path": event["image_path"],
                    "prompt": rule.vlm_prompt,
                    "max_tokens": int(max_tokens),
                }
            )
            st.info("VLM review queued. Live Monitor can continue refreshing while review runs.")
        else:
            update_event(
                event["event_id"],
                {
                    "vlm_review_status": "failed",
                    "status": "failed",
                    "error": "candidate event image was not available for VLM review",
                },
            )
    if event.get("stored_event"):
        st.caption(f"Recorded event {event['event_id']} in Event History.")
    else:
        st.caption("Updated latest snapshot only; this ordinary refresh was not retained as a long-term event.")


def render_live_monitor(use_sample_mode: bool, api_base_url: str) -> None:
    st.subheader("Live Event Stream")
    st.write("Turn camera snapshots into local EdgeLog events with Jetson YOLO TensorRT and simple state rules.")
    st.caption(
        "Auto refresh updates the latest snapshot only. Long-term history stores alerts, triggered events, "
        "reviews, rejects, errors/fallbacks, assistant summaries, and user-saved snapshots."
    )
    if "vlm_review_queue" in st.session_state:
        st.caption(f"VLM review pending: {st.session_state['vlm_review_queue'].qsize()}")

    monitoring_running = bool(st.session_state.get("monitoring_running", False))
    st.session_state["monitor_page_active"] = True

    left, right = st.columns([2, 1])
    with left:
        image_source, image_payload = _select_image_source(
            "Snapshot source",
            default_camera=not use_sample_mode,
            key_prefix="live",
        )
        interval_s = st.number_input(
            "Refresh interval seconds",
            min_value=1,
            max_value=30,
            value=2,
            step=1,
            key="live_refresh_interval",
        )
        control_cols = st.columns(2)
        with control_cols[0]:
            if not monitoring_running:
                if st.button("Start Monitoring", type="primary"):
                    st.session_state["monitoring_running"] = True
                    st.session_state["last_monitor_refresh_at"] = 0.0
                    st.rerun()
            else:
                if st.button("Stop Monitoring", type="secondary"):
                    st.session_state["monitoring_running"] = False
                    st.rerun()
        with control_cols[1]:
            capture_once = st.button("Capture Once")
        if monitoring_running:
            st.success("Event monitoring is running on this page.")
        else:
            st.info("Event monitoring is stopped. Use Capture Once or Start Monitoring.")
        if not hasattr(st, "fragment"):
            st.caption(
                "This Streamlit version does not support non-blocking auto refresh fragments. "
                "Use Capture Once for manual snapshots."
            )
    with right:
        with st.expander("Advanced routing settings"):
            task_type = st.selectbox("Local task", ["detect", "classify"], index=0, key="live_task")
            privacy = st.selectbox("Privacy", ["allow_remote", "local_only"], index=0, key="live_privacy")
            quality = st.selectbox("Quality", ["low", "medium", "high"], index=0, key="live_quality")
            latency_budget_ms = st.number_input(
                "Latency budget ms",
                min_value=100,
                max_value=60000,
                value=3000,
                step=100,
                key="live_latency",
            )
            max_tokens = st.number_input(
                "Max output tokens",
                min_value=32,
                max_value=2048,
                value=1024,
                step=32,
                key="live_tokens",
            )
        with st.expander("EdgeLog event rules"):
            active_rule = ensure_monitoring_rule_config()
            if st.session_state.pop("rule_sync_widgets", False) or any(
                key not in st.session_state for key in RULE_WIDGET_KEYS.values()
            ):
                sync_rule_widgets_from_config(active_rule)
            if st.session_state.get("rule_status_message"):
                st.success(st.session_state.pop("rule_status_message"))
            st.caption(
                "Edit values, then click Save Rule to apply them to event monitoring. "
                "Current active rule is used for Capture Once and Start Monitoring."
            )
            st.info(
                "Current active rule: "
                f"label={active_rule['watch_label']}, "
                f"confidence>={active_rule['confidence_threshold']:.2f}, "
                f"VLM confirmation={active_rule['require_vlm_confirmation']}, "
                f"privacy={active_rule['privacy']}."
            )
            st.checkbox("Enable candidate event trigger", key="rule_enabled")
            st.text_input("Watch label", key="rule_watch_label")
            st.slider("Confidence threshold", min_value=0.0, max_value=1.0, step=0.05, key="rule_confidence_threshold")
            st.number_input("Persistence frames", min_value=1, max_value=10, step=1, key="rule_persistence_frames")
            st.number_input("Cooldown seconds", min_value=0, max_value=300, step=5, key="rule_cooldown_seconds")
            roi_name = st.text_input("ROI name", value="watch_zone", key="edgelog_roi_name")
            roi_cols = st.columns(4)
            roi_x1 = roi_cols[0].number_input("ROI x1", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
            roi_y1 = roi_cols[1].number_input("ROI y1", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
            roi_x2 = roi_cols[2].number_input("ROI x2", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
            roi_y2 = roi_cols[3].number_input("ROI y2", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
            loitering_threshold_s = st.number_input(
                "Loitering threshold seconds",
                min_value=2,
                max_value=600,
                value=10,
                step=1,
            )
            st.checkbox("Require VLM confirmation", key="rule_require_vlm_confirmation")
            st.radio("Review privacy", ["allow_remote", "local_only"], horizontal=True, key="rule_privacy")
            st.text_area(
                "VLM confirmation prompt",
                height=80,
                key="rule_vlm_prompt",
            )
            rule_action_cols = st.columns(2)
            with rule_action_cols[0]:
                if st.button("Save Rule"):
                    st.session_state["monitoring_rule_config"] = read_rule_widgets()
                    st.session_state["rule_status_message"] = "Rule saved"
                    st.rerun()
            with rule_action_cols[1]:
                if st.button("Reset to Default"):
                    default_rule = get_default_monitoring_rule_config()
                    st.session_state["monitoring_rule_config"] = default_rule
                    st.session_state["rule_sync_widgets"] = True
                    st.session_state["monitoring_rule_state"] = {}
                    ensure_edgelog_engine().reset()
                    st.session_state["last_detection_signature"] = None
                    st.session_state["rule_status_message"] = "Rule reset to default"
                    st.rerun()
            user_save = st.checkbox("Save this snapshot as event", value=False)

    def capture() -> None:
        _run_live_monitor_capture(
            use_sample_mode=use_sample_mode,
            api_base_url=api_base_url,
            image_source=image_source,
            image_payload=image_payload,
            task_type=task_type,
            privacy=privacy,
            quality=quality,
            latency_budget_ms=int(latency_budget_ms),
            max_tokens=int(max_tokens),
            rule_enabled=bool(active_rule["enabled"]),
            watch_label=str(active_rule["watch_label"]),
            confidence_threshold=float(active_rule["confidence_threshold"]),
            persistence_frames=int(active_rule["persistence_frames"]),
            cooldown_seconds=float(active_rule["cooldown_seconds"]),
            require_vlm_confirmation=bool(active_rule["require_vlm_confirmation"]),
            rule_privacy=str(active_rule["privacy"]),
            vlm_prompt=str(active_rule["vlm_prompt"]),
            roi_name=roi_name,
            roi_x1=float(roi_x1),
            roi_y1=float(roi_y1),
            roi_x2=float(roi_x2),
            roi_y2=float(roi_y2),
            loitering_threshold_s=float(loitering_threshold_s),
            user_save=user_save,
        )

    if capture_once:
        capture()

    if st.session_state.get("monitoring_running", False) and hasattr(st, "fragment"):
        run_every = f"{int(interval_s)}s"

        @st.fragment(run_every=run_every)
        def live_monitor_fragment() -> None:
            capture()

        live_monitor_fragment()

    st.markdown("#### Recent EdgeLog events")
    recent_events = load_recent_events(limit=5)
    if not recent_events:
        st.caption("No retained events yet. Ordinary frames update only the latest snapshot.")
    else:
        st.dataframe(
            [
                {
                    "time": event.get("start_time") or event.get("timestamp"),
                    "event_type": event.get("event_type"),
                    "status": event.get("status"),
                    "risk": event.get("risk_level"),
                    "semantic": event.get("semantic_status"),
                    "objects": ", ".join(str(item) for item in event.get("objects", [])),
                }
                for event in recent_events
            ],
            use_container_width=True,
        )


def render_event_review(use_sample_mode: bool, api_base_url: str) -> None:
    st.subheader("Event Review")
    st.write("Use RTX semantic vision only for event review when privacy allows it. This is not the real-time monitor.")

    latest_image = st.session_state.get("latest_image_path")
    source_options = ["recent snapshot", "sample image", "upload image", "Jetson camera"] if latest_image else [
        "sample image",
        "upload image",
        "Jetson camera",
    ]

    left, right = st.columns([2, 1])
    with left:
        image_choice = st.selectbox("Review image", source_options, index=0, key="review_image_source")
        upload = None
        if image_choice == "upload image":
            upload = st.file_uploader("Upload event image", type=["jpg", "jpeg", "png"], key="review_upload")
        image_payload: Any
        if image_choice == "recent snapshot" and latest_image:
            image_payload = latest_image
        elif image_choice == "upload image":
            image_payload = upload or str(SAMPLE_IMAGE)
        elif image_choice == "Jetson camera":
            image_payload = None
        else:
            image_payload = str(SAMPLE_IMAGE)
        prompt = st.text_area(
            "Ask about this event",
            value="Is this monitoring event an alert? Return the structured JSON answer.",
            height=100,
        )
    with right:
        task_type = st.selectbox("Review task", ["scene_description", "vqa"], index=0)
        privacy = st.radio("Privacy mode", ["allow_remote", "local_only"], index=0, horizontal=True)
        quality = st.selectbox("Review quality", ["medium", "high", "low"], index=0)
        max_tokens = st.number_input("Vision max output tokens", min_value=32, max_value=1024, value=1024, step=32)
        latency_budget_ms = st.number_input(
            "Latency budget ms",
            min_value=100,
            max_value=60000,
            value=15000,
            step=500,
            key="review_latency",
        )
        if privacy == "allow_remote":
            st.caption("Send to RTX workstation when semantic review is required.")
            st.info("This image may be sent to the RTX workstation for semantic review.")
        else:
            st.caption("Privacy-safe mode blocks semantic vision offload.")

    submitted = st.button("Ask about this event", type="primary")
    if submitted:
        if use_sample_mode:
            result = select_vision_sample(task_type, privacy, quality)
            image_for_display = str(SAMPLE_IMAGE)
        else:
            backend_source = "camera" if image_choice == "Jetson camera" else "upload"
            with st.spinner("Reviewing event. Remote VLM routes can take 15-20 seconds."):
                result = call_vision_backend(
                    api_base_url,
                    image_source=backend_source,
                    image_path=image_payload or str(SAMPLE_IMAGE),
                    task_type=task_type,
                    privacy=privacy,
                    quality=quality,
                    latency_budget_ms=int(latency_budget_ms),
                    prompt=structured_vlm_prompt(prompt),
                    max_tokens=int(max_tokens),
                )
            image_for_display = _image_for_result(result, image_payload or str(SAMPLE_IMAGE))

        if privacy == "local_only" and result.get("route") == "reject":
            result["sent_to_remote"] = False
            st.warning(
                "Blocked by privacy policy: semantic vision would require sending the image "
                "to the remote workstation."
            )
        elif result.get("mode") == "sample fallback":
            st.warning("Real backend unavailable; showing committed sample fallback.")
        else:
            result["sent_to_remote"] = result.get("route") == "remote"

        _display_local_precheck(result, image_for_display, "Event snapshot with local YOLO precheck boxes")
        _display_routing_decision(result)
        st.markdown("#### VLM Review Status")
        st.write(result.get("generation_status") or result.get("status") or "completed")
        st.write(f"Sent to remote workstation: {bool(result.get('sent_to_remote'))}")
        _display_final_answer(result)
        event = _record_event(
            source="event_review",
            result=result,
            task_type=task_type,
            privacy=privacy,
            prompt=prompt,
            image_source=_image_source_for_event(result, image_payload or str(SAMPLE_IMAGE)),
        )
        st.session_state["latest_event_review_result"] = result
        st.session_state["latest_event_review_image"] = image_for_display
        st.session_state["latest_event_review_event_id"] = event["event_id"]
        st.caption(f"Recorded event {event['event_id']} in Event History.")
    elif st.session_state.get("latest_event_review_result"):
        st.info("Showing the latest Event Review result from this session.")
        result = st.session_state["latest_event_review_result"]
        image_for_display = st.session_state.get("latest_event_review_image", str(SAMPLE_IMAGE))
        _display_local_precheck(result, image_for_display, "Latest reviewed event with local YOLO precheck boxes")
        _display_routing_decision(result)
        st.markdown("#### VLM Review Status")
        st.write(result.get("generation_status") or result.get("status") or "completed")
        st.write(f"Sent to remote workstation: {bool(result.get('sent_to_remote'))}")
        _display_final_answer(result)
        if st.button("Refresh Review Status"):
            event_id = st.session_state.get("latest_event_review_event_id")
            if event_id:
                matches = [event for event in load_recent_events(limit=100) if event.get("event_id") == event_id]
                if matches:
                    st.json(matches[0])
                else:
                    st.warning("Review event is no longer in retained history.")


def render_monitoring_assistant(use_sample_mode: bool, api_base_url: str) -> None:
    st.subheader("Monitoring Assistant")
    st.write(
        "Monitoring Assistant is for event summaries, policy explanations, and backend status questions. "
        "It is not positioned as a generic chatbot."
    )
    recent_context = summarize_recent_events(limit=8)
    left, right = st.columns([2, 1])
    with left:
        user_prompt = st.text_area(
            "Assistant request",
            value="Summarize recent monitoring events and explain any routing decisions.",
            height=140,
        )
        with st.expander("Recent event context sent to the assistant"):
            st.text(recent_context)
    with right:
        task_type = st.selectbox("Assistant task", ["summary", "qa", "reasoning"], index=0)
        privacy = st.selectbox("Assistant privacy", ["allow_remote", "local_only"], index=0)
        quality = st.selectbox("Assistant quality", ["medium", "high", "low"], index=0)
        latency_budget_ms = st.number_input(
            "Latency budget ms",
            min_value=100,
            max_value=60000,
            value=5000,
            step=500,
            key="assistant_latency",
        )
        max_tokens = st.number_input("Max output tokens", min_value=32, max_value=2048, value=1024, step=32)
        timeout_s = st.number_input("Request timeout seconds", min_value=5, max_value=180, value=60, step=5)

    submitted = st.button("Ask Monitoring Assistant", type="primary")
    if submitted:
        prompt = (
            "You are the Monitoring Assistant for a Jetson local-first edge monitoring gateway. "
            "Use the recent event context when relevant.\n\n"
            f"Recent events:\n{recent_context}\n\n"
            f"User request:\n{user_prompt}"
        )
        if use_sample_mode:
            result = select_text_sample(prompt, task_type, privacy, quality, int(latency_budget_ms))
        else:
            result = call_text_backend(
                api_base_url,
                prompt,
                task_type,
                privacy,
                quality,
                int(latency_budget_ms),
                int(max_tokens),
                float(timeout_s),
            )
        if result.get("mode") == "sample fallback":
            st.warning("Real backend timed out/unavailable; showing committed sample fallback.")
        _display_routing_decision(result)
        if result.get("generation_status") == "incomplete_generation":
            st.error("Model did not produce a final answer before the output limit.")
            with st.expander("Debug raw model output"):
                st.text_area("Raw output", value=result.get("raw_output", ""), height=220)
        else:
            st.text_area("Assistant answer", value=result.get("full_response") or result.get("response_preview", ""), height=260)
        event = _record_event(
            source="monitoring_assistant",
            result=result,
            task_type=task_type,
            privacy=privacy,
            prompt=user_prompt,
        )
        st.session_state["latest_assistant_result"] = result
        st.caption(f"Recorded event {event['event_id']} in Event History.")
    elif st.session_state.get("latest_assistant_result"):
        st.info("Showing the latest Monitoring Assistant answer from this session.")
        result = st.session_state["latest_assistant_result"]
        _display_routing_decision(result)
        if result.get("generation_status") == "incomplete_generation":
            st.error("Model did not produce a final answer before the output limit.")
            with st.expander("Debug raw model output"):
                st.text_area("Raw output", value=result.get("raw_output", ""), height=220)
        else:
            st.text_area("Assistant answer", value=result.get("full_response") or result.get("response_preview", ""), height=260)


def render_event_history() -> None:
    st.subheader("Event History")
    st.write("Recent detection, review, reject, and assistant events are stored locally as JSONL.")
    policy = get_storage_policy()
    st.caption(
        "Storage policy: "
        f"max_events={policy.max_events}, max_images_mb={policy.max_images_mb}, "
        f"retention_days={policy.retention_days}. "
        "Latest ordinary snapshot is overwritten instead of retained forever."
    )
    route_filter = st.selectbox("Route filter", ["all", "local", "remote", "reject"], index=0)
    limit = st.number_input("Events to show", min_value=5, max_value=100, value=25, step=5)
    events = load_recent_events(limit=int(limit), route_filter=route_filter)

    if not events:
        st.info("No events have been recorded yet. Run Live Monitor, Event Review, or Monitoring Assistant first.")
        return

    summary_rows = [
        {
            "timestamp": event.get("timestamp"),
            "source": event.get("source"),
            "task_type": event.get("task_type"),
            "route": event.get("route"),
            "status": event.get("status"),
            "vlm_review_status": event.get("vlm_review_status"),
            "selected_backend": event.get("selected_backend"),
            "latency_ms": event.get("total_latency_ms"),
            "image": "expired"
            if event.get("image_missing")
            else ("stored" if event.get("stored_image") else "metadata only"),
            "remote": "sent to RTX" if event.get("sent_to_remote") else "local only",
            "expires_at": event.get("retention_expires_at"),
            "error": event.get("error"),
        }
        for event in events
    ]
    st.dataframe(summary_rows, use_container_width=True)

    for event in events:
        title = f"{event.get('timestamp', '')} | {event.get('source', '')} | {event.get('route', '')}"
        with st.expander(title):
            image_path = event.get("image_path")
            if event.get("image_missing"):
                st.info("Image expired by retention policy.")
            elif image_path and Path(image_path).exists():
                st.image(image_path, caption="Recorded event image", use_container_width=True)
            else:
                st.caption("Metadata-only event; no retained image is attached.")
            st.write(
                "Storage: "
                f"{'image stored' if event.get('stored_image') else 'metadata only'} | "
                f"{'sent to remote workstation' if event.get('sent_to_remote') else 'local only'} | "
                f"expires at {event.get('retention_expires_at', 'n/a')}"
            )
            st.json(event)


def render_event_search() -> None:
    st.subheader("Event Search")
    st.write("Search retained EdgeLog events by type, object labels, ROI, risk, and semantic description.")
    left, right = st.columns([2, 1])
    with left:
        query = st.text_input("Search events", placeholder="person, watch_zone, completed description...")
    with right:
        event_type = st.selectbox(
            "Event type",
            [
                "all",
                "person_enter_exit",
                "roi_intrusion",
                "object_change",
                "loitering",
                "assistant_summary",
                "privacy_reject",
                "backend_error",
            ],
        )
        risk_level = st.selectbox("Risk", ["all", "low", "medium", "high"])
        semantic_status = st.selectbox(
            "Semantic status",
            ["all", "not_required", "pending", "completed", "failed"],
        )
    results = search_events(
        query,
        event_type=event_type,
        risk_level=risk_level,
        semantic_status=semantic_status,
        limit=50,
    )
    st.caption(
        "MVP search uses local JSONL keyword/filter matching. It can be upgraded later to SQLite FTS5, embeddings, or FAISS."
    )
    if not results:
        st.info("No matching events found.")
        return
    st.dataframe(
        [
            {
                "start_time": event.get("start_time"),
                "event_type": event.get("event_type"),
                "risk": event.get("risk_level"),
                "semantic_status": event.get("semantic_status"),
                "objects": ", ".join(str(item) for item in event.get("objects", [])),
                "roi": event.get("roi_name"),
                "description": event.get("semantic_description") or event.get("final_answer_text"),
            }
            for event in results
        ],
        use_container_width=True,
    )
    for event in results[:10]:
        with st.expander(f"{event.get('event_type')} | {event.get('start_time')} | {event.get('risk_level')}"):
            image_path = event.get("keyframe_path") or event.get("image_path")
            if event.get("image_missing"):
                st.info("Image expired by retention policy.")
            elif image_path and Path(image_path).exists():
                st.image(image_path, caption="Event keyframe", use_container_width=True)
            st.write(event.get("semantic_description") or event.get("final_answer_text") or "No semantic description yet.")
            st.json(event)


def render_daily_summary(use_sample_mode: bool, api_base_url: str) -> None:
    st.subheader("Daily Summary")
    st.write("Generate a day-level summary from structured event metadata, not from raw video.")
    today = time.strftime("%Y-%m-%d")
    date_prefix = st.text_input("Date prefix", value=today)
    summary = build_daily_summary(date_prefix or None)
    st.metric("Total retained events", summary["total_events"])
    cols = st.columns(3)
    cols[0].metric("High risk events", len(summary["high_risk_events"]))
    cols[1].metric("Pending semantic reviews", len(summary["pending_semantic_reviews"]))
    cols[2].metric("Completed semantic descriptions", len(summary["completed_semantic_descriptions"]))
    st.markdown("#### Counts by type")
    st.json(summary["counts_by_type"])
    st.markdown("#### Timeline")
    st.text("\n".join(summary["timeline"]) or "No retained events for this date.")

    if st.button("Generate narrative summary with Monitoring Assistant"):
        prompt = (
            "You are EdgeLog's Monitoring Assistant. Generate a concise daily summary from this "
            "structured event table only. Do not infer from video that is not listed.\n\n"
            f"{summary}"
        )
        if use_sample_mode:
            result = select_text_sample(prompt, "summary", "allow_remote", "medium", 10000)
        else:
            result = call_text_backend(
                api_base_url,
                prompt,
                "summary",
                "allow_remote",
                "medium",
                10000,
                1024,
                120.0,
            )
        if result.get("mode") == "sample fallback":
            st.warning("Real backend unavailable; showing committed sample fallback.")
        st.text_area("Narrative daily summary", value=result.get("full_response") or result.get("response_preview", ""), height=260)
        event = _record_event(
            source="monitoring_assistant",
            result={
                **result,
                "event_type": "assistant_summary",
                "semantic_status": "completed",
                "semantic_description": result.get("full_response") or result.get("response_preview", ""),
            },
            task_type="summary",
            privacy="allow_remote",
            prompt=prompt,
        )
        st.caption(f"Recorded summary event {event['event_id']} in local history.")


def _status_label(ready: bool | None, mock: bool | None = None) -> str:
    if mock:
        return "Mock"
    if ready is True:
        return "Ready"
    if ready is False:
        return "Unavailable"
    return "Unknown"


def render_system_status(use_sample_mode: bool, api_base_url: str) -> None:
    st.subheader("System Status")
    st.write("Operational view of the monitoring stack.")
    if use_sample_mode:
        status = backend_status()
        st.info("Sample mode shows the expected backend roles, not live service health.")
        st.json(status)
        return

    health = real_backend_health(api_base_url)
    if not health.get("healthy"):
        st.error("Jetson Gateway is unavailable.")
        st.markdown(
            "- Run `python3 demo/run_interactive_stack.py`\n"
            "- Check the Router API base URL\n"
            "- Run `python3 demo/check_interactive_stack.py`"
        )
        st.json(health)
        return

    response = health.get("response") or {}
    local_ready = response.get("local_backend_available")
    remote_ready = response.get("remote_backend_available")
    vlm_ready = response.get("vision_remote_vlm_available")
    vlm_mock = response.get("vision_remote_is_mock")

    cols = st.columns(5)
    cols[0].metric("Jetson Gateway", "Ready")
    cols[1].metric("Jetson Local LLM", _status_label(local_ready))
    cols[2].metric("Jetson Local CV", "Ready")
    cols[3].metric("RTX Remote LLM", _status_label(remote_ready))
    cols[4].metric("RTX Remote VLM", _status_label(vlm_ready, vlm_mock))
    with st.expander("Raw health response"):
        st.json(response)


def render_model_policy() -> None:
    st.subheader("Model / Routing Policy")
    st.write("Read-only explanation of how EdgeLog splits real-time event memory from asynchronous semantic review.")

    st.markdown(
        """
| Role | Default backend | Product meaning |
| --- | --- | --- |
| Jetson local CV | YOLOv8n TensorRT FP16 | Real-time event trigger path for person enter/exit, ROI intrusion, object change, and loitering. |
| Jetson text default | Qwen3.5 0.8B Q4 | Local assistant responses when a summary/policy question should stay on the device. |
| RTX text fallback | Qwen3.5 4B Q4 | Heavier daily summaries and high-quality text review. |
| RTX semantic vision | Gemma 4 E2B-it Q4 + mmproj | Asynchronous event-level semantic descriptions after an event is already saved. |
| MobileNet-SSD | OpenCV DNN baseline | v0.5 system integration baseline and fallback reference. |
"""
    )
    st.markdown(
        """
- EdgeLog does not run VLM on every frame. VLM is an async semantic annotator for saved events.
- Local-only privacy blocks semantic offload and records a reject event instead.
- INT8 is treated as an experimental optimization because detection drift keeps it out of the default product path.
- The C++ worker is a hot-path exploration, not the default EdgeLog runtime.
"""
    )


def main() -> None:
    use_sample_mode, api_base_url = render_header()
    pages = [
        "Live Event Stream",
        "Event Search",
        "Daily Summary",
        "System Status",
        "Model / Routing Policy",
    ]
    page = st.sidebar.radio("Workbench page", pages, index=0)
    st.session_state["monitor_page_active"] = page == "Live Event Stream"
    if page != "Live Event Stream" and st.session_state.get("monitoring_running"):
        st.sidebar.info("Live Event Stream is paused while another page is active.")

    if page == "Live Event Stream":
        render_live_monitor(use_sample_mode, api_base_url)
    elif page == "Event Search":
        render_event_search()
    elif page == "Daily Summary":
        render_daily_summary(use_sample_mode, api_base_url)
    elif page == "System Status":
        render_system_status(use_sample_mode, api_base_url)
    elif page == "Model / Routing Policy":
        render_model_policy()


if __name__ == "__main__":
    main()
