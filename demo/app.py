from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st

from event_store import append_event, load_recent_events, summarize_recent_events
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
    page_title="Jetson Local-First Monitoring Gateway",
    layout="wide",
)


DEFAULT_GATEWAY_URL = "http://192.168.1.102:8000"


def default_gateway_url() -> str:
    return os.environ.get("EDGE_GATEWAY_URL", DEFAULT_GATEWAY_URL)


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
        "source": source,
        "prompt": prompt,
        "task_type": task_type,
        "privacy": privacy,
        "route": result.get("route"),
        "selected_backend": result.get("selected_backend") or result.get("local_cv_backend"),
        "local_cv_labels": _labels(result),
        "local_cv_latency_ms": _as_float(result.get("local_cv_inference_latency_ms")),
        "remote_latency_ms": _as_float(result.get("remote_latency_ms")),
        "total_latency_ms": _as_float(result.get("total_latency_ms")),
        "final_answer_text": result.get("final_answer_text")
        or result.get("full_response")
        or result.get("response_preview")
        or "",
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
) -> dict[str, Any]:
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
    st.session_state["latest_event"] = event
    if event.get("image_path"):
        st.session_state["latest_image_path"] = event["image_path"]
    return event


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
        st.metric("Local CV inference ms", result.get("local_cv_inference_latency_ms") or "n/a")
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
    st.title("Jetson Local-First Monitoring Gateway")
    st.caption(
        "A local-first edge monitoring workbench that uses Jetson for low-latency local detection "
        "and routes event review / summaries to RTX backends only when privacy and system constraints allow."
    )
    st.write(
        "Live monitoring stays on the Jetson fast path by default. "
        "Remote LLM/VLM backends are used for event review, summaries, and policy explanations when allowed."
    )

    left, right = st.columns([1, 2])
    with left:
        use_sample_mode = st.toggle("Use sample results", value=True)
    with right:
        api_base_url = st.text_input("Router API base URL", value=default_gateway_url())
        st.caption("Streamlit itself runs locally, but real backend mode should point to the Jetson Gateway.")
    st.info(f"Current mode: {mode_label(use_sample_mode)}")
    return use_sample_mode, api_base_url


def render_live_monitor(use_sample_mode: bool, api_base_url: str) -> None:
    st.subheader("Live Monitor")
    st.write("Capture a single snapshot and run the local Jetson YOLO TensorRT monitoring fast path.")

    left, right = st.columns([2, 1])
    with left:
        image_source, image_payload = _select_image_source(
            "Snapshot source",
            default_camera=not use_sample_mode,
            key_prefix="live",
        )
    with right:
        with st.expander("Advanced routing settings"):
            task_type = st.selectbox("Task", ["detect", "classify"], index=0, key="live_task")
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
                max_value=1024,
                value=128,
                step=32,
                key="live_tokens",
            )

    if st.button("Run Local Detection", type="primary"):
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

        if result.get("mode") == "sample fallback":
            st.warning("Real backend unavailable; showing committed sample fallback.")

        _display_local_precheck(result, image_for_display, "Current snapshot with local YOLO monitoring boxes")
        _display_routing_decision(result)
        st.success("Route/location: Processed on Jetson" if result.get("route") == "local" else "Route/location recorded by policy")
        event = _record_event(
            source="live_monitor",
            result=result,
            task_type=task_type,
            privacy=privacy,
            image_source=_image_source_for_event(result, image_payload or str(SAMPLE_IMAGE)),
        )
        st.caption(f"Recorded event {event['event_id']} in Event History.")


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
            value="Describe this monitoring event in one concise sentence.",
            height=100,
        )
    with right:
        task_type = st.selectbox("Review task", ["scene_description", "vqa"], index=0)
        privacy = st.radio("Privacy mode", ["allow_remote", "local_only"], index=0, horizontal=True)
        quality = st.selectbox("Review quality", ["medium", "high", "low"], index=0)
        max_tokens = st.number_input("Vision max output tokens", min_value=32, max_value=1024, value=384, step=32)
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
        else:
            st.caption("Privacy-safe mode blocks semantic vision offload.")

    if st.button("Ask about this event", type="primary"):
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
                    prompt=prompt,
                    max_tokens=int(max_tokens),
                )
            image_for_display = _image_for_result(result, image_payload or str(SAMPLE_IMAGE))

        if privacy == "local_only" and result.get("route") == "reject":
            st.warning(
                "Blocked by privacy policy: semantic vision would require sending the image "
                "to the remote workstation."
            )
        elif result.get("mode") == "sample fallback":
            st.warning("Real backend unavailable; showing committed sample fallback.")

        _display_local_precheck(result, image_for_display, "Event snapshot with local YOLO precheck boxes")
        _display_routing_decision(result)
        _display_final_answer(result)
        event = _record_event(
            source="event_review",
            result=result,
            task_type=task_type,
            privacy=privacy,
            prompt=prompt,
            image_source=_image_source_for_event(result, image_payload or str(SAMPLE_IMAGE)),
        )
        st.caption(f"Recorded event {event['event_id']} in Event History.")


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
        max_tokens = st.number_input("Max output tokens", min_value=32, max_value=2048, value=512, step=32)
        timeout_s = st.number_input("Request timeout seconds", min_value=5, max_value=180, value=60, step=5)

    if st.button("Ask Monitoring Assistant", type="primary"):
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
        st.text_area("Assistant answer", value=result.get("full_response") or result.get("response_preview", ""), height=260)
        event = _record_event(
            source="monitoring_assistant",
            result=result,
            task_type=task_type,
            privacy=privacy,
            prompt=user_prompt,
        )
        st.caption(f"Recorded event {event['event_id']} in Event History.")


def render_event_history() -> None:
    st.subheader("Event History")
    st.write("Recent detection, review, reject, and assistant events are stored locally as JSONL.")
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
            "selected_backend": event.get("selected_backend"),
            "latency_ms": event.get("total_latency_ms"),
            "error": event.get("error"),
        }
        for event in events
    ]
    st.dataframe(summary_rows, use_container_width=True)

    for event in events:
        title = f"{event.get('timestamp', '')} | {event.get('source', '')} | {event.get('route', '')}"
        with st.expander(title):
            image_path = event.get("image_path")
            if image_path and Path(image_path).exists():
                st.image(image_path, caption="Recorded event image", use_container_width=True)
            st.json(event)


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
    st.subheader("Model Policy")
    st.write("Read-only explanation of the current monitoring backend choices.")

    st.markdown(
        """
| Role | Default backend | Product meaning |
| --- | --- | --- |
| Jetson text default | Qwen3.5 0.8B Q4 | Local monitoring assistant responses when the task is short/private. |
| RTX text fallback | Qwen3.5 4B Q4 | Heavier summaries, code/reasoning, and high-quality text review. |
| Jetson local CV | YOLOv8n TensorRT FP16 | Local monitoring fast path for snapshots and object detection. |
| MobileNet-SSD | OpenCV DNN baseline | v0.5 system integration baseline and fallback reference. |
| RTX semantic vision | Gemma 4 E2B-it Q4 + mmproj | Event review and VQA when privacy permits image offload. |
"""
    )
    st.markdown(
        """
- INT8 is treated as an experimental optimization because detection drift keeps it out of the default product path.
- The C++ worker is a hot-path exploration, not the default workbench runtime.
- The router keeps local / remote / reject decisions explicit so privacy and system constraints remain visible.
"""
    )


def main() -> None:
    use_sample_mode, api_base_url = render_header()
    tabs = st.tabs(
        [
            "Live Monitor",
            "Event Review",
            "Monitoring Assistant",
            "Event History",
            "System Status",
            "Model Policy",
        ]
    )
    with tabs[0]:
        render_live_monitor(use_sample_mode, api_base_url)
    with tabs[1]:
        render_event_review(use_sample_mode, api_base_url)
    with tabs[2]:
        render_monitoring_assistant(use_sample_mode, api_base_url)
    with tabs[3]:
        render_event_history()
    with tabs[4]:
        render_system_status(use_sample_mode, api_base_url)
    with tabs[5]:
        render_model_policy()


if __name__ == "__main__":
    main()
