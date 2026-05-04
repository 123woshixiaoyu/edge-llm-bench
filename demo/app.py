from __future__ import annotations

import base64
import os
from io import BytesIO

import streamlit as st

from sample_results import (
    SAMPLE_IMAGE,
    backend_status,
    call_text_backend,
    call_vision_backend,
    draw_detections,
    key_result_tables,
    real_backend_health,
    select_text_sample,
    select_vision_sample,
)


st.set_page_config(
    page_title="Jetson-First Edge AI Inference Gateway",
    layout="wide",
)


DEFAULT_GATEWAY_URL = "http://192.168.1.102:8000"


def default_gateway_url() -> str:
    return os.environ.get("EDGE_GATEWAY_URL", DEFAULT_GATEWAY_URL)


def mode_label(use_sample_mode: bool) -> str:
    return "sample mode" if use_sample_mode else "real backend mode with sample fallback"


def render_header() -> tuple[bool, str]:
    st.title("Jetson-First Edge AI Inference Gateway")
    st.caption("Constraint-aware routing across local LLM/CV and remote LLM/VLM backends.")
    st.write(
        "This dashboard is a lightweight review surface for the edge gateway. "
        "Sample mode reads committed CSV/image evidence. Real mode is optional and falls back cleanly when services are not running."
    )
    left, right = st.columns([1, 2])
    with left:
        use_sample_mode = st.toggle("Use sample results", value=True)
    with right:
        api_base_url = st.text_input("Router API base URL", value=default_gateway_url())
        st.caption("Streamlit itself runs locally, but real backend mode should point to the Jetson Gateway.")
    st.info(f"Current mode: {mode_label(use_sample_mode)}")
    return use_sample_mode, api_base_url


def render_text_panel(use_sample_mode: bool, api_base_url: str) -> None:
    st.subheader("Text Task Router")
    left, right = st.columns([2, 1])
    with left:
        prompt = st.text_area(
            "Prompt",
            value="Explain why Q4 quantization is useful for a Jetson edge deployment.",
            height=140,
        )
    with right:
        task_type = st.selectbox("Task type", ["qa", "summary", "code", "reasoning"], index=0)
        privacy = st.selectbox("Privacy", ["allow_remote", "local_only"], index=0)
        quality = st.selectbox("Quality", ["low", "medium", "high"], index=1)
        latency_budget_ms = st.number_input("Latency budget ms", min_value=100, max_value=20000, value=3000, step=100)
        max_tokens = st.number_input("Max output tokens", min_value=32, max_value=2048, value=512, step=32)
        request_timeout_s = st.number_input("Request timeout seconds", min_value=5, max_value=180, value=60, step=5)
        if task_type in {"code", "reasoning"} or quality == "high":
            st.caption("Remote/high-quality text routes may need 120 seconds if the RTX backend is busy.")

    if st.button("Route Text Task", type="primary"):
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
                float(request_timeout_s),
            )

        if result.get("mode") == "sample fallback":
            st.warning("Real backend timed out/unavailable; showing committed sample fallback.")

        route_col, backend_col, latency_col = st.columns(3)
        route_col.metric("Route", result.get("route", "unknown"))
        backend_col.metric("Selected backend", result.get("selected_backend") or "none")
        latency_col.metric("Total latency ms", result.get("total_latency_ms") or "n/a")
        st.json(
            {
                "mode": result.get("mode"),
                "backend_latency_ms": result.get("backend_latency_ms"),
                "sample_request_id": result.get("sample_request_id"),
                "sample_source": result.get("sample_source"),
                "max_tokens_used": result.get("max_tokens_used"),
                "output_chars": result.get("output_chars"),
                "request_timeout_s": result.get("request_timeout_s"),
                "reasons": result.get("reasons", []),
            }
        )
        st.text_area("Response preview", value=result.get("response_preview", ""), height=160)
        if result.get("sample_truncated"):
            st.warning(result.get("sample_truncation_note") or "Sample mode only stores a preview.")
        with st.expander("Full response", expanded=not result.get("sample_truncated")):
            st.text_area(
                "Full model response",
                value=result.get("full_response") or result.get("response_preview", ""),
                height=260,
                label_visibility="collapsed",
            )


def render_vision_panel(use_sample_mode: bool, api_base_url: str) -> None:
    st.subheader("Vision Task Router")
    left, right = st.columns([2, 1])
    with left:
        image_source = st.selectbox(
            "Image source",
            ["sample image", "upload image", "Jetson camera"],
            index=0 if use_sample_mode else 2,
        )
        image_file = None
        if image_source == "upload image":
            image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        image_path = image_file if image_file is not None else str(SAMPLE_IMAGE)
    with right:
        task_type = st.selectbox("Vision task type", ["detect", "classify", "scene_description", "vqa"], index=0)
        privacy = st.selectbox("Vision privacy", ["allow_remote", "local_only"], index=0)
        quality = st.selectbox("Vision quality", ["low", "medium", "high"], index=0)
        latency_budget_ms = st.number_input("Vision latency budget ms", min_value=100, max_value=60000, value=3000, step=100)
        vision_max_tokens = st.number_input("Vision max output tokens", min_value=32, max_value=1024, value=384, step=32)
        prompt = st.text_area("Optional vision prompt", value="", height=92)

    if st.button("Route Vision Task", type="primary"):
        if use_sample_mode:
            result = select_vision_sample(task_type, privacy, quality)
        else:
            backend_source = "camera" if image_source == "Jetson camera" else "upload"
            with st.spinner("Calling Jetson Gateway. Remote VLM routes can take 15-20 seconds."):
                result = call_vision_backend(
                    api_base_url,
                    image_source=backend_source,
                    image_path=image_path,
                    task_type=task_type,
                    privacy=privacy,
                    quality=quality,
                    latency_budget_ms=int(latency_budget_ms),
                    prompt=prompt,
                    max_tokens=int(vision_max_tokens),
                )

        image_for_boxes = str(SAMPLE_IMAGE) if image_file is None else image_file
        if result.get("image_base64"):
            try:
                image_for_boxes = BytesIO(base64.b64decode(result["image_base64"]))
            except Exception:
                image_for_boxes = str(SAMPLE_IMAGE)
        precheck_detections = result.get("local_cv_precheck_detections") or result.get("detections", [])
        precheck_labels = result.get("local_cv_precheck_labels") or result.get("detected_labels", [])
        annotated = draw_detections(image_for_boxes, precheck_detections)

        st.markdown("### Local CV Precheck (Jetson)")
        st.caption(
            "This local detection runs before routing and provides fast edge-side visual evidence. "
            "For remote semantic tasks, the final answer comes from the remote VLM, not from these boxes."
        )
        preview_col, cv_col = st.columns([1, 1])
        with preview_col:
            if annotated is not None:
                caption = (
                    "Camera frame with local YOLO precheck boxes"
                    if image_source == "Jetson camera"
                    else "Uploaded image with local CV precheck boxes"
                )
                if use_sample_mode and image_source == "sample image":
                    caption = "Sample image with local CV precheck boxes"
                st.image(annotated, caption=caption, use_container_width=True)
            else:
                st.warning("Sample image not available.")
        with cv_col:
            st.metric("Local CV inference ms", result.get("local_cv_inference_latency_ms") or "n/a")
            st.json(
                {
                    "local_cv_backend": result.get("local_cv_backend"),
                    "local_cv_model": result.get("local_cv_model"),
                    "detected_labels": precheck_labels,
                    "detections": precheck_detections,
                    "capture_latency_ms": result.get("capture_latency_ms"),
                    "local_cv_inference_latency_ms": result.get("local_cv_inference_latency_ms"),
                    "local_cv_total_latency_ms": result.get("local_cv_total_latency_ms"),
                }
            )

        st.markdown("### Routing Decision")
        route_col, backend_col, latency_col = st.columns(3)
        route_col.metric("Route", result.get("route", "unknown"))
        backend_col.metric("Selected backend", result.get("selected_backend") or "none")
        latency_col.metric("Total latency ms", result.get("total_latency_ms") or "n/a")
        st.json(
            {
                "mode": result.get("mode"),
                "privacy": privacy,
                "quality": quality,
                "remote_is_mock": result.get("remote_is_mock"),
                "remote_latency_ms": result.get("remote_latency_ms"),
                "status": result.get("status"),
                "error": result.get("error"),
                "max_tokens_used": result.get("max_tokens_used"),
                "output_chars": result.get("output_chars"),
                "reasons": result.get("reasons"),
            }
        )

        st.markdown("### Final Routed Answer")
        final_source = result.get("final_answer_source") or "unknown"
        st.write(f"**Final answer source:** {final_source}")
        route = result.get("route")
        if route == "local":
            st.info("Local CV detection/classification is the final answer for this task.")
            st.json(
                {
                    "detected_labels": precheck_labels,
                    "detections": precheck_detections,
                }
            )
        elif route == "remote":
            st.info("The final semantic answer comes from the RTX remote VLM. YOLO boxes above are only local pre-analysis.")
            st.text_area(
                "Remote VLM response",
                value=result.get("final_answer_text") or result.get("remote_response_text", ""),
                height=180,
            )
            if result.get("sample_truncated"):
                st.warning(result.get("sample_truncation_note") or "Sample mode only stores a preview.")
            with st.expander("Full remote VLM response", expanded=not result.get("sample_truncated")):
                st.text_area(
                    "Full remote VLM response text",
                    value=result.get("full_response") or result.get("final_answer_text", ""),
                    height=280,
                    label_visibility="collapsed",
                )
        elif route == "reject":
            st.warning("Policy rejected this request. No model answer was produced.")
            st.json({"reject_reasons": result.get("reasons", [])})
        else:
            st.write(result.get("final_answer_text") or "")


def render_backend_panel(use_sample_mode: bool, api_base_url: str) -> None:
    st.subheader("Backend Status")
    st.caption(
        "For real text remote routes, verify the Jetson Gateway, RTX remote llama-server, "
        "and SSH tunnel are running before treating a fallback as a model failure."
    )
    if use_sample_mode:
        st.json(backend_status())
    else:
        st.json(real_backend_health(api_base_url))


def render_results_panel() -> None:
    st.subheader("Key Results Snapshot")
    tables = key_result_tables()

    st.markdown("**Model/backend scorecard recommendations**")
    st.dataframe(tables["backend_recommendations"], use_container_width=True)

    st.markdown("**Local CV runtime summary**")
    st.dataframe(tables["local_cv_runtime_summary"], use_container_width=True)

    st.markdown("**v0.7 reliability summary**")
    st.dataframe(tables["reliability_summary"], use_container_width=True)

    st.markdown(
        "- Qwen3.5 0.8B Q4_K_M is the Jetson local text default.\n"
        "- ONNXRuntime session reuse reduced SSD total latency from about 4930 ms to about 49 ms.\n"
        "- YOLOv8n TensorRT FP16 reduced local CV inference from 91.70 ms to 14.54 ms.\n"
        "- v0.5b validated a real non-mock remote VLM path; v0.7 adds prototype reliability evidence."
    )


def main() -> None:
    use_sample_mode, api_base_url = render_header()
    text_tab, vision_tab, status_tab, results_tab = st.tabs(
        ["Text Router", "Vision Router", "Backend Status", "Results Snapshot"]
    )
    with text_tab:
        render_text_panel(use_sample_mode, api_base_url)
    with vision_tab:
        render_vision_panel(use_sample_mode, api_base_url)
    with status_tab:
        render_backend_panel(use_sample_mode, api_base_url)
    with results_tab:
        render_results_panel()


if __name__ == "__main__":
    main()
