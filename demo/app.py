from __future__ import annotations

import streamlit as st

from sample_results import (
    SAMPLE_IMAGE,
    backend_status,
    call_text_backend,
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
        api_base_url = st.text_input("Router API base URL", value="http://127.0.0.1:8000")
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

    if st.button("Route Text Task", type="primary"):
        if use_sample_mode:
            result = select_text_sample(prompt, task_type, privacy, quality, int(latency_budget_ms))
        else:
            result = call_text_backend(api_base_url, prompt, task_type, privacy, quality, int(latency_budget_ms))

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
                "reasons": result.get("reasons", []),
            }
        )
        st.text_area("Response preview", value=result.get("response_preview", ""), height=160)


def render_vision_panel(use_sample_mode: bool) -> None:
    st.subheader("Vision Task Router")
    left, right = st.columns([2, 1])
    with left:
        image_file = st.file_uploader("Upload image, or leave empty to use the sample camera frame", type=["jpg", "jpeg", "png"])
        image_path = image_file if image_file is not None else str(SAMPLE_IMAGE)
    with right:
        task_type = st.selectbox("Vision task type", ["detect", "classify", "scene_description", "vqa"], index=0)
        privacy = st.selectbox("Vision privacy", ["allow_remote", "local_only"], index=0)
        quality = st.selectbox("Vision quality", ["low", "medium", "high"], index=0)

    if st.button("Route Vision Task", type="primary"):
        result = select_vision_sample(task_type, privacy, quality)
        if not use_sample_mode:
            result["mode"] = "sample fallback"
            result["reasons"] = [
                "real vision API is not invoked by this lightweight dashboard",
                *result.get("reasons", []),
            ]

        image_for_boxes = str(SAMPLE_IMAGE) if image_file is None else image_file
        annotated = draw_detections(image_for_boxes, result.get("detections", []))
        preview_col, decision_col = st.columns([1, 1])
        with preview_col:
            if annotated is not None:
                st.image(annotated, caption="Image preview with sample detections", use_column_width=True)
            else:
                st.warning("Sample image not available.")
        with decision_col:
            st.metric("Route", result.get("route", "unknown"))
            st.metric("Selected backend", result.get("selected_backend") or "none")
            st.metric("Local CV inference ms", result.get("local_cv_inference_latency_ms") or "n/a")
            st.json(
                {
                    "mode": result.get("mode"),
                    "remote_is_mock": result.get("remote_is_mock"),
                    "local_cv_backend": result.get("local_cv_backend"),
                    "local_cv_model": result.get("local_cv_model"),
                    "capture_latency_ms": result.get("capture_latency_ms"),
                    "local_cv_total_latency_ms": result.get("local_cv_total_latency_ms"),
                    "remote_latency_ms": result.get("remote_latency_ms"),
                    "total_latency_ms": result.get("total_latency_ms"),
                    "detected_labels": result.get("detected_labels"),
                    "detections": result.get("detections"),
                    "reasons": result.get("reasons"),
                }
            )


def render_backend_panel(use_sample_mode: bool, api_base_url: str) -> None:
    st.subheader("Backend Status")
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
        render_vision_panel(use_sample_mode)
    with status_tab:
        render_backend_panel(use_sample_mode, api_base_url)
    with results_tab:
        render_results_panel()


if __name__ == "__main__":
    main()
