from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .policy import load_router_config
from .router import TaskRouterService
from .schemas import ChatChoice, ChatCompletionResponse, ChatRequest, Message, RouteResponse, RouterStateUpdate


SERVING_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("EDGE_ROUTER_CONFIG_DIR", SERVING_ROOT / "configs"))

router_service = TaskRouterService(load_router_config(CONFIG_DIR), SERVING_ROOT)

app = FastAPI(
    title="Edge LLM Task Router",
    version="0.1.0",
    description="Jetson-first heterogeneous inference router for local-vs-remote LLM task assignment.",
)


@app.get("/health")
def health() -> dict:
    state = router_service.current_state()
    return {
        "status": "ok",
        "backend_mode": router_service.config.backend_mode,
        "local_backend_available": state.local_available,
        "remote_backend_available": state.remote_available,
        "local_queue_depth": state.local_queue_depth,
        "remote_queue_depth": state.remote_queue_depth,
        "jetson_temp_c": state.jetson_temp_c,
    }


@app.get("/metrics")
def metrics() -> dict:
    return router_service.metrics()


@app.post("/state")
def update_state(update: RouterStateUpdate) -> dict:
    state = router_service.update_state(update)
    return {"status": "ok", "state": state.model_dump()}


@app.post("/state/reset")
def reset_state() -> dict:
    state = router_service.reset_state()
    return {"status": "ok", "state": state.model_dump()}


@app.post("/v1/route", response_model=RouteResponse)
def route_request(request: ChatRequest) -> RouteResponse:
    request_id, analysis, decision, state, total_latency_ms = router_service.route(request)
    status = "rejected" if decision.route == "reject" else "success"
    router_service.log_decision(
        request_id=request_id,
        request=request,
        analysis=analysis,
        decision=decision,
        total_latency_ms=total_latency_ms,
        status=status,
        backend_latency_ms=0.0,
    )
    return RouteResponse(
        request_id=request_id,
        analysis=analysis,
        decision=decision,
        state=state,
        total_latency_ms=round(total_latency_ms, 2),
    )


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    start = time.perf_counter()
    request_id, analysis, decision, state, route_latency_ms = router_service.route(request)

    if decision.route == "reject":
        total_latency_ms = (time.perf_counter() - start) * 1000
        router_service.log_decision(
            request_id=request_id,
            request=request,
            analysis=analysis,
            decision=decision,
            total_latency_ms=total_latency_ms,
            status="rejected",
            backend_latency_ms=0.0,
        )
        return JSONResponse(
            status_code=400,
            content={
                "request_id": request_id,
                "error": "request rejected by routing policy",
                "analysis": analysis.model_dump(),
                "decision": decision.model_dump(),
                "state": state.model_dump(),
                "total_latency_ms": round(total_latency_ms, 2),
            },
        )

    backend = router_service.backend_for(decision)
    with router_service.backend_slot(decision.route):
        backend_result = backend.generate(request.messages, decision.selected_model or "unknown", request.max_tokens)
    total_latency_ms = (time.perf_counter() - start) * 1000

    if not backend_result.ok:
        router_service.log_decision(
            request_id=request_id,
            request=request,
            analysis=analysis,
            decision=decision,
            total_latency_ms=total_latency_ms,
            status="backend_error",
            backend_latency_ms=backend_result.latency_ms,
        )
        return JSONResponse(
            status_code=502,
            content={
                "request_id": request_id,
                "error": backend_result.error or "backend error",
                "analysis": analysis.model_dump(),
                "decision": decision.model_dump(),
                "state": state.model_dump(),
                "backend_latency_ms": round(backend_result.latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
            },
        )

    router_service.log_decision(
        request_id=request_id,
        request=request,
        analysis=analysis,
        decision=decision,
        total_latency_ms=total_latency_ms,
        status="success",
        backend_latency_ms=backend_result.latency_ms,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=decision.selected_model or "unknown",
        choices=[
            ChatChoice(
                message=Message(role="assistant", content=backend_result.text),
                finish_reason="stop",
            )
        ],
        route_decision=decision,
        usage={
            "estimated_prompt_tokens": analysis.estimated_prompt_tokens,
            "completion_tokens": None,
            "total_tokens": None,
        },
        backend_latency_ms=round(backend_result.latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
    )
    return response
