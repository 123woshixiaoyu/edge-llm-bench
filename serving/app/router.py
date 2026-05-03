from __future__ import annotations

import time
import threading
import uuid
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Iterator

from .analyzer import analyze_request
from .backends.base import Backend
from .backends.llamacpp import LlamaCppBackend
from .backends.mock import MockBackend
from .metrics import DecisionLogger
from .policy import PolicyEngine, RouterConfig
from .schemas import ChatRequest, RouteDecision, RouterState, RouterStateUpdate, TaskAnalysis


class TaskRouterService:
    def __init__(self, config: RouterConfig, serving_root: Path):
        self.config = config
        self.policy = PolicyEngine(config)
        self.serving_root = serving_root
        self.logger = DecisionLogger(serving_root / "results/raw/routing_decisions.jsonl")
        self.local_backend, self.remote_backend = self._build_backends()
        self._lock = threading.Lock()
        self.local_inflight = 0
        self.remote_inflight = 0
        self.jetson_temp_c: float | None = 55.0
        self._state_override = RouterStateUpdate()

    def _build_backends(self) -> tuple[Backend, Backend]:
        if self.config.backend_mode == "hybrid":
            return (
                LlamaCppBackend("local_llamacpp", self.config.local_base_url),
                MockBackend("remote_mock"),
            )
        if self.config.backend_mode == "llamacpp":
            return (
                LlamaCppBackend("local_llamacpp", self.config.local_base_url),
                LlamaCppBackend("remote_llamacpp", self.config.remote_base_url),
            )
        return MockBackend("local_mock"), MockBackend("remote_mock")

    def current_state(self, override: RouterState | None = None) -> RouterState:
        if override is not None:
            return override
        with self._lock:
            local_inflight = self.local_inflight
            remote_inflight = self.remote_inflight
            simulated = self._state_override
            jetson_temp_c = self.jetson_temp_c
        state = RouterState(
            local_available=self.local_backend.available(),
            remote_available=self.remote_backend.available(),
            local_queue_depth=local_inflight,
            remote_queue_depth=remote_inflight,
            jetson_temp_c=jetson_temp_c,
        )
        update = simulated.model_dump(exclude_none=True)
        if "local_queue_depth" in update:
            update["local_queue_depth"] = max(local_inflight, int(update["local_queue_depth"]))
        if "remote_queue_depth" in update:
            update["remote_queue_depth"] = max(remote_inflight, int(update["remote_queue_depth"]))
        return state.model_copy(update=update)

    def update_state(self, update: RouterStateUpdate) -> RouterState:
        with self._lock:
            self._state_override = update
            if update.jetson_temp_c is not None:
                self.jetson_temp_c = update.jetson_temp_c
        return self.current_state()

    def reset_state(self) -> RouterState:
        with self._lock:
            self._state_override = RouterStateUpdate()
            self.jetson_temp_c = 55.0
        return self.current_state()

    @contextmanager
    def backend_slot(self, route: str) -> Iterator[None]:
        if route not in {"local", "remote"}:
            yield
            return
        with self._lock:
            if route == "local":
                self.local_inflight += 1
            else:
                self.remote_inflight += 1
        try:
            yield
        finally:
            with self._lock:
                if route == "local":
                    self.local_inflight = max(0, self.local_inflight - 1)
                else:
                    self.remote_inflight = max(0, self.remote_inflight - 1)

    def route(self, request: ChatRequest) -> tuple[str, TaskAnalysis, RouteDecision, RouterState, float]:
        start = time.perf_counter()
        request_id = request.request_id or str(uuid.uuid4())
        analysis = analyze_request(request)
        state = self.current_state(request.state_override)
        decision = self.policy.decide(request, analysis, state)
        total_latency_ms = (time.perf_counter() - start) * 1000
        return request_id, analysis, decision, state, total_latency_ms

    def backend_for(self, decision: RouteDecision) -> Backend:
        return self.local_backend if decision.route == "local" else self.remote_backend

    def metrics(self) -> dict:
        state = self.current_state()
        snapshot = self.logger.snapshot()
        snapshot.update(
            {
                "current_local_queue_depth": state.local_queue_depth,
                "current_remote_queue_depth": state.remote_queue_depth,
                "jetson_temp_c": state.jetson_temp_c,
                "local_backend_available": state.local_available,
                "remote_backend_available": state.remote_available,
                "backend_mode": self.config.backend_mode,
            }
        )
        return snapshot

    def log_decision(
        self,
        *,
        request_id: str,
        request: ChatRequest,
        analysis: TaskAnalysis,
        decision: RouteDecision,
        total_latency_ms: float,
        status: str,
        backend_latency_ms: float | None = None,
    ) -> None:
        self.logger.log(
            {
                "request_id": request_id,
                "task_type": analysis.task_type,
                "estimated_prompt_tokens": analysis.estimated_prompt_tokens,
                "quality": request.quality,
                "privacy": request.privacy,
                "route": decision.route,
                "selected_model": decision.selected_model,
                "reasons": decision.reasons,
                "backend_latency_ms": backend_latency_ms,
                "total_latency_ms": round(total_latency_ms, 2),
                "status": status,
            }
        )
