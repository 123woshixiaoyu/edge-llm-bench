from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import ChatRequest, RouteDecision, RouterState, TaskAnalysis


DEFAULT_POLICY = {
    "max_local_prompt_tokens": 512,
    "max_total_prompt_tokens": 4096,
    "local_queue_limit": 2,
    "max_jetson_temp_c": 75,
    "low_latency_budget_ms": 5000,
    "too_low_complex_latency_ms": 1500,
}


DEFAULT_MODELS = {
    "local": {
        "default": "qwen35_08b_q4",
        "quality": "qwen35_4b_q4",
        "medium_candidate": "gemma4_e2b_q4",
    },
    "remote": {
        "default": "qwen35_4b_q4",
        "high_quality": "remote_large_model",
    },
}


@dataclass
class RouterConfig:
    policy: dict[str, Any]
    models: dict[str, Any]
    backend_mode: str = "mock"
    local_base_url: str = "http://127.0.0.1:8080"
    remote_base_url: str = "http://127.0.0.1:8081"


def load_yaml(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = dict(default)
    merged.update(data)
    return merged


def load_router_config(config_dir: Path) -> RouterConfig:
    policy = load_yaml(config_dir / "policy.yaml", DEFAULT_POLICY)
    models = load_yaml(config_dir / "models.yaml", DEFAULT_MODELS)
    return RouterConfig(
        policy=policy,
        models=models,
        backend_mode=str(policy.get("backend_mode", "mock")),
        local_base_url=str(policy.get("local_llama_base_url", "http://127.0.0.1:8080")),
        remote_base_url=str(policy.get("remote_llama_base_url", "http://127.0.0.1:8081")),
    )


class PolicyEngine:
    def __init__(self, config: RouterConfig):
        self.config = config
        self.policy = config.policy
        self.models = config.models

    def local_model(self, request: ChatRequest, analysis: TaskAnalysis) -> str:
        local_models = self.models.get("local", {})
        if request.quality == "high" or analysis.task_type in {"code", "reasoning", "long_context"}:
            return str(local_models.get("quality", local_models.get("default", "qwen35_08b_q4")))
        return str(local_models.get("default", "qwen35_08b_q4"))

    def remote_model(self, request: ChatRequest, analysis: TaskAnalysis) -> str:
        remote_models = self.models.get("remote", {})
        if request.quality == "high" or analysis.task_type in {"code", "reasoning", "long_context"}:
            return str(remote_models.get("high_quality", remote_models.get("default", "qwen35_4b_q4")))
        return str(remote_models.get("default", "qwen35_4b_q4"))

    def complex_task(self, analysis: TaskAnalysis, request: ChatRequest) -> bool:
        return (
            analysis.task_type in {"code", "reasoning", "long_context"}
            or request.quality == "high"
            or analysis.estimated_prompt_tokens > int(self.policy["max_local_prompt_tokens"])
        )

    def decide(self, request: ChatRequest, analysis: TaskAnalysis, state: RouterState) -> RouteDecision:
        reasons: list[str] = []
        max_total = int(self.policy["max_total_prompt_tokens"])
        max_local = int(self.policy["max_local_prompt_tokens"])
        queue_limit = int(self.policy["local_queue_limit"])
        max_temp = float(self.policy["max_jetson_temp_c"])
        too_low_latency = int(self.policy["too_low_complex_latency_ms"])

        if analysis.estimated_prompt_tokens > max_total:
            return RouteDecision(
                route="reject",
                selected_model=None,
                estimated_risk="high",
                reasons=[
                    f"estimated_prompt_tokens={analysis.estimated_prompt_tokens} exceeds max_total_prompt_tokens={max_total}",
                    "request is too large for the MVP routing policy",
                ],
            )

        if not state.local_available and not state.remote_available:
            return RouteDecision(
                route="reject",
                selected_model=None,
                estimated_risk="high",
                reasons=["both local and remote backends are unavailable"],
            )

        if request.latency_budget_ms < too_low_latency and self.complex_task(analysis, request):
            return RouteDecision(
                route="reject",
                selected_model=None,
                estimated_risk="high",
                reasons=[
                    f"latency_budget_ms={request.latency_budget_ms} is too low for task_type={analysis.task_type}",
                    "complex task would likely miss the latency budget",
                ],
            )

        if request.privacy == "local_only":
            if not state.local_available:
                return RouteDecision(
                    route="reject",
                    selected_model=None,
                    estimated_risk="high",
                    reasons=["privacy=local_only but local backend is unavailable"],
                )
            risk = "high" if self.complex_task(analysis, request) else "low"
            reasons.append("privacy=local_only requires local execution")
            if self.complex_task(analysis, request):
                reasons.append("task is complex for Jetson; routing locally with elevated risk")
            else:
                reasons.append("task fits local default policy")
            return RouteDecision(
                route="local",
                selected_model=self.local_model(request, analysis),
                estimated_risk=risk,
                reasons=reasons,
            )

        remote_triggers: list[str] = []
        if analysis.task_type in {"code", "reasoning", "long_context"}:
            remote_triggers.append(f"task_type={analysis.task_type} benefits from stronger backend")
        if request.quality == "high":
            remote_triggers.append("quality=high requests stronger model")
        if analysis.estimated_prompt_tokens > max_local:
            remote_triggers.append(
                f"estimated_prompt_tokens={analysis.estimated_prompt_tokens} exceeds max_local_prompt_tokens={max_local}"
            )
        if state.local_queue_depth >= queue_limit:
            remote_triggers.append(
                f"local_queue_depth={state.local_queue_depth} >= local_queue_limit={queue_limit}"
            )
        if state.jetson_temp_c is not None and state.jetson_temp_c >= max_temp:
            remote_triggers.append(f"jetson_temp_c={state.jetson_temp_c} >= max_jetson_temp_c={max_temp}")
        if not state.local_available:
            remote_triggers.append("local backend unavailable")

        if remote_triggers:
            if state.remote_available:
                return RouteDecision(
                    route="remote",
                    selected_model=self.remote_model(request, analysis),
                    estimated_risk="low" if request.quality == "high" else "medium",
                    reasons=remote_triggers,
                )
            if state.local_available and analysis.estimated_prompt_tokens <= max_local:
                return RouteDecision(
                    route="local",
                    selected_model=self.local_model(request, analysis),
                    estimated_risk="high",
                    reasons=remote_triggers + ["remote backend unavailable; degraded to local"],
                )
            return RouteDecision(
                route="reject",
                selected_model=None,
                estimated_risk="high",
                reasons=remote_triggers + ["remote backend unavailable and local fallback is not safe"],
            )

        if state.local_available:
            return RouteDecision(
                route="local",
                selected_model=self.local_model(request, analysis),
                estimated_risk="low",
                reasons=[
                    f"task_type={analysis.task_type} fits local default policy",
                    f"estimated_prompt_tokens={analysis.estimated_prompt_tokens} <= max_local_prompt_tokens={max_local}",
                    "Jetson state is healthy and local backend is available",
                ],
            )

        if state.remote_available:
            return RouteDecision(
                route="remote",
                selected_model=self.remote_model(request, analysis),
                estimated_risk="medium",
                reasons=["local backend unavailable; remote backend available"],
            )

        return RouteDecision(
            route="reject",
            selected_model=None,
            estimated_risk="high",
            reasons=["no route matched and no backend is available"],
        )
