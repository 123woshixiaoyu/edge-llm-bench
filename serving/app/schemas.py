from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TaskType = Literal["qa", "summary", "code", "reasoning", "long_context", "unknown"]
Quality = Literal["low", "medium", "high"]
Privacy = Literal["local_only", "allow_remote"]
Route = Literal["local", "remote", "reject"]
Risk = Literal["low", "medium", "high"]
VisionTaskType = Literal["detect", "classify", "vqa", "scene_description"]
VisionImageSource = Literal["camera", "upload", "sample"]


class Message(BaseModel):
    role: str
    content: str


class RouterState(BaseModel):
    local_available: bool = True
    remote_available: bool = True
    local_queue_depth: int = 0
    remote_queue_depth: int = 0
    jetson_temp_c: float | None = 55.0


class RouterStateUpdate(BaseModel):
    local_available: bool | None = None
    remote_available: bool | None = None
    local_queue_depth: int | None = None
    remote_queue_depth: int | None = None
    jetson_temp_c: float | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: list[Message]
    model: str | None = None
    stream: bool = False
    task_type: TaskType | None = None
    quality: Quality = "medium"
    latency_budget_ms: int = 5000
    privacy: Privacy = "allow_remote"
    max_tokens: int | None = None
    request_id: str | None = None
    state_override: RouterState | None = Field(
        default=None,
        description="Optional test-only state override for policy evaluation.",
    )

    def prompt_text(self) -> str:
        return "\n".join(message.content for message in self.messages)


class TaskAnalysis(BaseModel):
    prompt_chars: int
    estimated_prompt_tokens: int
    task_type: TaskType


class RouteDecision(BaseModel):
    route: Route
    selected_model: str | None
    reasons: list[str]
    estimated_risk: Risk


class RouteResponse(BaseModel):
    request_id: str
    analysis: TaskAnalysis
    decision: RouteDecision
    state: RouterState
    total_latency_ms: float


class ChatChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    route_decision: RouteDecision
    usage: dict[str, Any]
    backend_latency_ms: float
    total_latency_ms: float


class VisionAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_type: VisionTaskType
    privacy: Privacy = "allow_remote"
    quality: Quality = "medium"
    latency_budget_ms: int = 3000
    prompt: str | None = None
    image_source: VisionImageSource = "camera"
    image_base64: str | None = None
    use_yolo_trt: bool = True
    max_tokens: int = 128
    request_id: str | None = None


class VisionAnalyzeResponse(BaseModel):
    request_id: str
    route: Route
    selected_backend: str | None
    local_cv_backend: str | None
    local_cv_model: str | None
    remote_model: str | None
    remote_is_mock: bool
    capture_metadata: dict[str, Any]
    detected_labels: list[str]
    detections: list[dict[str, Any]]
    remote_response_text: str
    image_base64: str | None = None
    capture_latency_ms: float | None
    local_cv_inference_latency_ms: float | None
    local_cv_total_latency_ms: float | None
    remote_latency_ms: float | None
    total_latency_ms: float
    reasons: list[str]
    error: str = ""
    status: str = "success"
