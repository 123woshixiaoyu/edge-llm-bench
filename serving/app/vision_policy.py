from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .vision_analyzer import VisionAnalysis, VisionPrivacy, VisionQuality


VisionRoute = Literal["local", "remote", "reject"]


@dataclass
class VisionDecision:
    route: VisionRoute
    selected_model: str | None
    reasons: list[str]
    estimated_risk: str
    remote_is_mock: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class VisionPolicyEngine:
    def __init__(
        self,
        *,
        local_cv_model: str = "mobilenet_ssd_voc_opencv_dnn",
        remote_vlm_model: str = "mock_remote_vlm",
        remote_available: bool = True,
        max_local_latency_budget_ms: int = 3000,
    ):
        self.local_cv_model = local_cv_model
        self.remote_vlm_model = remote_vlm_model
        self.remote_available = remote_available
        self.max_local_latency_budget_ms = max_local_latency_budget_ms

    def decide(
        self,
        *,
        analysis: VisionAnalysis,
        privacy: VisionPrivacy,
        quality: VisionQuality,
        latency_budget_ms: int,
        local_cv_available: bool,
    ) -> VisionDecision:
        reasons: list[str] = []

        if analysis.task_type == "unknown":
            return VisionDecision(
                route="reject",
                selected_model=None,
                reasons=["unknown vision task_type; MVP only supports detect/classify/vqa/scene_description"],
                estimated_risk="high",
            )

        if latency_budget_ms < 500 and analysis.requires_semantic_reasoning:
            return VisionDecision(
                route="reject",
                selected_model=None,
                reasons=[
                    f"latency_budget_ms={latency_budget_ms} is too low for semantic vision reasoning",
                    "remote VLM would likely miss the budget",
                ],
                estimated_risk="high",
            )

        if privacy == "local_only" and analysis.requires_semantic_reasoning:
            return VisionDecision(
                route="reject",
                selected_model=None,
                reasons=[
                    "privacy=local_only prevents sending image to remote VLM",
                    "local CV baseline only supports detection/classification, not high-level semantic explanation",
                ],
                estimated_risk="high",
            )

        if analysis.local_cv_sufficient and quality != "high":
            if not local_cv_available:
                if privacy == "local_only":
                    return VisionDecision(
                        route="reject",
                        selected_model=None,
                        reasons=["privacy=local_only but local CV backend is unavailable"],
                        estimated_risk="high",
                    )
                if self.remote_available:
                    return VisionDecision(
                        route="remote",
                        selected_model=self.remote_vlm_model,
                        reasons=["local CV unavailable; privacy allows remote fallback"],
                        estimated_risk="medium",
                        remote_is_mock=True,
                    )
                return VisionDecision(
                    route="reject",
                    selected_model=None,
                    reasons=["local CV unavailable and remote VLM unavailable"],
                    estimated_risk="high",
                )
            reasons.append(f"task_type={analysis.task_type} is covered by the local CV baseline")
            reasons.append("quality is not high, so remote VLM is unnecessary")
            return VisionDecision(
                route="local",
                selected_model=self.local_cv_model,
                reasons=reasons,
                estimated_risk="low",
            )

        remote_reasons: list[str] = []
        if analysis.requires_semantic_reasoning:
            remote_reasons.append(f"task_type={analysis.task_type} requires semantic visual reasoning")
        if quality == "high":
            remote_reasons.append("quality=high requests stronger visual understanding")
        if latency_budget_ms > self.max_local_latency_budget_ms:
            remote_reasons.append("latency budget allows remote processing")

        if remote_reasons:
            if privacy == "local_only":
                return VisionDecision(
                    route="reject",
                    selected_model=None,
                    reasons=remote_reasons + ["privacy=local_only prevents remote image transfer"],
                    estimated_risk="high",
                )
            if self.remote_available:
                return VisionDecision(
                    route="remote",
                    selected_model=self.remote_vlm_model,
                    reasons=remote_reasons,
                    estimated_risk="medium",
                    remote_is_mock=True,
                )
            return VisionDecision(
                route="reject",
                selected_model=None,
                reasons=remote_reasons + ["remote VLM backend unavailable"],
                estimated_risk="high",
            )

        return VisionDecision(
            route="reject",
            selected_model=None,
            reasons=["no vision route matched"],
            estimated_risk="high",
        )
