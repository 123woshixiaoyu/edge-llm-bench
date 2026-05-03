from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


VisionTaskType = Literal["detect", "classify", "vqa", "scene_description", "unknown"]
VisionQuality = Literal["low", "medium", "high"]
VisionPrivacy = Literal["local_only", "allow_remote"]
ImageSource = Literal["camera", "file"]


@dataclass
class VisionAnalysis:
    task_type: VisionTaskType
    image_source: ImageSource
    requires_semantic_reasoning: bool
    local_cv_sufficient: bool

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_vision_task(task_type: str, image_source: str) -> VisionAnalysis:
    normalized_task = task_type if task_type in {"detect", "classify", "vqa", "scene_description"} else "unknown"
    normalized_source = image_source if image_source in {"camera", "file"} else "file"
    semantic = normalized_task in {"vqa", "scene_description"}
    local_cv_sufficient = normalized_task in {"detect", "classify"}
    return VisionAnalysis(
        task_type=normalized_task,  # type: ignore[arg-type]
        image_source=normalized_source,  # type: ignore[arg-type]
        requires_semantic_reasoning=semantic,
        local_cv_sufficient=local_cv_sufficient,
    )
