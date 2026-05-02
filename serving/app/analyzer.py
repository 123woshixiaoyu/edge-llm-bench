from __future__ import annotations

import math

from .schemas import ChatRequest, TaskAnalysis, TaskType


SUMMARY_TERMS = ["summarize", "summary", "\u603b\u7ed3", "\u6982\u62ec"]
REASONING_TERMS = ["reason", "plan", "why", "\u5206\u6790", "\u63a8\u7406", "\u89c4\u5212"]
CODE_TERMS = ["code", "python", "debug", "function", "traceback"]


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def infer_task_type(prompt: str, estimated_tokens: int) -> TaskType:
    lowered = prompt.lower()
    if estimated_tokens > 512 or len(prompt) > 2048:
        return "long_context"
    if any(term in lowered for term in CODE_TERMS):
        return "code"
    if any(term in lowered for term in SUMMARY_TERMS):
        return "summary"
    if any(term in lowered for term in REASONING_TERMS):
        return "reasoning"
    return "qa"


def analyze_request(request: ChatRequest) -> TaskAnalysis:
    prompt = request.prompt_text()
    estimated_tokens = estimate_tokens(prompt)
    task_type = request.task_type or infer_task_type(prompt, estimated_tokens)
    return TaskAnalysis(
        prompt_chars=len(prompt),
        estimated_prompt_tokens=estimated_tokens,
        task_type=task_type,
    )
