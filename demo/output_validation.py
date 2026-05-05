from __future__ import annotations

import json
import re
from typing import Any


FINAL_MARKERS = [
    "FINAL_ANSWER:",
    "<|channel|>final<|message|>",
    "<|start|>assistant<|channel|>final<|message|>",
]

THINKING_HINTS = [
    "thinking process",
    "analysis steps",
    "hidden chain-of-thought",
    "constraints",
    "let's think",
    "we need to",
    "i need to",
    "chain of thought",
]


def _extract_json(value: str) -> dict[str, Any] | None:
    text = value.strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("answer"):
            return parsed
    return None


def _looks_like_thinking_only(value: str) -> bool:
    text = value.strip().lower()
    if not text:
        return False
    hint_count = sum(1 for hint in THINKING_HINTS if hint in text)
    starts_like_process = bool(re.match(r"^(analysis|thinking|constraints|reasoning)\s*[:\-]", text))
    return hint_count >= 1 and (starts_like_process or "final_answer" not in text)


def validate_generation(raw_output: str | None) -> dict[str, Any]:
    raw = raw_output or ""
    stripped = raw.strip()
    if not stripped:
        return {
            "status": "incomplete_generation",
            "final_answer": "",
            "raw_output": raw,
            "reason": "empty_generation",
            "structured_answer": None,
        }

    for marker in FINAL_MARKERS:
        if marker in stripped:
            answer = stripped.split(marker, 1)[1].strip()
            if answer:
                structured = _extract_json(answer)
                return {
                    "status": "complete",
                    "final_answer": structured.get("answer", answer) if structured else answer,
                    "raw_output": raw,
                    "reason": f"final_marker:{marker}",
                    "structured_answer": structured,
                }

    structured = _extract_json(stripped)
    if structured:
        return {
            "status": "complete",
            "final_answer": str(structured.get("answer", "")).strip(),
            "raw_output": raw,
            "reason": "structured_json",
            "structured_answer": structured,
        }

    if _looks_like_thinking_only(stripped):
        return {
            "status": "incomplete_generation",
            "final_answer": "",
            "raw_output": raw,
            "reason": "thinking_without_final_answer",
            "structured_answer": None,
        }

    return {
        "status": "complete",
        "final_answer": stripped,
        "raw_output": raw,
        "reason": "plain_answer",
        "structured_answer": None,
    }


def structured_vlm_prompt(user_prompt: str) -> str:
    prompt = user_prompt.strip() or "Describe this monitoring event."
    return (
        "Return JSON only:\n"
        "{\n"
        '  "answer": "...",\n'
        '  "decision": "yes|no|unknown",\n'
        '  "confidence": 0.0,\n'
        '  "alert": true|false\n'
        "}\n"
        f"Monitoring question: {prompt}"
    )
