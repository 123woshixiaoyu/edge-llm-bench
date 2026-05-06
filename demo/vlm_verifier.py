from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any


FINAL_RE = re.compile(r"FINAL_ANSWER\s*:\s*(YES|NO|UNKNOWN)\b", re.IGNORECASE)
REASON_RE = re.compile(r"REASON\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class VerificationResult:
    verifier_backend: str
    semantic_status: str
    final_answer: str
    reason: str
    latency_ms: float
    raw_output: str
    error: str = ""


def parse_final_line_output(text: str) -> VerificationResult:
    start = time.perf_counter()
    match = FINAL_RE.search(text or "")
    if not match:
        return VerificationResult(
            verifier_backend="parser",
            semantic_status="failed",
            final_answer="UNKNOWN",
            reason="No FINAL_ANSWER line found.",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            raw_output=text,
            error="missing_final_answer",
        )
    answer = match.group(1).upper()
    reason_match = REASON_RE.search(text or "")
    reason = reason_match.group(1).strip() if reason_match else ""
    return VerificationResult(
        verifier_backend="parser",
        semantic_status=answer.lower(),
        final_answer=answer,
        reason=reason,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        raw_output=text,
    )


class MockSemanticVerifier:
    """Workflow verifier used until a SmolVLM2 final-line service is integrated."""

    def __init__(self, backend_name: str = "mock_final_line") -> None:
        self.backend_name = backend_name

    def verify(self, rule: dict[str, Any], proposal: dict[str, Any]) -> VerificationResult:
        start = time.perf_counter()
        rule_text = str(rule.get("natural_language_rule", "")).lower()
        proposal_text = " ".join(
            [
                str(proposal.get("proposal_reason", "")),
                str(proposal.get("trigger_type", "")),
                " ".join(str(item) for item in proposal.get("objects", [])),
                str(proposal.get("roi_name", "")),
            ]
        ).lower()

        if "force_no" in rule_text:
            answer = "NO"
            reason = "Mock rule requested a negative verifier result."
        elif "force_unknown" in rule_text:
            answer = "UNKNOWN"
            reason = "Mock rule requested an unknown verifier result."
        elif "force_fail" in rule_text:
            return VerificationResult(
                verifier_backend=self.backend_name,
                semantic_status="failed",
                final_answer="UNKNOWN",
                reason="Mock verifier failure.",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                raw_output="",
                error="mock_failure",
            )
        elif any(term in proposal_text for term in ["person", "roi", "object", "motion", "scene"]):
            answer = "YES"
            reason = "Candidate trigger is consistent with the semantic event rule."
        else:
            answer = "UNKNOWN"
            reason = "Candidate trigger lacks enough evidence."

        raw = f"FINAL_ANSWER: {answer}\nREASON: {reason}"
        return VerificationResult(
            verifier_backend=self.backend_name,
            semantic_status=answer.lower(),
            final_answer=answer,
            reason=reason,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            raw_output=raw,
        )


def verification_to_dict(result: VerificationResult) -> dict[str, Any]:
    return {
        "verifier_backend": result.verifier_backend,
        "semantic_status": result.semantic_status,
        "final_answer": result.final_answer,
        "reason": result.reason,
        "latency_ms": result.latency_ms,
        "raw_output": result.raw_output,
        "error": result.error,
    }
