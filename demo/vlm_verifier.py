from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


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


class SmolVLM2FastVerifier:
    """HTTP adapter for the RTX SmolVLM2 FINAL_ANSWER verifier service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8092", *, timeout_s: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def verify(self, rule: dict[str, Any], proposal: dict[str, Any]) -> VerificationResult:
        start = time.perf_counter()
        payload: dict[str, Any] = {
            "rule": rule.get("natural_language_rule") or rule.get("rule_name") or "",
            "question": "Does this candidate frame show that event?",
            "roi_name": proposal.get("roi_name"),
            "proposal_reason": proposal.get("proposal_reason") or "",
            "objects": proposal.get("objects") or [],
            "max_new_tokens": int(rule.get("max_new_tokens") or 32),
            "timeout_s": self.timeout_s,
        }
        if proposal.get("image_base64"):
            payload["image_base64"] = proposal.get("image_base64")
        image_path = proposal.get("keyframe_path")
        if image_path and "image_base64" not in payload:
            path = Path(str(image_path))
            if path.exists():
                try:
                    payload["image_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
                except OSError as exc:
                    return VerificationResult(
                        verifier_backend="smolvlm2_fast",
                        semantic_status="failed",
                        final_answer="",
                        reason="",
                        latency_ms=round((time.perf_counter() - start) * 1000, 2),
                        raw_output="",
                        error=f"could not read keyframe for verifier: {exc}",
                    )
            else:
                payload["image_path"] = str(image_path)

        request = urllib.request.Request(
            f"{self.base_url}/v1/verify_event",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s + 2.0) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            return VerificationResult(
                verifier_backend="smolvlm2_fast",
                semantic_status="failed",
                final_answer="",
                reason="",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                raw_output="",
                error=f"SmolVLM2 verifier unavailable: {exc}",
            )

        error = str(data.get("error") or "")
        final_answer = str(data.get("final_answer") or "").upper()
        semantic_status = final_answer.lower() if final_answer in {"YES", "NO", "UNKNOWN"} else "failed"
        if not data.get("parse_success") and not error:
            error = "SmolVLM2 output did not match FINAL_ANSWER protocol"
        return VerificationResult(
            verifier_backend=str(data.get("backend") or "smolvlm2_fast"),
            semantic_status=semantic_status if not error else "failed",
            final_answer=final_answer if not error else "",
            reason=str(data.get("reason") or ""),
            latency_ms=float(data.get("latency_ms") or round((time.perf_counter() - start) * 1000, 2)),
            raw_output=str(data.get("raw_text") or ""),
            error=error,
        )


def verifier_for_backend(backend: str, *, base_url: str = "http://127.0.0.1:8092") -> MockSemanticVerifier | SmolVLM2FastVerifier:
    if backend == "smolvlm2_fast":
        return SmolVLM2FastVerifier(base_url)
    return MockSemanticVerifier()


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
