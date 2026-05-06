from __future__ import annotations

import base64
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_BACKEND = "smolvlm2_256m_final_line"
FINAL_RE = re.compile(r"(?:FINAL_ANSWER|ANSWER)\s*:\s*(YES|NO|UNKNOWN)\b", re.IGNORECASE)
REASON_RE = re.compile(r"REASON\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


class VerifyEventRequest(BaseModel):
    rule: str
    question: str = "Does this candidate frame show that event?"
    image_path: str | None = None
    image_base64: str | None = None
    roi_name: str | None = None
    proposal_reason: str = ""
    objects: list[str] = Field(default_factory=list)
    max_new_tokens: int = 32
    timeout_s: float = 5.0


class VerifyEventResponse(BaseModel):
    backend: str = DEFAULT_BACKEND
    final_answer: str = ""
    reason: str = ""
    raw_text: str = ""
    parse_success: bool = False
    latency_ms: float = 0.0
    error: str = ""


class FastVerifierRuntime:
    def __init__(self) -> None:
        self.model_id = os.environ.get("SMOLVLM2_MODEL_ID", DEFAULT_MODEL_ID)
        self.backend = os.environ.get("SMOLVLM2_BACKEND_NAME", DEFAULT_BACKEND)
        self.local_files_only = os.environ.get("SMOLVLM2_LOCAL_FILES_ONLY", "1") not in {"0", "false", "False"}
        self.processor: Any = None
        self.model: Any = None
        self.torch: Any = None
        self.device = "unloaded"
        self.load_latency_ms: float | None = None
        self.load_error = ""

    @property
    def ready(self) -> bool:
        return self.model is not None and not self.load_error

    def load(self) -> None:
        if self.ready or self.load_error:
            return
        start = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.torch = torch
            self.processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=self.local_files_only)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    local_files_only=self.local_files_only,
                )
            except TypeError:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    dtype=dtype,
                    local_files_only=self.local_files_only,
                )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            if self.device == "cuda":
                torch.cuda.synchronize()
            self.load_latency_ms = round((time.perf_counter() - start) * 1000, 2)
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            self.load_error = str(exc)

    def generate(self, image_path: Path, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        self.load()
        if not self.ready:
            raise RuntimeError(self.load_error or "SmolVLM2 model is not loaded")
        from PIL import Image

        torch = self.torch
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        text = self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
        return text, latency_ms


runtime = FastVerifierRuntime()
app = FastAPI(title="EdgeLog SmolVLM2 Fast Verifier")


def build_prompt(request: VerifyEventRequest) -> str:
    objects = ", ".join(request.objects) if request.objects else "none"
    roi_text = request.roi_name or "not configured"
    return (
        "Answer in exactly this format on the final line:\n"
        "FINAL_ANSWER: YES|NO|UNKNOWN\n"
        "No other final text.\n"
        "Use YES only if the candidate frame and trigger evidence match the user rule. "
        "Use NO if they do not match. Use UNKNOWN if the image does not provide enough evidence.\n\n"
        f"User rule: {request.rule.strip()}\n"
        f"Candidate trigger reason: {request.proposal_reason.strip() or 'not provided'}\n"
        f"ROI: {roi_text}\n"
        f"Cheap trigger objects: {objects}\n"
        f"Question: {request.question.strip()}\n"
    )


def parse_final_line(text: str) -> tuple[bool, str, str, str]:
    matches = FINAL_RE.findall(text)
    unique = sorted({match.upper() for match in matches})
    if len(unique) != 1:
        stripped = (text or "").strip()
        single_token = re.fullmatch(r"(YES|NO|UNKNOWN)", stripped, flags=re.IGNORECASE)
        if not single_token:
            return False, "", "", "missing or conflicting FINAL_ANSWER line"
        unique = [single_token.group(1).upper()]
    reason_match = REASON_RE.search(text)
    reason = reason_match.group(1).strip().splitlines()[0] if reason_match else "single-token verifier response"
    return True, unique[0], reason[:300], ""


def request_image_path(request: VerifyEventRequest) -> tuple[Path | None, str, bool]:
    if request.image_base64:
        try:
            image_bytes = base64.b64decode(request.image_base64, validate=True)
        except Exception as exc:
            return None, f"invalid image_base64: {exc}", False
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        with tmp:
            tmp.write(image_bytes)
        return Path(tmp.name), "", True
    if request.image_path:
        path = Path(request.image_path)
        if path.exists():
            return path, "", False
        return None, f"image_path does not exist: {request.image_path}", False
    return None, "image_path or image_base64 is required", False


@app.on_event("startup")
def startup_load() -> None:
    if os.environ.get("SMOLVLM2_LOAD_ON_START", "1") not in {"0", "false", "False"}:
        runtime.load()


@app.get("/health")
def health() -> dict[str, Any]:
    runtime.load()
    return {
        "status": "ok" if runtime.ready else "error",
        "model": runtime.model_id,
        "backend": runtime.backend,
        "device": runtime.device,
        "ready": runtime.ready,
        "load_latency_ms": runtime.load_latency_ms,
        "error": runtime.load_error,
    }


@app.post("/v1/verify_event", response_model=VerifyEventResponse)
def verify_event(request: VerifyEventRequest) -> VerifyEventResponse:
    total_start = time.perf_counter()
    image_path, image_error, is_temp = request_image_path(request)
    if image_error or image_path is None:
        return VerifyEventResponse(
            backend=runtime.backend,
            latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
            error=image_error,
        )
    try:
        raw_text, _generate_ms = runtime.generate(image_path, build_prompt(request), max(1, request.max_new_tokens))
        parse_success, answer, reason, parse_error = parse_final_line(raw_text)
        return VerifyEventResponse(
            backend=runtime.backend,
            final_answer=answer,
            reason=reason,
            raw_text=raw_text,
            parse_success=parse_success,
            latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
            error="" if parse_success else parse_error,
        )
    except Exception as exc:
        return VerifyEventResponse(
            backend=runtime.backend,
            latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
            error=str(exc),
        )
    finally:
        if is_temp:
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                pass
