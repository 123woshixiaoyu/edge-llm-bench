from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/mnt/d/AI/Models/gemma4/E2B-it/gemma-4-E2B-it-Q4_K_M.gguf"
DEFAULT_MMPROJ = "/mnt/d/AI/Models/gemma4/E2B-it/mmproj-F16.gguf"
DEFAULT_CLI = str(REPO_ROOT / "llama.cpp/build/bin/llama-mtmd-cli")


class VisionCompletionRequest(BaseModel):
    image_base64: str | None = None
    image_mime_type: str = "image/jpeg"
    image_path: str | None = None
    prompt: str = "Describe the image in one short sentence."
    max_tokens: int = 128


class VisionCompletionResponse(BaseModel):
    ok: bool
    model: str
    text: str
    latency_ms: float
    error: str = ""


def settings() -> dict[str, str]:
    return {
        "model": os.environ.get("VLM_MODEL_PATH", DEFAULT_MODEL),
        "mmproj": os.environ.get("VLM_MMPROJ_PATH", DEFAULT_MMPROJ),
        "cli": os.environ.get("VLM_CLI_PATH", os.environ.get("LLAMA_MTMD_CLI", DEFAULT_CLI)),
        "model_name": os.environ.get("VLM_MODEL_NAME", "gemma4_e2b_it_q4_mmproj"),
        "ctx_size": os.environ.get("VLM_CTX_SIZE", "4096"),
        "gpu_layers": os.environ.get("VLM_GPU_LAYERS", "all"),
    }


def clean_generation(stdout: str, stderr: str) -> str:
    text = stdout.strip()
    if not text:
        return ""
    lines: list[str] = []
    skip_prefixes = (
        "ggml_",
        "common_",
        "llama_",
        "clip_",
        "load_",
        "print_info:",
        "sched_",
        "alloc_",
        "mtmd_",
        "main:",
        "WARN:",
        "---",
        "encoding image",
        "decoding image",
        "image decoded",
        "image slice",
        "warmup:",
    )
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(skip_prefixes):
            continue
        lines.append(stripped)
    cleaned = "\n".join(lines).strip()
    if "<|channel>final" in cleaned:
        cleaned = cleaned.split("<|channel>final", 1)[1].strip()
    if "<|channel>thought" in cleaned and "<|channel>final" not in cleaned:
        cleaned = cleaned.replace("<|channel>thought", "").strip()
    return cleaned[:4000]


def write_request_image(request: VisionCompletionRequest) -> tuple[Path | None, str]:
    if request.image_base64:
        suffix = ".jpg"
        if "png" in request.image_mime_type.lower():
            suffix = ".png"
        try:
            image_bytes = base64.b64decode(request.image_base64, validate=True)
        except Exception as exc:
            return None, f"invalid base64 image: {exc}"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with tmp:
            tmp.write(image_bytes)
        return Path(tmp.name), ""
    if request.image_path:
        path = Path(request.image_path)
        if path.exists():
            return path, ""
        return None, f"image_path does not exist on remote server: {request.image_path}"
    return None, "image_base64 or image_path is required"


def run_vlm(image_path: Path, request: VisionCompletionRequest) -> VisionCompletionResponse:
    cfg = settings()
    prompt = (
        "Answer directly in natural language. Do not mention hidden reasoning. "
        + request.prompt.strip()
    )
    cmd = [
        cfg["cli"],
        "-m",
        cfg["model"],
        "--mmproj",
        cfg["mmproj"],
        "--image",
        str(image_path),
        "-p",
        prompt,
        "-n",
        str(request.max_tokens),
        "--temp",
        "0",
        "-ngl",
        cfg["gpu_layers"],
        "-c",
        cfg["ctx_size"],
        "--no-warmup",
        "--jinja",
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(os.environ.get("VLM_SUBPROCESS_TIMEOUT_S", "180")),
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text = clean_generation(proc.stdout, proc.stderr)
        if proc.returncode != 0:
            error = (proc.stderr or proc.stdout).strip()[:1000]
            return VisionCompletionResponse(
                ok=False,
                model=cfg["model_name"],
                text=text,
                latency_ms=round(latency_ms, 2),
                error=error,
            )
        if not text:
            text = clean_generation(proc.stderr, "")
        return VisionCompletionResponse(
            ok=True,
            model=cfg["model_name"],
            text=text,
            latency_ms=round(latency_ms, 2),
            error="",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return VisionCompletionResponse(
            ok=False,
            model=cfg["model_name"],
            text="",
            latency_ms=round(latency_ms, 2),
            error=str(exc),
        )


app = FastAPI(
    title="Remote VLM Backend",
    version="0.5b",
    description="Small HTTP wrapper around llama.cpp multimodal CLI for Project 2 v0.5b.",
)


@app.get("/health")
def health() -> dict:
    cfg = settings()
    return {
        "status": "ok",
        "model": cfg["model_name"],
        "model_path_exists": Path(cfg["model"]).exists(),
        "mmproj_path_exists": Path(cfg["mmproj"]).exists(),
        "cli_path_exists": Path(cfg["cli"]).exists(),
    }


@app.post("/v1/vision/completions", response_model=VisionCompletionResponse)
def vision_completions(request: VisionCompletionRequest):
    image_path, error = write_request_image(request)
    temporary = bool(request.image_base64 and image_path)
    if error or image_path is None:
        return JSONResponse(
            status_code=400,
            content=VisionCompletionResponse(
                ok=False,
                model=settings()["model_name"],
                text="",
                latency_ms=0.0,
                error=error,
            ).model_dump(),
        )
    try:
        result = run_vlm(image_path, request)
        status_code = 200 if result.ok else 502
        return JSONResponse(status_code=status_code, content=result.model_dump())
    finally:
        if temporary:
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                pass
