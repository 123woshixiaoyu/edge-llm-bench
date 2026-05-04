from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


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
    resize_width: int | None = None
    mode: str = "subprocess_cli_mode"


class VisionCompletionResponse(BaseModel):
    ok: bool
    model: str
    text: str
    latency_ms: float
    error: str = ""
    timings: dict[str, Any] = Field(default_factory=dict)
    resized_width: int | None = None
    resized_height: int | None = None
    mode: str = "subprocess_cli_mode"


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
    if "FINAL_ANSWER:" in cleaned:
        cleaned = cleaned.split("FINAL_ANSWER:", 1)[1].strip()
    if "<channel|>" in cleaned:
        cleaned = cleaned.rsplit("<channel|>", 1)[1].strip()
    return cleaned[:4000]


def resize_image_if_requested(path: Path, resize_width: int | None) -> tuple[Path, int | None, int | None, float, str]:
    if not resize_width or resize_width <= 0:
        return path, None, None, 0.0, ""
    start = time.perf_counter()
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            if width <= resize_width:
                return path, width, height, round((time.perf_counter() - start) * 1000, 2), ""
            new_height = max(1, int(round(height * (resize_width / width))))
            resized = image.convert("RGB").resize((resize_width, new_height))
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            with tmp:
                resized.save(tmp.name, format="JPEG", quality=90)
            return Path(tmp.name), resize_width, new_height, round((time.perf_counter() - start) * 1000, 2), ""
    except Exception as exc:
        return path, None, None, round((time.perf_counter() - start) * 1000, 2), f"resize failed: {exc}"


def write_request_image(request: VisionCompletionRequest) -> tuple[Path | None, str, dict[str, Any], bool]:
    timings: dict[str, Any] = {}
    if request.image_base64:
        start_decode = time.perf_counter()
        suffix = ".jpg"
        if "png" in request.image_mime_type.lower():
            suffix = ".png"
        try:
            image_bytes = base64.b64decode(request.image_base64, validate=True)
        except Exception as exc:
            return None, f"invalid base64 image: {exc}", timings, False
        timings["server_base64_decode_ms"] = round((time.perf_counter() - start_decode) * 1000, 2)
        start_write = time.perf_counter()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with tmp:
            tmp.write(image_bytes)
        timings["server_image_write_ms"] = round((time.perf_counter() - start_write) * 1000, 2)
        return Path(tmp.name), "", timings, True
    if request.image_path:
        path = Path(request.image_path)
        if path.exists():
            timings["server_base64_decode_ms"] = 0.0
            timings["server_image_write_ms"] = 0.0
            return path, "", timings, False
        return None, f"image_path does not exist on remote server: {request.image_path}", timings, False
    return None, "image_base64 or image_path is required", timings, False


def run_vlm(image_path: Path, request: VisionCompletionRequest, timings: dict[str, Any], resized_width: int | None, resized_height: int | None) -> VisionCompletionResponse:
    cfg = settings()
    prompt = (
        "Return only the final answer. "
        "Do not include reasoning, thinking process, analysis steps, constraints, or hidden chain-of-thought.\n\n"
        f"{request.prompt.strip()}\n\n"
        "FINAL_ANSWER:"
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
        timings["server_subprocess_ms"] = round(latency_ms, 2)
        timings["server_total_ms"] = round(
            sum(float(timings.get(key, 0.0)) for key in ("server_base64_decode_ms", "server_image_write_ms", "server_resize_ms", "server_subprocess_ms")),
            2,
        )
        text = clean_generation(proc.stdout, proc.stderr)
        if proc.returncode != 0:
            error = (proc.stderr or proc.stdout).strip()[:1000]
            return VisionCompletionResponse(
                ok=False,
                model=cfg["model_name"],
                text=text,
                latency_ms=round(latency_ms, 2),
                error=error,
                timings=timings,
                resized_width=resized_width,
                resized_height=resized_height,
                mode=request.mode,
            )
        if not text:
            text = clean_generation(proc.stderr, "")
        return VisionCompletionResponse(
            ok=True,
            model=cfg["model_name"],
            text=text,
            latency_ms=round(latency_ms, 2),
            error="",
            timings=timings,
            resized_width=resized_width,
            resized_height=resized_height,
            mode=request.mode,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        timings["server_subprocess_ms"] = round(latency_ms, 2)
        timings["server_total_ms"] = round(
            sum(float(timings.get(key, 0.0)) for key in ("server_base64_decode_ms", "server_image_write_ms", "server_resize_ms", "server_subprocess_ms")),
            2,
        )
        return VisionCompletionResponse(
            ok=False,
            model=cfg["model_name"],
            text="",
            latency_ms=round(latency_ms, 2),
            error=str(exc),
            timings=timings,
            resized_width=resized_width,
            resized_height=resized_height,
            mode=request.mode,
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
    request_start = time.perf_counter()
    image_path, error, timings, temporary = write_request_image(request)
    if error or image_path is None:
        return JSONResponse(
            status_code=400,
            content=VisionCompletionResponse(
                ok=False,
                model=settings()["model_name"],
                text="",
                latency_ms=0.0,
                error=error,
                timings=timings,
                mode=request.mode,
            ).model_dump(),
        )
    resized_temporary = False
    try:
        resized_path, resized_width, resized_height, resize_ms, resize_error = resize_image_if_requested(image_path, request.resize_width)
        timings["server_resize_ms"] = resize_ms
        if resize_error:
            timings["server_resize_error"] = resize_error
        resized_temporary = resized_path != image_path
        result = run_vlm(resized_path, request, timings, resized_width, resized_height)
        result.timings["server_endpoint_total_ms"] = round((time.perf_counter() - request_start) * 1000, 2)
        status_code = 200 if result.ok else 502
        return JSONResponse(status_code=status_code, content=result.model_dump())
    finally:
        if resized_temporary:
            try:
                resized_path.unlink(missing_ok=True)
            except Exception:
                pass
        if temporary:
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                pass
