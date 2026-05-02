#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("EDGE_LLM_BENCH_ROOT", Path(__file__).resolve().parents[1]))
LLAMA_CPP = Path(os.environ.get("LLAMA_CPP_DIR", ROOT / "llama.cpp"))
LLAMA_COMPLETION = Path(os.environ.get("LLAMA_COMPLETION", LLAMA_CPP / "build/bin/llama-completion"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/mnt/d/AI/Models"))
DEFAULT_PROMPTS = ROOT / "prompts/benchmark_prompts.jsonl"
DEFAULT_OUT = ROOT / "results/raw/5090_baseline.csv"
DEFAULT_LOG_DIR = ROOT / "results/raw/logs"


def model_path(relative: str) -> str:
    return str(MODEL_ROOT / relative)


def cuda_env(base: dict[str, str]) -> dict[str, str]:
    env = base.copy()
    path_parts = [
        "/usr/local/cuda/bin",
        "/usr/local/cuda-13.2/bin",
        "/usr/lib/wsl/lib",
        env.get("PATH", ""),
    ]
    lib_parts = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-13.2/lib64",
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["PATH"] = ":".join(part for part in path_parts if part)
    env["LD_LIBRARY_PATH"] = ":".join(part for part in lib_parts if part)
    return env

MODELS = [
    {
        "model_key": "gemma4_e2b_q4_k_m",
        "model_name": "Gemma 4 E2B it",
        "quantization": "Q4_K_M",
        "path": model_path("gemma4/E2B-it/gemma-4-E2B-it-Q4_K_M.gguf"),
    },
    {
        "model_key": "gemma4_e2b_q8_0",
        "model_name": "Gemma 4 E2B it",
        "quantization": "Q8_0",
        "path": model_path("gemma4/E2B-it/gemma-4-E2B-it-Q8_0.gguf"),
    },
    {
        "model_key": "gemma4_e4b_q4_k_m",
        "model_name": "Gemma 4 E4B it",
        "quantization": "Q4_K_M",
        "path": model_path("gemma4/E4B-it/gemma-4-E4B-it-Q4_K_M.gguf"),
    },
    {
        "model_key": "qwen35_4b_q4_k_m",
        "model_name": "Qwen3.5 4B",
        "quantization": "Q4_K_M",
        "path": model_path("qwen3.5/Qwen3.5-4B-Q4_K_M.gguf"),
    },
    {
        "model_key": "qwen35_4b_q8_0",
        "model_name": "Qwen3.5 4B",
        "quantization": "Q8_0",
        "path": model_path("qwen3.5/Qwen3.5-4B-Q8_0.gguf"),
    },
    {
        "model_key": "qwen35_08b_q4_k_m",
        "model_name": "Qwen3.5 0.8B",
        "quantization": "Q4_K_M",
        "path": model_path("qwen3.5/Qwen3.5-0.8B-Q4_K_M.gguf"),
    },
]


@dataclass
class GpuSample:
    memory_mb: float | None = None
    power_w: float | None = None
    temp_c: float | None = None


def parse_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value or "")
    return float(match.group(0)) if match else None


def query_gpu() -> GpuSample:
    if not shutil.which("nvidia-smi"):
        return GpuSample()
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return GpuSample()
    if proc.returncode != 0 or not proc.stdout.strip():
        return GpuSample()
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    parts += ["", "", ""]
    return GpuSample(parse_number(parts[0]), parse_number(parts[1]), parse_number(parts[2]))


def parse_tegrastats_line(line: str) -> GpuSample:
    ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    temp_match = re.search(r"(?:gpu|GPU)@([0-9.]+)C", line)
    power_matches = re.findall(r"\b(?:VDD_IN|POM_5V_IN)\s+(\d+)mW", line)
    power_w = float(power_matches[0]) / 1000 if power_matches else None
    return GpuSample(
        memory_mb=float(ram_match.group(1)) if ram_match else None,
        power_w=power_w,
        temp_c=float(temp_match.group(1)) if temp_match else None,
    )


def monitor_tegrastats(stop: threading.Event, samples: list[GpuSample], interval_ms: int) -> None:
    if not shutil.which("tegrastats"):
        return
    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        while not stop.is_set() and proc.stdout:
            line = proc.stdout.readline()
            if line:
                samples.append(parse_tegrastats_line(line))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def monitor_gpu(stop: threading.Event, samples: list[GpuSample], interval_s: float, monitor_kind: str) -> None:
    if monitor_kind == "none":
        return
    if monitor_kind == "tegrastats":
        monitor_tegrastats(stop, samples, max(100, int(interval_s * 1000)))
        return
    while not stop.is_set():
        samples.append(query_gpu())
        stop.wait(interval_s)


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def parse_perf(output: str) -> dict:
    data: dict[str, float | int | str] = {}
    patterns = {
        "load_time_ms": r"load time =\s*([0-9.]+)\s*ms",
        "prompt_eval_time_ms": r"prompt eval time =\s*([0-9.]+)\s*ms",
        "decode_time_ms": r"common_perf_print:\s+eval time =\s*([0-9.]+)\s*ms",
        "total_time_ms": r"total time =\s*([0-9.]+)\s*ms",
        "prompt_tokens": r"prompt eval time =\s*[0-9.]+\s*ms\s*/\s*(\d+)\s*tokens",
        "decode_tokens": r"common_perf_print:\s+eval time =\s*[0-9.]+\s*ms\s*/\s*(\d+)\s*runs",
        "prompt_tps": r"prompt eval time =.*?,\s*([0-9.]+)\s*tokens per second",
        "decode_tps": r"common_perf_print:\s+eval time =.*?,\s*([0-9.]+)\s*tokens per second",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            data[key] = int(value) if key.endswith("tokens") else float(value)
    offload = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", output)
    if offload:
        data["gpu_layers_offloaded"] = int(offload.group(1))
        data["gpu_layers_total"] = int(offload.group(2))
    return data


def model_size_gib(path: Path) -> float:
    return path.stat().st_size / (1024**3)


def run_case(model: dict, prompt: dict, args: argparse.Namespace) -> dict:
    model_path = Path(model["path"])
    log_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model['model_key']}_{prompt['id']}.log"
    log_path = args.log_dir / log_name

    cmd = [
        str(LLAMA_COMPLETION),
        "-m",
        str(model_path),
        "-ngl",
        str(args.gpu_layers),
        "-c",
        str(prompt.get("ctx_size", args.ctx_size)),
        "-n",
        str(prompt.get("max_tokens", args.max_tokens)),
        "--temp",
        str(args.temperature),
        "-p",
        prompt["prompt"],
        "-no-cnv",
        "--no-display-prompt",
        "--perf",
        "--no-warmup",
    ]

    env = cuda_env(os.environ)

    samples: list[GpuSample] = []
    stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpu,
        args=(stop, samples, args.gpu_sample_interval, args.monitor),
        daemon=True,
    )
    monitor.start()
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(LLAMA_CPP),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_s,
    )
    elapsed_s = time.perf_counter() - start
    stop.set()
    monitor.join(timeout=2)
    if args.monitor == "nvidia-smi":
        samples.append(query_gpu())

    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    perf = parse_perf(output)
    mem_values = [s.memory_mb for s in samples if s.memory_mb is not None]
    power_values = [s.power_w for s in samples if s.power_w is not None]
    temp_values = [s.temp_c for s in samples if s.temp_c is not None]

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": args.hardware,
        "backend": args.backend,
        "llama_cpp_commit": args.llama_cpp_commit,
        "model_key": model["model_key"],
        "model_name": model["model_name"],
        "quantization": model["quantization"],
        "model_path": str(model_path),
        "model_size_gib": f"{model_size_gib(model_path):.3f}",
        "prompt_id": prompt["id"],
        "prompt_category": prompt["category"],
        "prompt_language": prompt["language"],
        "context_length": prompt.get("ctx_size", args.ctx_size),
        "requested_decode_tokens": prompt.get("max_tokens", args.max_tokens),
        "return_code": proc.returncode,
        "elapsed_wall_s": f"{elapsed_s:.3f}",
        "peak_gpu_memory_mb": max(mem_values) if mem_values else "",
        "max_gpu_power_w": max(power_values) if power_values else "",
        "max_gpu_temp_c": max(temp_values) if temp_values else "",
        "avg_gpu_power_w": f"{(sum(power_values) / len(power_values)):.2f}" if power_values else "",
        "log_path": str(log_path),
        "command": shlex.join(cmd),
    }
    row.update(perf)
    if proc.returncode != 0:
        row["error_tail"] = "\n".join(output.splitlines()[-20:])
    return row


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))
    all_rows = existing_rows + rows
    fieldnames: list[str] = []
    for row in all_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local llama.cpp CUDA benchmark cases and write CSV results.")
    parser.add_argument("--models", default=",".join(m["model_key"] for m in MODELS), help="Comma-separated model keys.")
    parser.add_argument("--prompt-ids", default="", help="Comma-separated prompt ids. Empty means all prompts.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.2)
    parser.add_argument("--monitor", choices=["nvidia-smi", "tegrastats", "none"], default="nvidia-smi")
    parser.add_argument("--backend", default="llama.cpp CUDA")
    parser.add_argument("--hardware", default="RTX 5090 Laptop 24GB / WSL2 Ubuntu-22.04")
    args = parser.parse_args()

    if not LLAMA_COMPLETION.exists():
        raise SystemExit(f"Missing llama-completion binary: {LLAMA_COMPLETION}")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(LLAMA_CPP),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    args.llama_cpp_commit = commit.stdout.strip() if commit.returncode == 0 else ""

    selected_model_keys = {m.strip() for m in args.models.split(",") if m.strip()}
    selected_prompt_ids = {p.strip() for p in args.prompt_ids.split(",") if p.strip()}
    models = [m for m in MODELS if m["model_key"] in selected_model_keys]
    prompts = load_prompts(args.prompts)
    if selected_prompt_ids:
        prompts = [p for p in prompts if p["id"] in selected_prompt_ids]
    if not models:
        raise SystemExit("No models selected.")
    if not prompts:
        raise SystemExit("No prompts selected.")

    rows = []
    for model in models:
        for prompt in prompts:
            print(f"RUN {model['model_key']} {prompt['id']}", flush=True)
            try:
                row = run_case(model, prompt, args)
            except subprocess.TimeoutExpired as exc:
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "hardware": args.hardware,
                    "backend": args.backend,
                    "llama_cpp_commit": args.llama_cpp_commit,
                    "model_key": model["model_key"],
                    "model_name": model["model_name"],
                    "quantization": model["quantization"],
                    "model_path": model["path"],
                    "model_size_gib": f"{model_size_gib(Path(model['path'])):.3f}",
                    "prompt_id": prompt["id"],
                    "prompt_category": prompt["category"],
                    "prompt_language": prompt["language"],
                    "context_length": prompt.get("ctx_size", args.ctx_size),
                    "requested_decode_tokens": prompt.get("max_tokens", args.max_tokens),
                    "return_code": "timeout",
                    "elapsed_wall_s": args.timeout_s,
                    "error_tail": str(exc),
                }
            rows.append(row)
            write_rows(args.out, [row])
            print(f"DONE {model['model_key']} {prompt['id']} rc={row.get('return_code')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
