#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "runtime_logs" / "interactive_stack"
STATE_PATH = LOG_DIR / "stack_state.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def http_json(url: str, timeout_s: float = 3.0) -> tuple[bool, dict[str, Any] | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        return True, json.loads(raw), ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, None, str(exc)


def wait_http_json(url: str, timeout_s: float = 60.0) -> tuple[bool, dict[str, Any] | None, str]:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        ok, data, error = http_json(url, timeout_s=3.0)
        if ok:
            return True, data, ""
        last_error = error
        time.sleep(2)
    return False, None, last_error


def pgrep(pattern: str) -> int | None:
    try:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and line.isdigit() and int(line) != os.getpid():
            return int(line)
    return None


def start_local(name: str, command: list[str], log_name: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    log_handle = log_path.open("ab")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=merged_env,
        start_new_session=True,
    )
    time.sleep(1)
    status = "started" if proc.poll() is None else f"exited_{proc.returncode}"
    return {
        "name": name,
        "location": "local",
        "managed": True,
        "pid": proc.pid,
        "status": status,
        "command": command,
        "log": str(log_path),
    }


def ssh_run(jetson_host: str, remote_command: str, timeout_s: float = 20.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-n", "-o", "BatchMode=yes", jetson_host, remote_command],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def remote_pgrep(jetson_host: str, pattern: str) -> int | None:
    command = f"pgrep -f {shlex.quote(pattern)} | head -1"
    try:
        result = ssh_run(jetson_host, command, timeout_s=10)
    except subprocess.SubprocessError:
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return int(line) if line.isdigit() else None


def remote_http_ready(jetson_host: str, url: str) -> bool:
    command = f"wget -qO- {shlex.quote(url)} >/dev/null"
    try:
        return ssh_run(jetson_host, command, timeout_s=8).returncode == 0
    except subprocess.SubprocessError:
        return False


def start_remote(
    *,
    jetson_host: str,
    jetson_project: str,
    name: str,
    script: str,
    log_name: str,
    pid_pattern: str,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    remote_log_dir = f"{jetson_project.rstrip('/')}/runtime_logs/interactive_stack"
    remote_log = f"{remote_log_dir}/{log_name}"
    env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in (env or {}).items())
    inner = (
        f"mkdir -p {shlex.quote(remote_log_dir)} && "
        f"cd {shlex.quote(jetson_project)} && "
        f"setsid env PROJECT_ROOT={shlex.quote(jetson_project)} {env_text} "
        f"bash {shlex.quote(script)} </dev/null >{shlex.quote(remote_log)} 2>&1 & "
        "pid=$!; echo $pid; exit 0"
    )
    command = f"bash -lc {shlex.quote(inner)}"
    stderr = ""
    try:
        result = ssh_run(jetson_host, command, timeout_s=20)
        pid_text = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        pid = int(pid_text) if pid_text.isdigit() else None
        stderr = result.stderr.strip()
        status = "started" if pid else "start_failed"
    except subprocess.TimeoutExpired:
        pid = remote_pgrep(jetson_host, pid_pattern)
        status = "started_after_ssh_timeout" if pid else "start_timeout"
        stderr = "SSH start command timed out after service launch; PID was recovered with pgrep." if pid else "SSH start command timed out and no matching service PID was found."
    return {
        "name": name,
        "location": "jetson",
        "managed": bool(pid),
        "pid": pid,
        "status": status,
        "command": command,
        "log": remote_log,
        "stderr": stderr,
    }


def record_already_ready(name: str, location: str, pid: int | None, health_url: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "location": location,
        "managed": False,
        "pid": pid,
        "status": "already_ready",
        "health_url": health_url,
        "note": "Service was already running; stop_interactive_stack.py will not kill it.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the interactive Jetson/RTX demo stack.")
    parser.add_argument("--jetson-host", default="rainbow@192.168.1.102")
    parser.add_argument("--jetson-project", default="/home/rainbow/edge-llm-bench")
    parser.add_argument("--gateway-url", default="http://192.168.1.102:8000")
    parser.add_argument("--skip-remote-vlm", action="store_true")
    parser.add_argument("--skip-fast-vlm-verifier", action="store_true")
    parser.add_argument("--skip-remote-llm", action="store_true")
    parser.add_argument("--skip-local-llm", action="store_true")
    parser.add_argument("--skip-gateway", action="store_true")
    parser.add_argument("--with-ui", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    services: dict[str, dict[str, Any]] = {}

    if not args.skip_remote_llm:
        ok, _, _ = http_json("http://127.0.0.1:8081/health")
        if ok:
            services["remote_llm"] = record_already_ready(
                "remote_llm",
                "local",
                pgrep("llama-server.*8081"),
                "http://127.0.0.1:8081/health",
            )
        else:
            services["remote_llm"] = start_local(
                "remote_llm",
                ["bash", "serving/scripts/run_remote_llama_server_5090.sh"],
                "remote_llm.log",
            )
    else:
        services["remote_llm"] = {"name": "remote_llm", "status": "skipped", "managed": False}

    if not args.skip_remote_vlm:
        ok, _, _ = http_json("http://127.0.0.1:8091/health")
        if ok:
            services["remote_vlm"] = record_already_ready(
                "remote_vlm",
                "local",
                pgrep("run_remote_vlm_server_5090.py"),
                "http://127.0.0.1:8091/health",
            )
        else:
            python_bin = REPO_ROOT / ".venv" / "bin" / "python"
            python_cmd = str(python_bin) if python_bin.exists() else "python3"
            services["remote_vlm"] = start_local(
                "remote_vlm",
                [python_cmd, "serving/scripts/run_remote_vlm_server_5090.py"],
                "remote_vlm.log",
            )
    else:
        services["remote_vlm"] = {"name": "remote_vlm", "status": "skipped", "managed": False}

    if not args.skip_fast_vlm_verifier:
        ok, _, _ = http_json("http://127.0.0.1:8092/health", timeout_s=8)
        if ok:
            services["fast_vlm_verifier"] = record_already_ready(
                "fast_vlm_verifier",
                "local",
                pgrep("run_fast_vlm_verifier_5090.py"),
                "http://127.0.0.1:8092/health",
            )
        else:
            python_bin = REPO_ROOT / ".venv-vlm" / "bin" / "python"
            python_cmd = str(python_bin) if python_bin.exists() else "python3"
            services["fast_vlm_verifier"] = start_local(
                "fast_vlm_verifier",
                [
                    python_cmd,
                    "serving/scripts/run_fast_vlm_verifier_5090.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8092",
                ],
                "fast_vlm_verifier.log",
            )
    else:
        services["fast_vlm_verifier"] = {"name": "fast_vlm_verifier", "status": "skipped", "managed": False}

    tunnel_forwards: list[str] = []
    if not args.skip_remote_llm:
        tunnel_forwards.extend(["-R", "18081:127.0.0.1:8081"])
    if not args.skip_remote_vlm:
        tunnel_forwards.extend(["-R", "18091:127.0.0.1:8091"])
    if tunnel_forwards:
        existing_tunnel = pgrep("ssh .*18081:127.0.0.1:8081|ssh .*18091:127.0.0.1:8091")
        tunnel_ready = True
        if not args.skip_remote_llm:
            tunnel_ready = tunnel_ready and remote_http_ready(args.jetson_host, "http://127.0.0.1:18081/health")
        if not args.skip_remote_vlm:
            tunnel_ready = tunnel_ready and remote_http_ready(args.jetson_host, "http://127.0.0.1:18091/health")
        if existing_tunnel and tunnel_ready:
            services["ssh_reverse_tunnel"] = record_already_ready("ssh_reverse_tunnel", "local", existing_tunnel)
        else:
            services["ssh_reverse_tunnel"] = start_local(
                "ssh_reverse_tunnel",
                ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", *tunnel_forwards, args.jetson_host],
                "ssh_reverse_tunnel.log",
            )
            if existing_tunnel and not tunnel_ready:
                services["ssh_reverse_tunnel"]["note"] = (
                    "An existing tunnel process was found but Jetson could not reach the forwarded ports, "
                    "so a new managed tunnel was started."
                )
    else:
        services["ssh_reverse_tunnel"] = {"name": "ssh_reverse_tunnel", "status": "skipped", "managed": False}

    if not args.skip_local_llm:
        if remote_http_ready(args.jetson_host, "http://127.0.0.1:8080/health"):
            services["jetson_local_llm"] = record_already_ready(
                "jetson_local_llm",
                "jetson",
                remote_pgrep(args.jetson_host, "llama-server.*8080"),
                "http://127.0.0.1:8080/health",
            )
        else:
            services["jetson_local_llm"] = start_remote(
                jetson_host=args.jetson_host,
                jetson_project=args.jetson_project,
                name="jetson_local_llm",
                script="serving/scripts/run_local_llama_server_jetson.sh",
                log_name="jetson_local_llm.log",
                pid_pattern="llama-server.*8080",
            )
    else:
        services["jetson_local_llm"] = {"name": "jetson_local_llm", "status": "skipped", "managed": False}

    if not args.skip_gateway:
        ok, gateway_data, _ = http_json(f"{args.gateway_url.rstrip('/')}/health")
        if ok:
            services["jetson_gateway"] = record_already_ready(
                "jetson_gateway",
                "jetson",
                remote_pgrep(args.jetson_host, "uvicorn serving.app.main:app"),
                f"{args.gateway_url.rstrip('/')}/health",
            )
            services["jetson_gateway"]["last_health"] = gateway_data
        else:
            services["jetson_gateway"] = start_remote(
                jetson_host=args.jetson_host,
                jetson_project=args.jetson_project,
                name="jetson_gateway",
                script="serving/scripts/run_gateway_jetson.sh",
                log_name="jetson_gateway.log",
                pid_pattern="uvicorn serving.app.main:app",
                env={"EDGE_ROUTER_CONFIG_DIR": "serving/configs_interactive_demo"},
            )
    else:
        services["jetson_gateway"] = {"name": "jetson_gateway", "status": "skipped", "managed": False}

    if args.with_ui:
        services["streamlit_ui"] = start_local(
            "streamlit_ui",
            ["streamlit", "run", "demo/app.py"],
            "streamlit_ui.log",
        )
    else:
        services["streamlit_ui"] = {"name": "streamlit_ui", "status": "skipped", "managed": False}

    print("Waiting for local RTX services...")
    health: dict[str, Any] = {}
    if not args.skip_remote_llm:
        ok, data, error = wait_http_json("http://127.0.0.1:8081/health", timeout_s=90)
        health["remote_llm"] = {"ready": ok, "data": data, "error": error}
    if not args.skip_remote_vlm:
        ok, data, error = wait_http_json("http://127.0.0.1:8091/health", timeout_s=60)
        health["remote_vlm"] = {"ready": ok, "data": data, "error": error}
    if not args.skip_fast_vlm_verifier:
        ok, data, error = wait_http_json("http://127.0.0.1:8092/health", timeout_s=90)
        health["fast_vlm_verifier"] = {"ready": ok and bool((data or {}).get("ready", ok)), "data": data, "error": error}

    print("Waiting for Jetson Gateway...")
    if not args.skip_gateway:
        deadline = time.time() + 120
        gateway_ready = False
        gateway_data: dict[str, Any] | None = None
        gateway_error = ""
        while time.time() < deadline:
            ok, data, error = http_json(f"{args.gateway_url.rstrip('/')}/health", timeout_s=5)
            gateway_data = data
            gateway_error = error
            if ok and data:
                local_ok = args.skip_local_llm or data.get("local_backend_available") is True
                remote_ok = args.skip_remote_llm or data.get("remote_backend_available") is True
                vlm_ok = args.skip_remote_vlm or data.get("vision_remote_vlm_available") is True
                vlm_real_ok = args.skip_remote_vlm or data.get("vision_remote_is_mock") is False
                if local_ok and remote_ok and vlm_ok and vlm_real_ok:
                    gateway_ready = True
                    break
            time.sleep(3)
        health["jetson_gateway"] = {"ready": gateway_ready, "data": gateway_data, "error": gateway_error}

    state = {
        "version": 1,
        "started_at": now_iso(),
        "repo_root": str(REPO_ROOT),
        "jetson_host": args.jetson_host,
        "jetson_project": args.jetson_project,
        "gateway_url": args.gateway_url,
        "log_dir": str(LOG_DIR),
        "services": services,
        "health": health,
    }
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"\nState: {STATE_PATH}")
    print(f"Logs:  {LOG_DIR}")
    for name, service in services.items():
        print(f"{name}: {service.get('status')} pid={service.get('pid')} managed={service.get('managed')}")

    print("\nHealth:")
    for name, result in health.items():
        print(f"- {name}: ready={result.get('ready')}")
        if result.get("data"):
            data = result["data"]
            if name == "jetson_gateway":
                print(f"  local_backend_available={data.get('local_backend_available')}")
                print(f"  remote_backend_available={data.get('remote_backend_available')}")
                print(f"  vision_remote_vlm_available={data.get('vision_remote_vlm_available')}")
                print(f"  vision_remote_is_mock={data.get('vision_remote_is_mock')}")
        if result.get("error") and not result.get("ready"):
            print(f"  error={result.get('error')}")

    failures = [name for name, result in health.items() if not result.get("ready")]
    if failures:
        print("\nSome services are not ready. Check the logs above, confirm model paths exist, and verify SSH/tunnel connectivity.")
        return 1
    print("\nInteractive stack is ready.")
    if not args.with_ui:
        print("Open the UI with: streamlit run demo/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
