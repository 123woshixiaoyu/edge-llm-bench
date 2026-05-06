#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "runtime_logs" / "interactive_stack" / "stack_state.json"


def http_json(url: str, timeout_s: float = 3.0) -> tuple[bool, dict[str, Any] | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        return True, json.loads(raw), ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, None, str(exc)


def local_pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def remote_pid_alive(jetson_host: str, pid: int | None) -> bool:
    if not pid:
        return False
    result = subprocess.run(
        ["ssh", "-n", "-o", "BatchMode=yes", jetson_host, f"kill -0 {int(pid)}"],
        capture_output=True,
        text=True,
        timeout=8,
    )
    return result.returncode == 0


def main() -> int:
    if not STATE_PATH.exists():
        print(f"No stack state found at {STATE_PATH}")
        print("Start the stack with: python3 demo/run_interactive_stack.py")
        return 1
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    jetson_host = state.get("jetson_host", "rainbow@192.168.1.102")
    gateway_url = state.get("gateway_url", "http://192.168.1.102:8000").rstrip("/")

    print(f"State: {STATE_PATH}")
    print(f"Started at: {state.get('started_at')}")
    print("\nProcesses:")
    for name, service in state.get("services", {}).items():
        status = service.get("status")
        managed = service.get("managed")
        pid = service.get("pid")
        location = service.get("location")
        alive = False
        if location == "local":
            alive = local_pid_alive(pid)
        elif location == "jetson":
            try:
                alive = remote_pid_alive(jetson_host, pid)
            except (subprocess.SubprocessError, OSError):
                alive = False
        else:
            alive = False
        print(f"- {name}: status={status} managed={managed} pid={pid} alive={alive}")

    endpoints = {
        "remote_llm": "http://127.0.0.1:8081/health",
        "remote_vlm": "http://127.0.0.1:8091/health",
        "fast_vlm_verifier": "http://127.0.0.1:8092/health",
        "jetson_gateway": f"{gateway_url}/health",
    }
    print("\nHealth endpoints:")
    all_ready = True
    inactive: list[str] = []
    for name, url in endpoints.items():
        service_status = state.get("services", {}).get(name, {}).get("status")
        if service_status in {"skipped", "stopped"}:
            print(f"- {name}: {service_status}")
            inactive.append(f"{name}:{service_status}")
            continue
        ok, data, error = http_json(url, timeout_s=5)
        endpoint_ready = ok
        print(f"- {name}: ready={ok} url={url}")
        if data and name == "jetson_gateway":
            local_backend_available = data.get("local_backend_available")
            remote_backend_available = data.get("remote_backend_available")
            vision_remote_vlm_available = data.get("vision_remote_vlm_available")
            vision_remote_is_mock = data.get("vision_remote_is_mock")
            print(f"  local_backend_available={local_backend_available}")
            print(f"  remote_backend_available={remote_backend_available}")
            print(f"  vision_remote_vlm_available={vision_remote_vlm_available}")
            print(f"  vision_remote_is_mock={vision_remote_is_mock}")
            services = state.get("services", {})
            remote_llm_skipped = services.get("remote_llm", {}).get("status") == "skipped"
            remote_vlm_skipped = services.get("remote_vlm", {}).get("status") == "skipped"
            endpoint_ready = (
                ok
                and local_backend_available is True
                and (remote_llm_skipped or remote_backend_available is True)
                and (remote_vlm_skipped or vision_remote_vlm_available is True)
                and (remote_vlm_skipped or vision_remote_is_mock is False)
            )
        elif data:
            print(f"  status={data.get('status', 'ok')} model={data.get('model')} backend={data.get('backend')}")
            if "ready" in data:
                print(f"  ready={data.get('ready')} device={data.get('device')}")
                endpoint_ready = ok and data.get("ready") is not False
        if error and not ok:
            print(f"  error={error}")
        if ok and name == "jetson_gateway" and not endpoint_ready:
            print("  error=gateway is reachable, but one or more required backends are unavailable")
        all_ready = all_ready and endpoint_ready

    if not all_ready:
        print("\nNot all services are ready. Check runtime_logs/interactive_stack/, SSH connectivity, and model paths.")
        return 1
    if inactive:
        print(f"\nCheck completed. Inactive services are expected from the saved state: {', '.join(inactive)}")
        return 0
    print("\nInteractive stack looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
