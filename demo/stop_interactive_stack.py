#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "runtime_logs" / "interactive_stack" / "stack_state.json"


def local_pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_local(pid: int | None) -> str:
    if not pid:
        return "missing_pid"
    if not local_pid_alive(pid):
        return "not_running"
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return "not_running"
    time.sleep(2)
    if local_pid_alive(pid):
        return "still_running"
    return "stopped"


def stop_remote(jetson_host: str, pid: int | None) -> str:
    if not pid:
        return "missing_pid"
    command = f"kill {int(pid)} 2>/dev/null || true; sleep 2; kill -0 {int(pid)} 2>/dev/null"
    try:
        result = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", jetson_host, command],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.SubprocessError as exc:
        return f"ssh_error: {exc}"
    return "still_running" if result.returncode == 0 else "stopped"


def main() -> int:
    if not STATE_PATH.exists():
        print(f"No stack state found at {STATE_PATH}")
        return 1
    state: dict[str, Any] = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    jetson_host = state.get("jetson_host", "rainbow@192.168.1.102")
    services = state.get("services", {})

    print(f"Stopping services from {STATE_PATH}")
    for name, service in services.items():
        if not service.get("managed"):
            service["stop_status"] = "not_managed"
            print(f"- {name}: not managed by this stack; leaving it alone")
            continue
        pid = service.get("pid")
        location = service.get("location")
        if location == "local":
            status = stop_local(pid)
        elif location == "jetson":
            status = stop_remote(jetson_host, pid)
        else:
            status = "skipped"
        service["stop_status"] = status
        service["status"] = "stopped" if status in {"stopped", "not_running"} else service.get("status")
        print(f"- {name}: pid={pid} stop_status={status}")

    state["stopped_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("Updated stack state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
