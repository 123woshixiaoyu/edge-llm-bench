from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DecisionLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []

    def log(self, row: dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self._lock:
            self._rows.append(row)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self._rows)
        total = len(rows)
        local_count = sum(1 for row in rows if row.get("route") == "local")
        remote_count = sum(1 for row in rows if row.get("route") == "remote")
        reject_count = sum(1 for row in rows if row.get("route") == "reject")
        backend_error_count = sum(1 for row in rows if row.get("status") == "backend_error")
        latencies = [float(row["total_latency_ms"]) for row in rows if row.get("total_latency_ms") is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return {
            "total_requests": total,
            "local_count": local_count,
            "remote_count": remote_count,
            "reject_count": reject_count,
            "backend_error_count": backend_error_count,
            "avg_total_latency_ms": round(avg_latency, 2),
        }
