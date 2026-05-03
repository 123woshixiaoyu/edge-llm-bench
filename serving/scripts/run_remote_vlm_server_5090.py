#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8091"))
    uvicorn.run("serving.app.remote_vlm_server:app", host=host, port=port)


if __name__ == "__main__":
    main()
