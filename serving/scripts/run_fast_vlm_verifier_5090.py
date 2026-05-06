#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EdgeLog SmolVLM2 fast semantic verifier server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    parser.add_argument("--backend-name", default="smolvlm2_256m_final_line")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face downloads instead of local cache only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.environ["SMOLVLM2_MODEL_ID"] = args.model_id
    os.environ["SMOLVLM2_BACKEND_NAME"] = args.backend_name
    os.environ["SMOLVLM2_LOCAL_FILES_ONLY"] = "0" if args.allow_download else "1"
    os.environ.setdefault("SMOLVLM2_LOAD_ON_START", "1")
    uvicorn.run("serving.app.fast_vlm_verifier_server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
