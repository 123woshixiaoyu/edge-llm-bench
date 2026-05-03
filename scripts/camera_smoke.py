#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def classify_camera(device: Path) -> str:
    name = device.name
    sys_path = Path("/sys/class/video4linux") / name / "device"
    try:
        real_path = str(sys_path.resolve()).lower()
    except Exception:
        real_path = ""
    if "usb" in real_path:
        return "usb"
    if "platform" in real_path or "tegra" in real_path or "csi" in real_path:
        return "csi_or_platform"
    return "unknown"


def list_devices() -> list[Path]:
    return sorted(Path("/dev").glob("video*"))


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def gst_element_available(name: str) -> bool:
    if not command_exists("gst-inspect-1.0"):
        return False
    proc = subprocess.run(
        ["gst-inspect-1.0", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    )
    return proc.returncode == 0 and "No such element" not in proc.stdout


def _argus_capture_command(out_image: Path) -> list[str]:
    return [
        "gst-launch-1.0",
        "-q",
        "nvarguscamerasrc",
        "sensor-id=0",
        "num-buffers=1",
        "!",
        "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1",
        "!",
        "nvvidconv",
        "!",
        "video/x-raw,format=I420",
        "!",
        "jpegenc",
        "!",
        "filesink",
        f"location={out_image}",
    ]


def _v4l2_capture_command(device: Path, out_image: Path) -> list[str]:
    return [
        "gst-launch-1.0",
        "-q",
        "v4l2src",
        f"device={device}",
        "num-buffers=1",
        "!",
        "videoconvert",
        "!",
        "jpegenc",
        "!",
        "filesink",
        f"location={out_image}",
    ]


def capture_with_gstreamer(
    device: Path | None, out_image: Path, prefer_argus: bool = False
) -> tuple[bool, str, float | None, str]:
    if not command_exists("gst-launch-1.0"):
        return False, "gst-launch-1.0 is not available", None, "GStreamer"
    out_image.parent.mkdir(parents=True, exist_ok=True)
    if prefer_argus and gst_element_available("nvarguscamerasrc"):
        cmd = _argus_capture_command(out_image)
        backend = "GStreamer Argus"
    elif device is not None:
        cmd = _v4l2_capture_command(device, out_image)
        backend = "GStreamer V4L2"
    elif gst_element_available("nvarguscamerasrc"):
        cmd = _argus_capture_command(out_image)
        backend = "GStreamer Argus"
    else:
        return False, "no V4L2 device and nvarguscamerasrc is not available", None, "GStreamer"

    start = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
    latency_ms = (time.perf_counter() - start) * 1000
    if proc.returncode != 0:
        return False, proc.stdout.strip()[:500], latency_ms, backend
    if out_image.exists() and out_image.stat().st_size > 0:
        return True, "", latency_ms, backend
    message = proc.stdout.strip()[:500] or "pipeline completed but produced no image"
    return False, message, latency_ms, backend


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one camera frame on Jetson for v0.5 prep.")
    parser.add_argument("--device", default=None, help="Optional device path, for example /dev/video0.")
    parser.add_argument("--out-json", type=Path, default=Path("results/raw/camera_smoke.json"))
    parser.add_argument("--out-image", type=Path, default=Path("results/figures/camera_smoke.jpg"))
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    result = {
        "ok": False,
        "device_path": args.device,
        "camera_type": "unknown",
        "backend": "OpenCV V4L2",
        "resolution": None,
        "fps_estimate": None,
        "capture_latency_ms": None,
        "stable": False,
        "permission_issue": False,
        "image_path": str(args.out_image),
        "video_devices": [str(path) for path in list_devices()],
        "media_devices": [str(path) for path in sorted(Path("/dev").glob("media*"))],
        "gstreamer_available": command_exists("gst-launch-1.0"),
        "v4l2src_available": gst_element_available("v4l2src"),
        "nvarguscamerasrc_available": gst_element_available("nvarguscamerasrc"),
        "error": "",
    }

    try:
        import cv2
    except Exception as exc:
        cv2 = None
        result["error"] = f"opencv import failed: {exc}"

    devices = [Path(args.device)] if args.device else list_devices()
    if not devices and cv2 is None:
        ok, error, latency_ms, backend = capture_with_gstreamer(
            None, args.out_image, prefer_argus=True
        )
        result.update(
            {
                "ok": ok,
                "backend": backend,
                "camera_type": "csi_or_platform" if result["media_devices"] else "unknown",
                "capture_latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
                "resolution": "1280x720" if backend == "GStreamer Argus" else None,
                "stable": ok,
                "error": error or result["error"],
            }
        )
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if ok else 1

    if not devices:
        result["error"] = "no /dev/video* devices found"
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 1

    last_error = ""
    for device in devices:
        result["device_path"] = str(device)
        result["camera_type"] = classify_camera(device)
        if cv2 is None:
            prefer_argus = result["camera_type"] == "csi_or_platform"
            ok, error, latency_ms, backend = capture_with_gstreamer(
                device, args.out_image, prefer_argus=prefer_argus
            )
            if ok:
                result.update(
                    {
                        "ok": True,
                        "backend": backend,
                        "capture_latency_ms": round(latency_ms or 0.0, 2),
                        "resolution": "1280x720" if backend == "GStreamer Argus" else None,
                        "stable": True,
                        "error": "",
                    }
                )
                break
            last_error = error
            continue
        cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
        if not cap.isOpened():
            last_error = f"failed to open {device}"
            if not device.exists():
                last_error += "; device path does not exist"
            continue

        frames_read = 0
        first_frame = None
        first_latency_ms = None
        start = time.perf_counter()
        for _ in range(max(1, args.frames)):
            frame_start = time.perf_counter()
            ok, frame = cap.read()
            latency_ms = (time.perf_counter() - frame_start) * 1000
            if ok and frame is not None:
                frames_read += 1
                if first_frame is None:
                    first_frame = frame
                    first_latency_ms = latency_ms
        elapsed_s = time.perf_counter() - start
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        if first_frame is None:
            last_error = f"opened {device} but no frames were read"
            continue

        args.out_image.parent.mkdir(parents=True, exist_ok=True)
        image_ok = cv2.imwrite(str(args.out_image), first_frame)
        result.update(
            {
                "ok": bool(image_ok),
                "resolution": f"{width}x{height}",
                "fps_estimate": round(frames_read / elapsed_s, 2) if elapsed_s > 0 else None,
                "capture_latency_ms": round(first_latency_ms or 0.0, 2),
                "stable": frames_read >= max(1, min(args.frames, 5)),
                "error": "" if image_ok else "failed to write image",
            }
        )
        break
    else:
        result["error"] = last_error
        result["permission_issue"] = "permission" in last_error.lower()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
