from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CameraFrame:
    ok: bool
    image_path: str
    device_path: str | None
    camera_type: str
    backend: str
    resolution: str | None
    capture_latency_ms: float | None
    stable: bool
    permission_issue: bool
    video_devices: list[str]
    media_devices: list[str]
    gstreamer_available: bool
    v4l2src_available: bool
    nvarguscamerasrc_available: bool
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def list_video_devices() -> list[Path]:
    return sorted(Path("/dev").glob("video*"))


def list_media_devices() -> list[Path]:
    return sorted(Path("/dev").glob("media*"))


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


def classify_camera(device: Path | None) -> str:
    if device is None:
        return "csi_or_platform" if list_media_devices() else "unknown"
    sys_path = Path("/sys/class/video4linux") / device.name / "device"
    try:
        real_path = str(sys_path.resolve()).lower()
    except Exception:
        real_path = ""
    if "usb" in real_path:
        return "usb"
    if "platform" in real_path or "tegra" in real_path or "csi" in real_path:
        return "csi_or_platform"
    return "unknown"


def _argus_capture_command(
    out_image: Path,
    *,
    sensor_id: int,
    width: int,
    height: int,
    framerate: int,
) -> list[str]:
    return [
        "gst-launch-1.0",
        "-q",
        "nvarguscamerasrc",
        f"sensor-id={sensor_id}",
        "num-buffers=1",
        "!",
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1",
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


def capture_frame(
    out_image: Path,
    *,
    device: str | Path | None = None,
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    framerate: int = 30,
    prefer_argus: bool | None = None,
    timeout_s: int = 20,
) -> CameraFrame:
    out_image = Path(out_image)
    out_image.parent.mkdir(parents=True, exist_ok=True)

    video_devices = list_video_devices()
    media_devices = list_media_devices()
    selected_device = Path(device) if device else (video_devices[0] if video_devices else None)
    camera_type = classify_camera(selected_device)
    gstreamer_available = command_exists("gst-launch-1.0")
    v4l2_available = gst_element_available("v4l2src")
    argus_available = gst_element_available("nvarguscamerasrc")

    if not gstreamer_available:
        return CameraFrame(
            ok=False,
            image_path=str(out_image),
            device_path=str(selected_device) if selected_device else None,
            camera_type=camera_type,
            backend="GStreamer",
            resolution=None,
            capture_latency_ms=None,
            stable=False,
            permission_issue=False,
            video_devices=[str(path) for path in video_devices],
            media_devices=[str(path) for path in media_devices],
            gstreamer_available=False,
            v4l2src_available=v4l2_available,
            nvarguscamerasrc_available=argus_available,
            error="gst-launch-1.0 is not available",
        )

    if prefer_argus is None:
        prefer_argus = camera_type == "csi_or_platform"

    if prefer_argus and argus_available:
        cmd = _argus_capture_command(
            out_image,
            sensor_id=sensor_id,
            width=width,
            height=height,
            framerate=framerate,
        )
        backend = "GStreamer Argus"
        resolution = f"{width}x{height}"
    elif selected_device and v4l2_available:
        cmd = _v4l2_capture_command(selected_device, out_image)
        backend = "GStreamer V4L2"
        resolution = None
    else:
        error = "no usable camera capture backend"
        if prefer_argus and not argus_available:
            error = "nvarguscamerasrc is not available"
        elif not selected_device:
            error = "no /dev/video* devices found"
        return CameraFrame(
            ok=False,
            image_path=str(out_image),
            device_path=str(selected_device) if selected_device else None,
            camera_type=camera_type,
            backend="GStreamer",
            resolution=None,
            capture_latency_ms=None,
            stable=False,
            permission_issue="permission" in error.lower(),
            video_devices=[str(path) for path in video_devices],
            media_devices=[str(path) for path in media_devices],
            gstreamer_available=gstreamer_available,
            v4l2src_available=v4l2_available,
            nvarguscamerasrc_available=argus_available,
            error=error,
        )

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    ok = proc.returncode == 0 and out_image.exists() and out_image.stat().st_size > 0
    error = ""
    if not ok:
        error = proc.stdout.strip()[:500] or "pipeline completed but produced no image"

    return CameraFrame(
        ok=ok,
        image_path=str(out_image),
        device_path=str(selected_device) if selected_device else None,
        camera_type=camera_type,
        backend=backend,
        resolution=resolution,
        capture_latency_ms=round(latency_ms, 2),
        stable=ok,
        permission_issue="permission" in error.lower(),
        video_devices=[str(path) for path in video_devices],
        media_devices=[str(path) for path in media_devices],
        gstreamer_available=gstreamer_available,
        v4l2src_available=v4l2_available,
        nvarguscamerasrc_available=argus_available,
        error=error,
    )
