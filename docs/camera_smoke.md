# Camera Smoke

This is v0.5 prep only. It checks whether a camera can be opened on Jetson and whether one non-sensitive test frame can be captured. It does not connect camera input to the router, does not run VLM inference, and does not introduce TensorRT.

Expected outputs:

- `results/raw/camera_smoke.json`
- `results/figures/camera_smoke.jpg`

Run on Jetson:

```bash
cd /home/rainbow/edge-llm-bench
python3 scripts/camera_smoke.py
```

The script records:

- device path such as `/dev/video0`
- rough camera type, based on the Linux device path
- backend, currently OpenCV V4L2
- resolution
- estimated FPS over a short burst
- first-frame capture latency
- whether frame capture looked stable
- permission or device errors, if any

If the captured frame contains private content, rerun the smoke test while pointing the camera at a neutral scene before committing the image.

## Current Result

The v0.5 prep smoke ran on Jetson but did not capture a frame yet.

Observed state:

- `ok`: false
- video devices: none; no `/dev/video*` path was present
- media devices: `/dev/media0`
- inferred camera type: `csi_or_platform`
- backend attempted: GStreamer
- GStreamer available: true
- `v4l2src` available: true
- `nvarguscamerasrc` available: false
- image saved: no
- error: `no V4L2 device and nvarguscamerasrc is not available`

This is a camera/driver exposure blocker rather than a router blocker. Next camera step is to make the camera appear as `/dev/video0` or install/enable the Jetson Argus GStreamer element so `nvarguscamerasrc` can capture a frame.
