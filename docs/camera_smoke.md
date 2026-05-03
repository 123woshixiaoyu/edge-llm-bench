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
- backend, currently OpenCV V4L2 for regular V4L2 devices or GStreamer Argus for Jetson CSI cameras
- resolution
- estimated FPS over a short burst
- first-frame capture latency
- whether frame capture looked stable
- permission or device errors, if any

If the captured frame contains private content, rerun the smoke test while pointing the camera at a neutral scene before committing the image.

## Current Result

The v0.5 prep smoke now captures a frame successfully on Jetson after enabling the IMX219-C CSI overlay for the 24-pin CSI header.

Successful output files:

- `results/raw/camera_smoke_after_jetson_io.json`
- `results/figures/camera_smoke_after_jetson_io.jpg`

Observed successful state:

- `ok`: true
- device path: `/dev/video0`
- media device: `/dev/media0`
- camera type: `csi_or_platform`
- backend: `GStreamer Argus`
- resolution: `1280x720`
- first-frame capture latency: about `1249 ms`
- stable: true
- permission issue: false
- error: none

Manual GStreamer validation also succeeded with `nvarguscamerasrc`, and Argus listed IMX219 sensor modes including `3280x2464`, `1920x1080`, and `1280x720`.

## Troubleshooting Notes

Initial smoke attempts failed because the Jetson camera stack was present only partially:

- no `/dev/video*` path was present
- `/dev/media0` was present
- `nvarguscamerasrc` was initially unavailable until the Jetson GStreamer packages were installed
- after installing the plugin, `nvarguscamerasrc` still reported `No cameras available`

The root cause was configuration rather than a bad camera. The IMX219 module was plugged into CAM1, which corresponds to the `Camera IMX219-C` overlay on the `Jetson 24pin CSI Connector`. Running Jetson-IO without a header number targeted the default `Jetson 40pin Header`, so the camera overlay was not applied. The correct configuration path is:

```bash
sudo /opt/nvidia/jetson-io/config-by-hardware.py -l
sudo /opt/nvidia/jetson-io/config-by-hardware.py -n 2="Camera IMX219-C"
sudo reboot
```

After reboot, `/dev/video0` appeared and Argus capture succeeded. The camera is ready for v0.5 prep, but it is still not connected to the router and no VLM path has been added.
