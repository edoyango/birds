# birds

Detects birds in camera feeds on an RK3588 board (developed on an Orange Pi 5 Pro) and keeps the clips that contain
them. A YOLO26 model runs INT8 on the NPU; frames are letterboxed by the RGA and clips are encoded with the hardware
video encoder.

## Layout

- `birdector/`: the Python package.
  - `extract-birbs` reads a camera or video, batches frames through the model, and writes clips with detections plus a
    `meta.csv`. It serves per-species counts to Prometheus.
  - `post-process` picks the best clips of a day and emails a summary.
  - `rknpu-exporter` serves NPU load to Prometheus.
- `src/`: the C/C++ library the package calls through ctypes: model loading, RGA letterbox, and YOLO26 decoding
  (`src/yolo26`).
- `training/`: training on an Intel Arc GPU, export, INT8 conversion with rknn-toolkit2, and accuracy checks. See
  [training/README.md](training/README.md).
- `models/`: the deployed `.rknn` model (Git LFS).
- `docker-compose.yml`: the two camera services, the NPU and node exporters, Prometheus and Grafana.

## Running on the board

```bash
git lfs pull                      # fetch models/*.rknn
docker build -t ghcr.io/edoyango/birds:latest .   # or pull the image named in .env
docker compose up -d
```

`.env` sets the image (`BIRDS_CONTAINER`). Each camera service takes, in order: the video device, the maximum number
of frames (0 for no limit), a file name prefix, the Prometheus port, the NPU cores and the batch size. The batch size
must equal the one the model was converted with (24 for the model in `models/`). `BIRDS_CONF` sets the score threshold
(default 0.25).

Clips go to `/bird-detections/<date>/`. `utils/post-process.sh` runs the daily summary and `utils/clean-old.sh` deletes
uncompressed clips older than a week.
