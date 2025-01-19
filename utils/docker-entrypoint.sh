#!/bin/bash

video="${1:-/dev/video0}"
max_frames=${2:-6000}
prefix="${3:-}"
prom_port="${4:-9093}"

extract-birbs \
    --model_path /app/models/yolov5-birbs.rknn \
    --video "$video" \
    --anchors /app/models/anchors.txt \
    --output-dir /output/$(date +%Y-%m-%d) \
    --target rk3588 \
    --max-frames $max_frames \
    --video-name-prefix "$prefix" \
    --metrics-port $prom_port
