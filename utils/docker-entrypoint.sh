#!/bin/bash

video="${1:-/dev/video0}"
max_frames=${2:-6000}
prefix="${3:-}"

python /app/extract-birbs.py \
    --model_path /app/models/yolov5-birbs.rknn \
    --video "$video" \
    --anchors /app/anchors.txt \
    --output-dir /output/$(date +%Y-%m-%d) \
    --target rk3588 \
    --max-frames $max_frames \
    --video-name-prefix "$prefix"
