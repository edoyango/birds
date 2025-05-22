#!/bin/bash

video="${1:-/dev/video0}"
max_frames=${2:-6000}
prefix="${3:-}"
prom_port="${4:-9093}"
npu_cores="${5:-'-1'}"
batch_size="${6:-1}"

extract-birbs \
    --model_path /app/models/yolov5-birbs.rknn \
    --video "$video" \
    --anchors /app/models/anchors.txt \
    --output-dir /output/$(date +%Y-%m-%d) \
    --max-frames $max_frames \
    --video-name-prefix "$prefix" \
    --metrics-port $prom_port \
    --npu-cores $npu_cores \
    --batch-size $batch_size
