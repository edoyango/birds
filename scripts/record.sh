#!/bin/bash

# record.sh

# records video for 1h and then launches the yolo analyser

dev="${1:-/dev/video0}"
duration="${2:-3570}"
prefix="$3"

set -eu

starttime=$(date +"%H-%M-%S")
output_vid=${prefix}$(basename $dev)-$starttime.mkv
today=$(date +"%Y-%m-%d")

ffmpeg -y -t $duration -f v4l2 -input_format yuyv422 -video_size 1280x720 -i $dev -vf scale=864:486 -c:v h264_v4l2m2m -b:v 2000K $output_vid

#birds_dir=/home/edwardy/birds
#mkdir -p $birds_dir/output
#docker run --gpus 0 -v $(pwd):$(pwd) -v $birds_dir:$birds_dir -w $(pwd) ultralytics/ultralytics:8.2.81 $birds_dir/scripts/extract-birbs.py -m $birds_dir/models/yolov10-birbs.pt -o $birds_dir/outputs -v $output_vid -c $birds_dir/models/yolov8-cls-birbs.pt

#rclone mkdir google-drive-birds:$today
rclone move $output_vid gdrive:birds/$today
