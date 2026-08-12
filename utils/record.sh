#!/bin/bash

today=$(date +%Y-%m-%d)
video=$1
prefix=$2

~/rknn-venv/bin/extract-birbs \
	--model_path ~/birds/models/yolov5-birbs.rknn \
	-v "$video" \
	--anchors ~/birds/models/anchors.txt \
	-o /bird-detections/$today -m 35400 -p "$prefix" > "/tmp/${prefix}record.log" 2> "/tmp/${prefix}record.err"
