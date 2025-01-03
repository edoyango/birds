#!/bin/bash

today=$(date +%Y-%m-%d)
video=$1
prefix=$2

~/rknn-venv/bin/python ~/birds/scripts/extract-birbs.py \
	--model_path ~/birds/models/yolov5-birbs.rknn \
	-v "$video" \
	--target rk3588 \
	--anchors ~/birds/scripts/anchors.txt \
	-o /bird-detections/$today -m 35400 -p $prefix > "/tmp/${prefix}record.log" 2> "/tmp/${prefix}record.err"
