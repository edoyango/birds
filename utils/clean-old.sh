#!/bin/bash

today="$(date +%Y-%m-%d)"
reference_date="${1:-$today}"
prev_date=$(date --date "$reference_date 7 days ago" +%Y-%m-%d)

rm $(find /bird-detections/$prev_date/ -name '*.mp4' | grep -v compressed)
