#!/bin/bash

video2rm=$(rclone lsf google-drive-birds:$(date --date "7 days ago" +"%Y-%m-%d") | grep -E "mkv|mp4")
for v in $video2rm
do
	rclone rm google-drive-birds:$v
done
