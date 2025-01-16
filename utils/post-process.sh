#!/bin/bash -l

set -eu

today=$(date +%Y-%m-%d)
output_dir=/output/$today

docker run \
    --net birds_bird_net \
    -v /bird-detections:/output \
    -v ./email-lists.csv:/app/email-lists.csv:ro \
    --entrypoint python \
    ghcr.io/edoyango/birds \
        /app/post-process.py $output_dir $output_dir/meta.csv /app/email-lists.csv
