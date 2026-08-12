#!/bin/bash -l

set -eu

: "${GMAIL_APP_PWD:?GMAIL_APP_PWD must be set in the environment (do not hardcode it here).}"

today=$(date +%Y-%m-%d)
output_dir=/output/$today

docker run \
    --net birds_bird_net \
    -v /bird-detections:/output \
    -v ./email-lists.csv:/app/email-lists.csv:ro \
    -e GMAIL_APP_PWD \
    --entrypoint post-process \
    ghcr.io/edoyango/birds \
        -p 11 $output_dir $output_dir/meta.csv /app/email-lists.csv
