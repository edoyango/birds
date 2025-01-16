#!/bin/bash -l

set -eu

today=$(date +%Y-%m-%d)
output_dir=/output/$today

python /app/post-process.py $output_dir $output_dir/meta.csv /app/email-lists.csv
