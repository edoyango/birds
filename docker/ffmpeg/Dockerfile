FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install libnvidia-encode-535 ffmpeg -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
