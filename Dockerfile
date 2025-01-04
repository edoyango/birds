FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install pkg-config libdrm-dev meson ninja-build git cmake g++ curl -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir 'torch<1.14' 'torchvision<0.15' --index-url https://download.pytorch.org/whl/cpu 'numpy<2'  && \
    pip install --no-cache-dir rknn-toolkit2 && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless
# get rknn driver
RUN curl https://raw.githubusercontent.com/airockchip/rknn-toolkit2/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so -o /usr/lib/librknnrt.so && mkdir /usr/lib64 && ln -s /usr/lib/librknnrt.so /usr/lib64/librknnrt.so

# build ffmpeg with acceleration
RUN mkdir -p /ffmpeg-dev && cd ffmpeg-dev && \
    git clone -b jellyfin-mpp --depth=1 https://github.com/nyanmisaka/mpp.git rkmpp && cd rkmpp && \
    mkdir rkmpp_build && cd rkmpp_build && \
    cmake \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TEST=OFF \
    .. && \
    make -j && make install
RUN cd /ffmpeg-dev && \
    git clone -b jellyfin-rga --depth=1 https://github.com/nyanmisaka/rk-mirrors.git rkrga && cd rkrga && \
    meson setup . rkrga_build \
    --prefix=/usr \
    --libdir=lib \
    --buildtype=release \
    --default-library=shared \
    -Dcpp_args=-fpermissive \
    -Dlibdrm=false \
    -Dlibrga_demo=false && \
    meson configure rkrga_build && \
    ninja -C rkrga_build install
RUN cd /ffmpeg-dev && \
    git clone --depth=1 https://github.com/nyanmisaka/ffmpeg-rockchip.git ffmpeg && \
    cd ffmpeg && \
    ./configure --prefix=/usr --enable-gpl --enable-version3 --enable-libdrm --enable-rkmpp --enable-rkrga && \
    make && make install

FROM python:3.10-slim AS runtime

# get runtime packages
RUN apt-get update && apt-get install libdrm2 libgcc-s1 libstdc++6 -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# /usr/local will contain all pip-installed packages
COPY --from=builder /usr/local /usr/local
COPY --from=builder /lib/librga.so.2 /lib/librga.so.2
COPY --from=builder /lib/librga.so.2.1.0 /lib/librga.so.2.1.0
COPY --from=builder /lib/aarch64-linux-gnu/librockchip_mpp.so.0 /lib/aarch64-linux-gnu/librockchip_mpp.so.0
COPY --from=builder /lib/aarch64-linux-gnu/librockchip_mpp.so.1 /lib/aarch64-linux-gnu/librockchip_mpp.so.1
COPY --from=builder /usr/bin/ffmpeg /usr/bin/ffmpeg
COPY --from=builder /usr/lib/librknnrt.so /usr/lib/librknnrt.so
COPY --from=builder /usr/lib64/librknnrt.so /usr/lib64/librknnrt.so

COPY ./scripts /app
COPY ./models /app/models
WORKDIR /app

CMD ["python", "/app/extract-birds.py", "--model_path", "/app/models/yolov5-birbs.rknn", "--video_path", "/dev/video0", "--anchors", "/app/anchors.txt"]