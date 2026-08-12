"""Tests for the pure/testable logic in birdector.extract_birbs.

inference_worker is exercised as a real multiprocessing.Process (via the
fake rknn_yolov5.model from conftest) rather than called in-process, since
that's how it actually runs in production and is what the shared-memory
frame pool / queue-based termination signaling depend on.
"""

import multiprocessing as mp
import time
import urllib.request

import numpy as np
import pytest

from birdector.frame_pool import FrameBufferPool
from tests.conftest import CLASS_NAMES, FakeDetection, FakeObjectDetectResult, NullModel


def test_count_detections_zero_fills_undetected_species(extract_birbs):
    infres = FakeObjectDetectResult(
        detections=[FakeDetection(0), FakeDetection(0), FakeDetection(2)],
        class_names=CLASS_NAMES,
    )
    counts = extract_birbs.count_detections(infres)
    assert counts[CLASS_NAMES[0]] == 2
    assert counts[CLASS_NAMES[2]] == 1
    # every other species should be present with an explicit 0, not missing
    for name in CLASS_NAMES:
        assert name in counts
    assert sum(counts.values()) == 3


def _run_inference_worker(extract_birbs, fake_rknn_yolov5, tmp_path, model_cls, frames, batch_size=4, port=19990):
    """Spawn a real inference_worker process fed `frames` through a
    FrameBufferPool, then the DONE sentinel. Returns (exitcode, out_dir)."""
    fake_rknn_yolov5.model = model_cls

    anchors_path = tmp_path / "anchors.txt"
    anchors_path.write_text("\n".join(["1.0"] * 18))
    out_dir = tmp_path / "out"

    shape = frames[0].shape if frames else (16, 16, 3)
    pool = FrameBufferPool(shape=shape, dtype=np.uint8, n_buffers=max(len(frames), 1) + 1)
    frame_queue = mp.Queue(maxsize=200)
    for frame in frames:
        idx = pool.put(frame, timeout=5)
        frame_queue.put(idx)
    frame_queue.put("DONE")

    p = mp.Process(
        target=extract_birbs.inference_worker,
        args=(
            "fake-model-path", anchors_path, batch_size, "testvideo", port,
            frame_queue, pool, 10, 50, out_dir, shape[1], shape[0], "test-", [-1],
        ),
    )
    p.start()
    p.join(timeout=20)
    pool.close()
    pool.unlink()
    return p.exitcode, out_dir


def test_final_partial_batch_contains_no_padding_frames(extract_birbs, fake_rknn_yolov5, tmp_path):
    batch_shapes_path = tmp_path / "batch_shapes.log"

    class RecordingModel(NullModel):
        def inference(self, frames):
            with open(batch_shapes_path, "a") as f:
                f.write(f"{frames.shape[0]}\n")
            for i in range(frames.shape[0]):
                assert frames[i].any(), f"batch index {i} is an all-zero padding frame"
            return [
                FakeObjectDetectResult(class_names=CLASS_NAMES, src_img=frames[i])
                for i in range(frames.shape[0])
            ]

    nframes = 21  # not a multiple of batch_size=8 -> exercises the tail batch
    frames = [np.ones((16, 16, 3), dtype=np.uint8) for _ in range(nframes)]

    exitcode, _ = _run_inference_worker(
        extract_birbs, fake_rknn_yolov5, tmp_path, RecordingModel, frames, batch_size=8, port=19991
    )
    assert exitcode == 0

    batch_shapes = [int(x) for x in batch_shapes_path.read_text().split()]
    assert batch_shapes == [8, 8, 5]
    assert sum(batch_shapes) == nframes


def test_done_with_no_frames_ever_loaded_does_not_crash(extract_birbs, fake_rknn_yolov5, tmp_path):
    calls_path = tmp_path / "calls.log"

    class CountingModel(NullModel):
        def inference(self, frames):
            calls_path.write_text("called")
            return []

    exitcode, _ = _run_inference_worker(
        extract_birbs, fake_rknn_yolov5, tmp_path, CountingModel, frames=[], batch_size=8, port=19992
    )
    assert exitcode == 0
    assert not calls_path.exists(), "inference() should never run on an empty/all-None batch"


def test_frame_queue_stall_exits_cleanly_instead_of_hanging(extract_birbs, fake_rknn_yolov5, tmp_path):
    fake_rknn_yolov5.model = NullModel
    anchors_path = tmp_path / "anchors.txt"
    anchors_path.write_text("\n".join(["1.0"] * 18))

    pool = FrameBufferPool(shape=(16, 16, 3), dtype=np.uint8, n_buffers=4)
    frame_queue = mp.Queue(maxsize=4)  # nothing is ever put on this queue

    p = mp.Process(
        target=extract_birbs.inference_worker,
        args=(
            "fake-model-path", anchors_path, 4, "testvideo", 19993,
            frame_queue, pool, 10, 50, tmp_path / "out", 16, 16, "test-", [-1],
        ),
    )
    p.start()
    p.join(timeout=20)  # must finish well before the process would be killed externally
    pool.close()
    pool.unlink()

    assert p.exitcode == 0
    assert not p.is_alive()


def test_bird_species_count_reflects_whole_batch_not_last_frame(extract_birbs, fake_rknn_yolov5, tmp_path):
    class OneSparrowPerFrameModel(NullModel):
        def inference(self, frames):
            return [
                FakeObjectDetectResult(
                    detections=[FakeDetection(0)],  # class 0 = one sparrow-equivalent
                    class_names=CLASS_NAMES,
                    src_img=frames[i],
                )
                for i in range(frames.shape[0])
            ]

    fake_rknn_yolov5.model = OneSparrowPerFrameModel
    anchors_path = tmp_path / "anchors.txt"
    anchors_path.write_text("\n".join(["1.0"] * 18))

    batch_size = 6
    pool = FrameBufferPool(shape=(16, 16, 3), dtype=np.uint8, n_buffers=200)
    frame_queue = mp.Queue(maxsize=200)
    frame = np.ones((16, 16, 3), dtype=np.uint8)
    for _ in range(batch_size):
        idx = pool.put(frame, timeout=5)
        frame_queue.put(idx)
    # deliberately don't send DONE yet - we want the worker alive to scrape /metrics

    port = 19989
    p = mp.Process(
        target=extract_birbs.inference_worker,
        args=(
            "fake-model-path", anchors_path, batch_size, "testvideo", port,
            frame_queue, pool, 10, 50, tmp_path / "out", 16, 16, "test-", [-1],
        ),
    )
    p.start()
    try:
        body = None
        for _ in range(30):
            time.sleep(1)
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2).read().decode()
                if f'bird_species_count{{species="{CLASS_NAMES[0]}"' in body:
                    break
            except Exception:
                continue
        assert body is not None, "never got a response from the metrics endpoint"
        # every one of the 6 frames contributed exactly one detection of
        # CLASS_NAMES[0] -> the gauge should read 6.0, not 1.0 (which is
        # what only reflecting the batch's last frame would produce)
        assert f'bird_species_count{{species="{CLASS_NAMES[0]}",video="testvideo"}} 6.0' in body
    finally:
        frame_queue.put("DONE")
        p.join(timeout=15)
        pool.close()
        pool.unlink()
