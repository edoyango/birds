"""Tests for the shared-memory frame buffer pool used to pass camera
frames between the reader and inference processes without pickling the
full frame on every hop."""

import multiprocessing as mp

import numpy as np
import pytest

from birdector.frame_pool import FrameBufferPool, FramePoolExhausted


@pytest.fixture
def pool():
    p = FrameBufferPool(shape=(48, 64, 3), dtype=np.uint8, n_buffers=4)
    yield p
    p.close()
    p.unlink()


def test_put_and_view_round_trips_data(pool):
    frame = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
    idx = pool.put(frame, timeout=1)
    assert np.array_equal(pool.view(idx), frame)


def test_release_makes_a_buffer_reusable(pool):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    idx = pool.put(frame, timeout=1)
    pool.release(idx)
    # with n_buffers=4, this only succeeds if idx actually came back to the
    # free list - otherwise only 3 buffers would be available
    for _ in range(4):
        pool.put(frame, timeout=1)


def test_exhausted_pool_raises_instead_of_hanging(pool):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    acquired = [pool.put(frame, timeout=1) for _ in range(pool.n_buffers)]
    with pytest.raises(FramePoolExhausted):
        pool.put(frame, timeout=0.2)
    for idx in acquired:
        pool.release(idx)


def _child_checksum(pool, idx, expected_checksum, result_path):
    view = pool.view(idx)
    ok = int(view.sum()) == expected_checksum
    pool.release(idx)
    pool.close()
    result_path.write_text(str(ok))


def test_data_is_visible_across_a_real_process_boundary(pool, tmp_path):
    """The whole point of this pool is transporting frames between
    processes - verify a forked child actually sees what the parent wrote,
    not just that the same-process view works."""
    frame = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
    idx = pool.put(frame, timeout=1)
    result_path = tmp_path / "result.txt"

    ctx = mp.get_context("fork")
    p = ctx.Process(
        target=_child_checksum, args=(pool, idx, int(frame.sum()), result_path)
    )
    p.start()
    p.join(timeout=10)

    assert p.exitcode == 0
    assert result_path.read_text() == "True"
