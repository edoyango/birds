"""Fixed-size pool of shared-memory frame buffers.

Passing a camera frame from the reader process to the inference process
through a plain multiprocessing.Queue means pickling a multi-megabyte numpy
array on every single frame, regardless of whether it ends up in a
recording. This pool lets the reader copy each frame into a preallocated
shared-memory buffer once and hand the *index* of that buffer (a few bytes)
through the queue instead - the inference process reads directly out of
shared memory and releases the buffer for reuse once it has copied what it
needs into its own batch array.
"""

from queue import Empty as _QueueEmpty
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


class FramePoolExhausted(Exception):
    """Raised by FrameBufferPool.put() when no buffer freed up in time."""


class FrameBufferPool:
    def __init__(self, shape: tuple, dtype, n_buffers: int):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.n_buffers = n_buffers
        nbytes = int(np.prod(self.shape)) * self.dtype.itemsize

        self._shms = [
            shared_memory.SharedMemory(create=True, size=nbytes)
            for _ in range(n_buffers)
        ]
        # every buffer starts out free; put()/release() move indices between
        # "checked out" (implicitly - held by whoever last got() one) and
        # here
        self._free_indices = mp.Queue(maxsize=n_buffers)
        for i in range(n_buffers):
            self._free_indices.put(i)

    def _array(self, idx: int) -> np.ndarray:
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self._shms[idx].buf)

    def put(self, array: np.ndarray, timeout: float = None) -> int:
        """Copy `array` into a free buffer and return its index.

        Raises FramePoolExhausted if no buffer becomes free within `timeout`
        seconds (or immediately, if timeout is 0/None and none is free).
        """
        try:
            idx = self._free_indices.get(timeout=timeout)
        except _QueueEmpty:
            raise FramePoolExhausted(
                f"No free frame buffer available within {timeout}s"
            )
        np.copyto(self._array(idx), array)
        return idx

    def view(self, idx: int) -> np.ndarray:
        """Return a zero-copy view of buffer `idx`.

        Only valid until release(idx) is called - copy out anything you
        need to keep before releasing.
        """
        return self._array(idx)

    def release(self, idx: int) -> None:
        """Return a buffer to the free pool once its contents have been read."""
        self._free_indices.put(idx)

    def close(self) -> None:
        """Close this process's shared-memory handles.

        Call once in every process that used the pool (including the one
        that created it) when that process is done with it.
        """
        for shm in self._shms:
            shm.close()

    def unlink(self) -> None:
        """Remove the shared-memory segments from the OS.

        Call exactly once, from the process that created the pool, only
        after every other process using it has finished.
        """
        for shm in self._shms:
            shm.unlink()
