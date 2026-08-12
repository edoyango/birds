"""Tests for birdector.rknn_yolov5.model - the Python side of the FFI
boundary with the compiled librknn_yolov5.so (built from src/yolov5).

The parent package asserts aarch64 Linux and loads that real .so at
import time, neither of which exist here, so both are mocked: platform
detection is patched to report aarch64 Linux, and ctypes.CDLL is patched
to hand back a fake library object whose init/inference/release
functions are plain MagicMocks. This tests the actual contract - what
arguments Python encodes and sends to the C functions, and how Python
reacts to the return codes they hand back - without needing the real
compiled model or target hardware.
"""

import ctypes
import platform
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class FakeCDLL:
    """Stands in for the ctypes.CDLL(librknn_yolov5.so) handle."""

    def __init__(self, path):
        self.path = path
        self.init_yolov5_model = MagicMock(return_value=0)
        self.inference_yolov5_model = MagicMock(return_value=0)
        self.release_yolov5_model = MagicMock(return_value=0)


@pytest.fixture
def rknn(monkeypatch):
    """Import the real birdector.rknn_yolov5 package with its platform
    check and compiled-library load both mocked out, and yield
    (module, fake_cdll_instance)."""
    fake_cdll_holder = {}

    def _fake_cdll(path):
        instance = FakeCDLL(path)
        fake_cdll_holder["instance"] = instance
        return instance

    previous = sys.modules.pop("birdector.rknn_yolov5", None)
    with patch.object(platform, "system", return_value="Linux"), \
         patch.object(platform, "machine", return_value="aarch64"), \
         patch.object(ctypes, "CDLL", _fake_cdll):
        import birdector.rknn_yolov5 as mod
        yield mod, fake_cdll_holder["instance"]

    sys.modules.pop("birdector.rknn_yolov5", None)
    if previous is not None:
        sys.modules["birdector.rknn_yolov5"] = previous


VALID_ANCHORS = np.ones((3, 6))


@pytest.mark.parametrize(
    "npu_cores,expected_mask",
    [
        ([-1], 0),  # RKNN_NPU_CORE_AUTO
        ([0], 1),  # RKNN_NPU_CORE_0
        ([1], 2),  # RKNN_NPU_CORE_1
        ([2], 4),  # RKNN_NPU_CORE_2
        ([0, 1], 3),  # CORE_0 | CORE_1
        ([0, 1, 2], 7),  # CORE_0 | CORE_1 | CORE_2
    ],
)
def test_core_mask_computed_from_npu_cores(rknn, npu_cores, expected_mask):
    mod, _fake_lib = rknn
    m = mod.model("fake.rknn", VALID_ANCHORS, npu_cores=npu_cores)
    assert m._core_mask == expected_mask


def test_npu_cores_must_all_be_int(rknn):
    mod, _fake_lib = rknn
    with pytest.raises(AssertionError):
        mod.model("fake.rknn", VALID_ANCHORS, npu_cores=[0, "1"])


def test_npu_cores_out_of_range_rejected_if_none_are_valid(rknn):
    mod, _fake_lib = rknn
    with pytest.raises(AssertionError):
        mod.model("fake.rknn", VALID_ANCHORS, npu_cores=[5])


def test_npu_cores_out_of_range_rejected_even_with_one_valid_element(rknn):
    """A mix of one valid and one out-of-range core must still be rejected -
    the range check is `all(...)`, not `any(...)`. With `any`, [0, 99]
    would slip past this assertion and fail later with a confusing raw
    KeyError out of _CORE_MASK_MAP instead of the intended message here."""
    mod, _fake_lib = rknn
    with pytest.raises(AssertionError):
        mod.model("fake.rknn", VALID_ANCHORS, npu_cores=[0, 99])


def test_init_passes_encoded_path_and_mask_to_the_c_library(rknn):
    mod, fake_lib = rknn
    mod.model("some/path.rknn", VALID_ANCHORS, npu_cores=[1])

    fake_lib.init_yolov5_model.assert_called_once()
    call_args = fake_lib.init_yolov5_model.call_args.args
    assert call_args[0] == b"some/path.rknn"
    assert call_args[2] == 2  # RKNN_NPU_CORE_1


def test_init_raises_when_the_c_library_reports_failure(rknn):
    mod, fake_lib = rknn
    fake_lib.init_yolov5_model.return_value = -1
    with pytest.raises(AssertionError):
        mod.model("fake.rknn", VALID_ANCHORS, npu_cores=[-1])


@pytest.mark.parametrize(
    "bad_anchors",
    [
        np.ones((2, 6)),  # wrong shape
        [[1] * 6] * 3,  # not a numpy array
    ],
)
def test_anchors_must_be_a_3x6_ndarray(rknn, bad_anchors):
    mod, _fake_lib = rknn
    with pytest.raises(AssertionError):
        mod.model("fake.rknn", bad_anchors, npu_cores=[-1])


def test_release_raises_when_the_c_library_reports_failure(rknn):
    mod, fake_lib = rknn
    m = mod.model("fake.rknn", VALID_ANCHORS, npu_cores=[-1])
    fake_lib.release_yolov5_model.return_value = 1
    with pytest.raises(AssertionError):
        m.release()


def test_inference_passes_batch_size_for_a_single_image(rknn):
    mod, fake_lib = rknn
    m = mod.model("fake.rknn", VALID_ANCHORS, npu_cores=[-1])

    img = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    result = m.inference(img)

    call_args = fake_lib.inference_yolov5_model.call_args.args
    bs_arg = call_args[4]
    assert isinstance(bs_arg, ctypes.c_int32)
    assert bs_arg.value == 1
    assert len(result) == 1
    # a single unbatched (h, w, c) image must come back whole, not sliced
    # down to img[0] (its first row) - see the fix in model.inference()
    assert result[0].src_img.shape == img.shape
    assert np.array_equal(result[0].src_img, img)
    assert np.shares_memory(result[0].src_img, img), "should be a view, not a copy"


def test_inference_passes_batch_size_for_a_batch(rknn):
    mod, fake_lib = rknn
    m = mod.model("fake.rknn", VALID_ANCHORS, npu_cores=[-1])

    batch = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    result = m.inference(batch)

    bs_arg = fake_lib.inference_yolov5_model.call_args.args[4]
    assert bs_arg.value == 4
    assert len(result) == 4
    for i, r in enumerate(result):
        assert np.array_equal(r.src_img, batch[i])
        assert np.shares_memory(r.src_img, batch), "should be a view, not a copy"


def test_inference_raises_when_the_c_library_reports_failure(rknn):
    mod, fake_lib = rknn
    m = mod.model("fake.rknn", VALID_ANCHORS, npu_cores=[-1])
    fake_lib.inference_yolov5_model.return_value = 7

    with pytest.raises(AssertionError):
        m.inference(np.zeros((8, 8, 3), dtype=np.uint8))
